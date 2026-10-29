# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Lossless compression of native torch.save ZIP archives.

Entries retain their exact names and uncompressed bytes, so ordinary torch.load
continues to work. Use inside checkpoint staging, before atomic publication.
The original remains intact until every compressed entry has been reread and
its SHA-256 verified. This changes storage only, never model precision/history.
"""
import hashlib
import os
from pathlib import Path
import time
import zipfile


class _SequentialArchive:
    # ZIP's normal seek-back header update flushes a buffered file for every
    # tensor. A streaming ZIP uses standard data descriptors instead and keeps
    # writes sequential, important for thousands of small runtime histories.
    def __init__(self,stream):self.stream=stream
    def write(self,data):return self.stream.write(data)
    def flush(self):return self.stream.flush()


def compact_torch_archive(path, *, level=1):
    path=Path(path)
    temporary=path.with_name(path.name+'.compressing')
    if temporary.exists():raise FileExistsError(temporary)
    original_bytes=path.stat().st_size;started=time.monotonic();hashes={}
    try:
        with zipfile.ZipFile(path,'r') as incoming,open(temporary,'wb',buffering=1024*1024) as destination,\
                zipfile.ZipFile(_SequentialArchive(destination),'w',
                compression=zipfile.ZIP_DEFLATED,compresslevel=level,allowZip64=True) as outgoing:
            names=incoming.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate native checkpoint archive entries')
            if not any(name.endswith('/data.pkl') for name in names):
                raise ValueError('Not a native torch.save ZIP archive')
            for entry in incoming.infolist():
                info=zipfile.ZipInfo(entry.filename,entry.date_time)
                info.compress_type=zipfile.ZIP_DEFLATED if entry.file_size>=4096 else zipfile.ZIP_STORED
                info.external_attr=entry.external_attr
                info._compresslevel=level
                digest=hashlib.sha256()
                with incoming.open(entry,'r') as source,outgoing.open(info,'w',force_zip64=True) as target:
                    while block:=source.read(1024*1024):
                        digest.update(block);target.write(block)
                hashes[entry.filename]=digest.hexdigest()
        with zipfile.ZipFile(temporary,'r') as archived:
            if archived.namelist()!=names:raise ValueError('Compaction changed native archive entry names')
            for name in names:
                digest=hashlib.sha256()
                with archived.open(name,'r') as source:
                    while block:=source.read(1024*1024):digest.update(block)
                if digest.hexdigest()!=hashes[name]:raise ValueError('Compaction changed native tensor/storage bytes')
        compressed_bytes=temporary.stat().st_size
        used=compressed_bytes<original_bytes
        if used:os.replace(temporary,path)
        else:temporary.unlink()
        return dict(contract='native_torch_zip_lossless_v1',original_bytes=original_bytes,
            stored_bytes=path.stat().st_size,compressed=used,level=level,
            seconds=time.monotonic()-started,entry_count=len(names),
            uncompressed_entry_sha256=hashes,verified='Every entry name and complete uncompressed byte sequence')
    except BaseException:
        if temporary.exists():temporary.unlink()
        raise
