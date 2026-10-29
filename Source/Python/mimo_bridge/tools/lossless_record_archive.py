# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Archive completed evidence without discarding any original bytes.

Preparation never removes the source. The receipt binds both complete files;
removal, when authorized, is a separate filesystem operation. Restoration checks
the complete original SHA-256 before publishing a regular, loadable source file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import time


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_archive(archive: Path, receipt: dict) -> None:
    if archive.stat().st_size != receipt['archive_bytes'] or digest(archive) != receipt['archive_sha256']:
        raise ValueError('Compressed evidence differs from its receipt')
    result = hashlib.sha256()
    size = 0
    with lzma.open(archive, 'rb') as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
            size += len(chunk)
    if size != receipt['original_bytes'] or result.hexdigest() != receipt['original_sha256']:
        raise ValueError('Archive does not reproduce the complete original evidence')


def prepare_archive(source: Path) -> dict:
    source = source.resolve(strict=True)
    archive = source.with_name(source.name + '.xz')
    receipt_path = source.with_name(source.name + '.archive.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        verify_archive(archive, receipt)
        if source.stat().st_size != receipt['original_bytes'] or digest(source) != receipt['original_sha256']:
            raise ValueError('Original evidence changed since archival')
        return receipt
    if archive.exists():
        raise FileExistsError(f'Archive without a verified receipt: {archive}')
    original_stat = source.stat()
    original_digest = digest(source)
    # Reserve for the worst case (incompressible input), plus concurrent live saves.
    if shutil.disk_usage(source.parent).free < original_stat.st_size + 4 * 2**30:
        raise OSError('Insufficient space to archive while retaining the original and recording reserve')
    temporary = archive.with_name(archive.name + '.preparing')
    with source.open('rb') as incoming, lzma.open(temporary, 'wb',
            filters=[dict(id=lzma.FILTER_LZMA2, preset=0, dict_size=64 * 2**20)]) as target:
        shutil.copyfileobj(incoming, target, 1024 * 1024)
    receipt = dict(schema='complete_evidence_archive_v1', original=str(source), archive=str(archive),
                   original_bytes=original_stat.st_size, original_sha256=original_digest,
                   original_mtime_ns=original_stat.st_mtime_ns,
                   archive_bytes=temporary.stat().st_size, archive_sha256=digest(temporary),
                   prepared_at=time.time(), all_original_bytes_preserved=True)
    verify_archive(temporary, receipt)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns) or digest(source) != original_digest:
        raise ValueError('Source changed while archiving; retain the source')
    os.replace(temporary, archive)
    receipt_tmp = receipt_path.with_name(receipt_path.name + '.preparing')
    receipt_tmp.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    os.replace(receipt_tmp, receipt_path)
    return receipt


def restore_archive(receipt_path: Path, destination: Path | None = None) -> Path:
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    archive = receipt_path.with_name(receipt_path.name.removesuffix('.archive.json') + '.xz')
    destination = Path(destination) if destination else receipt_path.with_name(receipt_path.name.removesuffix('.archive.json'))
    if destination.exists():
        if digest(destination) == receipt['original_sha256']:
            return destination
        raise FileExistsError('Restoration will not overwrite a different existing file')
    verify_archive(archive, receipt)
    if shutil.disk_usage(destination.parent).free < receipt['original_bytes'] + 64 * 2**20:
        raise OSError('Insufficient space for complete restoration')
    temporary = destination.with_name(destination.name + '.restoring')
    with lzma.open(archive, 'rb') as incoming, temporary.open('xb') as target:
        shutil.copyfileobj(incoming, target, 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    if temporary.stat().st_size != receipt['original_bytes'] or digest(temporary) != receipt['original_sha256']:
        raise ValueError('Restored original failed verification; destination was not published')
    # Exclusive publication: a concurrently created original must never be replaced.
    os.rename(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare-completed-study')
    prepare.add_argument('--study', type=Path, required=True)
    prepare.add_argument('--workers', type=int, choices=(1, 2), default=1)
    restore = sub.add_parser('restore')
    restore.add_argument('--receipt', type=Path, required=True)
    restore.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.command == 'restore':
        print(restore_archive(args.receipt, args.destination), flush=True)
        return
    root = args.study.resolve(strict=True)
    if json.loads((root / 'status.json').read_text(encoding='utf-8'))['stage'] != 'complete':
        raise ValueError('Only a completed study may be archived')
    paths = sorted(root.glob('*/event_*_branches.pt'))
    if not paths or any(not path.resolve().is_relative_to(root) or path.is_symlink() for path in paths):
        raise ValueError('No completed records, or a record escapes the study')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for receipt in executor.map(prepare_archive, paths):
            print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
