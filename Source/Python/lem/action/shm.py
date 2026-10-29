# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Action SHM — Windows named shared memory for the action buffer.

Python creates and writes; UE's ModelBridgeComponent reads.
Follows the same mmap pattern as tokenizer/transport/cpu_reader.py.
"""

from __future__ import annotations

import logging
import mmap
import struct

log = logging.getLogger(__name__)

# SHM-level header (separate from the 20-byte action frame header)
ACTION_MAGIC = 0x4C454D41      # "LEMA"
ACTION_VERSION = 2
ACTION_SHM_HEADER_SIZE = 64    # Reserved for magic, version, future fields

# Offsets within SHM header
OFF_SHM_MAGIC = 0              # uint32
OFF_SHM_VERSION = 4            # uint32


class ActionSHM:
    """Windows named shared memory region for action buffer output."""

    def __init__(self, shm_name: str, buffer_size: int):
        self.shm_name = shm_name
        self.buffer_size = buffer_size
        self._total_size = ACTION_SHM_HEADER_SIZE + buffer_size
        self._shm: mmap.mmap | None = None

    def open(self) -> bool:
        """Create/open SHM for writing. Returns True on success."""
        try:
            self._shm = mmap.mmap(
                -1, self._total_size,
                self.shm_name,
                access=mmap.ACCESS_WRITE,
            )
            # Zero entire region
            self._shm[:] = b'\x00' * self._total_size
            # Write magic + version
            struct.pack_into('<II', self._shm, OFF_SHM_MAGIC,
                             ACTION_MAGIC, ACTION_VERSION)
            return True
        except Exception as e:
            log.warning("Cannot create action SHM '%s': %s", self.shm_name, e)
            self._shm = None
            return False

    @property
    def buf(self) -> mmap.mmap | None:
        return self._shm

    @property
    def data_offset(self) -> int:
        """Offset within mmap where action frame data begins."""
        return ACTION_SHM_HEADER_SIZE

    @property
    def total_size(self) -> int:
        return self._total_size

    def close(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
