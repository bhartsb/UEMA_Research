# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
GPU Reader — D3D12→CUDA zero-copy vision import.

Imports named D3D12 shared texture handles from the UE pipeline's parity ring
into CUDA via mipmapped arrays. Each frame: wait fence → copy array→linear
→ DLPack → PyTorch tensors. ~0.5ms total (GPU-only copy, no CPU round trip).

Requires:
1. Pipeline must export named D3D12 handles (gpu_interop in manifest)
2. cuda_d3d12_interop.pyd built and importable
3. NVIDIA driver with CUDA support on the same GPU as D3D12
"""

import logging
import os
from typing import Optional, Tuple

from ..config import TokenizerConfig

log = logging.getLogger(__name__)


class GPUReader:
    """D3D12 → CUDA zero-copy vision texture reader."""

    def __init__(self, config: TokenizerConfig):
        self.available = False
        self.config = config
        self._interop = None

        if not config.gpu_interop_available:
            log.info("GPU reader: not available (no gpu_interop in manifest)")
            return

        try:
            import cuda_d3d12_interop as interop
            self._interop = interop
        except ImportError:
            log.warning("GPU reader: cuda_d3d12_interop.pyd not found")
            return

        try:
            # Initialize CUDA on the same GPU as D3D12 (matched by LUID)
            interop.init_cuda(config.adapter_luid_low, config.adapter_luid_high)

            # Import the shared D3D12 fence (named handle from UE pipeline)
            interop.import_d3d12_fence_named(config.fence_handle_name)

            # Import all 8 parity textures (2 slots × 4 textures)
            pano_alloc = int(config.texture_alloc_sizes.get("pano", 0))
            fovea_alloc = int(config.texture_alloc_sizes.get("fovea", 0))

            # Suppress C-level printf from import_d3d12_texture unless DEBUG
            _quiet = not log.isEnabledFor(logging.DEBUG)
            if _quiet:
                _devnull = os.open(os.devnull, os.O_WRONLY)
                _old_stdout = os.dup(1)
                _old_stderr = os.dup(2)
                os.dup2(_devnull, 1)
                os.dup2(_devnull, 2)
            try:
                imported = 0
                for slot_name, handles in config.parity_handle_names.items():
                    for tex_key, handle_name in handles.items():
                        if "pano" in tex_key:
                            w, h, alloc = config.panorama_width, config.panorama_height, pano_alloc
                        else:
                            w, h, alloc = config.fovea_width, config.fovea_height, fovea_alloc

                        tex_id = f"{slot_name}_{tex_key}"
                        interop.import_d3d12_texture(tex_id, handle_name, alloc, w, h)
                        imported += 1
                        log.debug("Imported texture: %s (%dx%d)", tex_id, w, h)
            finally:
                if _quiet:
                    os.dup2(_old_stdout, 1)
                    os.dup2(_old_stderr, 2)
                    os.close(_old_stdout)
                    os.close(_old_stderr)
                    os.close(_devnull)

            self.available = True
            log.info("GPU reader: initialized (%d parity textures imported)", imported)

        except Exception as e:
            log.error("GPU reader: initialization failed: %s", e)
            self._cleanup_partial()

    def read_vision_textures(self, slot: int, fence_value: int
                              ) -> Optional[Tuple]:
        """
        Read 4 vision textures from the parity ring via D3D12→CUDA zero-copy.

        Args:
            slot: Parity slot index (0 or 1), typically frame_number % 2
            fence_value: D3D12 fence value to wait on (= VisionFrameNumber)

        Returns:
            Tuple of 4 tensors (pano_l, pano_r, fovea_l, fovea_r) each [H, W, 4]
            uint8 on cuda:0, or None if read failed.
        """
        if not self.available or self._interop is None:
            return None

        try:
            import torch

            interop = self._interop
            slot_name = f"slot_{slot}"

            # 1. Wait for D3D12 fence (blocks CPU until GPU copy is done)
            interop.wait_fence(fence_value)

            # 2. Copy all 4 textures from CUDA array to linear device memory
            #    Runs on CUDA stream 0; DLPack + downstream ops are also stream 0,
            #    so no explicit synchronize() needed (stream-ordered).
            tex_keys = ["pano_left", "pano_right", "fovea_left", "fovea_right"]
            for key in tex_keys:
                interop.copy_texture_to_linear(f"{slot_name}_{key}")

            # 3. Wrap linear buffers as PyTorch tensors via DLPack (zero-copy)
            tensors = []
            for key in tex_keys:
                capsule = interop.to_dlpack_texture(f"{slot_name}_{key}")
                tensor = torch.from_dlpack(capsule)
                tensors.append(tensor)

            return tuple(tensors)

        except Exception as e:
            log.error("GPU reader: read failed: %s", e)
            return None

    def _cleanup_partial(self):
        """Best-effort cleanup after init failure."""
        if self._interop:
            try:
                self._interop.cleanup()
            except Exception:
                pass
        self.available = False

    def close(self):
        """Release all imported CUDA/D3D12 resources."""
        if self._interop:
            try:
                self._interop.cleanup()
            except Exception as e:
                log.warning("GPU reader cleanup error: %s", e)
            self._interop = None
        self.available = False
