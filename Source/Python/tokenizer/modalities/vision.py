# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Vision tokenizer — patch extraction + angular position encoding.

Handles 4 textures (pano L/R 2520x1560, fovea L/R 512x512) as BGRA8 bytes
or BGRA8 GPU tensors (from D3D12→CUDA zero-copy).
Converts to RGB float32, extracts 16x16 patches, computes angular positions.

Panorama angular LUT is precomputed once (static). Fovea angles are
recomputed per frame based on gaze direction.

Total output: 32,506 patches per frame (15,229 pano + 1,024 fovea) x 2 eyes.
"""

import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..config import TokenizerConfig
from .vision_kernels import load_vision_kernels

log = logging.getLogger(__name__)


class VisionTokenizer:
    """Stateful vision tokenizer with precomputed angular LUTs."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        device = config.device

        ps = config.patch_size  # 16

        # Panorama: crop to patch-aligned dimensions
        self.pano_crop_w = (config.panorama_width // ps) * ps    # 2512
        self.pano_crop_h = (config.panorama_height // ps) * ps   # 1552
        self.pano_cols = self.pano_crop_w // ps   # 157
        self.pano_rows = self.pano_crop_h // ps   # 97
        self.pano_patches = self.pano_cols * self.pano_rows  # 15,229

        # Fovea: clean division
        self.fovea_cols = config.fovea_width // ps   # 32
        self.fovea_rows = config.fovea_height // ps  # 32
        self.fovea_patches = self.fovea_cols * self.fovea_rows  # 1,024

        self.total_patches = (self.pano_patches + self.fovea_patches) * 2  # 32,506

        # Precompute panorama angular LUT [pano_patches, 2] (static)
        self.pano_angular_lut = self._precompute_pano_angles(device)

        # Precompute fovea patch center UVs [fovea_patches] (static grid, gaze offset dynamic)
        self._fovea_local_az, self._fovea_local_el = self._precompute_fovea_grid(device)

        # Precompute eye_id and source_id tags
        self.eye_left_pano = torch.zeros(self.pano_patches, dtype=torch.int32, device=device)
        self.eye_right_pano = torch.ones(self.pano_patches, dtype=torch.int32, device=device)
        self.eye_left_fovea = torch.zeros(self.fovea_patches, dtype=torch.int32, device=device)
        self.eye_right_fovea = torch.ones(self.fovea_patches, dtype=torch.int32, device=device)
        self.source_pano = torch.zeros(self.pano_patches, dtype=torch.int32, device=device)
        self.source_fovea = torch.ones(self.fovea_patches, dtype=torch.int32, device=device)

        # Pre-concatenated static tensors (eye_id and source_id never change)
        self._eye_id_cat = torch.cat([
            self.eye_left_pano, self.eye_right_pano,
            self.eye_left_fovea, self.eye_right_fovea,
        ], dim=0)
        self._source_id_cat = torch.cat([
            self.source_pano, self.source_pano,
            self.source_fovea, self.source_fovea,
        ], dim=0)

        # Pre-concatenated pano angular LUT (used for both eyes)
        self._pano_angular_cat = torch.cat([
            self.pano_angular_lut, self.pano_angular_lut,
        ], dim=0)  # [2*pano_patches, 2]

        # Try to load CUDA vision kernels
        self._cuda_kernels = load_vision_kernels()

    def _precompute_pano_angles(self, device: str) -> torch.Tensor:
        """
        Compute angular position (azimuth, elevation) in radians for each
        panorama patch center. Replicates UVToDirection() from PanoramaStitch.usf.

        Panorama covers:
          Horizontal: -105deg to +105deg (210deg total), left-to-right
          Vertical:   +65deg (top) to -65deg (bottom)
        """
        ps = self.config.patch_size
        fov_h = self.config.panorama_fov_h_deg
        fov_v = self.config.panorama_fov_v_deg
        orig_w = self.config.panorama_width
        orig_h = self.config.panorama_height

        angles = np.zeros((self.pano_patches, 2), dtype=np.float32)

        for row in range(self.pano_rows):
            for col in range(self.pano_cols):
                idx = row * self.pano_cols + col

                # Patch center pixel (in original pano coordinates)
                cx = col * ps + ps / 2
                cy = row * ps + ps / 2

                # Map to UV [0, 1]
                u = (cx + 0.5) / orig_w
                v = (cy + 0.5) / orig_h

                # UV to angle (matching shader)
                half_h = fov_h / 2.0
                half_v = fov_v / 2.0
                az_deg = -half_h + u * fov_h   # -105 to +105
                el_deg = half_v - v * fov_v    # +65 to -65

                angles[idx, 0] = math.radians(az_deg)
                angles[idx, 1] = math.radians(el_deg)

        return torch.from_numpy(angles).to(device)

    def _precompute_fovea_grid(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute local azimuth/elevation offsets for fovea patch centers.
        These are relative to gaze — add gaze direction per frame.
        """
        ps = self.config.patch_size
        fov = self.config.fovea_fov_deg
        w = self.config.fovea_width
        h = self.config.fovea_height
        half_fov_rad = math.radians(fov / 2.0)

        cols = torch.arange(self.fovea_cols, dtype=torch.float32, device=device)
        rows = torch.arange(self.fovea_rows, dtype=torch.float32, device=device)

        # Patch center UVs
        u = (cols * ps + ps / 2 + 0.5) / w  # [0, 1]
        v = (rows * ps + ps / 2 + 0.5) / h  # [0, 1]

        # Local angles relative to gaze center
        local_az = -half_fov_rad + u * (2.0 * half_fov_rad)  # left to right
        local_el = half_fov_rad - v * (2.0 * half_fov_rad)   # top to bottom

        # Meshgrid -> flatten
        el_grid, az_grid = torch.meshgrid(local_el, local_az, indexing='ij')

        return az_grid.reshape(-1), el_grid.reshape(-1)  # [fovea_patches] each

    def _extract_patches(self, rgb: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """
        Extract 16x16 patches from an RGB image tensor.

        Args:
            rgb: [H, W, 3] float32 on device (already cropped if needed)
            rows: number of patch rows
            cols: number of patch columns

        Returns:
            [rows*cols, 16, 16, 3] float32
        """
        ps = self.config.patch_size
        return (rgb
                .reshape(rows, ps, cols, ps, 3)
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, ps, ps, 3))

    def _bgra_bytes_to_rgb(self, bgra_bytes: bytes, height: int, width: int,
                            device: str) -> torch.Tensor:
        """Convert raw BGRA8 bytes to RGB float32 [H, W, 3] on GPU."""
        arr = np.frombuffer(bgra_bytes, dtype=np.uint8).reshape(height, width, 4)
        tex = torch.from_numpy(arr.copy()).to(device, non_blocking=True)
        # BGRA -> RGB: swap B and R, drop alpha
        rgb = tex[:, :, [2, 1, 0]].float() / 255.0
        return rgb

    def _bgra_tensor_to_rgb(self, bgra: torch.Tensor) -> torch.Tensor:
        """Convert BGRA8 uint8 tensor [H, W, 4] on GPU to RGB float32 [H, W, 3]."""
        return bgra[:, :, [2, 1, 0]].float() / 255.0

    def _bgra_to_patches_cuda(self, bgra: torch.Tensor, crop_h: int, crop_w: int,
                               patch_cols: int, n_patches: int) -> torch.Tensor:
        """Fused BGRA→RGB patches via custom CUDA kernel."""
        return self._cuda_kernels.bgra_to_rgb_patches(
            bgra, crop_h, crop_w, patch_cols, n_patches)

    def tokenize(
        self,
        pano_l_input: Union[bytes, torch.Tensor],
        pano_r_input: Union[bytes, torch.Tensor],
        fovea_l_input: Union[bytes, torch.Tensor],
        fovea_r_input: Union[bytes, torch.Tensor],
        gaze_azimuth_deg: float,
        gaze_elevation_deg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize 4 vision textures into patches with angular positions.

        Args:
            pano_l_input:  Left panorama — BGRA8 bytes or [H,W,4] uint8 tensor
            pano_r_input:  Right panorama — BGRA8 bytes or [H,W,4] uint8 tensor
            fovea_l_input: Left fovea — BGRA8 bytes or [H,W,4] uint8 tensor
            fovea_r_input: Right fovea — BGRA8 bytes or [H,W,4] uint8 tensor
            gaze_azimuth_deg: Current gaze azimuth in degrees
            gaze_elevation_deg: Current gaze elevation in degrees

        Returns:
            patches:     [32506, 16, 16, 3] float32 on device
            angular_pos: [32506, 2] float32 on device (azimuth, elevation in radians)
            eye_id:      [32506] int32 on device (0=left, 1=right)
            source_id:   [32506] int32 on device (0=panorama, 1=fovea)
        """
        cfg = self.config
        device = cfg.device

        # Fast path: CUDA kernel for GPU tensor inputs
        use_cuda = (self._cuda_kernels is not None
                    and isinstance(pano_l_input, torch.Tensor))

        if use_cuda:
            # Fused: crop + BGRA→RGB + uint8→float32 + /255 + patch reshape
            pano_patches_l = self._bgra_to_patches_cuda(
                pano_l_input, self.pano_crop_h, self.pano_crop_w,
                self.pano_cols, self.pano_patches)
            pano_patches_r = self._bgra_to_patches_cuda(
                pano_r_input, self.pano_crop_h, self.pano_crop_w,
                self.pano_cols, self.pano_patches)
            fovea_patches_l = self._bgra_to_patches_cuda(
                fovea_l_input, cfg.fovea_height, cfg.fovea_width,
                self.fovea_cols, self.fovea_patches)
            fovea_patches_r = self._bgra_to_patches_cuda(
                fovea_r_input, cfg.fovea_height, cfg.fovea_width,
                self.fovea_cols, self.fovea_patches)
        else:
            # Fallback: PyTorch ops
            pw = cfg.panorama_width
            ph = cfg.panorama_height
            fw = cfg.fovea_width
            fh = cfg.fovea_height

            if isinstance(pano_l_input, torch.Tensor):
                pano_l_rgb = self._bgra_tensor_to_rgb(pano_l_input)
            else:
                pano_l_rgb = self._bgra_bytes_to_rgb(pano_l_input, ph, pw, device)

            if isinstance(pano_r_input, torch.Tensor):
                pano_r_rgb = self._bgra_tensor_to_rgb(pano_r_input)
            else:
                pano_r_rgb = self._bgra_bytes_to_rgb(pano_r_input, ph, pw, device)

            if isinstance(fovea_l_input, torch.Tensor):
                fovea_l_rgb = self._bgra_tensor_to_rgb(fovea_l_input)
            else:
                fovea_l_rgb = self._bgra_bytes_to_rgb(fovea_l_input, fh, fw, device)

            if isinstance(fovea_r_input, torch.Tensor):
                fovea_r_rgb = self._bgra_tensor_to_rgb(fovea_r_input)
            else:
                fovea_r_rgb = self._bgra_bytes_to_rgb(fovea_r_input, fh, fw, device)

            pano_l_cropped = pano_l_rgb[:self.pano_crop_h, :self.pano_crop_w, :]
            pano_r_cropped = pano_r_rgb[:self.pano_crop_h, :self.pano_crop_w, :]

            pano_patches_l = self._extract_patches(pano_l_cropped, self.pano_rows, self.pano_cols)
            pano_patches_r = self._extract_patches(pano_r_cropped, self.pano_rows, self.pano_cols)
            fovea_patches_l = self._extract_patches(fovea_l_rgb, self.fovea_rows, self.fovea_cols)
            fovea_patches_r = self._extract_patches(fovea_r_rgb, self.fovea_rows, self.fovea_cols)

        # --- Fovea angular positions (dynamic per frame) ---
        gaze_az_rad = math.radians(gaze_azimuth_deg)
        gaze_el_rad = math.radians(gaze_elevation_deg)

        fovea_angles = torch.stack([
            self._fovea_local_az + gaze_az_rad,
            self._fovea_local_el + gaze_el_rad,
        ], dim=1)  # [fovea_patches, 2]

        # --- Concatenate: pano_L, pano_R, fovea_L, fovea_R ---
        patches = torch.cat([
            pano_patches_l, pano_patches_r,
            fovea_patches_l, fovea_patches_r,
        ], dim=0)  # [32506, 16, 16, 3]

        angular_pos = torch.cat([
            self._pano_angular_cat,          # both eyes share pano angles
            fovea_angles, fovea_angles,      # same gaze-relative for L/R fovea
        ], dim=0)  # [32506, 2]

        return patches, angular_pos, self._eye_id_cat, self._source_id_cat
