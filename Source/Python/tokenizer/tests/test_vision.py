# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Unit tests for the vision tokenizer.

Creates synthetic BGRA8 images with known patterns and verifies
patch extraction, angular encoding, and output shapes.
"""

import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokenizer.config import TokenizerConfig
from tokenizer.modalities.vision import VisionTokenizer


def make_gradient_bgra(width: int, height: int) -> bytes:
    """Create a BGRA8 gradient test image."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x, 0] = x * 255 // max(width - 1, 1)   # B
            img[y, x, 1] = y * 255 // max(height - 1, 1)   # G
            img[y, x, 2] = 128                              # R
            img[y, x, 3] = 255                              # A
    return img.tobytes()


def make_solid_bgra(width: int, height: int, b: int, g: int, r: int) -> bytes:
    """Create a solid-color BGRA8 image."""
    img = np.full((height, width, 4), [b, g, r, 255], dtype=np.uint8)
    return img.tobytes()


def test_vision_patch_counts():
    """Verify total patch count matches expectations."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)

    assert vt.pano_patches == 15229, f"Pano patches: {vt.pano_patches}"
    assert vt.fovea_patches == 1024, f"Fovea patches: {vt.fovea_patches}"
    assert vt.total_patches == 32506, f"Total patches: {vt.total_patches}"

    print(f"PASS: test_vision_patch_counts "
          f"(pano={vt.pano_patches}, fovea={vt.fovea_patches}, total={vt.total_patches})")


def test_vision_output_shapes():
    """Verify tokenize() output shapes and dtypes."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)

    pano_l = make_solid_bgra(config.panorama_width, config.panorama_height, 0, 0, 255)
    pano_r = make_solid_bgra(config.panorama_width, config.panorama_height, 0, 255, 0)
    fovea_l = make_solid_bgra(config.fovea_width, config.fovea_height, 255, 0, 0)
    fovea_r = make_solid_bgra(config.fovea_width, config.fovea_height, 0, 0, 255)

    patches, angles, eye_id, source_id = vt.tokenize(
        pano_l, pano_r, fovea_l, fovea_r, 0.0, 0.0
    )

    assert patches.shape == (32506, 16, 16, 3), f"Patches shape: {patches.shape}"
    assert angles.shape == (32506, 2), f"Angles shape: {angles.shape}"
    assert eye_id.shape == (32506,), f"Eye ID shape: {eye_id.shape}"
    assert source_id.shape == (32506,), f"Source ID shape: {source_id.shape}"

    assert patches.dtype == torch.float32
    assert angles.dtype == torch.float32
    assert eye_id.dtype == torch.int32
    assert source_id.dtype == torch.int32

    assert str(patches.device).startswith("cuda")

    print("PASS: test_vision_output_shapes")


def test_vision_color_conversion():
    """Verify BGRA->RGB conversion and [0,1] normalization."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)

    # Solid red: BGRA = (0, 0, 255, 255) -> RGB = (1.0, 0.0, 0.0)
    pano = make_solid_bgra(config.panorama_width, config.panorama_height, 0, 0, 255)
    fovea = make_solid_bgra(config.fovea_width, config.fovea_height, 0, 0, 255)

    patches, _, _, _ = vt.tokenize(pano, pano, fovea, fovea, 0.0, 0.0)

    sample = patches[0, 0, 0].cpu()  # first pixel of first patch
    assert abs(sample[0].item() - 1.0) < 0.01, f"R channel: {sample[0]}"
    assert abs(sample[1].item() - 0.0) < 0.01, f"G channel: {sample[1]}"
    assert abs(sample[2].item() - 0.0) < 0.01, f"B channel: {sample[2]}"

    print("PASS: test_vision_color_conversion")


def test_vision_angular_center():
    """Verify center patch has angular position near (0, 0)."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)

    # The center patch of the panorama LUT
    center_row = vt.pano_rows // 2
    center_col = vt.pano_cols // 2
    center_idx = center_row * vt.pano_cols + center_col

    center_angles = vt.pano_angular_lut[center_idx].cpu()
    az_deg = math.degrees(center_angles[0].item())
    el_deg = math.degrees(center_angles[1].item())

    # Center should be near (0, 0) within a patch width
    assert abs(az_deg) < 2.0, f"Center azimuth: {az_deg} deg"
    assert abs(el_deg) < 2.0, f"Center elevation: {el_deg} deg"

    print(f"PASS: test_vision_angular_center (az={az_deg:.2f}, el={el_deg:.2f} deg)")


def test_vision_angular_extremes():
    """Verify corner patches have expected extreme angles."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)

    # Top-left patch: should be near (-105, +65)
    tl_angles = vt.pano_angular_lut[0].cpu()
    tl_az = math.degrees(tl_angles[0].item())
    tl_el = math.degrees(tl_angles[1].item())
    assert tl_az < -100.0, f"Top-left azimuth: {tl_az}"
    assert tl_el > 60.0, f"Top-left elevation: {tl_el}"

    # Bottom-right patch
    br_idx = vt.pano_patches - 1
    br_angles = vt.pano_angular_lut[br_idx].cpu()
    br_az = math.degrees(br_angles[0].item())
    br_el = math.degrees(br_angles[1].item())
    assert br_az > 100.0, f"Bottom-right azimuth: {br_az}"
    assert br_el < -60.0, f"Bottom-right elevation: {br_el}"

    print(f"PASS: test_vision_angular_extremes "
          f"(TL: az={tl_az:.1f}, el={tl_el:.1f}; BR: az={br_az:.1f}, el={br_el:.1f})")


def test_vision_eye_source_tags():
    """Verify eye_id and source_id tag correctness."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)
    pano = make_solid_bgra(config.panorama_width, config.panorama_height, 128, 128, 128)
    fovea = make_solid_bgra(config.fovea_width, config.fovea_height, 128, 128, 128)

    _, _, eye_id, source_id = vt.tokenize(pano, pano, fovea, fovea, 0.0, 0.0)

    eye = eye_id.cpu()
    src = source_id.cpu()

    np_ = vt.pano_patches
    nf = vt.fovea_patches

    # Panorama left: eye=0, source=0
    assert (eye[:np_] == 0).all()
    assert (src[:np_] == 0).all()
    # Panorama right: eye=1, source=0
    assert (eye[np_:2*np_] == 1).all()
    assert (src[np_:2*np_] == 0).all()
    # Fovea left: eye=0, source=1
    assert (eye[2*np_:2*np_+nf] == 0).all()
    assert (src[2*np_:2*np_+nf] == 1).all()
    # Fovea right: eye=1, source=1
    assert (eye[2*np_+nf:] == 1).all()
    assert (src[2*np_+nf:] == 1).all()

    print("PASS: test_vision_eye_source_tags")


def test_vision_value_range():
    """Verify all patch values are in [0, 1]."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)
    pano = make_gradient_bgra(config.panorama_width, config.panorama_height)
    fovea = make_gradient_bgra(config.fovea_width, config.fovea_height)

    patches, _, _, _ = vt.tokenize(pano, pano, fovea, fovea, 0.0, 0.0)

    assert patches.min() >= 0.0, f"Min: {patches.min()}"
    assert patches.max() <= 1.0, f"Max: {patches.max()}"
    assert not torch.isnan(patches).any(), "NaN in patches"
    assert not torch.isinf(patches).any(), "Inf in patches"

    print("PASS: test_vision_value_range")


def test_vision_tensor_input():
    """Verify tokenize() accepts GPU tensors (D3D12→CUDA zero-copy path)."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)

    # Create BGRA8 uint8 tensors on GPU (simulating DLPack import from D3D12)
    device = config.device

    # Solid red: BGRA = (0, 0, 255, 255) -> RGB = (1.0, 0.0, 0.0)
    pano_tensor = torch.zeros(config.panorama_height, config.panorama_width, 4,
                               dtype=torch.uint8, device=device)
    pano_tensor[:, :, 2] = 255  # R channel in BGRA
    pano_tensor[:, :, 3] = 255  # Alpha

    fovea_tensor = torch.zeros(config.fovea_height, config.fovea_width, 4,
                                dtype=torch.uint8, device=device)
    fovea_tensor[:, :, 2] = 255
    fovea_tensor[:, :, 3] = 255

    patches, angles, eye_id, source_id = vt.tokenize(
        pano_tensor, pano_tensor, fovea_tensor, fovea_tensor, 0.0, 0.0
    )

    # Verify shapes match bytes path
    assert patches.shape == (32506, 16, 16, 3), f"Patches shape: {patches.shape}"
    assert angles.shape == (32506, 2), f"Angles shape: {angles.shape}"
    assert eye_id.shape == (32506,), f"Eye ID shape: {eye_id.shape}"
    assert source_id.shape == (32506,), f"Source ID shape: {source_id.shape}"

    assert patches.dtype == torch.float32
    assert str(patches.device).startswith("cuda")

    # Verify BGRA→RGB conversion: red input → R=1, G=0, B=0
    sample = patches[0, 0, 0].cpu()
    assert abs(sample[0].item() - 1.0) < 0.01, f"R channel: {sample[0]}"
    assert abs(sample[1].item() - 0.0) < 0.01, f"G channel: {sample[1]}"
    assert abs(sample[2].item() - 0.0) < 0.01, f"B channel: {sample[2]}"

    # Verify all values in [0, 1]
    assert patches.min() >= 0.0, f"Min: {patches.min()}"
    assert patches.max() <= 1.0, f"Max: {patches.max()}"

    print("PASS: test_vision_tensor_input")


def test_vision_tensor_mixed_input():
    """Verify tokenize() handles mixed bytes + tensor input."""
    config = TokenizerConfig()
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    vt = VisionTokenizer(config)
    device = config.device

    # Pano as bytes, fovea as tensor
    pano_bytes = make_solid_bgra(config.panorama_width, config.panorama_height, 0, 0, 255)
    fovea_tensor = torch.full((config.fovea_height, config.fovea_width, 4),
                               128, dtype=torch.uint8, device=device)
    fovea_tensor[:, :, 3] = 255

    patches, _, _, _ = vt.tokenize(
        pano_bytes, pano_bytes, fovea_tensor, fovea_tensor, 0.0, 0.0
    )

    assert patches.shape == (32506, 16, 16, 3), f"Mixed input shape: {patches.shape}"
    assert not torch.isnan(patches).any(), "NaN in mixed input patches"

    print("PASS: test_vision_tensor_mixed_input")


if __name__ == '__main__':
    test_vision_patch_counts()
    test_vision_output_shapes()
    test_vision_color_conversion()
    test_vision_angular_center()
    test_vision_angular_extremes()
    test_vision_eye_source_tags()
    test_vision_value_range()
    test_vision_tensor_input()
    test_vision_tensor_mixed_input()
    print("\nAll vision tests passed!")
