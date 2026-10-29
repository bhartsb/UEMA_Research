# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Unit tests for the manifest system.

Verifies manifest build, write, and pipeline manifest override logic.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokenizer.config import TokenizerConfig
from tokenizer.manifest import (
    build_tokenizer_manifest,
    write_tokenizer_manifest,
    load_pipeline_manifest,
    apply_manifest_overrides,
)


def test_manifest_build():
    """Verify all expected keys are present in the built manifest."""
    config = TokenizerConfig()
    manifest = build_tokenizer_manifest(config)

    assert "version" in manifest
    assert "generated_at" in manifest
    assert "frame_metadata" in manifest
    assert "modalities" in manifest

    mods = manifest["modalities"]
    expected_modalities = ["vision", "audio", "proprioception", "motor", "touch", "voice", "intermind"]
    for mod in expected_modalities:
        assert mod in mods, f"Missing modality: {mod}"

    # Vision specifics
    vis = mods["vision"]
    assert vis["patches_shape"] == [32506, 16, 16, 3]
    assert vis["panorama_patches_per_eye"] == 15229
    assert vis["fovea_patches_per_eye"] == 1024
    assert vis["eyes"] == 2
    assert vis["patch_size"] == 16

    # Audio specifics
    aud = mods["audio"]
    assert aud["spectral_shape"] == [256, 2]
    assert aud["fft_window_size"] == 1024
    assert aud["sample_rate"] == 48000

    # Proprio
    assert mods["proprioception"]["shape"] == [23, 12]

    # Motor
    assert mods["motor"]["handles_shape"] == [23, 10]

    # Voice
    assert mods["voice"]["shape"] == [56]
    assert len(mods["voice"]["fields"]) == 56

    print("PASS: test_manifest_build")


def test_manifest_write_read():
    """Verify manifest roundtrips through JSON correctly."""
    config = TokenizerConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenizer_manifest.json")
        written = write_tokenizer_manifest(config, output_path=path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["version"] == written["version"]
        assert loaded["modalities"]["vision"]["patches_shape"] == written["modalities"]["vision"]["patches_shape"]

    print("PASS: test_manifest_write_read")


def test_manifest_override():
    """Verify pipeline manifest values override config."""
    config = TokenizerConfig()
    assert config.fovea_fov_deg == 5.0
    assert config.audio_sample_rate == 48000

    # Simulate a pipeline manifest with different values
    pipeline = {
        "vision": {
            "fovea_fov_degrees": 10.0,
            "panorama_width": 2520,
        },
        "audio": {
            "sample_rate": 44100,
        },
    }

    config = apply_manifest_overrides(config, pipeline)

    assert config.fovea_fov_deg == 10.0, f"fovea_fov_deg: {config.fovea_fov_deg}"
    assert config.audio_sample_rate == 44100, f"sample_rate: {config.audio_sample_rate}"
    assert config.panorama_width == 2520  # unchanged

    print("PASS: test_manifest_override")


def test_manifest_load_missing():
    """Verify graceful handling of missing manifest file."""
    result = load_pipeline_manifest("/nonexistent/path/manifest.json")
    assert result is None
    print("PASS: test_manifest_load_missing")


if __name__ == '__main__':
    test_manifest_build()
    test_manifest_write_read()
    test_manifest_override()
    test_manifest_load_missing()
    print("\nAll manifest tests passed!")
