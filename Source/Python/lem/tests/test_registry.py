# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_registry.py — Tests for Phases 6-7 (encoder factories + registry).

Tests:
  - Each of 6 encoder factories produces [1, d_model] from correct input
  - EncoderRegistry builds only active encoders
  - Disabling a modality removes its encoder
  - Enabling voice adds its encoder
  - cross_handle_attention is an alias for cross_patch_attention
  - Missing config key raises KeyError
"""

import copy
import sys
from pathlib import Path

# achieved handle state (10) + executed command (9): what the motor ENCODER takes.
# Distinct from features_per_handle, which is the tokenizer's 10-wide wire format.
MOTOR_ENC_F = 19

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.config.resolver import resolve
from lem.model.encoders import ENCODER_FACTORIES
from lem.model.encoders.cross_patch import (
    build_cross_patch_attention,
    build_cross_handle_attention,
)
from lem.model.encoder_registry import EncoderRegistry

YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


@pytest.fixture
def resolved():
    with open(YAML_PATH) as f:
        master = yaml.safe_load(f)
    return resolve(copy.deepcopy(master))


@pytest.fixture
def registry(resolved):
    return EncoderRegistry(resolved)


# =========================================================================
# Encoder factories
# =========================================================================

class TestEncoderFactories:
    def test_swin_patch(self, resolved):
        enc_cfg = resolved["modalities"]["vision"]["encoder"]
        encoder = ENCODER_FACTORIES["swin_patch"](enc_cfg)
        patch_dim = enc_cfg["_patch_dim"]
        x = torch.randn(1, 100, patch_dim)
        ang = torch.randn(1, 100, 2)
        out = encoder(x, ang)
        assert out.shape == (1, 1, resolved["model"]["d_model"])

    def test_cross_patch_attention(self, resolved):
        enc_cfg = resolved["modalities"]["tactile"]["encoder"]
        encoder = ENCODER_FACTORIES["cross_patch_attention"](enc_cfg)
        pipe = resolved["modalities"]["tactile"]["pipeline"]
        x = torch.randn(1, pipe["n_sensors"], pipe["features_per_sensor"])
        out = encoder(x)
        assert out.shape == (1, 1, resolved["model"]["d_model"])

    def test_conv1d_stack(self, resolved):
        enc_cfg = resolved["modalities"]["audio"]["encoder"]
        encoder = ENCODER_FACTORIES["conv1d_stack"](enc_cfg)
        spec = torch.randn(1, 256, 2)
        spat = torch.randn(1, 2)
        out = encoder(spec, spat)
        assert out.shape == (1, 1, resolved["model"]["d_model"])

    def test_mlp_project(self, resolved):
        enc_cfg = resolved["modalities"]["proprioception"]["encoder"]
        encoder = ENCODER_FACTORIES["mlp_project"](enc_cfg)
        x = torch.randn(1, enc_cfg["_input_dim"])
        out = encoder(x)
        assert out.shape == (1, 1, resolved["model"]["d_model"])

    def test_cross_handle_attention(self, resolved):
        enc_cfg = resolved["modalities"]["motor"]["encoder"]
        encoder = ENCODER_FACTORIES["cross_handle_attention"](enc_cfg)
        # 19 = achieved (10, the tokenizer's wire format) + executed command (9).
        # The two numbers are different: features_per_handle stays 10.
        x = torch.randn(1, 24, MOTOR_ENC_F)
        out = encoder(x)
        assert out.shape == (1, 1, resolved["model"]["d_model"])

    def test_self_attention_pool(self, resolved):
        enc_cfg = resolved["modalities"]["intermind"]["encoder"]
        encoder = ENCODER_FACTORIES["self_attention_pool"](enc_cfg)
        embed_dim = enc_cfg["_embed_input_dim"]
        x = torch.randn(1, 5, embed_dim)
        out = encoder(x)
        assert out.shape == (1, 1, resolved["model"]["d_model"])

    def test_cross_handle_is_alias(self):
        assert build_cross_handle_attention is build_cross_patch_attention

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            ENCODER_FACTORIES["mlp_project"]({"type": "mlp_project"})


# =========================================================================
# Encoder registry
# =========================================================================

class TestEncoderRegistry:
    def test_builds_only_active(self, resolved, registry):
        assert set(registry.encoders.keys()) == set(resolved["_active_modalities"])

    def test_encode_returns_correct_shapes(self, resolved, registry):
        d_model = resolved["model"]["d_model"]
        patch_dim = resolved["modalities"]["vision"]["encoder"]["_patch_dim"]
        embed_dim = resolved["modalities"]["intermind"]["encoder"]["_embed_input_dim"]
        tac_pipe = resolved["modalities"]["tactile"]["pipeline"]
        frame = {
            "vision": {"patches": torch.randn(1, 100, patch_dim), "angular_pos": torch.randn(1, 100, 2)},
            "tactile": torch.randn(1, tac_pipe["n_sensors"], tac_pipe["features_per_sensor"]),
            "audio": {"spectral": torch.randn(1, 256, 2), "spatial": torch.randn(1, 2)},
            "proprioception": torch.randn(1, 288),
            "motor": torch.randn(1, 24, MOTOR_ENC_F),
            "intermind": torch.randn(1, 5, embed_dim),
        }
        tokens = registry.encode(frame)
        assert len(tokens) == 6
        for name, tok in tokens.items():
            assert tok.shape == (1, 1, d_model), f"{name}: {tok.shape}"

    def test_disable_removes_encoder(self):
        with open(YAML_PATH) as f:
            master = yaml.safe_load(f)
        master["modalities"]["tactile"]["enabled"] = False
        cfg = resolve(master)
        reg = EncoderRegistry(cfg)
        assert "tactile" not in reg.encoders
        assert len(reg.encoders) == 5

    def test_enable_voice_adds_encoder(self):
        with open(YAML_PATH) as f:
            master = yaml.safe_load(f)
        master["modalities"]["voice"]["enabled"] = True
        cfg = resolve(master)
        reg = EncoderRegistry(cfg)
        assert "voice" in reg.encoders
        assert len(reg.encoders) == 7

    def test_source_mask_disabled_skips(self, resolved, registry):
        patch_dim = resolved["modalities"]["vision"]["encoder"]["_patch_dim"]
        embed_dim = resolved["modalities"]["intermind"]["encoder"]["_embed_input_dim"]
        frame = {
            "vision": {"patches": torch.randn(1, 100, patch_dim), "angular_pos": torch.randn(1, 100, 2)},
            "tactile": torch.randn(1, 16, 8),
            "audio": {"spectral": torch.randn(1, 256, 2), "spatial": torch.randn(1, 2)},
            "proprioception": torch.randn(1, 288),
            "motor": torch.randn(1, 24, MOTOR_ENC_F),
            "intermind": torch.randn(1, 5, embed_dim),
        }
        tokens = registry.encode(frame, source_mask={"tactile": "disabled"})
        assert "tactile" not in tokens
        assert len(tokens) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
