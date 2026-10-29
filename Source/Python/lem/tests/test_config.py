# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_config.py — Tests for Phases 1-4 (config system).

Tests:
  - master_config.yaml loads with all defaults present
  - resolve() baseline scenario
  - resolve() all 6 companion scenarios
  - Disabling modality removes from active/heads
  - d_model change propagates
  - Body param change propagates to proprio _input_dim
  - Encoder default inheritance
  - d_internal=null resolves to d_model
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import yaml

# Ensure lem is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.config.defaults import get_defaults
from lem.config.resolver import resolve
from lem.config.loader import load_and_resolve

YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


def _expected_motor_input_dim(cfg):
    """Compute expected motor_controller._input_dim from resolved config."""
    d = cfg["model"]["d_model"]
    n_active = len(cfg["_active_modalities"])
    n_handles = cfg["body"]["n_handles"]
    fph = cfg["modalities"].get("motor", {}).get("pipeline", {}).get(
        "features_per_handle", cfg["model"]["motor_controller"]["output_per_handle"])
    oph = cfg["model"]["motor_controller"]["output_per_handle"]
    return d * n_active + n_handles * fph + n_handles * oph


@pytest.fixture
def master():
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def baseline(master):
    return resolve(copy.deepcopy(master))


# =========================================================================
# Phase 1: YAML loads with all defaults
# =========================================================================

class TestPhase1:
    def test_yaml_loads(self, master):
        assert master is not None

    def test_all_sections_present(self, master):
        for section in ["version", "body", "modalities", "encoder_defaults",
                        "model", "training", "checkpoint", "future"]:
            assert section in master, f"Missing section: {section}"

    def test_all_modalities_have_required_keys(self, master):
        for mod_name, mod_cfg in master["modalities"].items():
            for key in ["enabled", "data_source", "encoder", "pipeline",
                        "loss_ramp", "placeholder"]:
                assert key in mod_cfg, f"{mod_name} missing {key}"

    def test_defaults_match_yaml(self, master):
        defaults = get_defaults()
        assert defaults["body"]["n_handles"] == master["body"]["n_handles"]
        assert defaults["model"]["d_model"] == master["model"]["d_model"]


# =========================================================================
# Phase 2: resolve() scenarios
# =========================================================================

class TestPhase2:
    def test_baseline(self, baseline):
        d = baseline["model"]["d_model"]
        assert len(baseline["_active_modalities"]) == 6
        assert len(baseline["_prediction_head_modalities"]) == 5
        assert baseline["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(baseline)
        ctx = baseline["model"]["context_length"]
        n_mod = len(baseline["_active_modalities"])
        assert baseline["model"]["forward_model"]["_seq_length"] == ctx * n_mod
        # proprio: explicit pipeline.input_dim = 288 (tracked_bones(24)*12; bone 23 = eyes)
        assert baseline["modalities"]["proprioception"]["encoder"]["_input_dim"] == 288
        features_per_agent = baseline["modalities"]["intermind"]["pipeline"].get("features_per_agent", 3)
        assert baseline["modalities"]["intermind"]["encoder"]["_embed_input_dim"] == d + features_per_agent
        n_sensory = len([m for m in baseline["_active_modalities"] if m != "intermind"])
        assert baseline["model"]["wellness"]["approximate"]["_input_dim"] == d * n_sensory
        assert baseline["modalities"]["vision"]["encoder"]["_n_patches"] == 32506

    def test_scenario_disable_tactile(self, master):
        m = copy.deepcopy(master)
        m["modalities"]["tactile"]["enabled"] = False
        cfg = resolve(m)
        assert len(cfg["_active_modalities"]) == 5
        assert len(cfg["_prediction_head_modalities"]) == 4
        assert cfg["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(cfg)
        assert "tactile" not in cfg["_active_modalities"]

    def test_scenario_minimal(self, master):
        m = copy.deepcopy(master)
        m["modalities"]["tactile"]["enabled"] = False
        m["modalities"]["intermind"]["enabled"] = False
        cfg = resolve(m)
        assert len(cfg["_active_modalities"]) == 4
        assert len(cfg["_prediction_head_modalities"]) == 3
        assert cfg["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(cfg)

    def test_scenario_enable_voice(self, master):
        m = copy.deepcopy(master)
        m["modalities"]["voice"]["enabled"] = True
        cfg = resolve(m)
        assert len(cfg["_active_modalities"]) == 7
        assert len(cfg["_prediction_head_modalities"]) == 6
        assert cfg["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(cfg)
        assert "voice" in cfg["_active_modalities"]

    def test_scenario_scale_d_model(self, master):
        m = copy.deepcopy(master)
        m["model"]["d_model"] = 1024
        cfg = resolve(m)
        assert len(cfg["_active_modalities"]) == 6
        assert len(cfg["_prediction_head_modalities"]) == 5
        assert cfg["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(cfg)
        for mod_name in cfg["_active_modalities"]:
            assert cfg["modalities"][mod_name]["encoder"]["_d_output"] == 1024

    def test_scenario_ogc_delivery(self, master):
        m = copy.deepcopy(master)
        m["modalities"]["tactile"]["pipeline"]["shape"] = [200, 8]
        m["modalities"]["tactile"]["pipeline"]["n_sensors"] = 200
        m["modalities"]["tactile"]["pipeline"]["features_per_sensor"] = 8
        cfg = resolve(m)
        assert len(cfg["_active_modalities"]) == 6
        assert cfg["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(cfg)
        # Tactile SHM should reflect new size
        tac_shm = cfg["_shm_layout"]["tactile"]["aligned_size"]
        expected = (200 * 8 * 4 + 16 + 255) & ~255
        assert tac_shm == expected

    def test_motor_never_gets_prediction_head(self, baseline):
        assert "motor" not in baseline["_prediction_head_modalities"]
        assert "motor" in baseline["_active_modalities"]

    def test_d_internal_resolves_from_config(self, baseline):
        for mod_name in baseline["_active_modalities"]:
            enc = baseline["modalities"][mod_name]["encoder"]
            expected = enc.get("d_internal") or baseline["model"]["d_model"]
            assert enc["_d_internal"] == expected

    def test_encoder_default_inheritance(self, baseline):
        for mod_name in baseline["_active_modalities"]:
            enc = baseline["modalities"][mod_name]["encoder"]
            assert "bias" in enc
            assert "dropout" in enc
            assert "activation" in enc
            assert "norm_type" in enc
            assert "pool_type" in enc

    def test_body_param_change_propagates(self, master):
        """When pipeline.input_dim is null, formula fallback uses body params."""
        m = copy.deepcopy(master)
        m["body"]["n_joints"] = 100
        # Set input_dim to None to trigger formula fallback (overrides default)
        m["modalities"]["proprioception"]["pipeline"]["input_dim"] = None
        cfg = resolve(m)
        nj = 100
        nh = cfg["body"]["n_handles"]
        nb = cfg["body"]["n_bodysets"]
        expected = nj * 3 + nj * 3 + nh * 3 + nh * 7 + nb + nb + 1
        assert cfg["modalities"]["proprioception"]["encoder"]["_input_dim"] == expected


# =========================================================================
# Phase 4: End-to-end loader
# =========================================================================

class TestPhase4:
    def test_load_and_resolve(self):
        cfg = load_and_resolve(YAML_PATH, skip_validation=True)
        assert cfg["_active_modalities"] == [
            "vision", "tactile", "audio", "proprioception", "motor", "intermind"
        ]
        assert cfg["model"]["motor_controller"]["_input_dim"] == _expected_motor_input_dim(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
