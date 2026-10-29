# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Tier-3 sane defaults for all ~105 LEM config parameters.

resolve() merges these with user-provided values (user values win).
Any parameter omitted from master_config.yaml still gets a valid default.
"""


def _developmental_phases(steps_per_phase: int) -> dict:
    """The schedule, generated from lem.training.curriculum.PHASE_DEFINITIONS.

    Until 2026-08-27 this was a hand-written four-phase table (neonate < 10k steps,
    infant < 50k, toddler < 200k, child) that disagreed with the six-phase curriculum the
    trainer actually follows - two clocks, two sets of values, one model. There is one
    producer now; see curriculum.limits_for().
    """
    from lem.training.curriculum import phases_table
    return phases_table(steps_per_phase)


def get_defaults() -> dict:
    """Return the complete default config dict."""
    return {
        "version": "2.1",

        # -----------------------------------------------------------------
        # hardware — Device assignment for multi-GPU setups
        # -----------------------------------------------------------------
        "hardware": {
            "capture_device": "cuda:0",    # D3D12 interop, vision texture capture
            "training_device": "cuda:0",   # Model params, optimizer, gradients
            "multi_gpu": False,            # True = separate capture/training GPUs
            "torch_compile": {
                "enabled": False,          # torch.compile kernel fusion (PyTorch 2.0+)
                "mode": "default",         # "default" safe, "reduce-overhead" needs static shapes
            },
        },

        # -----------------------------------------------------------------
        # async_reader — Background frame reader thread
        # -----------------------------------------------------------------
        "async_reader": {
            "enabled": False,
            "ring_size": 3,
            "max_stale_age_ms": 50.0,
        },

        # -----------------------------------------------------------------
        # body
        # -----------------------------------------------------------------
        "body": {
            "n_handles": 24,
            "n_joints": 65,
            "n_bodysets": 10,
            "n_agents": 5,
            "tracked_bones": 24,
        },

        # -----------------------------------------------------------------
        # modalities
        # -----------------------------------------------------------------
        "modalities": {
            "vision": {
                "enabled": True,
                "data_source": "real",
                "loss_weight": 1.0,
                "rate_hz": 24,
                "pipeline": {
                    "format": "gpu_texture",
                    "panorama_resolution": [2520, 1560],
                    "fovea_resolution": [512, 512],
                    "panorama_fov_h_deg": 210.0,
                    "panorama_fov_v_deg": 130.0,
                    "fovea_fov_deg": 5.0,
                },
                "encoder": {
                    "type": "swin_patch",
                    "patch_size": 16,
                    "n_swin_layers": 4,
                    "n_heads": 8,
                    "window_size": 7,
                    "angular_pos_hidden": None,
                    "angular_pos_inputs": 2,   # (az, el); 4 adds which-eye and patch span
                    "patch_subsample": None,
                },
                "placeholder": {
                    "strategy": "gaussian",
                    "mean": 0.5,
                    "std": 0.15,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },

            "tactile": {
                "enabled": True,
                "data_source": "synthetic",
                "loss_weight": 0.5,
                "rate_hz": 60,
                "pipeline": {
                    "format": "sensor_array",
                    "n_sensors": 256,
                    "features_per_sensor": 10,
                    "shape": [256, 10],
                },
                "encoder": {
                    "type": "cross_patch_attention",
                    "per_patch_input_dim": 10,
                    "per_patch_hidden": None,
                    "n_cross_attn_layers": 2,
                    "n_heads": 4,
                },
                "placeholder": {
                    "strategy": "sparse_contact",
                    "n_active": 3,
                    "force_mean": 0.3,
                    "force_std": 0.15,
                    "temporal_correlation": 0.8,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },

            "audio": {
                "enabled": True,
                "data_source": "real",
                "loss_weight": 0.8,
                "rate_hz": 60,
                "pipeline": {
                    "format": "spectral",
                    "fft_n_bins": 256,
                    "n_channels": 2,
                    "sample_rate": 48000,
                    "spatial_features": 2,
                },
                "encoder": {
                    "type": "conv1d_stack",
                    "conv_channels": None,
                    "kernel_size": 3,
                    "spatial_branch_input_dim": 2,
                    "combine": "gate",
                    # absolute-frequency band path (09-09 feasibility review:
                    # conv+global-average provably erases WHERE on the
                    # frequency axis a sound lives). "on" adds zero-init
                    # parameters - enable at a segment launch only.
                    "freq_band_path": "off",
                    "n_freq_bands": 16,
                },
                "placeholder": {
                    "strategy": "gaussian",
                    "mean": 0.0,
                    "std": 0.01,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },

            "proprioception": {
                "enabled": True,
                "data_source": "real",
                "loss_weight": 1.0,
                "rate_hz": 60,
                "pipeline": {
                    "format": "flat_vector",
                    "input_dim": 288,  # tracked_bones(24) * 12 features - the eyes are bone 23
                },
                "encoder": {
                    "type": "mlp_project",
                    "hidden_dims": None,
                },
                "placeholder": {
                    "strategy": "gaussian",
                    "mean": 0.0,
                    "std": 0.1,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },

            "motor": {
                "enabled": True,
                "data_source": "real",
                "loss_weight": 1.0,
                "rate_hz": 60,
                "pipeline": {
                    "format": "handle_array",
                    "n_handles": 24,
                    "features_per_handle": 10,  # pos(3)+rot(4)+radius(1)+strength(1)+is_active(1)
                },
                "encoder": {
                    "type": "cross_handle_attention",
                    # 19 = achieved handle state (10) + executed command (9).
                    # The transition contract: the model is shown the order that carried
                    # this state to the next one, not only the pose it reached. LemLite
                    # carried this alone until 2026-08-25, which meant the cloud model
                    # would have started on the OLD contract. features_per_handle above
                    # stays 10: that is the tokenizer wire format, and the command is
                    # appended model-side.
                    "per_patch_input_dim": 19,
                    "per_patch_hidden": None,
                    "n_cross_attn_layers": 2,
                    "n_heads": 4,
                },
                "placeholder": {
                    "strategy": "gaussian",
                    "mean": 0.0,
                    "std": 0.05,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },

            "intermind": {
                "enabled": True,
                "data_source": "synthetic",
                "loss_weight": 0.0,
                "rate_hz": 60,
                "pipeline": {
                    "format": "agent_array",
                    "n_agents": 5,
                    "features_per_agent": 3,
                },
                "encoder": {
                    "type": "self_attention_pool",
                    "n_self_attn_layers": 2,
                    "n_heads": 4,
                },
                "placeholder": {
                    "strategy": "gaussian",
                    "mean": 0.0,
                    "std": 0.01,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },

            "voice": {
                "enabled": False,
                "data_source": "synthetic",
                "loss_weight": 0.3,
                "rate_hz": 60,
                "pipeline": {
                    "format": "flat_vector",
                    "input_dim": 56,
                },
                "encoder": {
                    "type": "mlp_project",
                    "hidden_dims": None,
                },
                "placeholder": {
                    "strategy": "gaussian",
                    "mean": 0.5,
                    "std": 0.2,
                },
                "loss_ramp": {
                    "enabled": False,
                    "from_weight": 0.1,
                    "to_weight": 1.0,
                    "over_steps": 1000,
                },
            },
        },

        # -----------------------------------------------------------------
        # encoder_defaults
        # -----------------------------------------------------------------
        "encoder_defaults": {
            "bias": False,
            "dropout": 0.0,
            "activation": "gelu",
            "norm_type": "layernorm",
            "pool_type": "learned_query",
            "pool_n_heads": 4,
            "ffn_expansion": 4,
            "d_internal": None,
        },

        # -----------------------------------------------------------------
        # model
        # -----------------------------------------------------------------
        "model": {
            "d_model": 768,
            "context_length": 32,
            "precision": "fp16",

            "forward_model": {
                "n_layers": 8,
                "n_heads": 12,
                "ffn_expansion": 4,
                "stages": {
                    "cross_modal": True,
                    "causal_temporal": True,
                    "cross_modal_bridge": True,
                },
            },

            "prediction_heads": {
                "n_heads": 4,
                "hidden_expansion": 2,
                "n_mlp_layers": 2,
            },

            "motor_controller": {
                "n_layers": 6,
                "hidden_dim": 512,
                "confidence_hidden": 128,
                "confidence_threshold": 0.8,
                "recursive": False,
                "output_per_handle": 9,
            },

            "gaze_head": {
                "enabled": False,
                "output_dim": 2,
                "hidden_dim": 128,
                "n_heads": 4,
                "azimuth_range": [-3.14159, 3.14159],
                "elevation_range": [-1.5708, 1.5708],
                "loss_weight": 0.1,
            },

            "voice_head": {
                "enabled": False,
                "output_dim": 80,
                "hidden_dim": 128,
                "n_heads": 4,
                "loss_weight": 0.0,
            },

            "wellness": {
                "loss_weight": 0.1,
                "grounded": {
                    "prediction_error": {
                        "input_dim": 1,
                        "hidden_dim": 32,
                    },
                    "tactile_comfort": {
                        "input_dim": 8,
                        "hidden_dim": 32,
                    },
                    "capability": {
                        "input_dim": 21,
                        "hidden_dim": 32,
                    },
                    "coherence": {
                        "input_dim": 1,
                        "hidden_dim": 32,
                    },
                    "motor_state": {
                        "input_dim": 216,  # 24 handles * 9 outputs
                        "hidden_dim": 64,
                    },
                },
                "approximate": {
                    "hidden_dims": [256, 128],
                    "n_outputs": 5,
                },
            },
        },

        # -----------------------------------------------------------------
        # training
        # -----------------------------------------------------------------
        "training": {
            "optimizer": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "grad_clip": 1.0,
            "warmup_steps": 1000,
            "batch_size": 1,
            "backward_every_n_frames": 1,  # 1 = every frame, N>1 = perception-only on N-1 frames

            "loss": {
                "source_scales": {
                    # deferred item 6 (2026-09-02): these drifted from
                    # master_config.yaml (synthetic 0.1 vs 0.0, stale 0.0 vs 0.3).
                    # The yaml always wins when present, so live behavior never
                    # changed - but a yaml-less loader would have trained against
                    # Gaussian noise at 0.1. A pin test now holds them equal.
                    "real": 1.0,
                    "synthetic": 0.0,
                    "stale": 0.3,
                },
                "vicreg": {
                    "enabled": False,
                    "variance_weight": 1.0,
                    "covariance_weight": 0.04,
                    "variance_target": 1.0,
                    "temporal_window": 0,
                },
            },

            "steps_per_phase": 50_000,
            "developmental_phases": _developmental_phases(50_000),
        },

        # -----------------------------------------------------------------
        # checkpoint
        # -----------------------------------------------------------------
        "checkpoint": {
            "format": "safetensors",
            "save_every_steps": 5000,
            "keep_last_n": 10,
            "progressive_resize": True,
            "checkpoint_dir": "checkpoints",
        },

        # -----------------------------------------------------------------
        # action_buffer
        # -----------------------------------------------------------------
        "action_buffer": {
            "enabled": True,
            "shm_name": "LEM_Action_Buffer",
            "buffer_size": 780,   # 20 header + 432 motor + 8 gaze + 320 voice
            "write_timeout_ms": 5,
        },

        # -----------------------------------------------------------------
        # future
        # -----------------------------------------------------------------
        "future": {
            "imagination_rollouts": {
                "enabled": False,
                "rollout_length": 16,
            },
            "dual_timescale": {
                "enabled": False,
                "fast_hz": 60,
                "slow_hz": 10,
            },
            "dreamzero": {
                "enabled": False,
                "replay_ratio": 0.25,
            },
        },
    }
