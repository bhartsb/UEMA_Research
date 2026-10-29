# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
LEM encoder factories — one per encoder type.

ENCODER_FACTORIES maps type string to factory function.
Each factory takes a resolved encoder config dict and returns an nn.Module.
"""

from typing import Callable, Dict

import torch.nn as nn

from .swin_patch import build_swin_patch
from .cross_patch import build_cross_patch_attention, build_cross_handle_attention
from .conv1d_stack import build_conv1d_stack
from .mlp_project import build_mlp_project
from .self_attention import build_self_attention_pool

ENCODER_FACTORIES: Dict[str, Callable[[dict], nn.Module]] = {
    "swin_patch": build_swin_patch,
    "cross_patch_attention": build_cross_patch_attention,
    "cross_handle_attention": build_cross_handle_attention,
    "conv1d_stack": build_conv1d_stack,
    "mlp_project": build_mlp_project,
    "self_attention_pool": build_self_attention_pool,
}
