# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
InterMind tokenizer — STUB.

Zero vector until the Many Minds hierarchy is implemented.
Shape will change when activated — consumers read from tokenizer manifest.
"""

import torch

from ..config import TokenizerConfig


def tokenize_intermind(config: TokenizerConfig) -> torch.Tensor:
    """Returns a zero vector [3] on device."""
    return torch.zeros(3, dtype=torch.float32, device=config.device)
