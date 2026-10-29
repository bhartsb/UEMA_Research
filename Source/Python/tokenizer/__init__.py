# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
LEM Tokenizer — model-agnostic multi-modal tokenizer for the UE5 pipeline.

Converts raw pipeline output (shared memory + GPU textures) into normalized,
fixed-shape, GPU-resident PyTorch tensors — the Grounded Register (P_t).
"""

from .config import TokenizerConfig, TokenizedFrame, FrameMetadata
from .tokenizer import LEMTokenizer

__all__ = ["LEMTokenizer", "TokenizerConfig", "TokenizedFrame", "FrameMetadata"]
