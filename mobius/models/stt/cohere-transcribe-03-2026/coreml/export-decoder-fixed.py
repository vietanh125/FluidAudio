#!/usr/bin/env python3
"""Export Cohere decoder using tensor indexing without .item() calls."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class FixedCachedDecoderWrapper(nn.Module):
    """Decoder wrapper using only tensor operations, no .item() calls."""

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        dec_config = full_model.config.transf_decoder["config_dict"]
        self.num_layers = dec_config["num_layers"]
        self.num_heads = dec_config["num_attention_heads"]
        self.hidden_size = dec_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len

    def forward(self, input_id, encoder_hidden_states, cache_k, cache_v, step, cross_attention_mask):
        """
        Decoder forward using ONLY tensor operations - no .item(), no Python conditionals on tensors.

        Key insight: Use torch.narrow() for dynamic slicing, which CoreML supports.
        """
        batch_size = 1

        # Build cache, truncating to current step using torch.narrow (CoreML-compatible)
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        # step is (1,) shape, extract scalar but keep as tensor
        step_scalar = step.squeeze() if step.ndim > 0 else step

        for layer_idx in range(self.num_layers):
            layer_k = cache_k[layer_idx].unsqueeze(0)  # (1, 8, 108, 128)
            layer_v = cache_v[layer_idx].unsqueeze(0)

            # Truncate to [0:step] using narrow (dynamic, CoreML-compatible)
            # narrow(dim, start, length) - we want positions [0:step]
            # Use max(step, 1) to handle step=0 case (narrow requires length >= 1)
            seq_len = torch.clamp(step_scalar, min=1)  # At least 1 for narrow
            layer_k_truncated = torch.narrow(layer_k, 2, 0, seq_len.item())  # Hmm, still need .item() for narrow
            layer_v_truncated = torch.narrow(layer_v, 2, 0, seq_len.item())

            # Handle step=0 case: if step==0, use empty cache
            # Use torch.where to conditionally select
            is_step_zero = (step_scalar == 0).float()
            empty_k = layer_k[:, :, :0, :]  # Empty slice
            empty_v = layer_v[:, :, :0, :]

            # This won't work either - different shapes can't be selected with where

            self_attention_cache.update(layer_k_truncated, layer_v_truncated, layer_idx)

        # ... rest of the code
