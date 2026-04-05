#!/usr/bin/env python3
"""Export a CoreML model that computes cross-attention caches from encoder output.

This avoids having to load both PyTorch and CoreML models simultaneously (OOM issue).
"""
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq


class CrossCacheComputer(nn.Module):
    """Computes all cross-attention K/V caches from encoder output."""

    def __init__(self, model):
        super().__init__()
        self.encoder_decoder_proj = model.encoder_decoder_proj
        self.num_layers = 8

        # Extract cross-attention key/value projections from each layer
        for i, layer in enumerate(model.transf_decoder._decoder.layers):
            cross_attn = layer.second_sub_layer
            setattr(self, f'cross_key_net_{i}', cross_attn.key_net)
            setattr(self, f'cross_value_net_{i}', cross_attn.value_net)

    def forward(self, encoder_hidden_states: torch.Tensor):
        """
        Args:
            encoder_hidden_states: (B, enc_seq_len, 1280) - encoder output

        Returns:
            8 cross-caches, each of shape (2, B, enc_seq_len, 8, 128)
        """
        # Project encoder outputs
        encoder_proj = self.encoder_decoder_proj(encoder_hidden_states)

        B = encoder_proj.shape[0]
        enc_seq_len = encoder_proj.shape[1]
        H = 8  # num_heads
        D = 128  # head_dim

        caches = []
        for i in range(self.num_layers):
            key_net = getattr(self, f'cross_key_net_{i}')
            value_net = getattr(self, f'cross_value_net_{i}')

            key = key_net(encoder_proj).view(B, enc_seq_len, H, D)
            value = value_net(encoder_proj).view(B, enc_seq_len, H, D)

            # Stack as (2, B, enc_seq_len, H, D)
            cross_cache = torch.stack([key, value], dim=0)
            caches.append(cross_cache)

        return tuple(caches)


def export_cross_cache_computer():
    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()

    print("Creating cross-cache computer...")
    computer = CrossCacheComputer(model)
    computer.eval()

    # Create dummy encoder output
    B = 1
    enc_seq_len = 375
    encoder_hidden = torch.randn(B, enc_seq_len, 1280)

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(computer, (encoder_hidden,), strict=False)

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="encoder_hidden_states", shape=(1, enc_seq_len, 1280))],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    output_path = "build/cross_cache_computer.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    print("\n✅ Done! Cross-cache computer exported.")
    print("\nOutputs (8 cross-caches):")
    print("  var_76, var_93, var_110, var_127, var_144, var_161, var_178, var_195")
    print("\nUsage in Swift:")
    print("  1. Run encoder to get encoder_hidden_states")
    print("  2. Call cross_cache_computer.prediction(encoder_hidden_states)")
    print("  3. Extract 8 cross-caches from output")
    print("  4. Use these caches with decoder_with_cache")


if __name__ == "__main__":
    export_cross_cache_computer()
