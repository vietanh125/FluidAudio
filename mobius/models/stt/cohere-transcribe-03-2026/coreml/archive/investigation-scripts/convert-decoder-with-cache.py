#!/usr/bin/env python3
"""Export Cohere decoder with explicit KV cache support for CoreML.

This follows the PocketTTS pattern for explicit KV cache management:
- Fixed-size cache buffers: (2, B, max_seq_len, H, D)
- Ring buffer indexing at max_seq_len
- Scatter operations for cache updates
"""
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq


class CohereDecoderWithCache(nn.Module):
    """Cohere decoder with explicit KV cache tensors for CoreML export.

    Architecture:
    - 8 decoder layers
    - 8 attention heads per layer
    - 128 head_dim
    - max_seq_len=512 (ring buffer)

    Cache Structure:
    - self_cache: (2, B, max_seq_len, H, D) - grows with generation
    - cross_cache: (2, B, enc_seq_len, H, D) - constant from encoder
    """

    def __init__(self, decoder_wrapper, proj_layer, num_layers: int = 8, max_seq_len: int = 512):
        super().__init__()
        self.transf_decoder = decoder_wrapper
        self.encoder_decoder_proj = proj_layer
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = 8
        self.head_dim = 128

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, 1) - single token
        position: torch.Tensor,   # (B,) - scalar position
        encoder_hidden_states: torch.Tensor,  # (B, enc_seq_len, 1280)
        # 8 self-attention caches (updated each step)
        self_cache0: torch.Tensor,  # (2, B, max_seq_len, H, D)
        self_cache1: torch.Tensor,
        self_cache2: torch.Tensor,
        self_cache3: torch.Tensor,
        self_cache4: torch.Tensor,
        self_cache5: torch.Tensor,
        self_cache6: torch.Tensor,
        self_cache7: torch.Tensor,
        # 8 cross-attention caches (constant)
        cross_cache0: torch.Tensor,  # (2, B, enc_seq_len, H, D)
        cross_cache1: torch.Tensor,
        cross_cache2: torch.Tensor,
        cross_cache3: torch.Tensor,
        cross_cache4: torch.Tensor,
        cross_cache5: torch.Tensor,
        cross_cache6: torch.Tensor,
        cross_cache7: torch.Tensor,
    ) -> tuple:
        """
        Returns:
            hidden_states: (B, 1, 1024) - decoder output
            new_position: (B,) - incremented position
            new_self_cache0-7: Updated self-attention caches
        """
        B = input_ids.shape[0]
        H = self.num_heads
        D = self.head_dim
        max_len = self.max_seq_len

        # Project encoder hidden states once
        encoder_proj = self.encoder_decoder_proj(encoder_hidden_states)

        # Collect self and cross caches
        self_caches_in = [
            self_cache0, self_cache1, self_cache2, self_cache3,
            self_cache4, self_cache5, self_cache6, self_cache7
        ]
        cross_caches_in = [
            cross_cache0, cross_cache1, cross_cache2, cross_cache3,
            cross_cache4, cross_cache5, cross_cache6, cross_cache7
        ]

        # Prepare updated self caches
        new_self_caches = []

        # Get embeddings
        positions_int = position.long().unsqueeze(-1)  # (B, 1)
        x = self.transf_decoder._embedding(input_ids, positions_int)  # (B, 1, 1024)

        # Process through each decoder layer
        for layer_idx, layer in enumerate(self.transf_decoder._decoder.layers):
            # Self-attention with cache
            self_cache = self_caches_in[layer_idx]  # (2, B, max_seq_len, H, D)

            # First sub-layer: self-attention
            self_attn = layer.first_sub_layer

            # Compute Q, K, V
            query = self_attn.query_net(x).view(B, 1, H, D)
            key = self_attn.key_net(x).view(B, 1, H, D)
            value = self_attn.value_net(x).view(B, 1, H, D)

            # Update cache with ring buffer
            # Position is a float tensor, convert to int for indexing
            pos_float = position.float()
            write_idx = (pos_float % max_len).long().view(B, 1, 1, 1).expand(B, 1, H, D)

            # Create new cache by scattering new K/V at position
            new_self_cache = self_cache.clone()
            new_self_cache[0] = new_self_cache[0].scatter(1, write_idx, key)
            new_self_cache[1] = new_self_cache[1].scatter(1, write_idx, value)

            # Get valid length for attention (up to current position + 1)
            valid_len = (pos_float + 1).long().clamp(max=max_len)

            # Extract cached K/V for attention
            # Shape: (B, valid_len, H, D)
            cached_key = new_self_cache[0][:, :valid_len.max().item()]
            cached_value = new_self_cache[1][:, :valid_len.max().item()]

            # Self-attention computation
            # Q: (B, 1, H, D), K: (B, seq_len, H, D)
            scores = torch.einsum('bqhd,bkhd->bhqk', query, cached_key) / (D ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            self_attn_out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, cached_value)
            self_attn_out = self_attn_out.reshape(B, 1, H * D)
            self_attn_out = self_attn.out_projection(self_attn_out)

            # Residual and norm
            x = layer.first_sub_layer_norm(x + self_attn_out)

            # Second sub-layer: cross-attention
            cross_cache = cross_caches_in[layer_idx]  # (2, B, enc_seq_len, H, D)
            cross_attn = layer.second_sub_layer

            # Query from decoder, K/V from cache
            query = cross_attn.query_net(x).view(B, 1, H, D)
            cached_key = cross_cache[0]  # (B, enc_seq_len, H, D)
            cached_value = cross_cache[1]

            # Cross-attention computation
            scores = torch.einsum('bqhd,bkhd->bhqk', query, cached_key) / (D ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            cross_attn_out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, cached_value)
            cross_attn_out = cross_attn_out.reshape(B, 1, H * D)
            cross_attn_out = cross_attn.out_projection(cross_attn_out)

            # Residual and norm
            x = layer.second_sub_layer_norm(x + cross_attn_out)

            # FFN
            ffn_out = layer.ffn(x)
            x = layer.final_layer_norm(x + ffn_out)

            new_self_caches.append(new_self_cache)

        # Final layer norm
        hidden_states = self.transf_decoder._decoder.layer_norm(x)

        # Increment position for next step
        new_position = position + 1.0

        return (
            hidden_states,
            new_position,
            *new_self_caches  # Return all 8 updated self caches
        )


def export_decoder_with_cache():
    """Export Cohere decoder with KV cache to CoreML."""
    print("Loading Cohere Transcribe model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()

    print("Creating decoder with cache wrapper...")
    decoder_with_cache = CohereDecoderWithCache(
        decoder_wrapper=model.transf_decoder,
        proj_layer=model.encoder_decoder_proj,
        num_layers=8,
        max_seq_len=512
    )
    decoder_with_cache.eval()

    # Create dummy inputs
    B = 1
    enc_seq_len = 375  # From encoder output
    max_seq_len = 512
    H = 8
    D = 128

    input_ids = torch.tensor([[13764]], dtype=torch.long)
    position = torch.tensor([0.0], dtype=torch.float32)
    encoder_hidden = torch.randn(B, enc_seq_len, 1280)

    # Initialize caches
    self_caches = [torch.zeros(2, B, max_seq_len, H, D) for _ in range(8)]
    cross_caches = [torch.randn(2, B, enc_seq_len, H, D) for _ in range(8)]

    print("Tracing decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(
            decoder_with_cache,
            (input_ids, position, encoder_hidden, *self_caches, *cross_caches),
            strict=False
        )

    print("Converting to CoreML...")

    # Define inputs
    inputs = [
        ct.TensorType(name="input_ids", shape=(B, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(B,), dtype=np.float32),
        ct.TensorType(name="encoder_hidden_states", shape=(B, enc_seq_len, 1280)),
    ]

    # Add self-attention cache inputs
    for i in range(8):
        inputs.append(
            ct.TensorType(name=f"self_cache{i}", shape=(2, B, max_seq_len, H, D))
        )

    # Add cross-attention cache inputs
    for i in range(8):
        inputs.append(
            ct.TensorType(name=f"cross_cache{i}", shape=(2, B, enc_seq_len, H, D))
        )

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    output_path = "build/decoder_with_cache.mlpackage"
    Path("build").mkdir(exist_ok=True)
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    print("\n✅ Done! Decoder with KV cache exported.")
    print("\nOutputs:")
    print("  - var_2384: hidden_states (B, 1, 1024)")
    print("  - var_2387: new_position (B,)")
    print("  - new_self_cache_X_internal_tensor_assign_2: Updated self caches (8 outputs)")
    print("\nNext: Export cross-cache computer with export-cross-cache-computer.py")


if __name__ == "__main__":
    export_decoder_with_cache()
