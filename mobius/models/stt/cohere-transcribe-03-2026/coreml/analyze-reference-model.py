#!/usr/bin/env python3
"""Analyze reference model's computation to understand the difference."""

import torch
import coremltools as ct
import numpy as np

print("="*70)
print("Analyzing Reference Model")
print("="*70)

# Load reference decoder
ref_decoder = ct.models.MLModel("barathwaj-models/cohere_decoder_cached.mlpackage")

# Create test inputs for step 3
encoder_hidden = np.random.randn(1, 376, 1024).astype(np.float16)
input_id = np.array([[16]], dtype=np.int32)
cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

# Fill cache with some values for first 3 positions (as if we decoded 3 tokens)
cache_k[:, :, :3, :] = np.random.randn(8, 8, 3, 128).astype(np.float16)
cache_v[:, :, :3, :] = np.random.randn(8, 8, 3, 128).astype(np.float16)

step = np.array([3], dtype=np.int32)
cross_mask = np.ones((1, 1, 1, 376), dtype=np.float16)

print("\nInput shapes:")
print(f"  encoder_hidden: {encoder_hidden.shape}")
print(f"  input_id: {input_id.shape}")
print(f"  cache_k: {cache_k.shape}")
print(f"  cache_v: {cache_v.shape}")
print(f"  step: {step.shape}")
print(f"  cross_mask: {cross_mask.shape}")

# Run reference at step 0
print("\n--- Step 0 ---")
output_0 = ref_decoder.predict({
    "encoder_hidden_states": encoder_hidden,
    "input_id": np.array([[13764]], dtype=np.int32),
    "cache_k": np.zeros((8, 8, 108, 128), dtype=np.float16),
    "cache_v": np.zeros((8, 8, 108, 128), dtype=np.float16),
    "step": np.array([0], dtype=np.int32),
    "cross_attention_mask": cross_mask,
})

print("Output keys:", list(output_0.keys()))
for key, value in output_0.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")

# Run reference at step 3
print("\n--- Step 3 ---")
output_3 = ref_decoder.predict({
    "encoder_hidden_states": encoder_hidden,
    "input_id": input_id,
    "cache_k": cache_k,
    "cache_v": cache_v,
    "step": step,
    "cross_attention_mask": cross_mask,
})

print("Output keys:", list(output_3.keys()))
for key, value in output_3.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
        if 'logits' in key.lower() or (len(value.shape) == 2 and value.shape[1] == 16384):
            print(f"    Top-5 indices: {np.argsort(value[0])[-5:][::-1]}")
            print(f"    Top-5 values: {np.sort(value[0])[-5:][::-1]}")

print("\n" + "="*70)
