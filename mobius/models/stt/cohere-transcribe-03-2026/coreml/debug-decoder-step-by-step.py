#!/usr/bin/env python3
"""Debug decoder step-by-step to find where it diverges."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset

print("="*70)
print("Debugging Decoder Step-by-Step")
print("="*70)

# Load LibriSpeech sample
print("\n[1/4] Loading test audio...")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", streaming=True)
sample = next(iter(dataset))
audio = np.array(sample["audio"]["array"], dtype=np.float32)
ground_truth = sample["text"].lower()
print(f"   Audio: {len(audio)} samples")
print(f"   Ground truth: {ground_truth}")

# Compute mel
print("\n[2/4] Computing mel spectrogram...")
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)
print(f"   Mel shape: {mel_padded.shape}")

# Load models
print("\n[3/4] Loading models...")
our_encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
our_decoder = ct.models.MLModel("build/cohere_decoder_cached.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
ref_decoder = ct.models.MLModel("barathwaj-models/cohere_decoder_cached.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
print("   ✓ Models loaded")

# Get encoder output
encoder_output = our_encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([3001], dtype=np.int32)
})
encoder_hidden = None
for key, value in encoder_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3:
        encoder_hidden = value
        break
print(f"   Encoder output: {encoder_hidden.shape}")

# Initialize caches
print("\n[4/4] Running step-by-step decoding...")
decoder_start_token_id = 13764

# Our decoder
our_tokens = [decoder_start_token_id]
our_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
our_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

# Reference decoder
ref_tokens = [decoder_start_token_id]
ref_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
ref_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

# Decode 5 steps
for step in range(5):
    print(f"\n--- Step {step} ---")

    # Our decoder
    our_input = {
        "input_id": np.array([[our_tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
        "cache_k": our_cache_k,
        "cache_v": our_cache_v,
    }
    our_output = our_decoder.predict(our_input)

    # Extract outputs
    our_logits = None
    for key, value in our_output.items():
        if hasattr(value, 'shape'):
            if len(value.shape) == 2 and value.shape[1] > 1000:
                our_logits = value
            elif 'cache_k' in key.lower() or key == 'new_cache_k':
                our_cache_k = value
            elif 'cache_v' in key.lower() or key == 'new_cache_v':
                our_cache_v = value

    our_next_token = int(np.argmax(our_logits[0]))
    our_tokens.append(our_next_token)

    # Reference decoder
    ref_input = {
        "input_id": np.array([[ref_tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
        "cache_k": ref_cache_k,
        "cache_v": ref_cache_v,
    }
    ref_output = ref_decoder.predict(ref_input)

    # Extract outputs
    ref_logits = None
    for key, value in ref_output.items():
        if hasattr(value, 'shape'):
            if len(value.shape) == 2 and value.shape[1] > 1000:
                ref_logits = value
            elif 'k' in key.lower():
                ref_cache_k = value
            else:
                ref_cache_v = value

    ref_next_token = int(np.argmax(ref_logits[0]))
    ref_tokens.append(ref_next_token)

    # Compare
    print(f"Our token: {our_next_token}, Ref token: {ref_next_token}")
    print(f"Match: {our_next_token == ref_next_token}")

    # Compare top-5 logits
    our_top5_indices = np.argsort(our_logits[0])[-5:][::-1]
    ref_top5_indices = np.argsort(ref_logits[0])[-5:][::-1]
    print(f"Our top-5 tokens: {our_top5_indices.tolist()}")
    print(f"Ref top-5 tokens: {ref_top5_indices.tolist()}")
    print(f"Our top-5 logits: {[float(our_logits[0][i]) for i in our_top5_indices]}")
    print(f"Ref top-5 logits: {[float(ref_logits[0][i]) for i in ref_top5_indices]}")

    # Compare cache statistics
    print(f"\nCache stats:")
    print(f"Our cache_k: min={our_cache_k.min():.6f}, max={our_cache_k.max():.6f}, mean={our_cache_k.mean():.6f}")
    print(f"Ref cache_k: min={ref_cache_k.min():.6f}, max={ref_cache_k.max():.6f}, mean={ref_cache_k.mean():.6f}")

    # Compare cache in valid region (up to current step+1)
    valid_len = step + 1
    our_valid_cache = our_cache_k[:, :, :valid_len, :]
    ref_valid_cache = ref_cache_k[:, :, :valid_len, :]
    cache_diff = np.abs(our_valid_cache - ref_valid_cache)
    print(f"Valid cache diff: max={cache_diff.max():.6f}, mean={cache_diff.mean():.6f}")

    if our_next_token != ref_next_token:
        print(f"\n⚠️  DIVERGENCE at step {step}!")
        break

print(f"\nFinal tokens:")
print(f"Our: {our_tokens}")
print(f"Ref: {ref_tokens}")
