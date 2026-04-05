#!/usr/bin/env python3
"""Debug preprocessor differences to understand the 0.43 max error."""
import torch
import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("=== Debugging Preprocessor Difference ===\n")

# Load audio
audio, sr = sf.read("test-audio.wav")
print(f"Audio: {len(audio)} samples @ {sr} Hz")

# Test 1: Transformers processor (reference)
print("\n=== Reference (Transformers) ===")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
ref_mel = inputs["input_features"].numpy()
print(f"Shape: {ref_mel.shape}")
print(f"Stats: min={ref_mel.min():.3f}, max={ref_mel.max():.3f}, mean={ref_mel.mean():.6f}, std={ref_mel.std():.3f}")

# Test 2: Our CoreML preprocessor
print("\n=== Our CoreML Preprocessor ===")
target_length = 480000
if len(audio) < target_length:
    audio_padded = np.pad(audio, (0, target_length - len(audio)), mode='constant')
else:
    audio_padded = audio[:target_length]

preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
prep_out = preprocessor.predict({
    "audio_signal": audio_padded.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
our_mel = prep_out["mel_features"]
print(f"Shape: {our_mel.shape}")
print(f"Stats: min={our_mel.min():.3f}, max={our_mel.max():.3f}, mean={our_mel.mean():.6f}, std={our_mel.std():.3f}")

# Comparison
print("\n=== Detailed Comparison ===")
min_len = min(ref_mel.shape[2], our_mel.shape[2])
ref_trimmed = ref_mel[:, :, :min_len]
our_trimmed = our_mel[:, :, :min_len]

diff = np.abs(ref_trimmed - our_trimmed)
print(f"Shape match: {ref_trimmed.shape} vs {our_trimmed.shape}")
print(f"\nDifference stats:")
print(f"  Max: {diff.max():.6f}")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Median: {np.median(diff):.6f}")
print(f"  95th percentile: {np.percentile(diff, 95):.6f}")
print(f"  99th percentile: {np.percentile(diff, 99):.6f}")

# Find where max difference occurs
max_idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\nMax difference at: batch={max_idx[0]}, mel_bin={max_idx[1]}, frame={max_idx[2]}")
print(f"  Reference value: {ref_trimmed[max_idx]:.6f}")
print(f"  Our value: {our_trimmed[max_idx]:.6f}")

# Check specific mel bins
print("\n=== Per Mel Bin Analysis ===")
for mel_bin in [0, 32, 64, 96, 127]:
    ref_bin = ref_trimmed[0, mel_bin, :]
    our_bin = our_trimmed[0, mel_bin, :]
    bin_diff = np.abs(ref_bin - our_bin)
    print(f"Mel bin {mel_bin}:")
    print(f"  Max diff: {bin_diff.max():.6f}, Mean diff: {bin_diff.mean():.6f}")
    print(f"  Ref mean: {ref_bin.mean():.6f}, Our mean: {our_bin.mean():.6f}")

# Check temporal regions
print("\n=== Temporal Analysis ===")
for region in ["Start (0-100)", "Middle (500-600)", "End (-100:-1)"]:
    if region.startswith("Start"):
        ref_region = ref_trimmed[:, :, :100]
        our_region = our_trimmed[:, :, :100]
    elif region.startswith("Middle"):
        ref_region = ref_trimmed[:, :, 500:600]
        our_region = our_trimmed[:, :, 500:600]
    else:
        ref_region = ref_trimmed[:, :, -100:]
        our_region = our_trimmed[:, :, -100:]

    region_diff = np.abs(ref_region - our_region)
    print(f"{region}: max_diff={region_diff.max():.6f}, mean_diff={region_diff.mean():.6f}")

# Check normalization
print("\n=== Normalization Check ===")
print("Reference per-mel-bin stats:")
for mel_bin in [0, 64, 127]:
    ref_bin = ref_trimmed[0, mel_bin, :]
    print(f"  Bin {mel_bin}: mean={ref_bin.mean():.6f}, std={ref_bin.std():.6f}")

print("\nOur per-mel-bin stats:")
for mel_bin in [0, 64, 127]:
    our_bin = our_trimmed[0, mel_bin, :]
    print(f"  Bin {mel_bin}: mean={our_bin.mean():.6f}, std={our_bin.std():.6f}")

print("\n=== Conclusion ===")
if diff.max() < 0.01:
    print("✅ Excellent match (< 0.01)")
elif diff.max() < 0.1:
    print("✅ Good match (< 0.1)")
elif diff.max() < 0.5:
    print("⚠️  Acceptable match (< 0.5)")
else:
    print(f"❌ Large difference ({diff.max():.3f})")
    print("Likely causes:")
    print("  - Different STFT implementation (window function, padding)")
    print("  - Different mel filterbank (edge frequencies, normalization)")
    print("  - Different normalization method")
    print("  - Float precision issues during conversion")
