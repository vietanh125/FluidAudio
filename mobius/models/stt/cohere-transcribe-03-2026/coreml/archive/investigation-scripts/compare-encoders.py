#!/usr/bin/env python3
"""Compare our encoder output with BarathwajAnandan's encoder output."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa

print("=== Comparing Encoder Outputs ===\n")

# Load models
print("1. Loading models...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
our_encoder = ct.models.MLModel("build/ultra_static_encoder.mlpackage")
their_encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage")
print("   ✓ Models loaded")

# Load audio
print("\n2. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

original_len = len(audio)
if len(audio) < 560000:
    audio = np.pad(audio, (0, 560000 - len(audio)))
else:
    audio = audio[:560000]

# Get mel spectrogram
print("\n3. Running frontend...")
frontend_out = frontend.predict({
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([original_len], dtype=np.int32)
})
mel = frontend_out["var_6916"]
print(f"   Mel shape: {mel.shape}")
print(f"   Mel stats: min={mel.min():.6f}, max={mel.max():.6f}, mean={mel.mean():.6f}")

# Run our encoder
print("\n4. Running OUR encoder...")
our_out = our_encoder.predict({"input_features": mel})
our_hidden = our_out["encoder_output"]
print(f"   Shape: {our_hidden.shape}")
print(f"   Stats: min={our_hidden.min():.6f}, max={our_hidden.max():.6f}, mean={our_hidden.mean():.6f}, std={our_hidden.std():.6f}")

# Run their encoder
print("\n5. Running BarathwajAnandan's encoder...")
their_out = their_encoder.predict({
    "input_features": mel,
    "feature_length": np.array([3501], dtype=np.int32)
})
their_hidden = their_out["var_6733"]
print(f"   Shape: {their_hidden.shape}")
print(f"   Stats: min={their_hidden.min():.6f}, max={their_hidden.max():.6f}, mean={their_hidden.mean():.6f}, std={their_hidden.std():.6f}")

# Compare
print("\n6. Comparison:")
diff = np.abs(our_hidden - their_hidden)
print(f"   Max absolute diff: {diff.max():.6f}")
print(f"   Mean absolute diff: {diff.mean():.6f}")
print(f"   Median absolute diff: {np.median(diff):.6f}")

# Check if they're similar
if diff.max() < 0.01:
    print("\n   ✅ EXCELLENT - Encoders match closely")
elif diff.max() < 0.1:
    print("\n   ✅ GOOD - Encoders are similar")
elif diff.max() < 1.0:
    print("\n   ⚠️  MODERATE - Some differences")
else:
    print("\n   ❌ LARGE DIFFERENCE - Encoders produce very different outputs")

# Show sample values
print("\n7. Sample values (first 10 elements of first frame):")
print(f"   Ours:   {our_hidden[0, 0, :10]}")
print(f"   Theirs: {their_hidden[0, 0, :10]}")
print(f"   Diff:   {diff[0, 0, :10]}")
