#!/usr/bin/env python3
"""Test if Cohere-style preprocessor has correct normalization."""
import numpy as np
import soundfile as sf
import librosa
import coremltools as ct

print("=== Testing Cohere-style Preprocessor Normalization ===\n")

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)\n")

# Test Cohere-style preprocessor
print("Testing Cohere-style preprocessor (build/cohere-preprocessor/):")
audio_30s = np.pad(audio, (0, 480000 - len(audio)), mode='constant')
preprocessor = ct.models.MLModel("build/cohere-preprocessor/preprocessor.mlpackage")
prep_out = preprocessor.predict({
    "audio_signal": audio_30s.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel = prep_out["mel_features"]
mel_length = int(prep_out["mel_length"][0])

print(f"   Mel shape: {mel.shape}")
print(f"   Valid mel length: {mel_length}")
print(f"   Mel statistics (full):")
print(f"     min={mel.min():.3f}, max={mel.max():.3f}, mean={mel.mean():.6f}, std={mel.std():.6f}")

# Check statistics on valid region only
mel_valid = mel[:, :, :mel_length]
print(f"   Mel statistics (valid region only, :, :, :{mel_length}):")
print(f"     min={mel_valid.min():.3f}, max={mel_valid.max():.3f}, mean={mel_valid.mean():.6f}, std={mel_valid.std():.6f}")

# Expected: mean should be ~0.0
if abs(mel_valid.mean()) < 0.01:
    print("   ✅ Normalization correct (mean ≈ 0)")
else:
    print(f"   ❌ Normalization incorrect (mean = {mel_valid.mean():.6f}, expected ≈ 0)")

# Also test old preprocessor for comparison
print("\nTesting old preprocessor (build/hf-upload/):")
preprocessor_old = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
prep_old_out = preprocessor_old.predict({
    "audio_signal": audio_30s.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel_old = prep_old_out["mel_features"]
mel_old_length = int(prep_old_out["mel_length"][0])

print(f"   Mel shape: {mel_old.shape}")
print(f"   Valid mel length: {mel_old_length}")

mel_old_valid = mel_old[:, :, :mel_old_length]
print(f"   Mel statistics (valid region):")
print(f"     min={mel_old_valid.min():.3f}, max={mel_old_valid.max():.3f}, mean={mel_old_valid.mean():.6f}, std={mel_old_valid.std():.6f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if abs(mel_valid.mean()) < 0.01 and abs(mel_old_valid.mean()) > 0.5:
    print("✅ Cohere-style preprocessor fixes the normalization issue!")
    print("   Next step: Use this preprocessor instead of the old one")
elif abs(mel_valid.mean()) < 0.01 and abs(mel_old_valid.mean()) < 0.01:
    print("✅ Both preprocessors have correct normalization")
    print("   The issue must be elsewhere")
else:
    print("❌ Both preprocessors have normalization issues")
    print("   Need to debug the normalization implementation")
