#!/usr/bin/env python3
"""Test: FluidInference encoder + projection vs BarathwajAnandan encoder."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa

print("=== Testing Encoder + Projection Fix ===\n")

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

audio_padded = np.pad(audio, (0, 480000 - len(audio)), mode='constant')
print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)\n")

# Test 1: FluidInference preprocessor + encoder + NEW projection
print("1. FluidInference Models WITH Projection:")
print("   Loading preprocessor...")
preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
prep_out = preprocessor.predict({
    "audio_signal": audio_padded.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel_features = prep_out["mel_features"]
mel_length = int(prep_out["mel_length"][0])
print(f"   Preprocessor: {mel_features.shape} (valid: {mel_length})")

print("   Loading encoder...")
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")
enc_out = encoder.predict({"input_features": mel_features})
encoder_1280 = enc_out["encoder_output"]
print(f"   Encoder output: {encoder_1280.shape} (1280-dim)")

print("   Loading projection...")
projection = ct.models.MLModel("build/encoder_decoder_proj.mlpackage")
proj_out = projection.predict({"encoder_output": encoder_1280})
our_encoder_1024 = proj_out["projected_output"]

# Trim to valid length
valid_encoder_len = int(mel_length * (encoder_1280.shape[1] / mel_features.shape[2]))
our_encoder_trimmed = our_encoder_1024[:, :valid_encoder_len, :]

print(f"   Projected output: {our_encoder_1024.shape} (1024-dim!)")
print(f"   Stats: min={our_encoder_trimmed.min():.3f}, max={our_encoder_trimmed.max():.3f}, mean={our_encoder_trimmed.mean():.6f}")
print(f"   First 5: {our_encoder_trimmed[0, 0, :5]}")

# Test 2: BarathwajAnandan's encoder
print("\n2. BarathwajAnandan's Encoder:")
print("   Loading frontend...")
audio_35s = np.pad(audio, (0, 560000 - len(audio)), mode='constant')
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage",
                            compute_units=ct.ComputeUnit.CPU_ONLY)
frontend_out = frontend.predict({
    "audio_samples": audio_35s.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([len(audio)], dtype=np.int32)
})
their_mel = max(frontend_out.values(), key=lambda x: x.size)
their_feature_length = int(frontend_out["cast_2"][0])
print(f"   Frontend: {their_mel.shape} (valid: {their_feature_length})")

print("   Loading encoder...")
their_encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage",
                                 compute_units=ct.ComputeUnit.CPU_ONLY)
their_enc_out = their_encoder.predict({
    "input_features": their_mel,
    "feature_length": np.array([int(their_feature_length)], dtype=np.int32)
})
their_encoder_output = list(their_enc_out.values())[0]

print(f"   Encoder output: {their_encoder_output.shape} (1024-dim)")
print(f"   Stats: min={their_encoder_output.min():.3f}, max={their_encoder_output.max():.3f}, mean={their_encoder_output.mean():.6f}")
print(f"   First 5: {their_encoder_output[0, 0, :5]}")

# Compare
print("\n3. Comparison:")
min_len = min(our_encoder_trimmed.shape[1], their_encoder_output.shape[1])
print(f"   Comparing first {min_len} frames...")

ours = our_encoder_trimmed[:, :min_len, :]
theirs = their_encoder_output[:, :min_len, :]

diff = np.abs(ours - theirs)
print(f"   Max difference: {diff.max():.6f}")
print(f"   Mean difference: {diff.mean():.6f}")
print(f"   Median difference: {np.median(diff):.6f}")

print(f"\n{'='*60}")
if diff.max() < 0.1:
    print("✅ SUCCESS! Encoders match closely!")
    print("The projection layer fixes the dimensional mismatch.")
    print("Our pipeline should now work with the decoder.")
elif diff.max() < 1.0:
    print("⚠️  Encoders are similar but not identical")
    print("This might be due to:")
    print("  - Different preprocessing (our preprocessor vs their frontend)")
    print("  - Different quantization (FP16 vs 6-bit)")
    print("Should still work for transcription.")
else:
    print("❌ Encoders are very different")
    print("There may be other architectural differences.")
print('='*60)
