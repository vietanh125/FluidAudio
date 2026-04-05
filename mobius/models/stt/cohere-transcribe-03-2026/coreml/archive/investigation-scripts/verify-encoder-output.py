#!/usr/bin/env python3
"""Verify that our CoreML encoder produces reasonable output."""
import torch
import numpy as np
import soundfile as sf
import librosa
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("=== Encoder Output Verification ===\n")

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
print(f"Audio: {len(audio)} samples @ 16000Hz")

# Test 1: Transformers (reference)
print("\n1. Transformers Reference:")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
model.eval()

with torch.no_grad():
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    encoder_output = model.encoder(inputs["input_features"])
    # Encoder returns tuple (hidden_states, ...)
    ref_encoder_out = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output

print(f"   Shape: {ref_encoder_out.shape}")
print(f"   Stats: min={ref_encoder_out.min():.3f}, max={ref_encoder_out.max():.3f}, mean={ref_encoder_out.mean():.6f}")
print(f"   First 5 values: {ref_encoder_out[0, 0, :5]}")

# Test 2: CoreML Preprocessor + Encoder
print("\n2. CoreML Pipeline:")
# Pad audio
audio_padded = np.pad(audio, (0, 480000 - len(audio)), mode='constant') if len(audio) < 480000 else audio[:480000]

preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")

prep_out = preprocessor.predict({
    "audio_signal": audio_padded.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel_features = prep_out["mel_features"]
mel_length = int(prep_out["mel_length"][0])
print(f"   Preprocessor output shape: {mel_features.shape} (valid length: {mel_length})")

# Encoder expects fixed 3000 frames
enc_out = encoder.predict({"input_features": mel_features})
coreml_encoder_out_full = enc_out["encoder_output"]

# Encoder downsamples by 8x (mel frames → encoder frames)
# 3000 mel → 375 encoder, so 1043 mel → ~130 encoder
downsampling_ratio = coreml_encoder_out_full.shape[1] / mel_features.shape[2]  # 375 / 3000 = 0.125 = 1/8
valid_encoder_length = int(mel_length * downsampling_ratio)
print(f"   Encoder output shape: {coreml_encoder_out_full.shape} (valid length: {valid_encoder_length})")

# Trim to valid length
coreml_encoder_out = coreml_encoder_out_full[:, :valid_encoder_length, :]
print(f"   Trimmed to: {coreml_encoder_out.shape}")
print(f"   Stats: min={coreml_encoder_out.min():.3f}, max={coreml_encoder_out.max():.3f}, mean={coreml_encoder_out.mean():.6f}")
print(f"   First 5 values: {coreml_encoder_out[0, 0, :5]}")

# Compare
print("\n3. Comparison:")
min_len = min(ref_encoder_out.shape[1], coreml_encoder_out.shape[1])
ref_trimmed = ref_encoder_out[:, :min_len, :].numpy()
coreml_trimmed = coreml_encoder_out[:, :min_len, :]

diff = np.abs(ref_trimmed - coreml_trimmed)
print(f"   Max difference: {diff.max():.6f}")
print(f"   Mean difference: {diff.mean():.6f}")
print(f"   Median difference: {np.median(diff):.6f}")

if diff.max() < 0.01:
    print("   ✅ Excellent match!")
elif diff.max() < 0.1:
    print("   ✅ Good match")
elif diff.max() < 1.0:
    print("   ⚠️  Acceptable match")
else:
    print(f"   ❌ Poor match - encoder output is wrong!")
