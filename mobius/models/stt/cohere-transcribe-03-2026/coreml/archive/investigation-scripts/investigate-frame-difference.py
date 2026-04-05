#!/usr/bin/env python3
"""Investigate why BarathwajAnandan gets 438 frames vs our 375 frames."""
import torch
import numpy as np
import soundfile as sf
import librosa
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("=== Investigating Frame Count Difference ===\n")

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)\n")

# Test 1: Our preprocessing
print("1. Our Preprocessing (FluidInference):")
audio_30s = np.pad(audio, (0, 480000 - len(audio)), mode='constant')
preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
prep_out = preprocessor.predict({
    "audio_signal": audio_30s.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
our_mel = prep_out["mel_features"]
our_mel_length = int(prep_out["mel_length"][0])
print(f"   Mel shape: {our_mel.shape}")
print(f"   Valid mel length: {our_mel_length}")

# Run encoder
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")
our_enc = encoder.predict({"input_features": our_mel})
our_encoder_out = our_enc["encoder_output"]
print(f"   Encoder output: {our_encoder_out.shape}")
print(f"   Downsampling ratio: {our_mel.shape[2] / our_encoder_out.shape[1]:.2f}x")
print(f"   Expected frames from {our_mel_length}: {our_mel_length / 8.0:.1f}")

# Test 2: Their preprocessing
print("\n2. Their Preprocessing (BarathwajAnandan):")
audio_35s = np.pad(audio, (0, 560000 - len(audio)), mode='constant')
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage",
                            compute_units=ct.ComputeUnit.CPU_ONLY)
frontend_out = frontend.predict({
    "audio_samples": audio_35s.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([len(audio)], dtype=np.int32)
})
their_mel = max(frontend_out.values(), key=lambda x: x.size)
their_mel_length = int(frontend_out["cast_2"][0])
print(f"   Mel shape: {their_mel.shape}")
print(f"   Valid mel length: {their_mel_length}")

# Run encoder
their_encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage",
                                 compute_units=ct.ComputeUnit.CPU_ONLY)
their_enc = their_encoder.predict({
    "input_features": their_mel,
    "feature_length": np.array([int(their_mel_length)], dtype=np.int32)
})
their_encoder_out = list(their_enc.values())[0]
print(f"   Encoder output: {their_encoder_out.shape}")
print(f"   Downsampling ratio: {their_mel.shape[2] / their_encoder_out.shape[1]:.2f}x")
print(f"   Expected frames from {their_mel_length}: {their_mel_length / 8.0:.1f}")

# Test 3: Transformers preprocessing
print("\n3. Transformers Preprocessing:")
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

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
print(f"   Input features shape: {inputs['input_features'].shape}")
print(f"   Length: {inputs['length'].item()}")

with torch.no_grad():
    encoder_out = model.encoder(inputs["input_features"])
    if isinstance(encoder_out, tuple):
        encoder_hidden = encoder_out[0]
    else:
        encoder_hidden = encoder_out

print(f"   Encoder output: {encoder_hidden.shape}")
print(f"   Downsampling ratio: {inputs['input_features'].shape[2] / encoder_hidden.shape[1]:.2f}x")

# Analysis
print(f"\n{'='*60}")
print("ANALYSIS")
print('='*60)

print(f"\nMel features comparison:")
print(f"  Ours:        {our_mel_length} frames → {our_encoder_out.shape[1]} encoder frames")
print(f"  Theirs:      {their_mel_length} frames → {their_encoder_out.shape[1]} encoder frames")
print(f"  Transformers: {inputs['input_features'].shape[2]} frames → {encoder_hidden.shape[1]} encoder frames")

print(f"\nDownsampling ratios:")
print(f"  Ours:        {our_mel.shape[2] / our_encoder_out.shape[1]:.2f}x")
print(f"  Theirs:      {their_mel.shape[2] / their_encoder_out.shape[1]:.2f}x")
print(f"  Transformers: {inputs['input_features'].shape[2] / encoder_hidden.shape[1]:.2f}x")

print(f"\nHypothesis:")
if abs(their_mel.shape[2] / their_encoder_out.shape[1] - 8.0) < 0.1:
    print("  ✅ Their encoder uses 8x downsampling (standard)")
else:
    print(f"  ⚠️  Their encoder uses {their_mel.shape[2] / their_encoder_out.shape[1]:.2f}x downsampling (unusual)")

if abs(our_mel.shape[2] / our_encoder_out.shape[1] - 8.0) < 0.1:
    print("  ✅ Our encoder uses 8x downsampling (standard)")
else:
    print(f"  ⚠️  Our encoder uses {our_mel.shape[2] / our_encoder_out.shape[1]:.2f}x downsampling (unusual)")

# Check actual mel feature values
print(f"\nMel features statistics:")
print(f"  Ours:        min={our_mel.min():.3f}, max={our_mel.max():.3f}, mean={our_mel.mean():.6f}")
print(f"  Theirs:      min={their_mel.min():.3f}, max={their_mel.max():.3f}, mean={their_mel.mean():.6f}")
print(f"  Transformers: min={inputs['input_features'].min():.3f}, max={inputs['input_features'].max():.3f}, mean={inputs['input_features'].mean():.6f}")
