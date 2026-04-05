#!/usr/bin/env python3
"""Test BarathwajAnandan's encoder with CPU-only execution to bypass ANE compilation error."""

import numpy as np
import soundfile as sf
import coremltools as ct

print("Testing BarathwajAnandan's encoder with CPU-ONLY execution")
print("="*70)

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    raise ValueError(f"Audio must be 16kHz, got {sr}Hz")

duration = len(audio) / 16000
print(f"Audio: test-librispeech-real.wav ({duration:.2f}s, {len(audio)} samples)")

# Pad to 560k
max_samples = 560000
if len(audio) < max_samples:
    audio_padded = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
else:
    audio_padded = audio[:max_samples]

# Frontend (CPU-only)
print("\n[1/3] Loading frontend with CPU-only...")
frontend = ct.models.MLModel(
    "build/barathwaj-models/cohere_frontend.mlpackage",
    compute_units=ct.ComputeUnit.CPU_ONLY
)
print("   ✓ Frontend loaded")

frontend_output = frontend.predict({
    "audio_samples": audio_padded.astype(np.float32).reshape(1, -1),
    "audio_length": np.array([len(audio_padded)], dtype=np.int32)
})

# Find mel spectrogram output (shape should be (1, 128, 3501))
mel = None
for key, value in frontend_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 128:
        mel = value
        print(f"   Mel: {value.shape} (key: {key})")
        break

if mel is None:
    raise ValueError(f"Could not find mel output. Available keys: {list(frontend_output.keys())}")

# Encoder (CPU-only)
print("\n[2/3] Loading encoder with CPU-only...")
try:
    encoder = ct.models.MLModel(
        "build/barathwaj-models/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    print("   ✓ Encoder loaded successfully with CPU-only!")

    print("\n[3/3] Running encoder inference...")
    encoder_output = encoder.predict({
        "input_features": mel.astype(np.float32),
        "feature_length": np.array([3501], dtype=np.int32)
    })

    print(f"   Available encoder outputs: {list(encoder_output.keys())}")

    # Find the encoder output (shape should be (1, 438, 1024))
    hidden_states = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: {value.shape}")
            if len(value.shape) == 3 and value.shape[1] == 438 and value.shape[2] == 1024:
                hidden_states = value
                print(f"   ✓ Found encoder output: {key}")

    if hidden_states is not None:
        print(f"\n✅ SUCCESS! Encoder works with CPU-only execution")
        print(f"   Output shape: {hidden_states.shape}")
        print(f"   Output range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
        print(f"\nConclusion: ANE compilation error can be bypassed with CPU_ONLY compute units")
    else:
        print(f"\n❌ Could not find (1, 438, 1024) output tensor")

except Exception as e:
    print(f"\n❌ Error loading encoder with CPU-only: {e}")
    import traceback
    traceback.print_exc()
