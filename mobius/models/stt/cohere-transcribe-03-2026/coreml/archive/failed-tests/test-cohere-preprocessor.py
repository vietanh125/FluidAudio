#!/usr/bin/env python3
"""Test Cohere's exact preprocessor with encoder to verify compatibility."""
import coremltools as ct
import numpy as np
import soundfile as sf
from transformers import AutoProcessor

print("=== Testing Cohere Preprocessor with Encoder ===\n")

# Load audio
print("Loading audio...")
audio, sr = sf.read("test-audio.wav")
target_length = 480000
if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
print(f"  Audio: {len(audio)} samples @ {sr} Hz")

# Test 1: Transformers processor (ground truth)
print("\n=== Test 1: Transformers Processor (Ground Truth) ===")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
ref_mel = inputs["input_features"].numpy()
ref_length = inputs["length"].numpy()

print(f"  Output shape: {ref_mel.shape}")
print(f"  Output length: {ref_length[0]}")
print(f"  Stats: min={ref_mel.min():.3f}, max={ref_mel.max():.3f}, mean={ref_mel.mean():.3f}, std={ref_mel.std():.3f}")
print(f"  First 10 values: {ref_mel[0, 0, :10]}")

# Test 2: Exported CoreML preprocessor
print("\n=== Test 2: Exported Cohere Preprocessor (CoreML) ===")
try:
    preprocessor = ct.models.MLModel("build/cohere-preprocessor/preprocessor.mlpackage")

    prep_out = preprocessor.predict({
        "audio_signal": audio.reshape(1, -1).astype(np.float32),
        "length": np.array([len(audio)], dtype=np.int32)
    })

    coreml_mel = prep_out["mel_features"]
    coreml_length = prep_out["mel_length"]

    print(f"  Output shape: {coreml_mel.shape}")
    print(f"  Output length: {coreml_length[0]}")
    print(f"  Stats: min={coreml_mel.min():.3f}, max={coreml_mel.max():.3f}, mean={coreml_mel.mean():.3f}, std={coreml_mel.std():.3f}")
    print(f"  First 10 values: {coreml_mel[0, 0, :10]}")

    # Compare
    print("\n=== Comparison ===")
    min_len = min(ref_mel.shape[2], coreml_mel.shape[2])
    diff = np.abs(ref_mel[:, :, :min_len] - coreml_mel[:, :, :min_len])

    print(f"  Max abs difference: {diff.max():.6f}")
    print(f"  Mean abs difference: {diff.mean():.6f}")

    if diff.max() < 0.01:
        print("  ✅ Excellent match! (< 0.01 difference)")
    elif diff.max() < 0.1:
        print("  ✅ Good match! (< 0.1 difference)")
    elif diff.max() < 1.0:
        print("  ⚠️  Acceptable match (< 1.0 difference)")
    else:
        print(f"  ❌ Poor match (difference: {diff.max():.3f})")

except FileNotFoundError:
    print("  ❌ Preprocessor not found. Run export-cohere-preprocessor.py first.")
    coreml_mel = None
    coreml_length = None

# Test 3: Encoder compatibility
print("\n=== Test 3: Encoder Compatibility ===")
if coreml_mel is not None:
    try:
        encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")

        # Try with CoreML preprocessor output
        print("Testing encoder with Cohere preprocessor output...")
        enc_out = encoder.predict({
            "input_features": coreml_mel
        })
        encoder_hidden = enc_out["encoder_output"]

        print(f"  ✅ Encoder works with Cohere preprocessor!")
        print(f"  Encoder output shape: {encoder_hidden.shape}")
        print(f"  Stats: min={encoder_hidden.min():.3f}, max={encoder_hidden.max():.3f}, mean={encoder_hidden.mean():.3f}")

        # Also test with reference (transformers) output
        print("\nTesting encoder with Transformers processor output...")
        enc_out_ref = encoder.predict({
            "input_features": ref_mel[:, :, :3000]  # Trim to 3000 if needed
        })
        encoder_hidden_ref = enc_out_ref["encoder_output"]

        print(f"  Encoder output shape: {encoder_hidden_ref.shape}")
        print(f"  Stats: min={encoder_hidden_ref.min():.3f}, max={encoder_hidden_ref.max():.3f}, mean={encoder_hidden_ref.mean():.3f}")

        # Compare encoder outputs
        print("\n=== Encoder Output Comparison ===")
        enc_diff = np.abs(encoder_hidden - encoder_hidden_ref)
        print(f"  Max abs difference: {enc_diff.max():.6f}")
        print(f"  Mean abs difference: {enc_diff.mean():.6f}")

        if enc_diff.max() < 0.01:
            print("  ✅ Encoder outputs match perfectly!")
        elif enc_diff.max() < 0.1:
            print("  ✅ Encoder outputs match well!")
        else:
            print(f"  ⚠️  Encoder outputs differ (max diff: {enc_diff.max():.3f})")

    except Exception as e:
        print(f"  ❌ Encoder test failed: {e}")

print("\n=== Summary ===")
print("If all tests pass, the Cohere preprocessor is working correctly!")
print("Next: Test full pipeline with Swift")
