#!/usr/bin/env python3
"""Export encoder with projection (1280→1024) matching BarathwajAnandan's approach."""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np

print("=== Exporting Encoder with Projection ===\n")

# Load model
print("1. Loading PyTorch model...")
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
print(f"   ✓ Model loaded")

# Create encoder wrapper with projection
class EncoderWithProjection(nn.Module):
    """Encoder + Projection that outputs 1024-dim like BarathwajAnandan."""
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, input_features, feature_length):
        """
        Args:
            input_features: (1, 128, time) mel spectrogram
            feature_length: (1,) actual valid frames
        Returns:
            encoder_output: (1, time_downsampled, 1024)
            encoder_length: (1,) downsampled length (feature_length / 8)
        """
        # Run encoder
        encoder_out = self.encoder(input_features)

        # Extract hidden states (encoder may return tuple or tensor)
        if isinstance(encoder_out, tuple):
            hidden_states = encoder_out[0]
        else:
            hidden_states = encoder_out

        # Apply projection: 1280 → 1024
        projected = self.projection(hidden_states)

        # Compute output length (encoder downsamples by 8x)
        encoder_length = (feature_length.float() / 8.0).long()

        return projected, encoder_length

# Create wrapper
print("\n2. Creating encoder wrapper with projection...")
encoder_wrapper = EncoderWithProjection(
    model.encoder,
    model.encoder_decoder_proj
)
encoder_wrapper.eval()
print(f"   ✓ Wrapper created")

# Test with real audio
print("\n3. Testing with real audio...")
import soundfile as sf
import librosa
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs["input_features"]
feature_length = inputs["length"]

print(f"   Input: {input_features.shape}, length: {feature_length.item()}")

with torch.no_grad():
    encoder_output, encoder_length = encoder_wrapper(input_features, feature_length)

print(f"   Output: {encoder_output.shape}, length: {encoder_length.item()}")
print(f"   Output dim: {encoder_output.shape[2]} (should be 1024)")

if encoder_output.shape[2] != 1024:
    print(f"   ❌ ERROR: Expected 1024-dim output, got {encoder_output.shape[2]}")
    exit(1)

# Try scripting first, fallback to tracing
print("\n4. Attempting export...")
try:
    print("   Trying torch.jit.script...")
    scripted_model = torch.jit.script(encoder_wrapper)
    print("   ✓ Model scripted successfully")
    export_model = scripted_model
except Exception as e:
    print(f"   Scripting failed: {e}")
    print("   Trying torch.jit.trace...")

    try:
        batch = 1
        n_mels = 128
        max_time = 3501  # Match BarathwajAnandan (35s at 100 fps)

        example_features = torch.randn(batch, n_mels, max_time)
        example_length = torch.tensor([1043], dtype=torch.int32)

        traced_model = torch.jit.trace(encoder_wrapper, (example_features, example_length))
        print("   ✓ Model traced")
        export_model = traced_model
    except Exception as e2:
        print(f"   ❌ Tracing also failed: {e2}")
        print("\nNeed to create a static encoder without dynamic operations.")
        exit(1)

# Convert to CoreML
print("\n5. Converting to CoreML...")

# Match BarathwajAnandan's input shape: (1, 128, 3501) for 35s audio
input_features_shape = ct.Shape(shape=(1, 128, 3501))
feature_length_shape = ct.Shape(shape=(1,))

mlmodel = ct.convert(
    export_model,
    inputs=[
        ct.TensorType(name="input_features", shape=input_features_shape, dtype=np.float32),
        ct.TensorType(name="feature_length", shape=feature_length_shape, dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="encoder_output", dtype=np.float16),
        ct.TensorType(name="encoder_length", dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)

# Save
output_path = "build/encoder_with_projection.mlpackage"
mlmodel.save(output_path)
print(f"   ✓ Saved to: {output_path}")

# Test CoreML model
print("\n6. Testing CoreML model...")
coreml_model = ct.models.MLModel(output_path)

# Pad input to 3501 frames
input_padded = torch.nn.functional.pad(input_features, (0, 3501 - input_features.shape[2]))

test_input = {
    "input_features": input_padded.numpy().astype(np.float32),
    "feature_length": feature_length.numpy().astype(np.int32)
}

coreml_output = coreml_model.predict(test_input)
coreml_encoder_out = coreml_output["encoder_output"]
coreml_encoder_len = coreml_output["encoder_length"]

print(f"   CoreML output: {coreml_encoder_out.shape}")
print(f"   CoreML length: {coreml_encoder_len}")
print(f"   Output dim: {coreml_encoder_out.shape[2]} (should be 1024)")

# Compare with PyTorch
pytorch_out = encoder_output.numpy()
# Trim CoreML output to match PyTorch
coreml_trimmed = coreml_encoder_out[:, :pytorch_out.shape[1], :]
max_diff = np.abs(pytorch_out - coreml_trimmed).max()
mean_diff = np.abs(pytorch_out - coreml_trimmed).mean()

print(f"\n7. Validation:")
print(f"   Max diff: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")

if max_diff < 0.1:
    print("   ✅ Good match!")
else:
    print(f"   ⚠️  Larger diff than expected")

print(f"\n{'='*60}")
print("SUCCESS! Encoder exported with projection")
print(f"Output: {output_path}")
print(f"Shape: (1, 128, 3501) → (1, 438, 1024)")
print(f"Output dim: 1024 (includes projection)")
print('='*60)
