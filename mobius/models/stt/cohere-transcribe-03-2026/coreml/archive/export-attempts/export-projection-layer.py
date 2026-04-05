#!/usr/bin/env python3
"""Export just the projection layer (1280 → 1024) to apply after existing encoder."""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq
import numpy as np

print("=== Exporting Projection Layer (1280 → 1024) ===\n")

# Load model to get projection weights
print("1. Loading PyTorch model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
model.eval()

print(f"   ✓ Model loaded")
print(f"   Projection layer: {model.encoder_decoder_proj}")
print(f"   Weight shape: {model.encoder_decoder_proj.weight.shape}")
print(f"   Bias: {model.encoder_decoder_proj.bias is not None}")

# Create simple projection module
class ProjectionLayer(nn.Module):
    """Simple linear projection: 1280 → 1024"""
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, encoder_output):
        """
        Args:
            encoder_output: (batch, time, 1280) encoder hidden states
        Returns:
            projected_output: (batch, time, 1024) projected hidden states
        """
        return self.projection(encoder_output)

# Create projection module
print("\n2. Creating projection module...")
projection_module = ProjectionLayer(model.encoder_decoder_proj)
projection_module.eval()
print("   ✓ Projection module created")

# Test with dummy input
print("\n3. Testing projection module...")
batch_size = 1
time = 375  # Example encoder output length
hidden_size = 1280

dummy_encoder_output = torch.randn(batch_size, time, hidden_size)

with torch.no_grad():
    output = projection_module(dummy_encoder_output)

print(f"   Input shape: {dummy_encoder_output.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output dim: {output.shape[2]} (should be 1024)")

if output.shape[2] != 1024:
    print("   ❌ ERROR: Output dimension is not 1024!")
    exit(1)

print("   ✓ Output dimension correct!")

# Convert to CoreML
print("\n4. Converting to CoreML...")

# Trace the model
traced_model = torch.jit.trace(projection_module, dummy_encoder_output)

# Define input shapes (variable time dimension)
encoder_output_shape = ct.Shape(shape=(1, ct.RangeDim(1, 1500), 1280))

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="encoder_output", shape=encoder_output_shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="projected_output", dtype=np.float16)
    ],
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)

# Save
output_path = "build/encoder_decoder_proj.mlpackage"
mlmodel.save(output_path)
print(f"   ✓ Saved to: {output_path}")

# Test CoreML model
print("\n5. Testing CoreML model...")
coreml_model = ct.models.MLModel(output_path)

test_input = {
    "encoder_output": dummy_encoder_output.numpy().astype(np.float16)
}

coreml_output = coreml_model.predict(test_input)
coreml_result = coreml_output["projected_output"]

print(f"   CoreML output shape: {coreml_result.shape}")
print(f"   CoreML output dim: {coreml_result.shape[2]} (should be 1024)")

# Compare with PyTorch
pytorch_output = output.numpy()
max_diff = np.abs(pytorch_output - coreml_result).max()
mean_diff = np.abs(pytorch_output - coreml_result).mean()

print(f"\n6. Validation:")
print(f"   Max diff: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")

if max_diff < 0.01:
    print("   ✅ Excellent match!")
elif max_diff < 0.1:
    print("   ✅ Good match")
else:
    print(f"   ⚠️  Difference might be too large")

print(f"\n{'='*60}")
print("SUCCESS! Projection layer exported.")
print(f"Output: {output_path}")
print(f"")
print("Pipeline:")
print("  1. preprocessor.mlpackage → mel features")
print("  2. encoder.mlpackage → 1280-dim output")
print("  3. encoder_decoder_proj.mlpackage → 1024-dim output")
print("  4. decoder.mlpackage → tokens")
print('='*60)
