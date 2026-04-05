#!/usr/bin/env python3
"""Convert ONNX models to CoreML."""
import onnx
from onnx_coreml import convert
import coremltools as ct
import numpy as np

print("=== Converting ONNX to CoreML ===\n")

# Step 1: Load ONNX encoder
print("1. Loading ONNX encoder...")
encoder_path = "onnx-models/onnx/encoder_model_fp16.onnx"
encoder_onnx = onnx.load(encoder_path)
print(f"   ✓ Loaded: {encoder_path}")

# Check encoder inputs/outputs
print("\n2. Encoder structure:")
print("   Inputs:")
for inp in encoder_onnx.graph.input:
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"     {inp.name}: {shape}")
print("   Outputs:")
for out in encoder_onnx.graph.output:
    shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    print(f"     {out.name}: {shape}")

# Step 3: Convert encoder to CoreML
print("\n3. Converting encoder to CoreML...")
try:
    encoder_coreml = convert(
        encoder_onnx,
        minimum_ios_deployment_target='13',
        preprocessing_args={
            'image_scale': 1.0
        }
    )
    
    encoder_output = "build/encoder_from_onnx.mlpackage"
    encoder_coreml.save(encoder_output)
    print(f"   ✓ Saved to: {encoder_output}")
    
except Exception as e:
    print(f"   ❌ Encoder conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Load ONNX decoder
print("\n4. Loading ONNX decoder...")
decoder_path = "onnx-models/onnx/decoder_model_merged_fp16.onnx"
decoder_onnx = onnx.load(decoder_path)
print(f"   ✓ Loaded: {decoder_path}")

# Check decoder inputs/outputs
print("\n5. Decoder structure:")
print("   Inputs:")
for inp in decoder_onnx.graph.input:
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"     {inp.name}: {shape}")
print("   Outputs:")
for out in decoder_onnx.graph.output:
    shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    print(f"     {out.name}: {shape}")

# Step 6: Convert decoder to CoreML
print("\n6. Converting decoder to CoreML...")
try:
    decoder_coreml = convert(
        decoder_onnx,
        minimum_ios_deployment_target='13',
        preprocessing_args={
            'image_scale': 1.0
        }
    )
    
    decoder_output = "build/decoder_from_onnx.mlpackage"
    decoder_coreml.save(decoder_output)
    print(f"   ✓ Saved to: {decoder_output}")
    
except Exception as e:
    print(f"   ❌ Decoder conversion failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ONNX → CoreML conversion complete!")
print("="*60)
