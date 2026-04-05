#!/usr/bin/env python3
"""Convert ONNX models to CoreML using coremltools directly."""
import coremltools as ct
import onnx

print("=== Converting ONNX to CoreML (Direct) ===\n")

# Step 1: Load and convert encoder
print("1. Loading ONNX encoder...")
encoder_path = "onnx-models/onnx/encoder_model_fp16.onnx"
encoder_onnx = onnx.load(encoder_path)
print(f"   ✓ Loaded: {encoder_path}")

print("\n2. Encoder structure:")
print("   Inputs:")
for inp in encoder_onnx.graph.input:
    shape = [dim.dim_value if dim.dim_value != 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
    print(f"     {inp.name}: {shape}")
print("   Outputs:")
for out in encoder_onnx.graph.output:
    shape = [dim.dim_value if dim.dim_value != 0 else 'dynamic' for dim in out.type.tensor_type.shape.dim]
    print(f"     {out.name}: {shape}")

print("\n3. Converting encoder to CoreML...")
try:
    encoder_coreml = ct.convert(
        encoder_onnx,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16
    )
    
    encoder_output = "build/encoder_from_onnx.mlpackage"
    encoder_coreml.save(encoder_output)
    print(f"   ✓ Saved to: {encoder_output}")
    
    # Print encoder spec
    spec = encoder_coreml.get_spec()
    print("\n   CoreML Encoder Info:")
    print(f"     Inputs: {[inp.name for inp in spec.description.input]}")
    print(f"     Outputs: {[out.name for out in spec.description.output]}")
    
except Exception as e:
    print(f"   ❌ Encoder conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Load and convert decoder
print("\n4. Loading ONNX decoder...")
decoder_path = "onnx-models/onnx/decoder_model_merged_fp16.onnx"
decoder_onnx = onnx.load(decoder_path)
print(f"   ✓ Loaded: {decoder_path}")

print("\n5. Decoder structure:")
print("   Inputs:")
for inp in decoder_onnx.graph.input:
    shape = [dim.dim_value if dim.dim_value != 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
    print(f"     {inp.name}: {shape}")
print("   Outputs:")
for out in decoder_onnx.graph.output:
    shape = [dim.dim_value if dim.dim_value != 0 else 'dynamic' for dim in out.type.tensor_type.shape.dim]
    print(f"     {out.name}: {shape}")

print("\n6. Converting decoder to CoreML...")
try:
    decoder_coreml = ct.convert(
        decoder_onnx,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16
    )
    
    decoder_output = "build/decoder_from_onnx.mlpackage"
    decoder_coreml.save(decoder_output)
    print(f"   ✓ Saved to: {decoder_output}")
    
    # Print decoder spec
    spec = decoder_coreml.get_spec()
    print("\n   CoreML Decoder Info:")
    print(f"     Inputs: {[inp.name for inp in spec.description.input]}")
    print(f"     Outputs: {[out.name for out in spec.description.output]}")
    
except Exception as e:
    print(f"   ❌ Decoder conversion failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ONNX → CoreML conversion complete!")
print("="*60)
