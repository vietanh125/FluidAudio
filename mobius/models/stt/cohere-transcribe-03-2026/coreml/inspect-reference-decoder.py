#!/usr/bin/env python3
"""Inspect reference decoder structure."""

import coremltools as ct

print("="*70)
print("Reference Decoder Inspection")
print("="*70)

ref_decoder = ct.models.MLModel("barathwaj-models/cohere_decoder_cached.mlpackage")

spec = ref_decoder.get_spec()

print("\nInputs:")
for input_desc in spec.description.input:
    print(f"  {input_desc.name}:")
    if input_desc.type.WhichOneof('Type') == 'multiArrayType':
        ma = input_desc.type.multiArrayType
        print(f"    Shape: {list(ma.shape)}")
        print(f"    DataType: {ma.dataType}")

print("\nOutputs:")
for output_desc in spec.description.output:
    print(f"  {output_desc.name}:")
    if output_desc.type.WhichOneof('Type') == 'multiArrayType':
        ma = output_desc.type.multiArrayType
        print(f"    Shape: {list(ma.shape)}")
        print(f"    DataType: {ma.dataType}")
