#!/usr/bin/env python3
"""Reverse engineer BarathwajAnandan's export approach by analyzing their CoreML models."""
import coremltools as ct
import json
import pprint

print("="*80)
print("REVERSE ENGINEERING BARATHWAJANANDAN'S EXPORT")
print("="*80)

models_to_analyze = {
    "Frontend": "build/barathwaj-models/cohere_frontend.mlpackage",
    "Encoder": "build/barathwaj-models/cohere_encoder.mlpackage",
    "Decoder (Full-Seq)": "build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage",
    "Decoder (Cached)": "build/barathwaj-models/cohere_decoder_cached.mlpackage",
    "Cross KV Projector": "build/barathwaj-models/cohere_cross_kv_projector.mlpackage",
}

def analyze_model(name, path):
    """Analyze a CoreML model's structure."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print('='*80)

    try:
        # Load with skip_model_load to get spec without loading weights
        spec = ct.utils.load_spec(path)

        print(f"\n📋 Model Type: {spec.WhichOneof('Type')}")

        # Get description
        desc = spec.description

        print(f"\n🔹 Inputs ({len(desc.input)}):")
        for inp in desc.input:
            print(f"   • {inp.name}")
            if inp.type.WhichOneof('Type') == 'multiArrayType':
                ma = inp.type.multiArrayType
                shape = list(ma.shape) if ma.shape else "dynamic"
                dtype = ma.dataType
                print(f"     Shape: {shape}")
                print(f"     Type: {dtype}")

        print(f"\n🔹 Outputs ({len(desc.output)}):")
        for out in desc.output:
            print(f"   • {out.name}")
            if out.type.WhichOneof('Type') == 'multiArrayType':
                ma = out.type.multiArrayType
                shape = list(ma.shape) if ma.shape else "dynamic"
                dtype = ma.dataType
                print(f"     Shape: {shape}")
                print(f"     Type: {dtype}")

        # Check metadata
        if desc.metadata:
            print(f"\n🔹 Metadata:")
            meta = desc.metadata
            if meta.author:
                print(f"   Author: {meta.author}")
            if meta.license:
                print(f"   License: {meta.license}")
            if meta.shortDescription:
                print(f"   Description: {meta.shortDescription}")
            if meta.userDefined:
                print(f"   User Defined:")
                for key, value in meta.userDefined.items():
                    print(f"     {key}: {value}")

        # Check if it's mlprogram
        if spec.WhichOneof('Type') == 'mlProgram':
            print(f"\n🔹 ML Program Details:")
            mlprog = spec.mlProgram
            print(f"   Functions: {len(mlprog.functions)}")
            if mlprog.functions:
                main_func = mlprog.functions.get('main')
                if main_func:
                    print(f"   Main function operations: {len(main_func.block_specializations)}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

# Analyze all models
for name, path in models_to_analyze.items():
    analyze_model(name, path)

# Load and analyze manifest
print(f"\n{'='*80}")
print("MANIFEST ANALYSIS")
print('='*80)

with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)

print("\n🔹 Architecture Config:")
print(f"   Model ID: {manifest['model_id']}")
print(f"   Precision: {manifest['precision']}")
print(f"   Quantization: {manifest['quantize']}")
print(f"   Encoder hidden size: {manifest['encoder_hidden_size']}")
print(f"   Decoder max length: {manifest['decoder_max_len']}")
print(f"   Max encoder frames: {manifest['max_encoder_frames']}")

print("\n🔹 Pipeline Flow:")
pipeline_keys = ['frontend', 'encoder', 'decoder', 'decoder_cached']
for key in pipeline_keys:
    if key in manifest:
        info = manifest[key]
        print(f"\n   {key.upper()}:")
        print(f"     Inputs: {info.get('inputs', 'N/A')}")
        print(f"     Outputs: {info.get('outputs', 'N/A')}")

print("\n🔹 Prompt Configuration:")
print(f"   Prompt IDs: {manifest['prompt_ids']}")
id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_tokens = [id_to_token[i] for i in manifest['prompt_ids']]
print(f"   Decoded: {prompt_tokens}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("""
1. Compare encoder output dimensions:
   - Check if encoder outputs 1024 or 1280

2. Look for projection layer:
   - Separate model? Integrated into encoder?

3. Frontend vs Preprocessor differences:
   - Input/output names
   - Architecture differences

4. Decoder architecture:
   - Attention mask requirements
   - KV cache structure
""")
