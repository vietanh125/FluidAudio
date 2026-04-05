import coremltools as ct

model = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")
spec = model.get_spec()

print("Decoder inputs:")
for inp in spec.description.input:
    name = inp.name
    shape = [d.constant.size if hasattr(d, 'constant') else 'flexible' for d in inp.type.multiArrayType.shape]
    dtype = str(inp.type.multiArrayType.dataType).split('.')[-1]
    print(f"  {name}: {shape} ({dtype})")

print("\nDecoder outputs:")
for out in spec.description.output:
    name = out.name
    shape = [d.constant.size if hasattr(d, 'constant') else 'flexible' for d in out.type.multiArrayType.shape]
    dtype = str(out.type.multiArrayType.dataType).split('.')[-1]
    print(f"  {name}: {shape} ({dtype})")
