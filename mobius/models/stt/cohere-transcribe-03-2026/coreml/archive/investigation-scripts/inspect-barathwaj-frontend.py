import coremltools as ct

model = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
spec = model.get_spec()

print("Frontend inputs:")
for inp in spec.description.input:
    print(f"  {inp.name}: {inp.type}")

print("\nFrontend outputs:")
for out in spec.description.output:
    print(f"  {out.name}: {out.type}")
