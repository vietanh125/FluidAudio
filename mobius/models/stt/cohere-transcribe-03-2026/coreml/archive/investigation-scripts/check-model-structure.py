from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)

print("Model attributes:")
print([attr for attr in dir(model) if not attr.startswith('_')])
