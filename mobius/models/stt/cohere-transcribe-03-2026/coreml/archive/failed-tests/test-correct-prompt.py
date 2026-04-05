#!/usr/bin/env python3
"""Test with the CORRECT prompt sequence from working CoreML conversion."""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import librosa
import json

# Load vocab
with open('build/hf-upload/vocab.json') as f:
    vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

# The CORRECT prompt from BarathwajAnandan's working conversion
CORRECT_PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]

print("Correct prompt sequence:")
for token_id in CORRECT_PROMPT:
    print(f"  {token_id}: {id_to_token[token_id]}")

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

print(f"\n{'='*60}")
print("Test with CORRECT prompt sequence")
print('='*60)

decoder_input_ids = torch.tensor([CORRECT_PROMPT], dtype=torch.long)

with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=50,
        num_beams=1,
        do_sample=False
    )

result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\nResult: \"{result}\"")
print(f"\nGenerated token IDs: {generated_ids[0].tolist()[:20]}")

# Decode tokens to see what was generated
print(f"\nDecoded tokens:")
for i, token_id in enumerate(generated_ids[0].tolist()[:20]):
    token_str = id_to_token.get(token_id, f"<UNK:{token_id}>")
    print(f"  {i}: {token_id:5d} = {token_str}")
