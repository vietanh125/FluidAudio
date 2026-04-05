#!/usr/bin/env python3
"""Test forcing English language token in decoder prefix."""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import librosa
import json

# Load vocab
with open('build/hf-upload/vocab.json') as f:
    vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

# Find special tokens
START_OF_TRANSCRIPT = 4   # <|startoftranscript|>
ENGLISH = 62              # <|en|>
NO_PNC = 6                # <|nopnc|>
NO_ITN = 9                # <|noitn|>
NO_TIMESTAMP = 11         # <|notimestamp|>
NO_DIARIZE = 13           # <|nodiarize|>

print(f"Special tokens:")
print(f"  START_OF_TRANSCRIPT = {START_OF_TRANSCRIPT} = {id_to_token[START_OF_TRANSCRIPT]}")
print(f"  ENGLISH = {ENGLISH} = {id_to_token[ENGLISH]}")
print(f"  NO_PNC = {NO_PNC} = {id_to_token[NO_PNC]}")
print(f"  NO_ITN = {NO_ITN} = {id_to_token[NO_ITN]}")
print(f"  NO_TIMESTAMP = {NO_TIMESTAMP} = {id_to_token[NO_TIMESTAMP]}")
print(f"  NO_DIARIZE = {NO_DIARIZE} = {id_to_token[NO_DIARIZE]}")

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
print("Test 1: Default generation (no prefix)")
print('='*60)

with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        max_new_tokens=50,
        num_beams=1,
        do_sample=False
    )

result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Result: \"{result[:100]}...\"")
print(f"First 10 tokens: {generated_ids[0, :10].tolist()}")

print(f"\n{'='*60}")
print("Test 2: With English language token prefix")
print('='*60)

# Try different prefix combinations
prefixes_to_test = [
    [START_OF_TRANSCRIPT, ENGLISH],
    [START_OF_TRANSCRIPT, ENGLISH, NO_TIMESTAMP],
    [START_OF_TRANSCRIPT, ENGLISH, NO_PNC, NO_ITN, NO_TIMESTAMP, NO_DIARIZE],
]

for prefix in prefixes_to_test:
    prefix_str = " → ".join([id_to_token[t] for t in prefix])
    print(f"\nPrefix: {prefix_str}")

    decoder_input_ids = torch.tensor([prefix], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=50,
            num_beams=1,
            do_sample=False
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"  Result: \"{result[:100]}{'...' if len(result) > 100 else ''}\"")
    print(f"  First 10 tokens: {generated_ids[0, :10].tolist()}")
