#!/usr/bin/env python3
"""Check what tokens transformers actually generates."""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import librosa
import json

# Load vocab
with open('build/hf-upload/vocab.json') as f:
    vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

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

# Generate with transformers
print("Generating with transformers...")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        max_new_tokens=50,
        num_beams=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )

# Show generated tokens
tokens = generated_ids.sequences[0].tolist()
print(f"\nGenerated {len(tokens)} tokens:")
print(f"Token IDs: {tokens}")

print(f"\nDecoded tokens:")
for i, token_id in enumerate(tokens[:20]):  # First 20
    token_str = id_to_token.get(token_id, f"<UNK:{token_id}>")
    print(f"  {i}: {token_id:5d} = {token_str}")

# Check what forced_decoder_ids would be
print(f"\nModel config:")
print(f"  forced_decoder_ids: {model.config.forced_decoder_ids}")
print(f"  begin_suppress_tokens: {model.config.begin_suppress_tokens}")
print(f"  suppress_tokens: {model.config.suppress_tokens if hasattr(model.config, 'suppress_tokens') else 'N/A'}")

# Check processor's special tokens
print(f"\nProcessor tokenizer special tokens:")
print(f"  bos_token: {processor.tokenizer.bos_token} = {processor.tokenizer.bos_token_id}")
print(f"  eos_token: {processor.tokenizer.eos_token} = {processor.tokenizer.eos_token_id}")
print(f"  pad_token: {processor.tokenizer.pad_token} = {processor.tokenizer.pad_token_id}")

# Check if there's a language setting
if hasattr(model.config, 'language'):
    print(f"  language: {model.config.language}")
if hasattr(model.config, 'task'):
    print(f"  task: {model.config.task}")
