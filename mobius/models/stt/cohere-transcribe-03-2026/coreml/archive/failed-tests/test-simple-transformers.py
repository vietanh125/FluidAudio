#!/usr/bin/env python3
"""Simple test using Cohere exactly as documented."""
import torch
from transformers import pipeline
import librosa
import soundfile as sf

print("=== Simple Transformers Pipeline Test ===\n")

# Load audio
import sys
audio_file = sys.argv[1] if len(sys.argv) > 1 else "test-librispeech-real.wav"
audio, sr = sf.read(audio_file)
print(f"Testing: {audio_file}")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

print(f"Audio: {len(audio)} samples ({len(audio)/sr:.2f}s) @ {sr}Hz")

# Use pipeline (recommended way)
print("\nInitializing ASR pipeline...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="CohereLabs/cohere-transcribe-03-2026",
    torch_dtype=torch.float32,
    device="cpu",
    trust_remote_code=True
)

print("Transcribing...")
result = pipe({"array": audio, "sampling_rate": sr})

print(f"\n✅ Result: {result}")
print(f"\nTranscription: \"{result['text']}\"")
