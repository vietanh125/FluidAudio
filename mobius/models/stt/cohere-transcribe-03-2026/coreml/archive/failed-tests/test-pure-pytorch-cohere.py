#!/usr/bin/env python3
"""Test pure PyTorch Cohere Transcribe 03-2026 to verify the model works at all."""
import torch
from transformers import pipeline
import librosa
import soundfile as sf
import sys

print("=== Pure PyTorch Cohere Transcribe Test ===\n")

# Test with multiple audio files
test_files = [
    "test-librispeech-real.wav",
    "test-real-speech.wav",
]

# Add command line arg if provided
if len(sys.argv) > 1:
    test_files = [sys.argv[1]]

print("Initializing ASR pipeline...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="CohereLabs/cohere-transcribe-03-2026",
    torch_dtype=torch.float32,
    device="cpu",
    trust_remote_code=True
)

for audio_file in test_files:
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {audio_file}")
        print('='*60)

        # Load audio
        audio, sr = sf.read(audio_file)
        print(f"Original: {len(audio)} samples @ {sr}Hz")

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
            print(f"Converted stereo to mono")

        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            print(f"Resampled to 16000Hz")

        print(f"Final: {len(audio)} samples ({len(audio)/sr:.2f}s) @ {sr}Hz")

        # Check audio statistics
        print(f"Audio stats: min={audio.min():.3f}, max={audio.max():.3f}, mean={audio.mean():.6f}")

        # Transcribe
        print("\nTranscribing...")
        result = pipe({"array": audio, "sampling_rate": sr})

        print(f"\n✅ Result:")
        print(f"Text: \"{result['text'][:200]}{'...' if len(result['text']) > 200 else ''}\"")
        print(f"Full length: {len(result['text'])} characters")

    except FileNotFoundError:
        print(f"❌ File not found: {audio_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Also test with a simple sine wave to see what happens
print(f"\n{'='*60}")
print(f"Testing: Synthetic 1kHz sine wave (1 second)")
print('='*60)

import numpy as np
duration = 1.0
sr = 16000
t = np.linspace(0, duration, int(sr * duration))
sine_wave = 0.5 * np.sin(2 * np.pi * 1000 * t)

print(f"Sine wave: {len(sine_wave)} samples @ {sr}Hz")
print(f"Stats: min={sine_wave.min():.3f}, max={sine_wave.max():.3f}, mean={sine_wave.mean():.6f}")

print("\nTranscribing...")
result = pipe({"array": sine_wave.astype(np.float32), "sampling_rate": sr})

print(f"\n✅ Result:")
print(f"Text: \"{result['text']}\"")

# Test with silence
print(f"\n{'='*60}")
print(f"Testing: Pure silence (1 second)")
print('='*60)

silence = np.zeros(sr, dtype=np.float32)
print(f"Silence: {len(silence)} samples @ {sr}Hz")

print("\nTranscribing...")
result = pipe({"array": silence, "sampling_rate": sr})

print(f"\n✅ Result:")
print(f"Text: \"{result['text']}\"")

print(f"\n{'='*60}")
print("Summary: Check if any output makes sense")
print('='*60)
