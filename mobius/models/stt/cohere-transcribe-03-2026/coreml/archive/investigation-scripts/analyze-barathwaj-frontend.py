#!/usr/bin/env python3
"""Analyze BarathwajAnandan's frontend to reverse-engineer parameters."""
import coremltools as ct
import numpy as np
import soundfile as sf

# Load their frontend
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")

# Test with simple sine wave
sr = 16000
duration = 1.0
freq = 440
t = np.linspace(0, duration, int(sr * duration))
audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

# Pad to 560,000
audio_padded = np.pad(audio, (0, 560000 - len(audio)))

# Run through frontend
output = frontend.predict({
    "audio_samples": audio_padded.reshape(1, -1),
    "audio_length": np.array([len(audio)], dtype=np.int32)
})

mel = output["var_6916"]
print(f"Mel output shape: {mel.shape}")
print(f"Mel stats: min={mel.min():.3f}, max={mel.max():.3f}, mean={mel.mean():.3f}")

# Check the spec
spec = frontend.get_spec()
print(f"\nFrontend model info:")
print(f"Type: {spec.WhichOneof('Type')}")

# For comparison, let's use PyTorch to make mel spectrogram
import torch
import torchaudio

# Try standard parameters
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=160,
    win_length=1024,
    n_mels=128,
    f_min=0.0,
    f_max=8000.0,
)

audio_torch = torch.from_numpy(audio)
mel_torch = mel_transform(audio_torch.unsqueeze(0))
mel_torch_log = torch.log10(torch.clamp(mel_torch, min=1e-10))

print(f"\nPyTorch mel shape: {mel_torch_log.shape}")
print(f"PyTorch mel stats: min={mel_torch_log.numpy().min():.3f}, max={mel_torch_log.numpy().max():.3f}, mean={mel_torch_log.numpy().mean():.3f}")

# Compare first frame
print(f"\nBarathwaj mel first frame stats: min={mel[0, :, 0].min():.3f}, max={mel[0, :, 0].max():.3f}")
print(f"PyTorch mel first frame stats: min={mel_torch_log[0, :, 0].numpy().min():.3f}, max={mel_torch_log[0, :, 0].numpy().max():.3f}")
