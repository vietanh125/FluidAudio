#!/usr/bin/env python3
"""Export fixed preprocessor with per-feature normalization.

The original preprocessor was missing per-feature normalization that
Cohere's FilterbankFeatures does. This script exports a fixed version.
"""
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path


class PreprocessorWrapper(nn.Module):
    """Mel spectrogram preprocessor with per-feature normalization."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, hop_length: int = 160):
        super().__init__()
        import torchaudio.transforms as T

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_signal: (batch, samples)
            length: (batch,)

        Returns:
            mel_features: (batch, n_mels, time) - normalized
            mel_length: (batch,)
        """
        # Compute mel spectrogram
        mel = self.mel_transform(audio_signal)
        mel_db = self.amplitude_to_db(mel)

        # Compute expected output length
        expected_len = (length.float() / self.mel_transform.hop_length).long()
        max_len = expected_len.max().item()
        mel_db = mel_db[:, :, :max_len]

        # Per-feature normalization
        batch_size = mel_db.shape[0]
        actual_frames = mel_db.shape[2]
        norm_len = torch.full((batch_size,), actual_frames, dtype=torch.long, device=mel_db.device)

        mel_features = self._normalize_per_feature(mel_db, norm_len)
        return mel_features, expected_len

    def _normalize_per_feature(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Per-feature (per mel bin) normalization across time."""
        batch_size, n_mels, max_time = x.shape

        # Create mask for valid time steps
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)

        # Compute mean per feature
        x_sum = torch.where(valid_mask.unsqueeze(1), x, torch.zeros_like(x)).sum(dim=2)
        x_count = valid_mask.sum(dim=1, keepdim=True)
        x_mean = x_sum / x_count.clamp(min=1)

        # Compute std per feature
        x_centered = x - x_mean.unsqueeze(2)
        x_var_sum = torch.where(valid_mask.unsqueeze(1), x_centered ** 2, torch.zeros_like(x)).sum(dim=2)
        x_std = torch.sqrt(x_var_sum / x_count.clamp(min=2))
        x_std = x_std + 1e-5

        # Normalize
        normalized = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        return normalized


print("Creating preprocessor...")
wrapper = PreprocessorWrapper(sample_rate=16000, n_mels=128, hop_length=160)
wrapper.eval()

# Dummy inputs (30s audio at 16kHz)
dummy_audio = torch.randn(1, 480000)
dummy_length = torch.tensor([480000], dtype=torch.long)

print("Tracing model...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (dummy_audio, dummy_length), strict=False)

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="audio_signal", shape=(1, 480000)),
        ct.TensorType(name="length", shape=(1,), dtype=np.int32)
    ],
    minimum_deployment_target=ct.target.iOS17,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

output_dir = Path("build/fixed-preprocessor")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "preprocessor.mlpackage"

print(f"Saving to {output_path}...")
mlmodel.save(str(output_path))

print("✅ Done! Fixed preprocessor exported.")
print("\nFeatures:")
print("  - Per-feature normalization (matching Cohere's FilterbankFeatures)")
print("  - Z-score normalization per mel bin")
print("  - Mean ~0, std ~1")
print("\nNext steps:")
print("1. Copy to build/hf-upload/: cp -r build/fixed-preprocessor/preprocessor.mlpackage build/hf-upload/")
print("2. Test with encoder and decoder")
