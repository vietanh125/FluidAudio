#!/usr/bin/env python3
"""Export a Cohere-style preprocessor that matches FilterbankFeatures behavior.

Instead of trying to export the complex FilterbankFeatures module,
we reimplement the key operations in a simpler, CoreML-compatible way.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
import librosa
from pathlib import Path
from transformers import AutoProcessor

print("Loading Cohere processor to get parameters...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

filterbank = processor.feature_extractor.filterbank

print("\n=== FilterbankFeatures Parameters ===")
print(f"Sample rate: {filterbank.sample_rate}")
print(f"Window size: {filterbank.win_length}")
print(f"Hop length: {filterbank.hop_length}")
print(f"N_FFT: {filterbank.n_fft}")
print(f"N_mels: {filterbank.nfilt}")
print(f"Preemphasis: {filterbank.preemph}")
print(f"Normalize: {filterbank.normalize}")


class CohereStylePreprocessor(nn.Module):
    """Cohere-style preprocessing matching FilterbankFeatures behavior.

    Implements the same operations as Cohere's FilterbankFeatures but in a
    CoreML-compatible way.
    """

    def __init__(
        self,
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=512,
        n_mels=128,
        preemph=0.97,
        log_zero_guard=2**-24,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.preemph = preemph
        self.log_zero_guard = log_zero_guard

        # Create mel filterbank using librosa (same as NeMo)
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0,
            fmax=sample_rate / 2,
            norm="slaney"
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Hann window
        window = torch.hann_window(win_length, periodic=False)
        self.register_buffer("window", window)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        """
        Args:
            audio_signal: (B, samples)
            length: (B,) - int32

        Returns:
            features: (B, n_mels, time) - normalized mel spectrogram
            seq_len: (B,) - int32
        """
        B = audio_signal.shape[0]
        length_float = length.float()

        # 1. Pre-emphasis filter (if enabled)
        if self.preemph is not None and self.preemph > 0:
            # x[n] = x[n] - preemph * x[n-1]
            audio_signal = torch.cat([
                audio_signal[:, :1],
                audio_signal[:, 1:] - self.preemph * audio_signal[:, :-1]
            ], dim=1)

        # 2. STFT
        # Pad for centered STFT
        pad_amount = self.n_fft // 2
        audio_padded = F.pad(audio_signal.unsqueeze(1), (pad_amount, pad_amount), mode='constant')
        audio_padded = audio_padded.squeeze(1)

        # Compute STFT
        stft_out = torch.stft(
            audio_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
            pad_mode='constant'
        )

        # 3. Magnitude spectrogram
        magnitude = torch.abs(stft_out)  # (B, freq, time)

        # Square for power
        power = magnitude ** 2

        # 4. Mel filterbank
        mel_spec = torch.matmul(self.mel_basis, power)  # (B, n_mels, time)

        # 5. Log
        mel_spec = torch.log(mel_spec + self.log_zero_guard)

        # 6. Compute output sequence length
        seq_len = (length_float / self.hop_length).long()

        # Trim to exact length
        max_len = seq_len.max().item()
        mel_spec = mel_spec[:, :, :max_len]

        # 7. Per-feature normalization (like Cohere)
        mel_spec = self._normalize_per_feature(mel_spec, seq_len)

        return mel_spec, seq_len

    def _normalize_per_feature(self, x: torch.Tensor, seq_len: torch.Tensor):
        """Per-feature (per mel bin) normalization."""
        B, n_mels, max_time = x.shape

        # Create mask for valid frames
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(B, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)  # (B, time)

        # Compute mean per feature
        x_sum = torch.where(valid_mask.unsqueeze(1), x, torch.zeros_like(x)).sum(dim=2)  # (B, n_mels)
        x_count = valid_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        x_mean = x_sum / x_count.clamp(min=1)  # (B, n_mels)

        # Compute std per feature
        x_centered = x - x_mean.unsqueeze(2)
        x_var_sum = torch.where(valid_mask.unsqueeze(1), x_centered ** 2, torch.zeros_like(x)).sum(dim=2)
        x_std = torch.sqrt(x_var_sum / (x_count - 1).clamp(min=1))  # (B, n_mels)

        # Add small constant for stability
        x_std = x_std + 1e-5

        # Normalize
        normalized = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

        return normalized


print("\n=== Creating Cohere-style preprocessor ===")
preprocessor = CohereStylePreprocessor(
    sample_rate=filterbank.sample_rate,
    win_length=filterbank.win_length,
    hop_length=filterbank.hop_length,
    n_fft=filterbank.n_fft,
    n_mels=filterbank.nfilt,
    preemph=filterbank.preemph,
)
preprocessor.eval()

# Test
print("\nTesting with dummy input...")
dummy_audio = torch.randn(1, 480000)
dummy_length = torch.tensor([480000], dtype=torch.int32)

with torch.no_grad():
    features, seq_len = preprocessor(dummy_audio, dummy_length)
    print(f"Output shape: {features.shape}")
    print(f"Output length: {seq_len.item()}")
    print(f"Stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}, std={features.std():.3f}")

print("\n=== Tracing model ===")
with torch.no_grad():
    traced = torch.jit.trace(
        preprocessor,
        (dummy_audio, dummy_length),
        strict=False
    )

print("\n=== Converting to CoreML ===")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="audio_signal", shape=(1, 480000), dtype=np.float32),
        ct.TensorType(name="length", shape=(1,), dtype=np.int32)
    ],
    outputs=[
        ct.TensorType(name="mel_features"),
        ct.TensorType(name="mel_length")
    ],
    minimum_deployment_target=ct.target.iOS17,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

output_dir = Path("build/cohere-preprocessor")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "preprocessor.mlpackage"

print(f"\nSaving to {output_path}...")
mlmodel.save(str(output_path))

print("\n✅ Done! Cohere-style preprocessor exported.")
print("\nFeatures:")
print("  - Matches FilterbankFeatures behavior")
print("  - Pre-emphasis filter (0.97)")
print("  - STFT with Hann window")
print("  - Mel filterbank (slaney norm)")
print("  - Log compression")
print("  - Per-feature normalization")
print("\nNext steps:")
print("1. Test with encoder:")
print("   uv run python test-cohere-preprocessor.py")
print("2. Copy to build/hf-upload/:")
print("   cp -r build/cohere-preprocessor/preprocessor.mlpackage build/hf-upload/")
print("3. Test Swift pipeline:")
print("   swift test-swift-cache.swift")
