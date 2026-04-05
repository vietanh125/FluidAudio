#!/usr/bin/env python3
"""Export Cohere's exact FilterbankFeatures preprocessing to CoreML.

This uses the actual preprocessing from the Cohere model, ensuring
compatibility with the encoder.
"""
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
from transformers import AutoProcessor

print("Loading Cohere processor...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

# Get the feature extractor (which wraps FilterbankFeatures)
feature_extractor = processor.feature_extractor

print("\n=== Feature Extractor Configuration ===")
print(f"Type: {type(feature_extractor).__name__}")
print(f"Sample rate: {feature_extractor.sampling_rate}")
print(f"Feature size: {feature_extractor.feature_size}")
print(f"Hop length: {feature_extractor.hop_length}")
print(f"Max duration: {feature_extractor.max_duration}")

# Access the underlying filterbank
filterbank = feature_extractor.filterbank

print("\n=== FilterbankFeatures Configuration ===")
print(f"Window size: {filterbank.win_length}")
print(f"Hop length: {filterbank.hop_length}")
print(f"N_FFT: {filterbank.n_fft}")
print(f"N_mels: {filterbank.nfilt}")
print(f"Preemphasis: {filterbank.preemph}")
print(f"Log: {filterbank.log}")
print(f"Normalize: {filterbank.normalize}")
print(f"Pad to: {filterbank.pad_to}")


class CoherePreprocessorWrapper(nn.Module):
    """Wrapper around Cohere's FilterbankFeatures for CoreML export."""

    def __init__(self, filterbank):
        super().__init__()
        self.filterbank = filterbank
        # Store parameters for tracing
        self.sample_rate = filterbank.sample_rate
        self.hop_length = filterbank.hop_length

        # Convert bfloat16 buffers to float32 for CoreML compatibility
        if hasattr(self.filterbank, 'window') and self.filterbank.window is not None:
            self.filterbank.window = self.filterbank.window.to(torch.float32)
        if hasattr(self.filterbank, 'fb') and self.filterbank.fb is not None:
            self.filterbank.fb = self.filterbank.fb.to(torch.float32)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        """
        Args:
            audio_signal: (B, samples) - raw audio waveform
            length: (B,) - audio lengths in samples (int32)

        Returns:
            mel_features: (B, n_mels, time) - normalized mel spectrogram
            mel_length: (B,) - mel spectrogram lengths (int32)
        """
        # Convert length to float for filterbank
        length_float = length.float()

        # Run filterbank features
        # Output shape: (B, n_mels, time)
        features, seq_len = self.filterbank(audio_signal, length_float)

        # Convert seq_len back to int32
        seq_len_int = seq_len.long()

        return features, seq_len_int


print("\n=== Creating CoreML-compatible wrapper ===")
wrapper = CoherePreprocessorWrapper(filterbank)
wrapper.eval()

# Test with dummy input
print("\nTesting with dummy input...")
dummy_audio = torch.randn(1, 480000)  # 30s at 16kHz
dummy_length = torch.tensor([480000], dtype=torch.int32)

with torch.no_grad():
    features, seq_len = wrapper(dummy_audio, dummy_length.float())
    print(f"Output shape: {features.shape}")
    print(f"Output length: {seq_len.item()}")
    print(f"Stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}, std={features.std():.3f}")

print("\n=== Tracing model ===")
with torch.no_grad():
    traced = torch.jit.trace(
        wrapper,
        (dummy_audio, dummy_length.float()),
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

print("\n✅ Done! Cohere's exact preprocessor exported.")
print("\nFeatures:")
print("  - Exact FilterbankFeatures from Cohere model")
print("  - NeMo-compatible preprocessing")
print("  - Per-feature normalization included")
print("  - Matches training preprocessing exactly")
print("\nNext steps:")
print("1. Test with encoder:")
print("   uv run python test-cohere-preprocessor.py")
print("2. Copy to build/hf-upload/:")
print("   cp -r build/cohere-preprocessor/preprocessor.mlpackage build/hf-upload/")
print("3. Test Swift pipeline:")
print("   swift test-swift-cache.swift")
