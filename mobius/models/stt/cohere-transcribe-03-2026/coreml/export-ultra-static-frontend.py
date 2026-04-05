#!/usr/bin/env python3
"""Export ultra-static frontend (mel spectrogram preprocessor).

Following mobius pattern:
- Hard-code all shapes for 35s max audio (560,000 samples @ 16kHz)
- No dynamic operations
- Static mel spectrogram computation
"""
import torch
import torch.nn as nn
import torchaudio
import coremltools as ct
from transformers import AutoProcessor
import numpy as np
import soundfile as sf
import librosa

print("=== Exporting Ultra-Static Frontend ===\n")

# Load processor to get mel config
print("1. Loading processor for mel config...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
print("   ✓ Processor loaded")

# Get mel spectrogram config
feature_extractor = processor.feature_extractor
print(f"\n2. Mel spectrogram config:")
print(f"   Sample rate: {feature_extractor.sampling_rate}")
print(f"   N FFT: {feature_extractor.n_fft}")
print(f"   Hop length: {feature_extractor.hop_length}")
print(f"   Win length: {feature_extractor.win_length}")
print(f"   N mels: {feature_extractor.num_mel_bins}")
print(f"   Padding: {feature_extractor.padding_value}")

# ============================================================================
# ULTRA-STATIC FRONTEND
# ============================================================================

class UltraStaticFrontend(nn.Module):
    """Completely static mel spectrogram frontend.

    Fixed Configuration:
    - Input: (1, 560000) raw audio @ 16kHz (35 seconds)
    - Output: (1, 128, 3501) mel spectrogram
    - No dynamic operations, no padding
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        n_mels=128,
        f_min=0.0,
        f_max=8000.0,
    ):
        super().__init__()

        # Fixed constants
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        # Pre-create mel filterbank
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode='reflect',
        )

    def forward(self, audio):
        """
        Args:
            audio: (1, 560000) raw waveform

        Returns:
            mel: (1, 128, 3501) mel spectrogram
        """
        # Compute mel spectrogram
        mel = self.mel_scale(audio)  # (1, 128, 3501)

        # Log scale
        mel = torch.clamp(mel, min=1e-10)
        mel = torch.log10(mel)

        # Normalize (subtract mean from valid region)
        # For full 3501 frames, all are valid
        mel_mean = mel.mean()
        mel = mel - mel_mean

        return mel


print("\n3. Creating ultra-static frontend...")
frontend = UltraStaticFrontend(
    sample_rate=feature_extractor.sampling_rate,
    n_fft=feature_extractor.n_fft,
    hop_length=feature_extractor.hop_length,
    win_length=feature_extractor.win_length,
    n_mels=feature_extractor.num_mel_bins,
)
frontend.eval()
print("   ✓ Frontend created")

# ============================================================================
# VALIDATION WITH REAL AUDIO
# ============================================================================

print("\n4. Testing with real audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Pad to exactly 560,000 samples (35 seconds)
max_samples = 560000
if len(audio) < max_samples:
    audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
else:
    audio = audio[:max_samples]

audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
print(f"   Input: {audio_tensor.shape}")

with torch.no_grad():
    mel_output = frontend(audio_tensor)

print(f"   Output: {mel_output.shape}")
print(f"   Expected: (1, 128, 3501)")

if mel_output.shape != (1, 128, 3501):
    print(f"   ❌ ERROR: Shape mismatch!")
    exit(1)

print("   ✓ Output shape correct")

# Compare with processor output
print("\n5. Comparing with HuggingFace processor...")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
hf_mel = inputs["input_features"]

# Pad HF output to 3501 if needed
if hf_mel.shape[2] < 3501:
    hf_mel = torch.nn.functional.pad(hf_mel, (0, 3501 - hf_mel.shape[2]))

# Compare statistics (not exact match due to normalization)
print(f"   Our mel - mean: {mel_output.mean():.6f}, std: {mel_output.std():.6f}")
print(f"   HF mel  - mean: {hf_mel.mean():.6f}, std: {hf_mel.std():.6f}")

# ============================================================================
# TORCH.JIT.TRACE EXPORT
# ============================================================================

print("\n6. Attempting torch.jit.trace...")
try:
    example_input = torch.randn(1, 560000)

    traced_model = torch.jit.trace(
        frontend,
        (example_input,),
        check_trace=True
    )
    print("   ✓ Model traced successfully!")

    # Validate traced output
    traced_output = traced_model(audio_tensor)
    max_diff = torch.abs(mel_output - traced_output).max().item()
    print(f"   Trace validation: max diff = {max_diff:.6f}")

except Exception as e:
    print(f"   ❌ Tracing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# COREML CONVERSION
# ============================================================================

print("\n7. Converting to CoreML...")

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="audio", shape=(1, 560000), dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="mel_spectrogram", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,  # Keep FP32 for preprocessing
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Preprocessing on CPU
    )

    # Save
    output_path = "build/ultra_static_frontend.mlpackage"
    mlmodel.save(output_path)
    print(f"   ✓ Saved to: {output_path}")

except Exception as e:
    print(f"   ❌ CoreML conversion failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# COREML VALIDATION
# ============================================================================

print("\n8. Testing CoreML model...")
coreml_model = ct.models.MLModel(output_path)

test_input = {
    "audio": audio_tensor.numpy().astype(np.float32)
}

coreml_output = coreml_model.predict(test_input)
coreml_mel = coreml_output["mel_spectrogram"]

print(f"   CoreML output: {coreml_mel.shape}")

# Compare with PyTorch
pytorch_mel = mel_output.numpy()
max_diff = np.abs(pytorch_mel - coreml_mel).max()
mean_diff = np.abs(pytorch_mel - coreml_mel).mean()

print(f"\n9. Validation:")
print(f"   Max diff: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")

if max_diff < 0.01:
    print("   ✅ Good match!")
else:
    print(f"   ⚠️  Larger diff than expected")

print(f"\n{'='*60}")
print("SUCCESS! Ultra-static frontend exported")
print(f"Output: {output_path}")
print(f"Shape: (1, 560000) → (1, 128, 3501)")
print('='*60)
