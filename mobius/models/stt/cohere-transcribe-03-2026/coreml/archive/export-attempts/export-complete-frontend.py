#!/usr/bin/env python3
"""Export complete ultra-static frontend (mel spectrogram).

Following mobius pattern - completely static with hard-coded shapes.
"""
import torch
import torch.nn as nn
import torchaudio
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa

print("=== Exporting Complete Ultra-Static Frontend ===\n")

# ============================================================================
# ULTRA-STATIC FRONTEND
# ============================================================================

class UltraStaticFrontend(nn.Module):
    """Completely static mel spectrogram frontend.

    Fixed Configuration:
    - Input: (1, 560000) raw audio @ 16kHz (35 seconds)
    - Output: (1, 128, 3501) normalized mel spectrogram
    - No dynamic operations
    """

    def __init__(self):
        super().__init__()

        # Fixed mel spectrogram parameters (standard for ASR)
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
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
            mel: (1, 128, 3501) normalized mel spectrogram
        """
        # Compute mel spectrogram
        mel = self.mel_scale(audio)  # (1, 128, 3501)

        # Log scale
        mel = torch.clamp(mel, min=1e-10)
        mel = torch.log10(mel)

        # Normalize (subtract mean)
        mel_mean = mel.mean()
        mel = mel - mel_mean

        return mel


print("1. Creating ultra-static frontend...")
frontend = UltraStaticFrontend()
frontend.eval()
print("   ✓ Frontend created")

# ============================================================================
# VALIDATION WITH REAL AUDIO
# ============================================================================

print("\n2. Testing with real audio...")
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

print(f"   ✓ Output shape correct")
print(f"   Mel stats: min={mel_output.min():.3f}, max={mel_output.max():.3f}, mean={mel_output.mean():.6f}")

# ============================================================================
# TORCH.JIT.TRACE EXPORT
# ============================================================================

print("\n3. Attempting torch.jit.trace...")
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

print("\n4. Converting to CoreML...")

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
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Preprocessing on CPU
        pass_pipeline=ct.PassPipeline.EMPTY,  # Skip problematic optimizations
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

print("\n5. Testing CoreML model...")
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

print(f"\n6. Validation:")
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
