#!/usr/bin/env python3
"""Export ultra-static encoder with NO dynamic operations.

Following mobius pattern:
- Hard-code all shapes for 35s max input (3501 mel frames)
- Materialize all operations statically
- No conditional branching
- No shape-dependent operations
"""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import soundfile as sf
import librosa

print("=== Exporting Ultra-Static Encoder ===\n")

# Load original model
print("1. Loading source model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
model.eval()
print("   ✓ Model loaded")

# Extract components
original_encoder = model.encoder
projection = model.encoder_decoder_proj

# ============================================================================
# ULTRA-STATIC ENCODER - NO DYNAMIC OPERATIONS
# ============================================================================

class UltraStaticEncoder(nn.Module):
    """Completely static encoder with hard-coded shapes.

    Fixed Configuration:
    - Input: (1, 128, 3501) mel spectrogram
    - Output: (1, 438, 1024) encoder hidden states
    - Max audio: 35 seconds
    - No dynamic operations, no conditionals, no shape checks
    """

    def __init__(self, encoder, projection):
        super().__init__()

        # Copy ConvSubsampling layers (NO dynamic checks)
        self.conv = encoder.pre_encode.conv
        self.out = encoder.pre_encode.out

        # Copy encoder layers
        self.layers = encoder.layers

        # Copy projection
        self.projection = projection

        # Hard-coded constants
        self.d_model = 1280
        self.max_encoder_frames = 438  # ceil(3501 / 8)

        # Pre-materialize positional encoding (NO runtime creation)
        self.register_buffer('pos_enc', self._create_static_pos_enc())

    def _create_static_pos_enc(self):
        """Create static positional encoding buffer (NO dynamic ops)."""
        max_len = 2000  # Large enough for 438 frames
        d_model = self.d_model

        # Create sinusoidal encoding
        position = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(position.size(0), d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, 2*max_len-1, d_model)

    def forward(self, input_features):
        """
        Args:
            input_features: (1, 128, 3501) mel spectrogram

        Returns:
            encoder_output: (1, 438, 1024) projected encoder states
        """
        batch_size = 1  # Hard-coded

        # === ConvSubsampling (NO dynamic checks) ===
        # Input: (1, 128, 3501)
        x = input_features.transpose(1, 2).unsqueeze(1)  # (1, 1, 3501, 128)

        # Run conv layers DIRECTLY (no _needs_conv_split check)
        # Hardcoded length tensor for 3501 frames
        lengths = torch.tensor([3501], dtype=torch.int32)
        x, lengths = self.conv(x, lengths)

        # Reshape and linear
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.out(x)  # (1, 438, 1280)

        # === Positional Encoding (NO dynamic creation) ===
        input_len = x.size(1)  # Should be 438
        center_pos = self.pos_enc.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pos_enc[:, start_pos:end_pos]  # (1, 438, 1280)

        # === Attention Masks (STATIC) ===
        max_audio_length = input_len  # 438

        # Hard-code attention mask for 438 frames
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool)

        # Hard-code padding mask (all valid for full 3501 input)
        encoder_lengths = torch.tensor([438], dtype=torch.int32)
        pad_mask = torch.arange(0, max_audio_length).expand(1, -1) < encoder_lengths.unsqueeze(-1)

        # Combine masks
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(
            pad_mask_for_att_mask,
            pad_mask_for_att_mask.transpose(1, 2)
        )
        att_mask = torch.logical_and(att_mask, pad_mask_for_att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask

        # === Encoder Layers (NO modifications needed) ===
        for layer in self.layers:
            x = layer(x, pos_emb, mask=att_mask, pad_mask=pad_mask)

        # === Projection: 1280 → 1024 ===
        projected = self.projection(x)  # (1, 438, 1024)

        return projected


print("\n2. Creating ultra-static encoder...")
static_encoder = UltraStaticEncoder(original_encoder, projection)
static_encoder.eval()
print("   ✓ Static encoder created")

# ============================================================================
# VALIDATION WITH REAL AUDIO
# ============================================================================

print("\n3. Testing with real audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs["input_features"]

# Pad to exactly 3501 frames
input_padded = torch.nn.functional.pad(input_features, (0, 3501 - input_features.shape[2]))
print(f"   Input: {input_padded.shape}")

with torch.no_grad():
    encoder_output = static_encoder(input_padded)

print(f"   Output: {encoder_output.shape}")
print(f"   Expected: (1, 438, 1024)")

if encoder_output.shape != (1, 438, 1024):
    print(f"   ❌ ERROR: Shape mismatch!")
    exit(1)

print("   ✓ Output shape correct")

# ============================================================================
# TORCH.JIT.TRACE EXPORT
# ============================================================================

print("\n4. Attempting torch.jit.trace...")
try:
    # Use fixed-shape example
    example_input = torch.randn(1, 128, 3501)

    traced_model = torch.jit.trace(
        static_encoder,
        (example_input,),
        check_trace=True
    )
    print("   ✓ Model traced successfully!")

    # Validate traced output
    traced_output = traced_model(input_padded)
    max_diff = torch.abs(encoder_output - traced_output).max().item()
    print(f"   Trace validation: max diff = {max_diff:.6f}")

except Exception as e:
    print(f"   ❌ Tracing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# COREML CONVERSION
# ============================================================================

print("\n5. Converting to CoreML...")

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_features", shape=(1, 128, 3501), dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="encoder_output", dtype=np.float16)
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # Save
    output_path = "build/ultra_static_encoder.mlpackage"
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

print("\n6. Testing CoreML model...")
coreml_model = ct.models.MLModel(output_path)

test_input = {
    "input_features": input_padded.numpy().astype(np.float32)
}

coreml_output = coreml_model.predict(test_input)
coreml_encoder_out = coreml_output["encoder_output"]

print(f"   CoreML output: {coreml_encoder_out.shape}")

# Compare with PyTorch
pytorch_out = encoder_output.numpy()
max_diff = np.abs(pytorch_out - coreml_encoder_out).max()
mean_diff = np.abs(pytorch_out - coreml_encoder_out).mean()

print(f"\n7. Validation:")
print(f"   Max diff: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")

if max_diff < 0.1:
    print("   ✅ Good match!")
else:
    print(f"   ⚠️  Larger diff than expected")

print(f"\n{'='*60}")
print("SUCCESS! Ultra-static encoder exported")
print(f"Output: {output_path}")
print(f"Shape: (1, 128, 3501) → (1, 438, 1024)")
print('='*60)
