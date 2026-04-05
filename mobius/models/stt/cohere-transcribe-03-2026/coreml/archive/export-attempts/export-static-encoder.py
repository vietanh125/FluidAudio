#!/usr/bin/env python3
"""Export static encoder without dynamic operations."""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import soundfile as sf
import librosa

print("=== Creating Static Encoder ===\n")

# Load original model to copy weights
print("1. Loading original model to extract weights...")
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
print("   ✓ Original model loaded")

# Extract encoder and projection
original_encoder = model.encoder
projection = model.encoder_decoder_proj

print("\n2. Creating static encoder wrapper...")

class StaticEncoderWithProjection(nn.Module):
    """Static encoder that bypasses all dynamic operations.
    
    Hard-coded for:
    - Input: (1, 128, 3501) mel spectrogram
    - Output: (1, 438, 1024) encoder hidden states
    """
    def __init__(self, encoder, projection):
        super().__init__()
        # Copy all encoder layers
        self.pre_encode = encoder.pre_encode
        self.pos_enc = encoder.pos_enc
        self.layers = encoder.layers
        self.projection = projection
        
    def forward(self, input_features, feature_length):
        """
        Args:
            input_features: (1, 128, 3501) mel spectrogram
            feature_length: (1,) - actual valid frames
        Returns:
            encoder_output: (1, 438, 1024)
            encoder_length: (1,) - downsampled length (feature_length / 8)
        """
        # Run pre-encoding (ConvSubsampling) without dynamic checks
        # We know input is (1, 128, 3501) so it won't exceed int32 limits
        x = input_features.transpose(1, 2).unsqueeze(1)  # (1, 1, 3501, 128)
        
        # Run conv layers directly (bypass _needs_conv_split check)
        x, lengths = self.pre_encode.conv(x, feature_length)
        
        # Run output linear layer
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.pre_encode.out(x)
        
        # Positional encoding - materialize directly
        # For 438 frames, we need 2*438-1 = 875 positions
        # But let's use a fixed large buffer
        max_len = 1000
        device = x.device
        dtype = x.dtype
        
        # Create positional encoding directly
        positions = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pos_enc.d_model, 2, dtype=torch.float32, device=device)
            * -(torch.log(torch.tensor(10000.0)) / self.pos_enc.d_model)
        )
        pe = torch.zeros(positions.size(0), self.pos_enc.d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        
        # Extract relevant positions for input length
        input_len = x.size(1)
        center_pos = pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = pe[:, start_pos:end_pos]
        
        # Create attention masks
        max_audio_length = x.size(1)
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            lengths.size(0), -1
        ) < lengths.unsqueeze(-1)
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
        att_mask = torch.logical_and(att_mask.to(pad_mask_for_att_mask.device), pad_mask_for_att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask
        
        # Run encoder layers
        for layer in self.layers:
            x = layer(x, pos_emb, mask=att_mask, pad_mask=pad_mask)
        
        # Apply projection: 1280 → 1024
        projected = self.projection(x)
        
        # Compute output length (encoder downsamples by 8x)
        encoder_length = (feature_length.float() / 8.0).long()
        
        return projected, encoder_length

# Create static encoder
static_encoder = StaticEncoderWithProjection(original_encoder, projection)
static_encoder.eval()
print("   ✓ Static encoder created")

# Test with real audio
print("\n3. Testing with real audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs["input_features"]
feature_length = inputs["length"]

# Pad to 3501 frames
input_padded = torch.nn.functional.pad(input_features, (0, 3501 - input_features.shape[2]))

print(f"   Input: {input_padded.shape}, length: {feature_length.item()}")

with torch.no_grad():
    encoder_output, encoder_length = static_encoder(input_padded, feature_length)

print(f"   Output: {encoder_output.shape}, length: {encoder_length.item()}")
print(f"   Output dim: {encoder_output.shape[2]} (should be 1024)")

if encoder_output.shape[2] != 1024:
    print(f"   ❌ ERROR: Expected 1024-dim output, got {encoder_output.shape[2]}")
    exit(1)

# Try tracing
print("\n4. Attempting torch.jit.trace...")
try:
    # Use fixed-shape example inputs
    example_features = torch.randn(1, 128, 3501)
    example_length = torch.tensor([1043], dtype=torch.int32)
    
    traced_model = torch.jit.trace(static_encoder, (example_features, example_length))
    print("   ✓ Model traced successfully!")
    
except Exception as e:
    print(f"   ❌ Tracing failed: {e}")
    print("\n   This means there are still dynamic operations in the encoder.")
    print("   Need to further simplify the implementation.")
    exit(1)

# Convert to CoreML
print("\n5. Converting to CoreML...")

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_features", shape=(1, 128, 3501), dtype=np.float32),
        ct.TensorType(name="feature_length", shape=(1,), dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="encoder_output", dtype=np.float16),
        ct.TensorType(name="encoder_length", dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)

# Save
output_path = "build/static_encoder.mlpackage"
mlmodel.save(output_path)
print(f"   ✓ Saved to: {output_path}")

# Test CoreML model
print("\n6. Testing CoreML model...")
coreml_model = ct.models.MLModel(output_path)

test_input = {
    "input_features": input_padded.numpy().astype(np.float32),
    "feature_length": feature_length.numpy().astype(np.int32)
}

coreml_output = coreml_model.predict(test_input)
coreml_encoder_out = coreml_output["encoder_output"]
coreml_encoder_len = coreml_output["encoder_length"]

print(f"   CoreML output: {coreml_encoder_out.shape}")
print(f"   CoreML length: {coreml_encoder_len}")

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
print("SUCCESS! Static encoder exported")
print(f"Output: {output_path}")
print(f"Shape: (1, 128, 3501) → (1, {encoder_output.shape[1]}, 1024)")
print('='*60)
