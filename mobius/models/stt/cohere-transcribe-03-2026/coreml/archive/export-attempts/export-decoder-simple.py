#!/usr/bin/env python3
"""Export Cohere decoder WITHOUT KV cache (simpler, for testing preprocessor).

This is a simplified decoder export just to test the preprocessor with the encoder.
For full KV cache support, use convert-decoder-with-cache.py (needs debugging).
"""
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq

print("Loading Cohere Transcribe model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
model.eval()

class SimpleDecoderWrapper(nn.Module):
    """Simple wrapper that uses the decoder's forward pass directly."""

    def __init__(self, transf_decoder, encoder_decoder_proj):
        super().__init__()
        self.transf_decoder = transf_decoder
        self.encoder_decoder_proj = encoder_decoder_proj

    def forward(self, input_ids, encoder_hidden_states):
        """
        Args:
            input_ids: (B, seq_len)
            encoder_hidden_states: (B, enc_seq_len, 1280)

        Returns:
            hidden_states: (B, seq_len, 1024)
        """
        # Project encoder states
        encoder_proj = self.encoder_decoder_proj(encoder_hidden_states)

        # Run decoder
        output = self.transf_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_proj
        )

        return output

print("Creating simple decoder wrapper...")
wrapper = SimpleDecoderWrapper(
    transf_decoder=model.transf_decoder,
    encoder_decoder_proj=model.encoder_decoder_proj
)
wrapper.eval()

# Create dummy inputs
B = 1
seq_len = 10  # Small sequence for testing
enc_seq_len = 375

input_ids = torch.randint(0, 16384, (B, seq_len))
encoder_hidden = torch.randn(B, enc_seq_len, 1280)

print("Testing wrapper...")
with torch.no_grad():
    output = wrapper(input_ids, encoder_hidden)
    print(f"  Output shape: {output.shape}")

print("\nTracing model...")
with torch.no_grad():
    traced = torch.jit.trace(
        wrapper,
        (input_ids, encoder_hidden),
        strict=False
    )

print("\nConverting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=(1, 375, 1280), dtype=np.float32)
    ],
    outputs=[
        ct.TensorType(name="hidden_states")
    ],
    minimum_deployment_target=ct.target.iOS17,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

output_path = Path("build/decoder_simple.mlpackage")
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"\nSaving to {output_path}...")
mlmodel.save(str(output_path))

print("\n✅ Done! Simple decoder exported.")
print("\nNote: This decoder does NOT have KV cache support.")
print("It's meant for testing the preprocessor + encoder pipeline.")
print("\nFor KV cache support, fix convert-decoder-with-cache.py")
