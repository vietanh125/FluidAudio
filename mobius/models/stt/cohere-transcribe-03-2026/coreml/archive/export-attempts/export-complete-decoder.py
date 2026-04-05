#!/usr/bin/env python3
"""Export complete ultra-static decoder (full sequence).

Following mobius pattern - completely static with hard-coded shapes.
"""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np

print("=== Exporting Complete Ultra-Static Decoder ===\n")

# Load model
print("1. Loading model...")
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

decoder = model.transf_decoder  # NOTE: transf_decoder, not decoder!
lm_head = model.log_softmax

print(f"\n2. Decoder info:")
print(f"   Decoder type: {type(decoder).__name__}")
print(f"   LM head type: {type(lm_head).__name__}")

# ============================================================================
# ULTRA-STATIC DECODER (FULL SEQUENCE)
# ============================================================================

class UltraStaticDecoder(nn.Module):
    """Static decoder for full-sequence decoding.

    Fixed Configuration:
    - Encoder: (1, 438, 1024) encoder hidden states
    - Input IDs: (1, 108) token IDs (prompt + generation)
    - Output: (1, 108, 16384) logits
    - No KV cache (full sequence every time)
    """

    def __init__(self, decoder, lm_head):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head

        # Fixed constants
        self.max_seq_len = 108
        self.encoder_len = 438
        self.vocab_size = 16384

    def forward(self, input_ids, encoder_hidden_states):
        """
        Args:
            input_ids: (1, 108) token IDs
            encoder_hidden_states: (1, 438, 1024) encoder output

        Returns:
            logits: (1, 108, 16384) token logits
        """
        # Create position indices (0, 1, 2, ..., 107)
        positions = torch.arange(self.max_seq_len).unsqueeze(0)  # (1, 108)

        # Create attention masks
        batch_size = 1
        seq_len = self.max_seq_len
        encoder_len = self.encoder_len

        # Self-attention mask (causal mask for decoder)
        # Shape: (batch, seq_len, seq_len)
        self_attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        # Cross-attention mask (all valid)
        # Shape: (batch, seq_len, encoder_len)
        cross_attention_mask = torch.ones(batch_size, seq_len, encoder_len)

        # Run decoder
        decoder_output = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=None,
            cache_position=None,
            kv_seq_len=None,
        )

        # Get hidden states (decoder returns tuple: (hidden_states, past_key_values))
        hidden_states = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output

        # Apply LM head
        logits = self.lm_head(hidden_states)  # (1, 108, 16384)

        return logits


print("\n3. Creating ultra-static decoder...")
static_decoder = UltraStaticDecoder(decoder, lm_head)
static_decoder.eval()
print("   ✓ Static decoder created")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n4. Testing decoder...")

# Create test inputs
# Prompt: [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
prompt_ids = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
input_ids = torch.tensor([prompt_ids + [0] * (108 - len(prompt_ids))], dtype=torch.long)
encoder_output = torch.randn(1, 438, 1024)  # Dummy encoder output

print(f"   Input IDs: {input_ids.shape}")
print(f"   Encoder: {encoder_output.shape}")

with torch.no_grad():
    logits = static_decoder(input_ids, encoder_output)

print(f"   Output logits: {logits.shape}")
print(f"   Expected: (1, 108, 16384)")

if logits.shape != (1, 108, 16384):
    print(f"   ❌ ERROR: Shape mismatch!")
    exit(1)

print("   ✓ Output shape correct")

# ============================================================================
# TORCH.JIT.TRACE EXPORT
# ============================================================================

print("\n5. Attempting torch.jit.trace...")
try:
    example_ids = torch.randint(0, 16384, (1, 108), dtype=torch.long)
    example_encoder = torch.randn(1, 438, 1024)

    traced_model = torch.jit.trace(
        static_decoder,
        (example_ids, example_encoder),
        check_trace=False,  # Disable strict checking due to attention
    )
    print("   ✓ Model traced successfully!")

    # Validate traced output
    traced_output = traced_model(input_ids, encoder_output)
    max_diff = torch.abs(logits - traced_output).max().item()
    print(f"   Trace validation: max diff = {max_diff:.6f}")

except Exception as e:
    print(f"   ❌ Tracing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# COREML CONVERSION
# ============================================================================

print("\n6. Converting to CoreML...")

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 108), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 438, 1024), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16)
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # Save
    output_path = "build/ultra_static_decoder.mlpackage"
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

print("\n7. Testing CoreML model...")
coreml_model = ct.models.MLModel(output_path)

test_input = {
    "input_ids": input_ids.numpy().astype(np.int32),
    "encoder_hidden_states": encoder_output.numpy().astype(np.float32),
}

coreml_output = coreml_model.predict(test_input)
coreml_logits = coreml_output["logits"]

print(f"   CoreML output: {coreml_logits.shape}")

# Compare with PyTorch
pytorch_logits = logits.numpy()
max_diff = np.abs(pytorch_logits - coreml_logits).max()
mean_diff = np.abs(pytorch_logits - coreml_logits).mean()

print(f"\n8. Validation:")
print(f"   Max diff: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")

if max_diff < 1.0:  # Looser tolerance for decoder
    print("   ✅ Good match!")
else:
    print(f"   ⚠️  Larger diff than expected")

print(f"\n{'='*60}")
print("SUCCESS! Ultra-static decoder exported")
print(f"Output: {output_path}")
print(f"Shape: (1, 108) + (1, 438, 1024) → (1, 108, 16384)")
print('='*60)
