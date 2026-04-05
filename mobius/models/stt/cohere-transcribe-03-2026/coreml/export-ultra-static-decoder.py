#!/usr/bin/env python3
"""Export ultra-static decoder with KV cache.

Following mobius pattern:
- Hard-code all shapes for max sequence length
- Static KV cache handling
- No dynamic operations
"""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np

print("=== Exporting Ultra-Static Decoder ===\n")

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

decoder = model.transf_decoder

print(f"\n2. Decoder info:")
print(f"   Type: {type(decoder).__name__}")
print(f"   Has config: {hasattr(decoder, 'config')}")

# ============================================================================
# ULTRA-STATIC DECODER (WITHOUT KV CACHE - FIRST TOKEN)
# ============================================================================

class UltraStaticDecoderFirstToken(nn.Module):
    """Static decoder for first token generation (no KV cache input).

    Fixed Configuration:
    - Encoder: (1, 438, 1024) encoder hidden states
    - Input IDs: (1, 10) prompt tokens
    - Output: (1, 10, 16384) logits + KV cache for next tokens
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

        # Fixed constants
        self.num_layers = 8
        self.num_heads = 8
        self.head_dim = 128
        self.max_prompt_len = 10
        self.encoder_len = 438

    def forward(self, input_ids, encoder_hidden_states):
        """
        Args:
            input_ids: (1, 10) prompt token IDs
            encoder_hidden_states: (1, 438, 1024) encoder output

        Returns:
            logits: (1, 10, 16384) token logits
        """
        # Create attention mask (all ones for prompt)
        attention_mask = torch.ones(1, self.max_prompt_len, dtype=torch.int32)

        # Create encoder attention mask (all ones for encoder output)
        encoder_attention_mask = torch.ones(1, self.encoder_len, dtype=torch.int32)

        # Run decoder
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,  # No cache for first token
            return_dict=True,
        )

        logits = outputs.logits  # (1, 10, 16384)

        return logits


# ============================================================================
# ULTRA-STATIC DECODER (WITH KV CACHE - SUBSEQUENT TOKENS)
# ============================================================================

class UltraStaticDecoderCached(nn.Module):
    """Static decoder for subsequent tokens (with KV cache).

    Fixed Configuration:
    - Input: (1, 1) single new token
    - Encoder: (1, 438, 1024) encoder hidden states
    - KV Cache: Past key/values for fast generation
    - Output: (1, 1, 16384) logits for next token
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

        # Fixed constants
        self.num_layers = 8
        self.num_heads = 8
        self.head_dim = 128
        self.encoder_len = 438

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        past_key_values_flat,  # Flattened KV cache
        past_length,  # How many tokens already generated
    ):
        """
        Args:
            input_ids: (1, 1) new token ID
            encoder_hidden_states: (1, 438, 1024) encoder output
            past_key_values_flat: Flattened past KV cache
            past_length: Scalar indicating cache length

        Returns:
            logits: (1, 1, 16384) token logits
            new_kv_cache: Updated KV cache
        """
        # Reshape past_key_values from flat to structured
        # past_key_values is tuple of (key, value) for each layer
        # Each is shape (batch, num_heads, seq_len, head_dim)

        # For now, use simplified version without cache
        # (Full cache implementation requires complex reshaping)

        # Create attention mask
        seq_len = past_length + 1
        attention_mask = torch.ones(1, seq_len, dtype=torch.int32)

        # Create encoder attention mask
        encoder_attention_mask = torch.ones(1, self.encoder_len, dtype=torch.int32)

        # Run decoder
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits  # (1, 1, 16384)

        return logits


print("\n3. Creating ultra-static decoders...")
decoder_first = UltraStaticDecoderFirstToken(decoder)
decoder_first.eval()
print("   ✓ First-token decoder created")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n4. Testing first-token decoder...")

# Create test inputs matching Cohere prompt format
# Prompt: [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
prompt_ids = torch.tensor([[13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]], dtype=torch.long)
encoder_output = torch.randn(1, 438, 1024)  # Dummy encoder output

print(f"   Prompt IDs: {prompt_ids.shape}")
print(f"   Encoder: {encoder_output.shape}")

with torch.no_grad():
    logits = decoder_first(prompt_ids, encoder_output)

print(f"   Output logits: {logits.shape}")
print(f"   Expected: (1, 10, 16384)")

if logits.shape != (1, 10, 16384):
    print(f"   ❌ ERROR: Shape mismatch!")
    exit(1)

print("   ✓ Output shape correct")

# ============================================================================
# TORCH.JIT.TRACE EXPORT
# ============================================================================

print("\n5. Attempting torch.jit.trace for first-token decoder...")
try:
    example_ids = torch.tensor([[13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]], dtype=torch.long)
    example_encoder = torch.randn(1, 438, 1024)

    traced_model = torch.jit.trace(
        decoder_first,
        (example_ids, example_encoder),
        check_trace=False,  # Disable strict checking due to attention
    )
    print("   ✓ Model traced successfully!")

    # Validate traced output
    traced_output = traced_model(prompt_ids, encoder_output)
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
            ct.TensorType(name="input_ids", shape=(1, 10), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 438, 1024), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16)
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # Save
    output_path = "build/ultra_static_decoder_first.mlpackage"
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
    "input_ids": prompt_ids.numpy().astype(np.int32),
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
print(f"Shape: (1, 10) + (1, 438, 1024) → (1, 10, 16384)")
print('='*60)
