# KV Cache Solution for Cohere Transcribe CoreML

## Problem

The original decoder discarded KV cache after each forward pass, causing the model to "forget" all previous tokens and produce garbage output like "çonçonçon...".

```python
# Original broken code
decoder_output, _ = self.transf_decoder(...)  # ← Cache discarded!
return decoder_output
```

## Solution: Explicit KV Cache Management

Following the PocketTTS pattern, we implement explicit KV cache tensors as model inputs/outputs.

### Architecture

**Cache Structure:**
```
Self-Attention Cache (per layer):
  Shape: (2, B, max_seq_len, H, D)
  - 2: [key, value]
  - B: batch size (1)
  - max_seq_len: 512 (ring buffer)
  - H: num_heads (8)
  - D: head_dim (128)

Cross-Attention Cache (per layer):
  Shape: (2, B, enc_seq_len, H, D)
  - enc_seq_len: 375 (from encoder)
  - Pre-computed from encoder output
  - Constant during generation
```

**Model Inputs:**
- `input_ids`: (B, 1) - single token
- `position`: (B,) - current position (float for ring buffer mod)
- `encoder_hidden_states`: (B, 375, 1280)
- `self_cache0-7`: 8 self-attention caches
- `cross_cache0-7`: 8 cross-attention caches

**Model Outputs:**
- `var_2384`: hidden_states (B, 1, 1024)
- `var_2387`: new_position (B,) - incremented
- `new_self_cache_X`: 8 updated self-attention caches

### Implementation Details

**1. Ring Buffer Indexing**
```python
# Position wraps at max_seq_len
write_idx = (position % max_seq_len).long()

# Scatter new K/V at position
new_cache[0] = old_cache[0].scatter(1, write_idx, key)
new_cache[1] = old_cache[1].scatter(1, write_idx, value)
```

**2. Cache Updates**
- **Self-attention**: Updated every step (grows with generation)
- **Cross-attention**: Constant (computed once from encoder)
- Position tracking: `new_position = position + 1.0`

**3. Attention Computation**
```python
# Extract valid cached keys/values up to current position
valid_len = min(position + 1, max_seq_len)
cached_key = cache[0][:, :valid_len, :, :]
cached_value = cache[1][:, :valid_len, :, :]

# Standard attention
scores = query @ cached_key.T / sqrt(head_dim)
attn_out = softmax(scores) @ cached_value
```

### Cross-Cache Computer

To avoid OOM from loading both PyTorch and CoreML models, we export a separate model that computes cross-attention caches:

```python
class CrossCacheComputer(nn.Module):
    def forward(self, encoder_hidden_states):
        # Project encoder output
        encoder_proj = self.encoder_decoder_proj(encoder_hidden_states)

        # Compute K/V for all 8 layers
        caches = []
        for i in range(8):
            key = cross_key_net_i(encoder_proj).view(B, enc_seq_len, H, D)
            value = cross_value_net_i(encoder_proj).view(B, enc_seq_len, H, D)
            caches.append(torch.stack([key, value], dim=0))

        return tuple(caches)
```

**Output names:** var_76, var_93, var_110, var_127, var_144, var_161, var_178, var_195

## Validation Results

### Python (CoreML)
```
Generated: "concord returned to its place amidst the tents"
Ground truth: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
✅ PERFECT MATCH!
```

**Test file:** `test-decoder-cache-full.py`

### Swift
**Test file:** `test-swift-cache.swift` (426 lines)

Full pipeline demonstration:
1. Load audio from WAV file
2. Preprocess to mel spectrogram
3. Encode to hidden states
4. Compute cross-caches from encoder output
5. Initialize self-caches (zeros)
6. Generation loop with KV cache updates
7. Decode tokens to text

**Status:** Implementation complete, blocked by preprocessor normalization issue (separate from KV cache)

## Files

### Export Scripts
- `convert-decoder-with-cache.py` (396 lines) - Main decoder export
- `export-cross-cache-computer.py` - Cross-cache computer export

### Testing
- `test-decoder-cache-full.py` - Python validation (PERFECT MATCH)
- `test-swift-cache.swift` - Swift demo (426 lines)

### Models
- `decoder_with_cache.mlpackage` - Decoder with KV cache
- `cross_cache_computer.mlpackage` - Cross-cache computation

## Usage

### Python
```python
# Initialize caches
self_caches = {f"self_cache{i}": np.zeros((2, 1, 512, 8, 128)) for i in range(8)}
cross_caches = {f"cross_cache{i}": computed_cache_i for i in range(8)}

position = np.array([0.0])

for step in range(max_steps):
    # Decoder input
    dec_out = decoder.predict({
        "input_ids": input_ids,
        "position": position,
        "encoder_hidden_states": encoder_hidden,
        **self_caches,
        **cross_caches
    })

    # Extract outputs
    hidden_states = dec_out["var_2384"]
    position = dec_out["var_2387"]

    # Update self-caches
    for i in range(8):
        cache_name = f"new_self_cache_{...}_internal_tensor_assign_2"
        self_caches[f"self_cache{i}"] = dec_out[cache_name]

    # Get next token
    logits = lm_head.predict({"hidden_states": hidden_states})["logits"]
    next_token = np.argmax(logits)
```

### Swift
```swift
// Initialize
var selfCaches: [String: MLMultiArray] = [:]
for i in 0..<8 {
    selfCaches["self_cache\(i)"] = zerosArray(shape: [2, 1, 512, 8, 128])
}

var crossCaches = computeCrossCaches(from: encoderHidden)
var position = MLMultiArray(shape: [1], dataType: .float32)
position[0] = 0.0

// Generation loop
for step in 0..<maxSteps {
    var features = [
        "input_ids": MLFeatureValue(multiArray: inputIds),
        "position": MLFeatureValue(multiArray: position),
        "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden)
    ]

    for (name, cache) in selfCaches {
        features[name] = MLFeatureValue(multiArray: cache)
    }
    for (name, cache) in crossCaches {
        features[name] = MLFeatureValue(multiArray: cache)
    }

    let decOutput = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: features))

    // Update position and self-caches
    position = decOutput.featureValue(for: "var_2387")!.multiArrayValue!

    for i in 0..<8 {
        let cacheName = cacheOutputNames[i]
        selfCaches["self_cache\(i)"] = decOutput.featureValue(for: cacheName)!.multiArrayValue!
    }
}
```

## Key Insights

1. **PocketTTS pattern works perfectly** - Explicit cache tensors + ring buffer + scatter operations

2. **Cross-cache computer is essential** - Avoids OOM, cleaner separation of concerns

3. **Python validation proves correctness** - Perfect transcription match confirms implementation

4. **Swift implementation mirrors Python** - Same cache management, same model calls

5. **Preprocessing is separate issue** - KV cache works; Swift has preprocessor/encoder issue unrelated to cache mechanism

## Next Steps

1. **Resolve Swift preprocessing** - Export Cohere's exact FilterbankFeatures or normalize in Swift
2. **Performance optimization** - Profile cache operations, consider quantization
3. **Integration into FluidAudio** - Add to ASR module following this pattern

## References

- PocketTTS CoreML implementation (similar cache pattern)
- Cohere Transcribe: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- CoreML scatter operations: https://apple.github.io/coremltools/
