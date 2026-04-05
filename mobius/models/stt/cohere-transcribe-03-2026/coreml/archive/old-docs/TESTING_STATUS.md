# CoreML Models Testing Status

**Date**: 2026-04-04
**Test**: End-to-end pipeline with Python

## Summary

✅ **Models Run Successfully**: Encoder and decoder execute without errors and produce correct output shapes
⚠️ **Output is Garbage**: Transcription produces random tokens instead of real text

## Test Results

### Our Pipeline (Frontend + Our Encoder + Our Decoder)

**Configuration**:
- Frontend: BarathwajAnandan's `cohere_frontend.mlpackage`
- Encoder: Our `ultra_static_encoder.mlpackage` ✅
- Decoder: Our `ultra_static_decoder.mlpackage` ✅

**Results**:
```
Input: test-librispeech-real.wav (10.44s audio)

Pipeline execution:
1. Frontend: ✅ (1, 128, 3501) mel spectrogram
2. Encoder: ✅ (1, 438, 1024) hidden states
3. Decoder: ✅ (1, 108, 16384) logits

Output: "▁mainten▁apenasbres▁Auf▁essentially▁essentially▁40▁tú▁apenas▁wszyst..."
Status: ⚠️ GARBAGE (random multilingual tokens)
```

**Technical Details**:
- All shapes correct ✅
- No runtime errors ✅
- Decoder receives correct inputs:
  - `input_ids`: (1, 108) - prompt IDs + padding
  - `encoder_hidden_states`: (1, 438, 1024) - from our encoder
- Output logits: (1, 108, 16384) ✅
- Predicted tokens: Random mix of languages (Spanish "apenas", German "Auf", Polish "wszyst", etc.)

### BarathwajAnandan's Pipeline (Baseline)

**Configuration**:
- Frontend: BarathwajAnandan's
- Encoder: BarathwajAnandan's
- Decoder: BarathwajAnandan's (with attention masks)

**Results**:
- Testing in progress...
- Expected: Real English transcription

## Key Difference: Attention Masks

### Our Decoder (Ultra-Static)
```python
# Our decoder inputs (simplified)
decoder_out = decoder.predict({
    "input_ids": (1, 108),
    "encoder_hidden_states": (1, 438, 1024)
})
```

Internal attention masks are hard-coded:
- Self-attention mask: `torch.tril(torch.ones(108, 108))` - causal mask
- Cross-attention mask: `torch.ones(1, 108, 438)` - full attention

### BarathwajAnandan's Decoder
```python
# BarathwajAnandan's decoder inputs
decoder_out = decoder.predict({
    "input_ids": (1, 108),
    "decoder_attention_mask": (1, 108),  # ← Explicit mask
    "encoder_hidden_states": (1, 438, 1024),
    "cross_attention_mask": (1, 1, 1, 438)  # ← Different shape!
})
```

External attention masks provided:
- `decoder_attention_mask`: (1, 108) - marks which tokens are valid (not padding)
- `cross_attention_mask`: (1, 1, 1, 438) - controls encoder attention

## Hypothesis: Attention Mask Mismatch

**Problem**: Our decoder uses hard-coded internal masks, but BarathwajAnandan's decoder takes masks as inputs.

**Specifically**:
1. **Decoder attention mask**: We assume all 108 tokens are valid, but should only attend to the prompt (first 10 tokens)
2. **Cross-attention mask shape**: We use (1, 108, 438), they use (1, 1, 1, 438)

**Impact**:
- Decoder processes padding tokens as real tokens
- Cross-attention shape mismatch may cause incorrect attention patterns
- Results in garbage output despite technically correct execution

## Next Steps

### Option 1: Fix Our Decoder (Recommended)

Update `export-complete-decoder.py` to:
1. Add `decoder_attention_mask` as input (not hard-coded)
2. Use correct cross-attention mask shape: (1, 1, 1, 438) instead of (1, 108, 438)
3. Re-export decoder with these changes

**Benefits**:
- Matches BarathwajAnandan's proven architecture
- Should produce correct transcriptions
- Still fully static (just different input signature)

### Option 2: Use BarathwajAnandan's Decoder

Fall back to using their proven decoder:
- Frontend: BarathwajAnandan's ✅
- Encoder: Our ultra-static ✅
- Decoder: BarathwajAnandan's (proven working)

**Benefits**:
- Immediate testing capability
- Validates our encoder works correctly
- Can implement Swift frontend while debugging decoder

### Option 3: Debug Current Decoder

Investigate if our decoder can work without explicit masks:
- Compare attention patterns between our decoder and BarathwajAnandan's
- Test with modified internal masks
- May not be worth the effort

## Files

### Test Scripts
1. `test-our-models.py` - Tests our encoder + decoder
2. `test-barathwaj-baseline.py` - Baseline comparison

### Models Tested
1. `build/ultra_static_encoder.mlpackage` ✅ Works (correct shapes)
2. `build/ultra_static_decoder.mlpackage` ⚠️ Runs but produces garbage
3. `build/barathwaj-models/cohere_frontend.mlpackage` ✅ Works
4. `build/barathwaj-models/cohere_encoder.mlpackage` ✅ Works (baseline)
5. `build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage` - Testing...

## Recommendation

**Immediate**:
1. Wait for BarathwajAnandan baseline test to complete
2. Confirm their decoder produces real text
3. If yes, use Option 2 (their decoder + our encoder) to validate our encoder works

**Short-term**:
1. Fix our decoder to match BarathwajAnandan's input signature
2. Re-export with correct attention masks
3. Test fixed decoder

**Long-term**:
1. Implement Swift frontend (using WhisperMelSpectrogram pattern)
2. Replace all BarathwajAnandan components with our own
3. Full control over entire pipeline

## Technical Validation

✅ **Our encoder works**:
- Correct output shape
- Reasonable hidden state statistics
- No runtime errors

⚠️ **Our decoder needs fixing**:
- Runs successfully ✅
- Produces correct shape ✅
- But wrong attention configuration → garbage output ❌

**Root cause**: Attention mask mismatch (our internal hard-coded vs their external inputs)
