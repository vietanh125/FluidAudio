# Encoder Export is Broken

**Date**: 2026-04-04
**Status**: ❌ Our ultra-static encoder produces incorrect output

## Test Result

**Configuration**:
- Frontend: BarathwajAnandan's `cohere_frontend.mlpackage` ✅
- Encoder: Our `ultra_static_encoder.mlpackage` ❌
- Decoder: BarathwajAnandan's `cohere_decoder_fullseq_masked.mlpackage` ✅

**Input**: `test-librispeech-real.wav` (10.44s English speech)

**Output**:
```
<|startofcontext|><|startoftranscript|><|emo:undefined|>.<|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>▁L
```

**Expected**: Real English transcription (e.g., "concord returned to its place...")

**Result**: ❌ **GARBAGE** - Decoder outputs special tokens and control codes instead of real text

## What This Means

Our encoder's hidden states are incompatible with the decoder. The decoder receives the encoder output but produces garbage, which means:

1. **Encoder output shape is correct**: (1, 438, 1024) ✅
2. **Encoder output values are wrong**: The numerical values don't match expectations ❌

## Possible Causes

### 1. Normalization Issue
The mel spectrogram might not be normalized correctly before encoding:
- BarathwajAnandan's frontend outputs mel with mean ≈ 0
- Our encoder might expect different normalization
- Or our encoder might be applying incorrect internal normalization

### 2. Projection Layer Issue
The 1280→1024 projection in our encoder might be:
- Applied incorrectly
- Missing layer norm
- Using wrong weights

### 3. Positional Encoding Issue
Our pre-materialized positional encoding might be:
- Computed incorrectly
- Wrong shape/indexing
- Missing some transformation

### 4. Attention Mask Issue
Our hard-coded attention masks might be:
- Wrong shape
- Wrong values
- Missing padding mask logic

### 5. Layer Ordering Issue
The encoder layers might be:
- Applied in wrong order
- Missing some operation
- Using wrong activation functions

## Debugging Steps

### Step 1: Compare Encoder Outputs Numerically ✅ IN PROGRESS
Script: `compare-encoders.py`
- Run both encoders on same mel input
- Compare hidden state values
- If they differ significantly → encoder export is broken
- If they match closely → problem is elsewhere

### Step 2: Check PyTorch vs CoreML Encoder
Export script: `export-ultra-static-encoder.py`
- Line 119: Test with real audio passed ✅
- Line 148: Trace validation showed max diff 0.004490 ✅
- But: This was with dummy random audio, not real mel spectrogram

**Need to validate**:
- PyTorch encoder on real mel → hidden states
- CoreML encoder on same mel → hidden states
- Compare numerically

### Step 3: Check Encoder Architecture
Compare our wrapper with BarathwajAnandan's:
- Input normalization
- Positional encoding application
- Projection layer placement
- Output format

## Recommendation

**Option 1: Debug Current Encoder** 📊
1. Run `compare-encoders.py` to see numerical diff
2. If large diff: Review ultra-static wrapper implementation
3. Check each component (pos_enc, projection, masks)
4. Re-export with fixes

**Option 2: Use BarathwajAnandan's Encoder** ✅ IMMEDIATE
1. Use their proven working encoder temporarily
2. Focus on getting decoder working
3. Come back to encoder later

**Option 3: Start Fresh** 🔄
1. Re-read BarathwajAnandan's encoder architecture
2. Create new ultra-static wrapper matching their approach exactly
3. Export and test incrementally

## Files

### Test Scripts
- `test-isolate-encoder.py` - Isolates our encoder (FAILED ❌)
- `compare-encoders.py` - Compares encoder outputs (RUNNING)

### Export Script
- `export-ultra-static-encoder.py` - Our encoder export (NEEDS FIX)

### Models
- `build/ultra_static_encoder.mlpackage` - Our broken encoder ❌
- `build/barathwaj-models/cohere_encoder.mlpackage` - Working baseline ✅

## Next Action

**Wait for `compare-encoders.py` to finish** - This will show us exactly how different the encoder outputs are, which will guide the debugging approach.

If encoders produce very different outputs (max diff > 1.0):
→ Major issue in encoder export, need to review architecture

If encoders produce similar outputs (max diff < 0.1):
→ Subtle issue, possibly in final projection or normalization

If encoders match closely (max diff < 0.01):
→ Problem might not be in encoder, investigate decoder
