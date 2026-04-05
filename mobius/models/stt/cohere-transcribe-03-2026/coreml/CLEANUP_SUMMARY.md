# Cleanup Summary - 2026-04-04

## Files Cleaned Up

### ✅ Kept (Working & Essential)

**Python Scripts:**
- `test-autoregressive-decode.py` - ✅ **Working pipeline** (KEEP THIS!)
- `test-transformers-ground-truth.py` - PyTorch reference
- `export-ultra-static-encoder.py` - Export methodology reference
- `export-ultra-static-decoder.py` - Export methodology reference
- `export-ultra-static-frontend.py` - Export methodology reference

**Documentation:**
- `README.md` - Clean overview and quick start
- `status.md` - Complete reverse engineering log (updated with final solution)
- `TEST_RESULTS.md` - Detailed test results

**Other:**
- `.venv312/` - Python 3.12 virtual environment
- `build/barathwaj-models/` - Working CoreML models
- `test-librispeech-real.wav` - Test audio file

### 🗑️ Archived

**Failed Test Attempts (24 files):**
- Moved to `archive/failed-tests/`
- Includes: test-barathwaj-*.py, test-cohere-*.py, test-cpu-*.py, etc.
- These were experiments that led to dead ends

**Investigation Scripts (17 files):**
- Moved to `archive/investigation-scripts/`
- Includes: analyze-*.py, check-*.py, compare-*.py, inspect-*.py, etc.
- One-off scripts used during reverse engineering

**Failed Export Attempts (11 files):**
- Moved to `archive/export-attempts/`
- Includes: export-cohere-*.py, export-complete-*.py, export-encoder-*.py, etc.
- Early export attempts before finding working approach

**Outdated Documentation (8 files):**
- Moved to `archive/old-docs/`
- Includes: ENCODER_BROKEN.md, KV_CACHE_*.md, PIPELINE.md, etc.
- Historical documents superseded by current findings

## Summary

**Before Cleanup:**
- 26 test scripts
- 20+ investigation scripts
- 14 export scripts
- 12 markdown files
- **Total: 70+ files**

**After Cleanup:**
- 5 essential Python scripts
- 3 documentation files
- 4 archive directories
- **Total: ~12 files in main directory**

## What Changed

### Documentation
- ✅ `README.md` - Completely rewritten with clean quick start
- ✅ `status.md` - Appended final working solution
- ✅ `TEST_RESULTS.md` - Updated with autoregressive decoding findings

### Key Discovery Added to status.md
```python
# Working autoregressive decoder implementation
# See test-autoregressive-decode.py for complete code
```

## Archive Structure

```
archive/
├── failed-tests/           # 24 test attempts (historical)
├── investigation-scripts/  # 17 investigation scripts
├── export-attempts/        # 11 failed export attempts
└── old-docs/              # 8 outdated markdown files
```

All archived files are preserved for historical reference but removed from main directory.

## Result

Clean, focused directory with only working code and current documentation. Easy to understand what works and how to use it.
