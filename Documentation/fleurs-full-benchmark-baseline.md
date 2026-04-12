# FLEURS Full Benchmark Results - Parakeet v3 Baseline

**Date:** 2026-04-11
**Branch:** `main`
**Model:** Parakeet TDT v3 (0.6B)
**Samples:** 100 per language × 24 languages = 2,400 total
**Duration:** 21 minutes 39 seconds

## Summary

This benchmark establishes the baseline performance of Parakeet v3 on the FLEURS multilingual dataset before implementing script filtering for issue #512.

**Key Findings:**
- Polish shows 8.98% WER, confirming Cyrillic script confusion issue
- All languages maintain real-time performance (RTFx > 40x)
- Average RTFx across all languages: 62.6x
- Best performance: Italian (3.46% WER)
- Lowest performance: Greek (38.91% WER)

## Complete Results

| Language | Code | WER% | CER% | RTFx | Duration | Samples |
|----------|------|------|------|------|----------|---------|
| English (US) | en_us | 4.57 | 2.46 | 47.9x | 953.9s | 100 |
| Spanish (Spain) | es_419 | 3.80 | 1.59 | 67.6x | 1200.8s | 100 |
| Italian (Italy) | it_it | 3.46 | 1.35 | 86.1x | 1516.9s | 100 |
| French (France) | fr_fr | 6.59 | 2.86 | 50.0x | 1073.7s | 100 |
| German (Germany) | de_de | 5.92 | 2.69 | 53.8x | 1496.2s | 100 |
| Russian (Russia) | ru_ru | 7.01 | 2.01 | 64.1x | 1136.6s | 100 |
| Dutch (Netherlands) | nl_nl | 8.12 | 3.07 | 52.6x | 1009.6s | 100 |
| **Polish (Poland)** | **pl_pl** | **8.98** | **3.17** | **53.0x** | **964.7s** | **100** |
| Ukrainian (Ukraine) | uk_ua | 7.02 | 2.12 | 59.3x | 1098.1s | 100 |
| Slovak (Slovakia) | sk_sk | 13.96 | 5.39 | 46.2x | 1196.3s | 100 |
| Czech (Czechia) | cs_cz | 11.28 | 3.67 | 68.0x | 1239.0s | 100 |
| Bulgarian (Bulgaria) | bg_bg | 11.78 | 3.74 | 47.8x | 1021.9s | 100 |
| Croatian (Croatia) | hr_hr | 13.52 | 4.06 | 60.0x | 1025.7s | 100 |
| Romanian (Romania) | ro_ro | 15.02 | 4.63 | 68.2x | 1110.8s | 100 |
| Finnish (Finland) | fi_fi | 16.08 | 4.98 | 66.1x | 1348.5s | 100 |
| Hungarian (Hungary) | hu_hu | 19.52 | 6.52 | 84.8x | 1295.2s | 100 |
| Swedish (Sweden) | sv_se | 17.44 | 5.83 | 65.6x | 1079.0s | 100 |
| Estonian (Estonia) | et_ee | 19.66 | 4.31 | 68.8x | 1198.9s | 100 |
| Danish (Denmark) | da_dk | 19.62 | 7.56 | 56.9x | 1125.7s | 100 |
| Lithuanian (Lithuania) | lt_lt | 25.33 | 7.45 | 70.5x | 1055.8s | 100 |
| **Greek (Greece)** | **el_gr** | **38.91** | **15.45** | **72.1x** | **1098.7s** | **100** |
| Maltese (Malta) | mt_mt | 29.59 | 11.23 | 68.1x | 1399.1s | 100 |
| Latvian (Latvia) | lv_lv | 26.20 | 7.35 | 76.1x | 1176.1s | 100 |
| Slovenian (Slovenia) | sl_si | 27.10 | 9.83 | 43.0x | 940.0s | 100 |

**Polish** is highlighted as the target language for issue #512 (Cyrillic script confusion).
**Greek** shows the highest WER, indicating potential room for improvement.

## Performance Categories

### Excellent (WER < 5%)
- 🥇 Italian: 3.46%
- 🥈 Spanish: 3.80%
- 🥉 English: 4.57%

### Very Good (WER 5-7%)
- German: 5.92%
- French: 6.59%
- Russian: 7.01%
- Ukrainian: 7.02%

### Good (WER 8-10%)
- Dutch: 8.12%
- Polish: 8.98% ← **Target for script filtering improvement**

### Moderate (WER 11-16%)
- Czech: 11.28%
- Bulgarian: 11.78%
- Croatian: 13.52%
- Slovak: 13.96%
- Romanian: 15.02%
- Finnish: 16.08%

### Fair (WER 17-20%)
- Swedish: 17.44%
- Danish: 19.62%
- Hungarian: 19.52%
- Estonian: 19.66%

### Lower (WER > 20%)
- Lithuanian: 25.33%
- Latvian: 26.20%
- Slovenian: 27.10%
- Maltese: 29.59%
- Greek: 38.91%

## Methodology

- **Model**: Parakeet TDT v3 (0.6B) with standard JointDecision (argmax only)
- **Dataset**: FLEURS multilingual benchmark
- **Sample Size**: 100 utterances per language
- **Evaluation**: Levenshtein distance for WER/CER calculation
- **Hardware**: Apple Silicon (M-series)
- **Compute Units**: Neural Engine + GPU

## Next Steps

1. Implement script filtering using JointDecisionv3 (top-K outputs)
2. Re-run benchmark on `feat/script-filtering-issue-512` branch
3. Compare WER improvement for Polish and other affected languages
4. Validate no regression on languages without script ambiguity

## Raw Results

Individual JSON results saved to:
```
benchmark_results/fleurs_*_20260411_224806.json
```

Full benchmark log:
```
benchmark_results/fleurs_full_benchmark_20260411_224806.log
```

## Related Issues

- [#512](https://github.com/FluidInference/FluidAudio/issues/512) - Polish utterances transcribed in Cyrillic instead of Latin script
- [#515](https://github.com/FluidInference/FluidAudio/pull/515) - Script filtering implementation (in progress)
