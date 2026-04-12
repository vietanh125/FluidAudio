#!/bin/bash
# Run FLEURS full multilingual benchmark (100 samples x 24 languages = 2,400 samples) with sleep prevention.
#
# Benchmarks all 24 languages supported by Parakeet TDT v3:
#   Best (WER < 5%): en_us, es_419, it_it, fr_fr, de_de
#   Good (5-10%): ru_ru, nl_nl, pl_pl, uk_ua, sk_sk
#   Moderate (10-15%): cs_cz, bg_bg, hr_hr, ro_ro, fi_fi
#   Lower (>15%): hu_hu, sv_se, et_ee, da_dk, lt_lt, el_gr, mt_mt, lv_lv, sl_si
#
# Usage:
#   ./Scripts/fleurs_full_benchmark.sh
#
# The script downloads FLEURS data automatically if needed.
# Uses caffeinate to prevent sleep so you can close the lid.
# Results are saved to benchmark_results/ with timestamps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/fleurs_full_benchmark_${TIMESTAMP}.log"
SAMPLES_PER_LANG=100

# All 24 supported languages
LANGUAGES=(
    # Best performing (WER < 5%)
    "en_us" "es_419" "it_it" "fr_fr" "de_de"
    # Good performance (WER 5-10%)
    "ru_ru" "nl_nl" "pl_pl" "uk_ua" "sk_sk"
    # Moderate performance (WER 10-15%)
    "cs_cz" "bg_bg" "hr_hr" "ro_ro" "fi_fi"
    # Lower performance (WER > 15%)
    "hu_hu" "sv_se" "et_ee" "da_dk" "lt_lt" "el_gr" "mt_mt" "lv_lv" "sl_si"
)

MODELS_DIR="$HOME/Library/Application Support/FluidAudio/Models"

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Verify Parakeet v3 models exist
verify_models() {
    local v3_dir="$MODELS_DIR/parakeet-tdt-0.6b-v3"
    for f in Preprocessor.mlmodelc Encoder.mlmodelc Decoder.mlmodelc JointDecision.mlmodelc parakeet_vocab.json; do
        if [[ ! -e "$v3_dir/$f" ]]; then
            log "MISSING  v3: $v3_dir/$f"
            return 1
        fi
    done
    return 0
}

log "=== Verifying Parakeet v3 models ==="
if ! verify_models; then
    log ""
    log "ERROR: Parakeet v3 models missing."
    log "Please run ASR benchmark first to download models."
    exit 1
fi
log "Parakeet v3 models verified. FLEURS data will download automatically if needed."

log "=== FLEURS full benchmark: $SAMPLES_PER_LANG samples x ${#LANGUAGES[@]} languages = $(( SAMPLES_PER_LANG * ${#LANGUAGES[@]} )) total ==="
log "Results directory: $RESULTS_DIR"

cd "$PROJECT_DIR"

# Build release if not already built
if [[ ! -x ".build/release/fluidaudiocli" ]]; then
    log "Building release binary..."
    swift build -c release 2>&1 | tail -1 | tee -a "$LOG_FILE"
fi
CLI="$PROJECT_DIR/.build/release/fluidaudiocli"

# caffeinate -s: prevent sleep even on AC power / lid closed
# caffeinate -i: prevent idle sleep
caffeinate -si -w $$ &
CAFFEINATE_PID=$!
log "caffeinate started (PID $CAFFEINATE_PID) — safe to close the lid"

SUITE_START=$(date +%s)

# Run all languages
LANG_NAMES=(
    "English (US)" "Spanish (Spain)" "Italian (Italy)" "French (France)" "German (Germany)"
    "Russian (Russia)" "Dutch (Netherlands)" "Polish (Poland)" "Ukrainian (Ukraine)" "Slovak (Slovakia)"
    "Czech (Czechia)" "Bulgarian (Bulgaria)" "Croatian (Croatia)" "Romanian (Romania)" "Finnish (Finland)"
    "Hungarian (Hungary)" "Swedish (Sweden)" "Estonian (Estonia)" "Danish (Denmark)" "Lithuanian (Lithuania)"
    "Greek (Greece)" "Maltese (Malta)" "Latvian (Latvia)" "Slovenian (Slovenia)"
)

for i in "${!LANGUAGES[@]}"; do
    lang="${LANGUAGES[$i]}"
    name="${LANG_NAMES[$i]}"
    label="fleurs_${lang}"
    output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- [$((i+1))/${#LANGUAGES[@]}] $name ($lang): starting ($SAMPLES_PER_LANG samples) ---"
    start_time=$(date +%s)

    "$CLI" fleurs-benchmark \
        --languages "$lang" \
        --samples "$SAMPLES_PER_LANG" \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"

    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    log "--- $name: finished in ${elapsed}s — $output_file ---"
done

SUITE_END=$(date +%s)
SUITE_ELAPSED=$(( SUITE_END - SUITE_START ))
SUITE_HOURS=$(( SUITE_ELAPSED / 3600 ))
SUITE_MINS=$(( (SUITE_ELAPSED % 3600) / 60 ))
SUITE_SECS=$(( SUITE_ELAPSED % 60 ))

log "=== All benchmarks complete in ${SUITE_HOURS}h ${SUITE_MINS}m ${SUITE_SECS}s ==="
log "Results:"
ls -lh "$RESULTS_DIR"/*_${TIMESTAMP}.json 2>/dev/null | tee -a "$LOG_FILE"

# Extract WER from all results
log ""
log "=== WER Summary (100 samples per language) ==="
log ""
printf "%-30s %10s %10s %10s\n" "Language" "WER%" "CER%" "RTFx" | tee -a "$LOG_FILE"
printf "%-30s %10s %10s %10s\n" "------------------------------" "----------" "----------" "----------" | tee -a "$LOG_FILE"

extract_metrics() {
    local json_file="$1"
    if [[ -f "$json_file" ]]; then
        python3 -c "
import json
d = json.load(open('$json_file'))
wer = round(d['summary']['averageWER']*100, 2)
cer = round(d['summary']['averageCER']*100, 2)
rtfx = round(d['summary']['averageRTFx'], 1)
print(f'{wer}\t{cer}\t{rtfx}')
" 2>/dev/null || echo "N/A\tN/A\tN/A"
    else
        echo "N/A\tN/A\tN/A"
    fi
}

for i in "${!LANGUAGES[@]}"; do
    lang="${LANGUAGES[$i]}"
    name="${LANG_NAMES[$i]}"
    json_file="$RESULTS_DIR/fleurs_${lang}_${TIMESTAMP}.json"

    metrics=$(extract_metrics "$json_file")
    wer=$(echo "$metrics" | cut -f1)
    cer=$(echo "$metrics" | cut -f2)
    rtfx=$(echo "$metrics" | cut -f3)

    printf "%-30s %9s%% %9s%% %9sx\n" "$name ($lang)" "$wer" "$cer" "$rtfx" | tee -a "$LOG_FILE"
done

log ""
log "✅ Full FLEURS benchmark complete"
log "Total samples processed: $(( SAMPLES_PER_LANG * ${#LANGUAGES[@]} ))"
log "Results saved to: $RESULTS_DIR/*_${TIMESTAMP}.json"

# caffeinate will exit automatically since the parent process ($$) exits
