# FluidAudio Scripts

Python utility scripts for FluidAudio development and dataset management.

## download_fleurs.py

Downloads FLEURS multilingual speech dataset from HuggingFace and organizes it for FluidAudio benchmarks.

### Usage

**Via FluidAudio CLI (Recommended):**
```bash
# Download all 14 Cohere-supported languages (100 samples each)
swift run fluidaudiocli download --dataset fleurs

# Then benchmark:
swift run fluidaudiocli cohere-benchmark --dataset fleurs --languages en_us,ja_jp,fr_fr
```

**Direct Python Usage:**
```bash
# Download specific languages
python3 Scripts/download_fleurs.py --languages en_us,ja_jp,fr_fr --samples 100

# Download all Cohere-supported languages
python3 Scripts/download_fleurs.py --all --samples 500

# Custom output directory
python3 Scripts/download_fleurs.py --languages en_us --samples 50 --output-dir ~/my_datasets
```

### Requirements

```bash
pip install datasets soundfile numpy
```

### Supported Languages

14 languages supported by Cohere Transcribe:
- English (en_us)
- French (fr_fr)
- German (de_de)
- Spanish (es_419)
- Italian (it_it)
- Portuguese (pt_br)
- Dutch (nl_nl)
- Polish (pl_pl)
- Greek (el_gr)
- Arabic (ar_eg)
- Japanese (ja_jp)
- Chinese Mandarin (cmn_hans_cn)
- Korean (ko_kr)
- Vietnamese (vi_vn)

### Output Structure

Files are saved to `~/Library/Application Support/FluidAudio/Datasets/fleurs/`:

```
fleurs/
├── en_us/
│   ├── en_us.trans.txt      # Format: file_id transcription
│   ├── sample_001.wav       # 16kHz mono WAV
│   ├── sample_002.wav
│   └── ...
├── ja_jp/
│   ├── ja_jp.trans.txt
│   ├── sample_001.wav
│   └── ...
└── ...
```

### Dataset Source

Uses `google/fleurs` dataset from HuggingFace, which provides:
- ~12,000+ samples across 30+ languages
- Professional transcriptions
- 16kHz audio quality
- CC-BY-4.0 License

Reference: https://huggingface.co/datasets/google/fleurs
