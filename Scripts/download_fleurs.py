#!/usr/bin/env python3
"""Download FLEURS dataset from HuggingFace and organize for FluidAudio.

Downloads audio files from google/fleurs dataset and organizes them into the structure
expected by FluidAudio's Cohere benchmark:

    ~/Library/Application Support/FluidAudio/Datasets/fleurs/
    ├── en_us/
    │   ├── en_us.trans.txt
    │   ├── sample_001.wav
    │   └── ...
    ├── ja_jp/
    │   └── ...
    └── ...

Usage:
    python3 download_fleurs.py --languages en_us,ja_jp,fr_fr --samples 100
    python3 download_fleurs.py --all --samples 500
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
import numpy as np

# All FLEURS languages supported by Cohere (14 total)
COHERE_LANGUAGES = [
    "en_us",
    "fr_fr",
    "de_de",
    "es_419",
    "it_it",
    "pt_br",
    "nl_nl",
    "pl_pl",
    "el_gr",
    "ar_eg",
    "ja_jp",
    "cmn_hans_cn",
    "ko_kr",
    "vi_vn",
]


def download_fleurs_language(language: str, output_dir: Path, max_samples: int = None, split: str = "train"):
    """Download FLEURS data for a specific language."""
    print(f"\n[{language}] Loading from google/fleurs...")

    try:
        # Load dataset with trust_remote_code for google/fleurs
        ds = load_dataset("google/fleurs", language, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load {language}: {e}")
        return 0

    # Create language directory
    lang_dir = output_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Prepare transcript file
    trans_file = lang_dir / f"{language}.trans.txt"
    trans_lines = []

    downloaded = 0
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # Extract sample data
        audio_array = np.array(sample['audio']['array'], dtype=np.float32)
        sample_rate = sample['audio']['sampling_rate']
        transcription = sample['transcription']
        sample_id = sample['id']

        # Create filename from sample ID
        audio_filename = f"{sample_id}.wav"
        audio_path = lang_dir / audio_filename

        # Write audio file (convert to 16kHz mono WAV)
        sf.write(audio_path, audio_array, sample_rate)

        # Add to transcript file (format: file_id transcription)
        trans_lines.append(f"{sample_id} {transcription}")

        downloaded += 1
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i + 1} samples...")

    # Write transcript file
    with open(trans_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(trans_lines))

    print(f"✅ {language}: {downloaded} samples downloaded to {lang_dir}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download FLEURS dataset for FluidAudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--languages", "-l",
        type=str,
        help="Comma-separated language codes (e.g., en_us,ja_jp,fr_fr)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 14 Cohere-supported languages"
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Maximum samples per language (default: 100)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to download (default: train)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory (default: ~/Library/Application Support/FluidAudio/Datasets/fleurs)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path.home() / "Library" / "Application Support" / "FluidAudio" / "Datasets" / "fleurs"

    # Determine languages to download
    if args.all:
        languages = COHERE_LANGUAGES
    elif args.languages:
        languages = [lang.strip() for lang in args.languages.split(",")]
    else:
        print("❌ Error: Must specify --languages or --all")
        parser.print_help()
        exit(1)

    # Validate languages
    invalid = [lang for lang in languages if lang not in COHERE_LANGUAGES]
    if invalid:
        print(f"❌ Invalid language codes: {', '.join(invalid)}")
        print(f"Valid codes: {', '.join(COHERE_LANGUAGES)}")
        exit(1)

    print("="*70)
    print(f"FLEURS Download")
    print("="*70)
    print(f"Languages: {', '.join(languages)}")
    print(f"Samples per language: {args.samples}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Download each language
    total_downloaded = 0
    for i, lang in enumerate(languages, 1):
        print(f"\n[{i}/{len(languages)}] Downloading {lang}...")
        count = download_fleurs_language(lang, output_dir, args.samples, args.split)
        total_downloaded += count

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Total languages: {len(languages)}")
    print(f"Total samples: {total_downloaded}")
    print(f"Location: {output_dir}")
    print("\nYou can now use these datasets with:")
    print("  swift run fluidaudiocli cohere-benchmark --dataset fleurs --languages en_us,ja_jp,...")
    print("="*70)


if __name__ == "__main__":
    main()
