# FluidAudio Documentation

## Quick Start

- [ASR Getting Started](ASR/GettingStarted.md) — Speech recognition basics
- [Diarization Getting Started](Diarization/GettingStarted.md) — Speaker identification
- [VAD Getting Started](VAD/GettingStarted.md) — Voice activity detection

## Reference

- [Models Overview](Models.md) — Available models and when to use each
- [API Reference](API.md) — Swift API documentation
- [CLI Reference](CLI.md) — Command-line interface
- [Benchmarks](Benchmarks.md) — Performance metrics

## Automatic Speech Recognition (ASR)

### Getting Started
- [ASR Getting Started](ASR/GettingStarted.md)
- [Directory Structure](ASR/DirectoryStructure.md)

### Models
- [TDT-CTC 110M](ASR/TDT-CTC-110M.md)
- [Nemotron](ASR/Nemotron.md)
- [Qwen3 ASR](ASR/Qwen3-ASR.md)
- [CTC Decoder Guide](CtcDecoderGuide.md)

### Customization
- [Custom Vocabulary](ASR/CustomVocabulary.md)
- [Custom Pronunciation](ASR/CustomPronunciation.md)
- [Post-Processing (ITN)](ASR/PostProcessing.md)

### Advanced
- [Last Chunk Handling](ASR/LastChunkHandling.md)
- [Manual Model Loading](ASR/ManualModelLoading.md)
- [Parakeet Benchmarks](ASR/benchmarks100.md)

## Diarization

### Getting Started
- [Diarization Getting Started](Diarization/GettingStarted.md)
- [Speaker Manager](Diarization/SpeakerManager.md)
- [DiarizerTimeline](Diarization/DiarizerTimeline.md)

### Models
- [LS-EEND](Diarization/LS-EEND.md)
- [Sortformer](Diarization/Sortformer.md)

### Evaluation
- [AMI Subset Benchmark](Diarization/BenchmarkAMISubset.md)
- [Investigation Report](Diarization/InvestigationReport.md)

## Voice Activity Detection (VAD)

- [VAD Getting Started](VAD/GettingStarted.md)
- [Segmentation](VAD/Segmentation.md)

## Text-to-Speech (TTS)

### Models
- [Kokoro](TTS/Kokoro.md)
- [PocketTTS](TTS/PocketTTS.md)

### Configuration
- [SSML](TTS/SSML.md)
- [Voice Quality Comparison](TTS/voice-quality.md)

## Developer Guides

- [Audio Conversion](Guides/AudioConversion.md)
- [Model Conversion](ModelConversion.md)
