#if os(macOS)
import FluidAudio
import Foundation

/// Command to transcribe audio files using Cohere Transcribe 03-2026.
enum CohereTranscribeCommand {
    private static let logger = AppLogger(category: "CohereTranscribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var modelDir: String?
        var language: CohereAsrConfig.Language?

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--language", "-l":
                if i + 1 < arguments.count {
                    let langStr = arguments[i + 1]
                    if let lang = CohereAsrConfig.Language(from: langStr) {
                        language = lang
                    } else {
                        logger.warning(
                            "Unknown language '\(langStr)'. Use --help to see supported languages."
                        )
                    }
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        await transcribe(audioFile: audioFile, modelDir: modelDir, language: language)
    }

    private static func transcribe(
        audioFile: String,
        modelDir: String?,
        language: CohereAsrConfig.Language?
    ) async {
        guard #available(macOS 15, iOS 18, *) else {
            logger.error("Cohere Transcribe requires macOS 15 or later")
            return
        }

        do {
            // Load models
            let manager = CohereAsrManager()

            if let dir = modelDir {
                logger.info("Loading Cohere Transcribe models from: \(dir)")
                let dirURL = URL(fileURLWithPath: dir)
                try await manager.loadModels(from: dirURL)
            } else {
                logger.info("Downloading Cohere Transcribe models from HuggingFace...")
                let cacheDir = try await CohereAsrModels.download()
                try await manager.loadModels(from: cacheDir)
            }

            // Load and resample audio to 16kHz mono
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(samples.count) / Double(CohereAsrConfig.sampleRate)
            logger.info(
                "Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 16kHz"
            )

            // Transcribe
            let langDesc = language?.englishName ?? "auto-detect"
            logger.info("Transcribing (language: \(langDesc))...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let text = try await manager.transcribe(
                audioSamples: samples,
                language: language,
                maxNewTokens: 512
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            let rtfx = duration / elapsed

            // Output
            logger.info(String(repeating: "=", count: 50))
            logger.info("COHERE TRANSCRIBE 03-2026")
            logger.info(String(repeating: "=", count: 50))
            print(text)
            logger.info("")
            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", elapsed))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")

        } catch {
            logger.error("Cohere Transcribe failed: \(error)")
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe 03-2026 Command

            Usage: fluidaudio cohere-transcribe <audio_file> [options]

            Options:
                --help, -h              Show this help message
                --model-dir <path>      Path to local model directory (skips download)
                --language, -l <code>   Language hint (e.g., en, fr, de, zh, ja, ko, vi, ar)

            Supported languages (14 total):
                en   English             fr   French              de   German
                it   Italian             es   Spanish             pt   Portuguese
                el   Greek               nl   Dutch               pl   Polish
                zh   Chinese (Mandarin)  ja   Japanese            ko   Korean
                vi   Vietnamese          ar   Arabic

            Benchmark Performance (FLEURS dataset, 100 samples/language):
                Western languages: 3.8-9.3% WER average
                Asian languages: 0-7.3% CER average
                  - Chinese: ~0% CER (near-perfect)
                  - Vietnamese: 3.43% CER
                  - Korean: 3.48% CER
                  - Japanese: 7.25% CER

            Examples:
                fluidaudio cohere-transcribe audio.wav
                fluidaudio cohere-transcribe chinese.wav --language zh
                fluidaudio cohere-transcribe meeting.wav --model-dir /path/to/cohere-transcribe
            """
        )
    }
}
#endif
