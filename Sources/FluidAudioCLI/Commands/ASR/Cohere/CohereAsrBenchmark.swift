#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Benchmark for Cohere Transcribe 03-2026 supporting FLEURS (14 languages).
///
/// Runs inference through `CohereAsrManager` with WER/CER evaluation.
enum CohereAsrBenchmark {
    private static let logger = AppLogger(category: "CohereBenchmark")

    /// Map FLEURS language codes to CohereAsrConfig.Language (14 languages).
    private static let fleursToCohere: [String: CohereAsrConfig.Language] = [
        "en_us": .english,
        "fr_fr": .french,
        "de_de": .german,
        "it_it": .italian,
        "es_419": .spanish,
        "pt_br": .portuguese,
        "el_gr": .greek,
        "nl_nl": .dutch,
        "pl_pl": .polish,
        "cmn_hans_cn": .chinese,
        "ja_jp": .japanese,
        "ko_kr": .korean,
        "vi_vn": .vietnamese,
        "ar_eg": .arabic,
    ]

    static func runCLI(arguments: [String]) async {
        var maxFiles: Int? = nil
        var modelDir: String? = nil
        var outputFile = "cohere_asr_benchmark_results.json"
        var languages: [String] = ["en_us"]
        var fleursDir: String? = nil

        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--languages":
                if i + 1 < arguments.count {
                    languages = arguments[i + 1].components(separatedBy: ",").map {
                        $0.trimmingCharacters(in: .whitespaces)
                    }
                    i += 1
                }
            case "--fleurs-dir":
                if i + 1 < arguments.count {
                    fleursDir = arguments[i + 1]
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        logger.info("Cohere Transcribe 03-2026 Benchmark")
        logger.info("  Languages: \(languages.joined(separator: ", "))")
        logger.info("  Max files: \(maxFiles?.description ?? "all")")
        logger.info("  Model dir: \(modelDir ?? "auto-download")")
        logger.info("  Output: \(outputFile)")

        guard #available(macOS 15, iOS 18, *) else {
            logger.error("Cohere Transcribe requires macOS 15 or later")
            exit(1)
        }

        do {
            // 1. Load Cohere Transcribe models
            let manager = CohereAsrManager()
            if let dir = modelDir {
                logger.info("Loading models from \(dir)")
                try await manager.loadModels(from: URL(fileURLWithPath: dir))
            } else {
                logger.info("Downloading Cohere Transcribe models...")
                let cacheDir = try await CohereAsrModels.download()
                try await manager.loadModels(from: cacheDir)
            }

            // 2. Run FLEURS benchmark
            try await runFleursBenchmark(
                manager: manager,
                languages: languages,
                maxFiles: maxFiles,
                fleursDir: fleursDir,
                outputFile: outputFile
            )

        } catch {
            logger.error("Benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - FLEURS Benchmark

    @available(macOS 15, iOS 18, *)
    private static func runFleursBenchmark(
        manager: CohereAsrManager,
        languages: [String],
        maxFiles: Int?,
        fleursDir: String?,
        outputFile: String
    ) async throws {
        let baseFleursDir: URL
        if let dir = fleursDir {
            baseFleursDir = URL(fileURLWithPath: dir)
        } else {
            baseFleursDir =
                FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/FluidAudio/FLEURS")
        }

        for language in languages {
            let languageDir = baseFleursDir.appendingPathComponent(language)

            // Auto-download if not present
            if !FileManager.default.fileExists(atPath: languageDir.path) {
                logger.info("FLEURS data not found for \(language), downloading...")
                do {
                    try await downloadFLEURSLanguage(
                        language: language,
                        targetDir: languageDir,
                        maxFiles: maxFiles
                    )
                } catch {
                    logger.error("Failed to download FLEURS \(language): \(error.localizedDescription)")
                    continue
                }
            }

            let allFiles = try collectFLEURSFiles(language: language, directory: languageDir)
            let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
            logger.info("Collected \(files.count) files for FLEURS \(language)")

            let cohereLang = fleursToCohere[language]
            let results = try await runBenchmarkLoop(
                manager: manager,
                files: files.map { ($0.fileName, $0.audioPath, $0.transcript) },
                language: cohereLang
            )

            let langOutputFile: String
            if languages.count > 1 {
                let base = (outputFile as NSString).deletingPathExtension
                let ext = (outputFile as NSString).pathExtension
                langOutputFile = "\(base)_\(language).\(ext.isEmpty ? "json" : ext)"
            } else {
                langOutputFile = outputFile
            }

            let summary = CohereBenchmarkSummary(results: results)
            printSummary(summary: summary, datasetLabel: "FLEURS \(language)")
            try writeJSON(
                results: results,
                summary: summary,
                outputFile: langOutputFile,
                language: language
            )
        }
    }

    // MARK: - FLEURS Download

    /// Download FLEURS data for a language from HuggingFace.
    private static func downloadFLEURSLanguage(
        language: String,
        targetDir: URL,
        maxFiles: Int?
    ) async throws {
        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let datasetRepo = "FluidInference/fleurs-full"
        logger.info("Downloading from HuggingFace: \(datasetRepo)/\(language)...")

        // List files in the language directory
        let apiURL = try ModelRegistry.apiDatasets(datasetRepo, "tree/main/\(language)")
        let (listData, _) = try await DownloadUtils.fetchWithAuth(from: apiURL)

        guard let items = try JSONSerialization.jsonObject(with: listData) as? [[String: Any]] else {
            throw CohereAsrError.generationFailed("Could not parse file list from HuggingFace")
        }

        // Find transcript file and audio files
        var audioFiles: [String] = []
        let transFile = targetDir.appendingPathComponent("\(language).trans.txt")

        for item in items {
            guard let itemPath = item["path"] as? String,
                let itemType = item["type"] as? String,
                itemType == "file"
            else { continue }

            let fileName = URL(fileURLWithPath: itemPath).lastPathComponent

            if fileName == "\(language).trans.txt" {
                // Download transcript file
                let downloadURL = try ModelRegistry.resolveDataset(datasetRepo, itemPath)
                let transData = try await DownloadUtils.fetchHuggingFaceFile(
                    from: downloadURL,
                    description: "\(language) transcript"
                )
                try transData.write(to: transFile, options: .atomic)

                let transcriptContent = String(data: transData, encoding: .utf8) ?? ""
                let lines = transcriptContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
                logger.info("Downloaded \(lines.count) transcriptions")
            } else if fileName.hasSuffix(".wav") {
                audioFiles.append(itemPath)
            }
        }

        // Download audio files
        let maxDownload = maxFiles ?? audioFiles.count
        var downloadedCount = 0

        for audioPath in audioFiles.prefix(maxDownload) {
            let fileName = URL(fileURLWithPath: audioPath).lastPathComponent
            let audioFile = targetDir.appendingPathComponent(fileName)

            // Skip if already exists and valid
            if FileManager.default.fileExists(atPath: audioFile.path) {
                if isValidAudioFile(audioFile) {
                    downloadedCount += 1
                    continue
                }
                try? FileManager.default.removeItem(at: audioFile)
            }

            // Download audio file
            let downloadURL = try ModelRegistry.resolveDataset(datasetRepo, audioPath)
            let audioData = try await DownloadUtils.fetchHuggingFaceFile(
                from: downloadURL,
                description: "\(language)/\(fileName)"
            )
            try audioData.write(to: audioFile, options: .atomic)
            downloadedCount += 1

            if downloadedCount % 20 == 0 {
                logger.info("Downloaded \(downloadedCount)/\(maxDownload) audio files...")
            }
        }

        logger.info("Downloaded \(downloadedCount) audio files for \(language)")
    }

    private static func isValidAudioFile(_ url: URL) -> Bool {
        do {
            _ = try AVAudioFile(forReading: url)
            return true
        } catch {
            return false
        }
    }

    // MARK: - FLEURS File Collection

    private static func collectFLEURSFiles(
        language: String, directory: URL
    ) throws -> [BenchmarkAudioFile] {
        let transFile = directory.appendingPathComponent("\(language).trans.txt")
        guard FileManager.default.fileExists(atPath: transFile.path) else {
            throw CohereAsrError.generationFailed(
                "Transcript file not found: \(transFile.path)"
            )
        }

        let content = try String(contentsOf: transFile)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        var files: [BenchmarkAudioFile] = []

        for line in lines {
            // Format: file_id transcription
            guard let spaceIndex = line.firstIndex(of: " ") else { continue }
            let fileId = String(line[line.startIndex..<spaceIndex])
            let transcript = String(line[line.index(after: spaceIndex)...])

            // Try .wav first, then .flac
            let wavPath = directory.appendingPathComponent("\(fileId).wav")
            let flacPath = directory.appendingPathComponent("\(fileId).flac")

            let audioPath: URL
            if FileManager.default.fileExists(atPath: wavPath.path) {
                audioPath = wavPath
            } else if FileManager.default.fileExists(atPath: flacPath.path) {
                audioPath = flacPath
            } else {
                continue
            }

            files.append(
                BenchmarkAudioFile(
                    fileName: audioPath.lastPathComponent,
                    audioPath: audioPath,
                    transcript: transcript
                )
            )
        }

        return files
    }

    // MARK: - Shared Benchmark Loop

    @available(macOS 15, iOS 18, *)
    private static func runBenchmarkLoop(
        manager: CohereAsrManager,
        files: [(fileName: String, audioPath: URL, transcript: String)],
        language: CohereAsrConfig.Language?
    ) async throws -> [CohereBenchmarkResult] {
        var results: [CohereBenchmarkResult] = []
        let audioConverter = AudioConverter()

        for (index, file) in files.enumerated() {
            do {
                logger.info("[\(index + 1)/\(files.count)] \(file.fileName)")

                let samples = try audioConverter.resampleAudioFile(path: file.audioPath.path)
                let audioLength = Double(samples.count) / Double(CohereAsrConfig.sampleRate)

                let inferenceStart = CFAbsoluteTimeGetCurrent()
                let hypothesis = try await manager.transcribe(
                    audioSamples: samples,
                    language: language,
                    maxNewTokens: 512
                )
                let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart

                let metrics = WERCalculator.calculateWERAndCER(
                    hypothesis: hypothesis, reference: file.transcript
                )

                let result = CohereBenchmarkResult(
                    fileName: file.fileName,
                    hypothesis: hypothesis,
                    reference: file.transcript,
                    wer: metrics.wer,
                    cer: metrics.cer,
                    audioLength: audioLength,
                    processingTime: inferenceTime
                )
                results.append(result)

                let rtfx = audioLength / inferenceTime
                let werPct = metrics.wer * 100
                let cerPct = metrics.cer * 100
                logger.info(
                    "  WER: \(String(format: "%.1f", werPct))% | CER: \(String(format: "%.1f", cerPct))% | RTFx: \(String(format: "%.1f", rtfx))x | \(String(format: "%.2f", audioLength))s audio in \(String(format: "%.2f", inferenceTime))s"
                )
                if werPct > 50.0 {
                    logger.info("  REF: \(file.transcript)")
                    logger.info("  HYP: \(hypothesis)")
                }
            } catch {
                logger.error("Failed \(file.fileName): \(error)")
            }

            // Give system time to reclaim memory every 25 files.
            if (index + 1) % 25 == 0 {
                logger.info("Memory cleanup pause...")
                try? await Task.sleep(for: .seconds(1))
            }
        }

        return results
    }

    // MARK: - Summary & Output

    private static func printSummary(summary: CohereBenchmarkSummary, datasetLabel: String) {
        guard summary.filesProcessed > 0 else {
            logger.error("No results produced")
            return
        }

        print("")
        print("--- Cohere Transcribe 03-2026 Benchmark Results ---")
        print("   Dataset: \(datasetLabel)")
        print("   Files processed: \(summary.filesProcessed)")
        print("   Average WER: \(String(format: "%.1f", summary.avgWER * 100))%")
        print("   Median WER: \(String(format: "%.1f", summary.medianWER * 100))%")
        print("   Average CER: \(String(format: "%.1f", summary.avgCER * 100))%")
        print("   Median CER: \(String(format: "%.1f", summary.medianCER * 100))%")
        print("   Median RTFx: \(String(format: "%.1f", summary.medianRTFx))x")
        print(
            "   Overall RTFx: \(String(format: "%.1f", summary.overallRTFx))x (\(String(format: "%.1f", summary.totalAudio))s / \(String(format: "%.1f", summary.totalInference))s)"
        )
    }

    private static func writeJSON(
        results: [CohereBenchmarkResult],
        summary: CohereBenchmarkSummary,
        outputFile: String,
        language: String
    ) throws {
        guard !results.isEmpty else { return }

        let summaryDict: [String: Any] = [
            "model": "cohere-transcribe-03-2026",
            "dataset": "fleurs",
            "language": language,
            "filesProcessed": summary.filesProcessed,
            "averageWER": summary.avgWER,
            "medianWER": summary.medianWER,
            "averageCER": summary.avgCER,
            "medianCER": summary.medianCER,
            "medianRTFx": summary.medianRTFx,
            "overallRTFx": summary.overallRTFx,
            "totalAudioDuration": summary.totalAudio,
            "totalInferenceTime": summary.totalInference,
        ]

        let jsonResults = results.map { r -> [String: Any] in
            [
                "fileName": r.fileName,
                "hypothesis": r.hypothesis,
                "reference": r.reference,
                "wer": r.wer,
                "cer": r.cer,
                "audioLength": r.audioLength,
                "processingTime": r.processingTime,
                "rtfx": r.audioLength / r.processingTime,
            ]
        }

        let output: [String: Any] = [
            "summary": summaryDict,
            "results": jsonResults,
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputFile))
        logger.info("Results written to \(outputFile)")
    }

    // MARK: - Usage

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe 03-2026 Benchmark

            Usage: fluidaudio cohere-benchmark [options]

            Options:
                --languages <list>      FLEURS language codes, comma-separated (default: en_us)
                --max-files <number>    Max files to process (default: all)
                --model-dir <path>      Local model directory (skips download)
                --fleurs-dir <path>     FLEURS data directory (default: ~/Library/Application Support/FluidAudio/FLEURS)
                --output <file>         Output JSON path (default: cohere_asr_benchmark_results.json)
                --help, -h              Show this help

            Examples:
                # English
                fluidaudio cohere-benchmark --languages en_us --max-files 100

                # Multiple languages
                fluidaudio cohere-benchmark --languages en_us,fr_fr,de_de,zh_cn

                # All 14 Cohere languages
                fluidaudio cohere-benchmark --languages en_us,fr_fr,de_de,it_it,es_419,pt_br,el_gr,nl_nl,pl_pl,cmn_hans_cn,ja_jp,ko_kr,vi_vn,ar_eg

            Supported FLEURS languages (14 total):

            Western (10 languages):
                en_us   English         fr_fr   French          de_de   German
                it_it   Italian         es_419  Spanish         pt_br   Portuguese
                el_gr   Greek           nl_nl   Dutch           pl_pl   Polish
                ar_eg   Arabic

            Asian (4 languages):
                cmn_hans_cn  Chinese (Mandarin)     ja_jp   Japanese
                ko_kr        Korean                 vi_vn   Vietnamese

            All languages auto-download from FluidInference/fleurs-full.
            """
        )
    }
}

// MARK: - Types

private struct CohereBenchmarkResult {
    let fileName: String
    let hypothesis: String
    let reference: String
    let wer: Double
    let cer: Double
    let audioLength: Double
    let processingTime: Double
}

private struct BenchmarkAudioFile {
    let fileName: String
    let audioPath: URL
    let transcript: String
}

private struct CohereBenchmarkSummary {
    let filesProcessed: Int
    let avgWER: Double
    let medianWER: Double
    let avgCER: Double
    let medianCER: Double
    let medianRTFx: Double
    let overallRTFx: Double
    let totalAudio: Double
    let totalInference: Double

    init(results: [CohereBenchmarkResult]) {
        guard !results.isEmpty else {
            self.filesProcessed = 0
            self.avgWER = 0
            self.medianWER = 0
            self.avgCER = 0
            self.medianCER = 0
            self.medianRTFx = 0
            self.overallRTFx = 0
            self.totalAudio = 0
            self.totalInference = 0
            return
        }

        self.filesProcessed = results.count
        self.avgWER = results.map(\.wer).reduce(0, +) / Double(results.count)
        self.avgCER = results.map(\.cer).reduce(0, +) / Double(results.count)
        self.totalAudio = results.map(\.audioLength).reduce(0, +)
        self.totalInference = results.map(\.processingTime).reduce(0, +)
        self.overallRTFx = totalAudio / totalInference
        self.medianWER = results.map(\.wer).sorted()[results.count / 2]
        self.medianCER = results.map(\.cer).sorted()[results.count / 2]

        let sortedRTFx = results.map { $0.audioLength / $0.processingTime }.sorted()
        self.medianRTFx = sortedRTFx[sortedRTFx.count / 2]
    }
}
#endif
