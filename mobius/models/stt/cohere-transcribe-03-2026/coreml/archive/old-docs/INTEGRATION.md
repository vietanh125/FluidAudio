# Integration Guide: Cohere Transcribe with KV Cache

This guide shows how to integrate the KV cache decoder into FluidAudio.

## Architecture

```
FluidAudio/Sources/FluidAudio/ASR/CohereTranscribe/
├── CohereAsrManager.swift          # Main ASR manager (similar to Qwen3AsrManager)
├── CohereDecoder.swift             # KV cache decoder wrapper
├── CohereCrossCacheComputer.swift  # Cross-cache computation
└── CohereModels.swift              # Model loading and configuration
```

## Step 1: Model Loading

```swift
// CohereModels.swift
actor CohereModels {
    let preprocessor: MLModel
    let encoder: MLModel
    let crossCacheComputer: MLModel
    let decoder: MLModel
    let lmHead: MLModel

    init(modelDir: URL) async throws {
        preprocessor = try await Self.loadModel(at: modelDir.appendingPathComponent("preprocessor.mlpackage"))
        encoder = try await Self.loadModel(at: modelDir.appendingPathComponent("encoder.mlpackage"))
        crossCacheComputer = try await Self.loadModel(at: modelDir.appendingPathComponent("cross_cache_computer.mlpackage"))
        decoder = try await Self.loadModel(at: modelDir.appendingPathComponent("decoder_with_cache.mlpackage"))
        lmHead = try await Self.loadModel(at: modelDir.appendingPathComponent("lm_head.mlpackage"))
    }

    private static func loadModel(at url: URL) async throws -> MLModel {
        let compiledURL = try MLModel.compileModel(at: url)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        return try MLModel(contentsOf: compiledURL, configuration: config)
    }
}
```

## Step 2: KV Cache Management

```swift
// CohereDecoder.swift
actor CohereDecoder {
    private let decoder: MLModel
    private let lmHead: MLModel

    // Cache state
    private var selfCaches: [String: MLMultiArray]
    private var crossCaches: [String: MLMultiArray]
    private var position: MLMultiArray

    init(decoder: MLModel, lmHead: MLModel) {
        self.decoder = decoder
        self.lmHead = lmHead
        self.selfCaches = [:]
        self.crossCaches = [:]
        self.position = try! MLMultiArray(shape: [1], dataType: .float32)
    }

    func reset() {
        // Reset self-attention caches
        for i in 0..<8 {
            selfCaches["self_cache\(i)"] = try! MLMultiArray(shape: [2, 1, 512, 8, 128], dataType: .float32)
        }
        position[0] = 0.0
    }

    func setCrossCaches(_ caches: [String: MLMultiArray]) {
        self.crossCaches = caches
    }

    func decode(
        tokenId: Int,
        encoderHidden: MLMultiArray
    ) async throws -> Int {
        // Prepare input
        let inputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIds[[0, 0] as [NSNumber]] = NSNumber(value: tokenId)

        var features: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "position": MLFeatureValue(multiArray: position),
            "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden)
        ]

        // Add caches
        for (name, cache) in selfCaches {
            features[name] = MLFeatureValue(multiArray: cache)
        }
        for (name, cache) in crossCaches {
            features[name] = MLFeatureValue(multiArray: cache)
        }

        let decInput = try MLDictionaryFeatureProvider(dictionary: features)

        // Run decoder
        let decOutput = try decoder.prediction(from: decInput)

        // Extract hidden states
        guard let hiddenStates = decOutput.featureValue(for: "var_2384")?.multiArrayValue else {
            throw ASRError.decoderFailed
        }

        // Update position
        guard let newPosition = decOutput.featureValue(for: "var_2387")?.multiArrayValue else {
            throw ASRError.decoderFailed
        }
        position = newPosition

        // Update self-attention caches
        let cacheNames = [
            "new_self_cache_1_internal_tensor_assign_2",
            "new_self_cache_3_internal_tensor_assign_2",
            "new_self_cache_5_internal_tensor_assign_2",
            "new_self_cache_7_internal_tensor_assign_2",
            "new_self_cache_9_internal_tensor_assign_2",
            "new_self_cache_11_internal_tensor_assign_2",
            "new_self_cache_13_internal_tensor_assign_2",
            "new_self_cache_internal_tensor_assign_2"
        ]

        for (i, cacheName) in cacheNames.enumerated() {
            if let cache = decOutput.featureValue(for: cacheName)?.multiArrayValue {
                selfCaches["self_cache\(i)"] = cache
            }
        }

        // Get next token via LM head
        let lmInput = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenStates)
        ])

        let lmOutput = try lmHead.prediction(from: lmInput)
        guard let logits = lmOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.lmHeadFailed
        }

        return argmax(logits)
    }

    private func argmax(_ array: MLMultiArray) -> Int {
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        var maxIdx = 0
        var maxVal = ptr[0]
        for i in 1..<array.count {
            if ptr[i] > maxVal {
                maxVal = ptr[i]
                maxIdx = i
            }
        }
        return maxIdx
    }
}
```

## Step 3: Main ASR Manager

```swift
// CohereAsrManager.swift
public actor CohereAsrManager {
    private let models: CohereModels
    private let decoder: CohereDecoder
    private let vocabulary: [Int: String]

    // Prefix tokens (similar to Qwen3)
    private let PREFIX = [13764, 4, 16, 62, 6, 9, 11, 13]
    private let EOS_TOKEN = 3

    public init(modelDir: URL, vocabPath: URL) async throws {
        models = try await CohereModels(modelDir: modelDir)
        decoder = CohereDecoder(decoder: models.decoder, lmHead: models.lmHead)

        // Load vocabulary
        let data = try Data(contentsOf: vocabPath)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Int]
        var vocab: [Int: String] = [:]
        for (token, id) in json {
            vocab[id] = token
        }
        self.vocabulary = vocab
    }

    public func transcribe(audio: [Float]) async throws -> String {
        // 1. Preprocess
        let melFeatures = try await preprocess(audio: audio)

        // 2. Encode
        let encoderHidden = try await encode(melFeatures: melFeatures)

        // 3. Compute cross-attention caches
        let crossCaches = try await computeCrossCaches(encoderHidden: encoderHidden)

        // 4. Reset decoder state
        decoder.reset()
        decoder.setCrossCaches(crossCaches)

        // 5. Generate tokens
        var tokens: [Int] = []

        // Process prefix
        for tokenId in PREFIX {
            _ = try await decoder.decode(tokenId: tokenId, encoderHidden: encoderHidden)
        }

        // Generate up to maxTokens
        let maxTokens = 100
        for _ in 0..<maxTokens {
            let lastToken = tokens.last ?? PREFIX.last!
            let nextToken = try await decoder.decode(tokenId: lastToken, encoderHidden: encoderHidden)

            if nextToken == EOS_TOKEN {
                break
            }

            tokens.append(nextToken)
        }

        // 6. Decode to text
        return decodeTokens(tokens)
    }

    private func preprocess(audio: [Float]) async throws -> MLMultiArray {
        // Pad to 30s
        var paddedAudio = audio
        let targetLength = 480000
        if audio.count < targetLength {
            paddedAudio += [Float](repeating: 0, count: targetLength - audio.count)
        }

        let audioArray = try MLMultiArray(shape: [1, paddedAudio.count], dataType: .float32)
        for (i, sample) in paddedAudio.enumerated() {
            audioArray[[0, i] as [NSNumber]] = NSNumber(value: sample)
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: paddedAudio.count)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "length": MLFeatureValue(multiArray: lengthArray)
        ])

        let output = try models.preprocessor.prediction(from: input)

        // Handle output names
        if let mf = output.featureValue(for: "mel_features")?.multiArrayValue {
            return mf
        } else {
            let names = Array(output.featureNames).sorted()
            return output.featureValue(for: names[0])!.multiArrayValue!
        }
    }

    private func encode(melFeatures: MLMultiArray) async throws -> MLMultiArray {
        let melLength = try MLMultiArray(shape: [1], dataType: .int32)
        melLength[0] = NSNumber(value: melFeatures.shape[2].intValue)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: melFeatures),
            "length": MLFeatureValue(multiArray: melLength)
        ])

        let output = try models.encoder.prediction(from: input)
        return output.featureValue(for: "encoder_output")!.multiArrayValue!
    }

    private func computeCrossCaches(encoderHidden: MLMultiArray) async throws -> [String: MLMultiArray] {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden)
        ])

        let output = try models.crossCacheComputer.prediction(from: input)

        let cacheNames = ["var_76", "var_93", "var_110", "var_127", "var_144", "var_161", "var_178", "var_195"]
        var caches: [String: MLMultiArray] = [:]

        for (i, name) in cacheNames.enumerated() {
            caches["cross_cache\(i)"] = output.featureValue(for: name)!.multiArrayValue!
        }

        return caches
    }

    private func decodeTokens(_ tokens: [Int]) -> String {
        var text = tokens.compactMap { vocabulary[$0] }.joined()

        // Clean up special tokens
        text = text
            .replacingOccurrences(of: "<|startoftranscript|>", with: "")
            .replacingOccurrences(of: "<|endoftext|>", with: "")
            .replacingOccurrences(of: "<|emo:undefined|>", with: "")
            .replacingOccurrences(of: "<|en|>", with: "")
            .replacingOccurrences(of: "<|nopnc|>", with: "")
            .replacingOccurrences(of: "<|noitn|>", with: "")
            .replacingOccurrences(of: "<|notimestamp|>", with: "")
            .replacingOccurrences(of: "<|nodiarize|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return text
    }
}
```

## Step 4: CLI Integration

```swift
// FluidAudioCLI/Commands/CohereTranscribe.swift
struct CohereTranscribeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "cohere-transcribe",
        abstract: "Transcribe audio using Cohere Transcribe with KV cache"
    )

    @Argument(help: "Path to audio file")
    var audioPath: String

    @Option(help: "Model directory")
    var modelDir: String = "~/.fluidaudio/models/cohere-transcribe-03-2026"

    func run() async throws {
        let audio = try loadAudio(from: audioPath)

        let modelURL = URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath)
        let vocabURL = modelURL.appendingPathComponent("vocab.json")

        let manager = try await CohereAsrManager(modelDir: modelURL, vocabPath: vocabURL)
        let transcript = try await manager.transcribe(audio: audio)

        print(transcript)
    }
}
```

## Step 5: Testing

```swift
// Tests/FluidAudioTests/CohereAsrTests.swift
final class CohereAsrTests: XCTestCase {
    func testTranscription() async throws {
        let modelDir = URL(fileURLWithPath: "/path/to/models")
        let vocabPath = modelDir.appendingPathComponent("vocab.json")

        let manager = try await CohereAsrManager(modelDir: modelDir, vocabPath: vocabPath)

        // Load test audio
        let audio = try loadTestAudio()

        // Transcribe
        let result = try await manager.transcribe(audio: audio)

        // Verify
        XCTAssertFalse(result.isEmpty)
        XCTAssertFalse(result.contains("çon")) // No garbage output
    }
}
```

## Performance Considerations

1. **Model Loading**: Compile once, cache compiled models
2. **Cache Size**: 8 × (2 × 512 × 8 × 128 × 4 bytes) ≈ 4 MB per inference
3. **Batch Size**: Keep B=1 for streaming use cases
4. **Quantization**: Consider INT8 quantization for models (not caches)

## Error Handling

```swift
enum CohereAsrError: Error, LocalizedError {
    case modelLoadFailed(String)
    case preprocessingFailed
    case encodingFailed
    case decodingFailed
    case lmHeadFailed
    case invalidAudio

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let model):
            return "Failed to load model: \(model)"
        case .preprocessingFailed:
            return "Audio preprocessing failed"
        case .encodingFailed:
            return "Encoder failed"
        case .decodingFailed:
            return "Decoder failed"
        case .lmHeadFailed:
            return "LM head failed"
        case .invalidAudio:
            return "Invalid audio input"
        }
    }
}
```

## Model Download

Follow FluidAudio's existing model download pattern:

```swift
// Model registry entry
ModelRegistry.register(
    id: "cohere-transcribe-03-2026",
    repoId: "FluidInference/cohere-transcribe-03-2026-coreml",
    files: [
        "preprocessor.mlpackage",
        "encoder.mlpackage",
        "cross_cache_computer.mlpackage",
        "decoder_with_cache.mlpackage",
        "lm_head.mlpackage",
        "vocab.json"
    ]
)
```

## Next Steps

1. Resolve Swift preprocessing issue (see KV_CACHE_STATUS.md)
2. Add Cohere ASR to FluidAudio module
3. Profile performance and optimize
4. Add streaming support (chunk audio into 30s segments)
5. Add language detection support
