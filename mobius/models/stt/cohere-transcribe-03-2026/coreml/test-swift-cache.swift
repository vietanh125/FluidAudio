#!/usr/bin/env swift
/**
 Test Cohere Transcribe with KV cache in pure Swift

 Usage:
   swift test-swift-cache.swift

 Requirements:
   - macOS 14.0+
   - CoreML models in build/hf-upload/ and build/
   - vocab.json, test-audio.wav in current dir
*/

import Foundation
import CoreML
import Accelerate
import AVFoundation

// MARK: - Helper Functions

func loadVocabulary() -> [Int: String]? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: "build/hf-upload/vocab.json")) else {
        print("❌ Could not load vocab.json")
        return nil
    }
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
        print("❌ Could not parse vocab.json")
        return nil
    }

    var vocab: [Int: String] = [:]
    for (token, id) in json {
        vocab[id] = token
    }
    return vocab
}

func loadAudioFile(path: String) -> (audio: [Float], sampleRate: Int)? {
    let url = URL(fileURLWithPath: path)
    guard let audioFile = try? AVAudioFile(forReading: url) else {
        print("❌ Could not load audio file")
        return nil
    }

    let format = audioFile.processingFormat
    let frameCount = UInt32(audioFile.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        print("❌ Could not create audio buffer")
        return nil
    }

    try? audioFile.read(into: buffer)
    let channelData = buffer.floatChannelData![0]
    let audio = Array(UnsafeBufferPointer<Float>(start: channelData, count: Int(frameCount)))

    return (audio, Int(format.sampleRate))
}

func createMLMultiArray(shape: [Int], dataType: MLMultiArrayDataType = .float32) -> MLMultiArray? {
    try? MLMultiArray(shape: shape as [NSNumber], dataType: dataType)
}

func zerosArray(shape: [Int]) -> MLMultiArray? {
    guard let array = createMLMultiArray(shape: shape) else { return nil }
    // MLMultiArray is already zero-initialized
    return array
}

func argmax(_ array: MLMultiArray) -> Int {
    let count = array.count
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    var maxVal = ptr[0]
    var maxIdx = 0

    for i in 0..<count {
        if ptr[i] > maxVal {
            maxVal = ptr[i]
            maxIdx = i
        }
    }
    return maxIdx
}

// MARK: - Main

print("=== Cohere Transcribe with KV Cache (Swift) ===\n")

// Load vocabulary
print("Loading vocabulary...")
guard let vocab = loadVocabulary() else {
    fatalError("Failed to load vocabulary")
}
print("  Loaded \(vocab.count) tokens")

// Load audio
print("\nLoading audio...")
guard let (audio, sampleRate) = loadAudioFile(path: "test-audio.wav") else {
    fatalError("Failed to load audio")
}
print("  Audio: \(audio.count) samples @ \(sampleRate) Hz")

// Pad to 30s if needed
var paddedAudio = audio
let targetLength = 480000
if audio.count < targetLength {
    paddedAudio = audio + [Float](repeating: 0, count: targetLength - audio.count)
}

// Compile and load models
print("\nCompiling and loading CoreML models...")
let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU

func compileAndLoad(_ path: String) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    print("  Compiling \(URL(fileURLWithPath: path).lastPathComponent)...")
    let compiledURL = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiledURL, configuration: config)
}

let preprocessor: MLModel
let encoder: MLModel
let crossCacheComputer: MLModel
let decoder: MLModel
let lmHead: MLModel

do {
    preprocessor = try compileAndLoad("build/hf-upload/preprocessor.mlpackage")
    encoder = try compileAndLoad("build/hf-upload/encoder.mlpackage")
    crossCacheComputer = try compileAndLoad("build/cross_cache_computer.mlpackage")
    decoder = try compileAndLoad("build/decoder_with_cache.mlpackage")
    lmHead = try compileAndLoad("build/hf-upload/lm_head.mlpackage")
    print("  ✓ All models loaded")
} catch {
    print("❌ Failed: \(error)")
    fatalError("Cannot continue")
}
print("  ✓ Loaded all models")

// Preprocess
print("\nPreprocessing audio...")
guard let audioArray = createMLMultiArray(shape: [1, paddedAudio.count]) else {
    fatalError("Failed to create audio array")
}
for (i, sample) in paddedAudio.enumerated() {
    audioArray[[0, i] as [NSNumber]] = NSNumber(value: sample)
}

guard let lengthArray = createMLMultiArray(shape: [1], dataType: .int32) else {
    fatalError("Failed to create length array")
}
lengthArray[0] = NSNumber(value: paddedAudio.count)

let prepInput = try! MLDictionaryFeatureProvider(dictionary: [
    "audio_signal": MLFeatureValue(multiArray: audioArray),
    "length": MLFeatureValue(multiArray: lengthArray)
])

guard let prepOutput = try? preprocessor.prediction(from: prepInput) else {
    fatalError("Preprocessing failed")
}

// Get mel features (handle both old and new output names)
let melFeatures: MLMultiArray
let melLength: MLMultiArray

if let mf = prepOutput.featureValue(for: "mel_features")?.multiArrayValue,
   let ml = prepOutput.featureValue(for: "mel_length")?.multiArrayValue {
    melFeatures = mf
    melLength = ml
} else {
    // Try auto-generated names
    let outputNames = Array(prepOutput.featureNames).sorted()
    guard let mf = prepOutput.featureValue(for: outputNames[0])?.multiArrayValue,
          let ml = prepOutput.featureValue(for: outputNames[1])?.multiArrayValue else {
        print("❌ Could not get preprocessor outputs. Available names: \(prepOutput.featureNames)")
        fatalError("Preprocessing failed")
    }
    melFeatures = mf
    melLength = ml
}
print("  Mel features: \(melFeatures.shape)")

// Encode
print("\nEncoding...")
let encInput = try! MLDictionaryFeatureProvider(dictionary: [
    "input_features": MLFeatureValue(multiArray: melFeatures),
    "length": MLFeatureValue(multiArray: melLength)
])

guard let encOutput = try? encoder.prediction(from: encInput),
      let encoderHidden = encOutput.featureValue(for: "encoder_output")?.multiArrayValue else {
    fatalError("Encoding failed")
}
print("  Encoder output: \(encoderHidden.shape)")

// Initialize caches
print("\nInitializing KV caches...")
var selfCaches: [String: MLMultiArray] = [:]
for i in 0..<8 {
    guard let cache = zerosArray(shape: [2, 1, 512, 8, 128]) else {
        fatalError("Failed to create self cache \(i)")
    }
    selfCaches["self_cache\(i)"] = cache
}

// Compute cross-attention caches from encoder output
print("Computing cross-attention caches from encoder output...")
let crossCacheInput = try! MLDictionaryFeatureProvider(dictionary: [
    "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden)
])

guard let crossCacheOutput = try? crossCacheComputer.prediction(from: crossCacheInput) else {
    fatalError("Failed to compute cross caches")
}

let crossCacheNames = ["var_76", "var_93", "var_110", "var_127", "var_144", "var_161", "var_178", "var_195"]
var crossCaches: [String: MLMultiArray] = [:]

for (i, outputName) in crossCacheNames.enumerated() {
    guard let cache = crossCacheOutput.featureValue(for: outputName)?.multiArrayValue else {
        fatalError("Failed to get cross cache \(i)")
    }
    crossCaches["cross_cache\(i)"] = cache
}

print("  ✓ Initialized 8 self-attention caches (zeros)")
print("  ✓ Computed 8 cross-attention caches (from THIS encoder output)")
print("  ✅ Using REAL cross-caches - should produce PERFECT transcription!")

// Generation
print("\nStarting generation...\n")

let PREFIX = [13764, 4, 16, 62, 6, 9, 11, 13]
var tokens: [Int] = []

guard var position = createMLMultiArray(shape: [1], dataType: .float32) else {
    fatalError("Failed to create position array")
}
position[0] = 0.0

let maxSteps = 100

for step in 0..<(PREFIX.count + maxSteps) {
    // Get current token
    let tokenId: Int
    if step < PREFIX.count {
        tokenId = PREFIX[step]
    } else {
        guard let lastToken = tokens.last else { break }
        tokenId = lastToken
    }

    // Prepare decoder input
    guard let inputIds = createMLMultiArray(shape: [1, 1], dataType: .int32) else {
        fatalError("Failed to create input_ids")
    }
    inputIds[[0, 0] as [NSNumber]] = NSNumber(value: tokenId)

    // Build feature dictionary
    var features: [String: MLFeatureValue] = [
        "input_ids": MLFeatureValue(multiArray: inputIds),
        "position": MLFeatureValue(multiArray: position),
        "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden)
    ]

    // Add all caches
    for (name, cache) in selfCaches {
        features[name] = MLFeatureValue(multiArray: cache)
    }
    for (name, cache) in crossCaches {
        features[name] = MLFeatureValue(multiArray: cache)
    }

    let decInput = try! MLDictionaryFeatureProvider(dictionary: features)

    // Run decoder
    guard let decOutput = try? decoder.prediction(from: decInput) else {
        print("❌ Decoder failed at step \(step)")
        break
    }

    // Extract outputs
    guard let hiddenStates = decOutput.featureValue(for: "var_2384")?.multiArrayValue,
          let newPosition = decOutput.featureValue(for: "var_2387")?.multiArrayValue else {
        print("❌ Failed to extract decoder outputs")
        break
    }

    // Update position
    position = newPosition

    // Update self-attention caches
    let cacheOutputNames = [
        "new_self_cache_1_internal_tensor_assign_2",
        "new_self_cache_3_internal_tensor_assign_2",
        "new_self_cache_5_internal_tensor_assign_2",
        "new_self_cache_7_internal_tensor_assign_2",
        "new_self_cache_9_internal_tensor_assign_2",
        "new_self_cache_11_internal_tensor_assign_2",
        "new_self_cache_13_internal_tensor_assign_2",
        "new_self_cache_internal_tensor_assign_2"
    ]

    for (i, cacheName) in cacheOutputNames.enumerated() {
        if let updatedCache = decOutput.featureValue(for: cacheName)?.multiArrayValue {
            selfCaches["self_cache\(i)"] = updatedCache
        }
    }

    // Get logits
    let lmInput = try! MLDictionaryFeatureProvider(dictionary: [
        "hidden_states": MLFeatureValue(multiArray: hiddenStates)
    ])

    guard let lmOutput = try? lmHead.prediction(from: lmInput),
          let logits = lmOutput.featureValue(for: "logits")?.multiArrayValue else {
        print("❌ LM head failed")
        break
    }

    // Get next token
    let nextToken = argmax(logits)

    // Print progress
    if step < PREFIX.count {
        if step < 3 {
            let tokenStr = vocab[tokenId] ?? "?"
            print("  Prefix \(step): token=\(tokenId) '\(tokenStr)'")
        }
    } else {
        if step == PREFIX.count {
            print("\nGeneration:")
        }

        let tokenStr = vocab[nextToken] ?? "?"
        if (step - PREFIX.count) < 15 {
            print("  Step \(step - PREFIX.count + 1): token=\(nextToken) '\(tokenStr)'")
        }

        if nextToken == 3 {  // EOS
            print("\n  → EOS at step \(step - PREFIX.count + 1)")
            break
        }
    }

    tokens.append(nextToken)
}

// Decode tokens
print("\nDecoding...")
var generatedText = ""
for token in tokens {
    if let tokenStr = vocab[token] {
        generatedText += tokenStr
    }
}

// Clean up special tokens
generatedText = generatedText
    .replacingOccurrences(of: "<|startoftranscript|>", with: "")
    .replacingOccurrences(of: "<|endoftext|>", with: "")
    .replacingOccurrences(of: "<|emo:undefined|>", with: "")
    .replacingOccurrences(of: "<|en|>", with: "")
    .replacingOccurrences(of: "<|nopnc|>", with: "")
    .replacingOccurrences(of: "<|noitn|>", with: "")
    .replacingOccurrences(of: "<|notimestamp|>", with: "")
    .replacingOccurrences(of: "<|nodiarize|>", with: "")
    .trimmingCharacters(in: .whitespacesAndNewlines)

print("\n=== Results ===")
print("Total tokens: \(tokens.count)")
print("\nGenerated: \"\(generatedText)\"")

// Load ground truth
if let gtData = try? Data(contentsOf: URL(fileURLWithPath: "test-audio-groundtruth.txt")),
   let groundTruth = String(data: gtData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) {
    print("Ground truth: \"\(groundTruth)\"")

    if generatedText.lowercased() == groundTruth.lowercased() {
        print("\n✅ PERFECT MATCH!")
    } else if groundTruth.lowercased().contains(generatedText.lowercased()) ||
              generatedText.lowercased().contains(groundTruth.lowercased()) {
        print("\n⚠️  PARTIAL MATCH")
    } else {
        print("\n❌ NO MATCH")
    }
}
