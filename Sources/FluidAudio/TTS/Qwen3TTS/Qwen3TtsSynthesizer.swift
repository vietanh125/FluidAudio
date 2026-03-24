import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

/// Qwen3-TTS 6-model CoreML synthesizer.
///
/// Pipeline (Argmax-style, matching `inference.py`):
/// 1. Build prefill embeddings: TextProjector(text) + CodeEmbedder(codec) per position
/// 2. CodeDecoder prefill: feed each embedding one at a time with KV cache
/// 3. Autoregressive decode loop:
///    a. MultiCodeDecoder: hidden_states + CB0 → CB1-CB15
///    b. Sum all 16 codec embeddings + tts_pad → CodeDecoder step → next CB0
/// 4. SpeechDecoder: all codec frames → audio waveform
public struct Qwen3TtsSynthesizer {

    static let logger = AppLogger(category: "Qwen3TtsSynthesizer")

    private enum Context {
        @TaskLocal static var modelStore: Qwen3TtsModelStore?
    }

    static func withModelStore<T>(
        _ store: Qwen3TtsModelStore,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelStore.withValue(store) {
            try await operation()
        }
    }

    static func currentModelStore() throws -> Qwen3TtsModelStore {
        guard let store = Context.modelStore else {
            throw TTSError.processingFailed(
                "Qwen3TtsSynthesizer requires a model store context.")
        }
        return store
    }

    // MARK: - Public Types

    /// Result of a Qwen3-TTS synthesis operation.
    public struct SynthesisResult: Sendable {
        /// WAV audio data (24kHz).
        public let audio: Data
        /// Raw Float32 audio samples.
        public let samples: [Float]
        /// Number of codec tokens generated.
        public let tokenCount: Int
    }

    // MARK: - Public API

    /// Synthesize audio from text.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - tokenIds: Pre-tokenized text IDs.
    ///   - useSpeaker: Whether to use speaker embedding (default: true).
    ///   - language: Language for synthesis (default: "english").
    ///   - config: Optional sampling configuration for parameter tuning.
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        tokenIds: [Int]? = nil,
        useSpeaker: Bool = true,
        language: String = Qwen3TtsConstants.defaultLanguage,
        config: Qwen3TtsSamplingConfig? = nil
    ) async throws -> SynthesisResult {
        let store = try currentModelStore()
        let samplingConfig = config ?? .default

        logger.info("Qwen3-TTS synthesizing: '\(text)'")

        guard let textTokens = tokenIds else {
            throw TTSError.processingFailed(
                "Qwen3-TTS requires pre-tokenized input. Please provide tokenIds.")
        }

        // 1. Build prefill embeddings
        let prefillStart = Date()
        let prefillEmbeds = try await buildPrefillEmbeddings(
            textTokens: textTokens,
            useSpeaker: useSpeaker,
            language: language,
            store: store
        )
        let prefillBuildTime = Date().timeIntervalSince(prefillStart)
        logger.info("Built \(prefillEmbeds.count) prefill embeddings in \(String(format: "%.2f", prefillBuildTime))s")

        // 2. CodeDecoder prefill
        let cdPrefillStart = Date()
        var cdState = CodeDecoderKVState()
        var lastOutput: CodeDecoderOutput!

        for emb in prefillEmbeds {
            lastOutput = try await runCodeDecoderStep(
                inputEmbeds: emb, state: &cdState, store: store)
        }
        let cdPrefillTime = Date().timeIntervalSince(cdPrefillStart)
        logger.info(
            "CodeDecoder prefill: \(prefillEmbeds.count) positions in \(String(format: "%.2f", cdPrefillTime))s"
        )

        // 3. Sample first CB0 from prefill logits
        var logits = extractFloatArray(from: lastOutput.logits)

        suppressControlTokens(&logits)
        suppressEos(&logits)  // min_new_tokens: suppress EOS for step 0
        let firstCb0 = sampleGreedy(logits: logits)
        var generatedCb0s: [Int] = [firstCb0]

        logger.info("First CB0: \(firstCb0)")

        // 4. Autoregressive decode loop
        let decodeStart = Date()
        var allFrames: [[Int]] = []
        var currentCb0 = firstCb0
        var currentHidden = lastOutput.hiddenStates

        // Cache tts_pad embedding for decode loop
        let textProjector = try await store.textProjector()
        let codeEmbedder = try await store.codeEmbedder()
        let multiCodeEmbedder = try await store.multiCodeEmbedder()

        // Pre-load tts_pad in appropriate format (tensor or array)
        let useTensorPath = {
            if #available(macOS 15.0, iOS 18.0, *) {
                return true
            }
            return false
        }()

        // Load tts_pad as tensor or array depending on path
        var ttsPadTensorOpt: Any? = nil  // Type-erased to avoid @available issues
        let ttsPadEmbed: MLMultiArray

        if #available(macOS 15.0, iOS 18.0, *), useTensorPath {
            let tensor = try await runTextProjectorTensor(textProjector, tokenId: Qwen3TtsConstants.ttsPadTokenId)
            ttsPadTensorOpt = tensor
            ttsPadEmbed = try await materializeTensorToArray(tensor)
        } else {
            ttsPadEmbed = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsPadTokenId)
        }

        // PERFORMANCE: No KV cache template needed - each frame will create fresh arrays
        // The first frame will call getModelStridedKVCaches(), subsequent frames will
        // reuse the model's output arrays from the previous frame's final position.
        var mcdKeyTemplate: MLMultiArray? = nil
        var mcdValTemplate: MLMultiArray? = nil

        for step in 0..<Qwen3TtsConstants.maxCodecTokens {
            // MultiCodeDecoder: hidden + CB0 → CB1-CB15
            let (cb1to15, newKeyTemplate, newValTemplate) = try await runMultiCodeDecoder(
                hiddenStates: currentHidden,
                cb0Token: currentCb0,
                codeEmbedder: codeEmbedder,
                multiCodeEmbedder: multiCodeEmbedder,
                kvKeyTemplate: mcdKeyTemplate,
                kvValTemplate: mcdValTemplate,
                store: store,
                config: samplingConfig
            )
            // Save the output KV caches as templates for next frame
            mcdKeyTemplate = newKeyTemplate
            mcdValTemplate = newValTemplate

            let frame = [currentCb0] + cb1to15
            allFrames.append(frame)

            // Build decode input: sum(all 16 codec embeddings) + tts_pad
            let decodeInput: MLMultiArray

            if #available(macOS 15.0, iOS 18.0, *), useTensorPath {
                // MLTensor fast path - deferred computation, zero-copy
                var codecSumTensor = try await runCodeEmbedderTensor(codeEmbedder, tokenId: currentCb0)

                // Add cb1-cb15 using deferred tensor addition
                for cbIdx in 0..<15 {
                    let linIdx = cbIdx * Qwen3TtsConstants.codecVocabSize + cb1to15[cbIdx]
                    let cbTensor = try await runMultiCodeEmbedderTensor(multiCodeEmbedder, linearizedId: linIdx)
                    codecSumTensor = codecSumTensor + cbTensor
                }

                // Add tts_pad overlay
                let ttsPadTensor = ttsPadTensorOpt as! MLTensor
                codecSumTensor = codecSumTensor + ttsPadTensor

                // Materialize tensor to MLMultiArray
                // The tensor operations are deferred until now, making this more efficient than scalar loops
                decodeInput = try await materializeTensorToArray(codecSumTensor)
            } else {
                // Legacy path with vDSP
                let cb0Embed = try runCodeEmbedder(codeEmbedder, tokenId: currentCb0)
                var codecSum = extractFloatArray(from: cb0Embed)

                for cbIdx in 0..<15 {
                    let linIdx = cbIdx * Qwen3TtsConstants.codecVocabSize + cb1to15[cbIdx]
                    let cbEmbed = try runMultiCodeEmbedder(multiCodeEmbedder, linearizedId: linIdx)
                    let cbFloats = extractFloatArray(from: cbEmbed)
                    for i in 0..<codecSum.count {
                        codecSum[i] += cbFloats[i]
                    }
                }

                let padFloats = extractFloatArray(from: ttsPadEmbed)
                for i in 0..<codecSum.count {
                    codecSum[i] += padFloats[i]
                }

                decodeInput = try createEmbedding(from: codecSum)
            }

            // CodeDecoder step
            let cdOutput = try await runCodeDecoderStep(
                inputEmbeds: decodeInput, state: &cdState, store: store)
            currentHidden = cdOutput.hiddenStates

            // Sample next CB0
            var nextLogits = extractFloatArray(from: cdOutput.logits)
            suppressControlTokens(&nextLogits)
            if step >= 1 {
                // Allow EOS after min_new_tokens=2 (step 0 was first token, step 1 is second)
            } else {
                suppressEos(&nextLogits)
            }
            applyEnhancedPenalties(
                &nextLogits,
                generatedIds: generatedCb0s,
                repetitionPenalty: samplingConfig.cb0RepetitionPenalty,
                frequencyPenalty: samplingConfig.cb0FrequencyPenalty,
                presencePenalty: samplingConfig.cb0PresencePenalty
            )
            let nextCb0 = sampleGreedy(logits: nextLogits)

            if nextCb0 == Qwen3TtsConstants.codecEosId {
                logger.info("EOS at step \(step + 1)")
                break
            }

            if cdState.position >= Qwen3TtsConstants.cdKvLen - 1 {
                logger.info("KV cache full at step \(step + 1)")
                break
            }

            generatedCb0s.append(nextCb0)
            currentCb0 = nextCb0
        }

        let decodeTime = Date().timeIntervalSince(decodeStart)
        let fps = Double(allFrames.count) / max(decodeTime, 0.001)
        logger.info(
            "Decoded \(allFrames.count) frames in \(String(format: "%.2f", decodeTime))s"
                + " (\(String(format: "%.1f", fps)) frames/s)"
        )

        // 5. SpeechDecoder: codes → audio
        let speechStart = Date()
        let audioSamples = try await runSpeechDecoder(
            allFrames: allFrames, store: store)
        let speechTime = Date().timeIntervalSince(speechStart)
        logger.info("SpeechDecoder: \(String(format: "%.2f", speechTime))s")

        // 6. Trim to actual frame count
        let expectedSamples = allFrames.count * Qwen3TtsConstants.samplesPerFrame
        let frameTrimmed: [Float]
        if expectedSamples < audioSamples.count {
            frameTrimmed = Array(audioSamples.prefix(expectedSamples))
        } else {
            frameTrimmed = audioSamples
        }

        // Strip leading/trailing silence
        let trimmedSamples = trimSilence(
            frameTrimmed, sampleRate: Qwen3TtsConstants.audioSampleRate)

        // Apply audio post-processing (de-esser, filtering, normalization)
        let processedSamples = applyPostProcessing(
            trimmedSamples,
            sampleRate: Qwen3TtsConstants.audioSampleRate,
            config: samplingConfig
        )

        // 7. Encode as WAV
        let audioData = try AudioWAV.data(
            from: processedSamples,
            sampleRate: Double(Qwen3TtsConstants.audioSampleRate)
        )

        let duration = Double(processedSamples.count) / Double(Qwen3TtsConstants.audioSampleRate)
        logger.info("Audio duration: \(String(format: "%.2f", duration))s")

        return SynthesisResult(
            audio: audioData,
            samples: processedSamples,
            tokenCount: allFrames.count
        )
    }

    // MARK: - Prefill Embedding Construction

    /// Build dual-embedding prefill sequence matching inference.py.
    ///
    /// Layout: role(3) + control(4) + speaker?(0-1) + bos(1) + text(N) + eos(1) + final(1)
    private static func buildPrefillEmbeddings(
        textTokens: [Int],
        useSpeaker: Bool,
        language: String,
        store: Qwen3TtsModelStore
    ) async throws -> [MLMultiArray] {
        let textProjector = try await store.textProjector()
        let codeEmbedder = try await store.codeEmbedder()

        var embeds: [MLMultiArray] = []

        // [0:3] Role: text_proj only (no codec overlay)
        for tokenId in Qwen3TtsConstants.rolePrefixTokens {
            embeds.append(try runTextProjector(textProjector, tokenId: tokenId))
        }

        // Cache tts_pad, tts_bos, tts_eos embeddings
        let ttsPad = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsPadTokenId)
        let ttsBos = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsBosTokenId)
        let ttsEos = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsEosTokenId)

        // [3:7] Control: tts_pad + codec_emb([think, think_bos, lang, think_eos])
        let langId =
            Qwen3TtsConstants.languageIds[language] ?? Qwen3TtsConstants.languageIds["english"]!
        let codecCtrlTokens = [
            Qwen3TtsConstants.codecThinkId,
            Qwen3TtsConstants.codecThinkBosId,
            langId,
            Qwen3TtsConstants.codecThinkEosId,
        ]
        for ctok in codecCtrlTokens {
            let codecEmb = try runCodeEmbedder(codeEmbedder, tokenId: ctok)
            embeds.append(try addEmbeddings(ttsPad, codecEmb))
        }

        // [7] Optional speaker embedding
        if useSpeaker, let speakerData = await store.speaker() {
            let speakerEmbed = try createEmbedding(from: speakerData)
            embeds.append(try addEmbeddings(ttsPad, speakerEmbed))
        }

        // Control: tts_bos + codec_emb(codec_pad)
        let codecPadEmb = try runCodeEmbedder(codeEmbedder, tokenId: Qwen3TtsConstants.codecPadId)
        embeds.append(try addEmbeddings(ttsBos, codecPadEmb))

        // Text: text_proj(token) + codec_emb(codec_pad) for each token
        for tokenId in textTokens {
            let textEmb = try runTextProjector(textProjector, tokenId: tokenId)
            embeds.append(try addEmbeddings(textEmb, codecPadEmb))
        }

        // EOS: text_proj(tts_eos) + codec_emb(codec_pad)
        embeds.append(try addEmbeddings(ttsEos, codecPadEmb))

        // Final: tts_pad + codec_emb(codec_bos)
        let codecBosEmb = try runCodeEmbedder(
            codeEmbedder, tokenId: Qwen3TtsConstants.codecBosId)
        embeds.append(try addEmbeddings(ttsPad, codecBosEmb))

        return embeds
    }

    // MARK: - CodeDecoder

    /// KV cache state for the CodeDecoder (28-layer transformer).
    private struct CodeDecoderKVState {
        var keyCache: MLMultiArray
        var valueCache: MLMultiArray
        var position: Int = 0

        init() {
            // [1, 28672, 1, 256] float16
            let shape: [NSNumber] = [
                1, NSNumber(value: Qwen3TtsConstants.cdKvDim), 1,
                NSNumber(value: Qwen3TtsConstants.cdKvLen),
            ]
            keyCache = try! MLMultiArray(shape: shape, dataType: .float16)
            valueCache = try! MLMultiArray(shape: shape, dataType: .float16)
        }
    }

    private struct CodeDecoderOutput {
        let logits: MLMultiArray
        let hiddenStates: MLMultiArray
    }

    /// Run a single CodeDecoder step (prefill or decode).
    private static func runCodeDecoderStep(
        inputEmbeds: MLMultiArray,
        state: inout CodeDecoderKVState,
        store: Qwen3TtsModelStore
    ) async throws -> CodeDecoderOutput {
        let model = try await store.codeDecoder()
        let pos = state.position
        let kvLen = Qwen3TtsConstants.cdKvLen

        // key_padding_mask [1, 256] float16: 0..pos = 0.0, rest = -10000.0
        let keyMask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        for i in 0..<kvLen {
            keyMask[i] = NSNumber(value: i <= pos ? Float(0.0) : Float(-10000.0))
        }

        // kv_cache_update_mask [1, 256] float16: only pos = 1.0
        let updateMask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        for i in 0..<kvLen {
            updateMask[i] = NSNumber(value: i == pos ? Float(1.0) : Float(0.0))
        }

        let cacheLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLenArr[0] = NSNumber(value: pos)

        // Cast input_embeds to float16
        let f16Input = try toFloat16(inputEmbeds)

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": f16Input,
            "cache_length": cacheLenArr,
            "key_padding_mask": keyMask,
            "kv_cache_update_mask": updateMask,
            "key_cache": state.keyCache,
            "value_cache": state.valueCache,
        ])

        let output = try await model.compatPrediction(from: features, options: MLPredictionOptions())

        guard let newKeyCache = output.featureValue(for: "new_key_cache")?.multiArrayValue,
            let newValueCache = output.featureValue(for: "new_value_cache")?.multiArrayValue,
            let hiddenStates = output.featureValue(for: "hidden_states")?.multiArrayValue,
            let logits = output.featureValue(for: "logits")?.multiArrayValue
        else {
            throw TTSError.processingFailed("Missing CodeDecoder outputs")
        }

        state.keyCache = newKeyCache
        state.valueCache = newValueCache
        state.position += 1

        return CodeDecoderOutput(logits: logits, hiddenStates: hiddenStates)
    }

    // MARK: - MultiCodeDecoder

    /// Run MultiCodeDecoder to generate CB1-CB15 from hidden_states + CB0.
    ///
    /// Returns: (CB tokens, final key cache, final value cache)
    /// The final KV caches can be reused as templates for the next frame.
    private static func runMultiCodeDecoder(
        hiddenStates: MLMultiArray,
        cb0Token: Int,
        codeEmbedder: MLModel,
        multiCodeEmbedder: MLModel,
        kvKeyTemplate: MLMultiArray?,
        kvValTemplate: MLMultiArray?,
        store: Qwen3TtsModelStore,
        config: Qwen3TtsSamplingConfig
    ) async throws -> ([Int], MLMultiArray, MLMultiArray) {
        let model = try await store.multiCodeDecoder()
        let kvLen = Qwen3TtsConstants.mcdKvLen

        // Get initial KV caches: either from cached template (subsequent frames)
        // or from warmup prediction (first frame only)
        var (mcdKey, mcdVal): (MLMultiArray, MLMultiArray)
        if let keyTemplate = kvKeyTemplate, let valTemplate = kvValTemplate {
            // Reuse previous frame's final KV caches as template, then zero them
            mcdKey = keyTemplate
            mcdVal = valTemplate
            // Zero in-place using memset for performance
            memset(mcdKey.dataPointer, 0, mcdKey.count * MemoryLayout<Float16>.size)
            memset(mcdVal.dataPointer, 0, mcdVal.count * MemoryLayout<Float16>.size)
        } else {
            // First frame: run warmup prediction to get properly-strided arrays
            (mcdKey, mcdVal) = try await getModelStridedKVCaches(model: model, kvLen: kvLen)
        }

        // Position 0: feed hidden_states
        let (mask0, umask0) = try makeMcdMasks(pos: 0, kvLen: kvLen)
        let cacheLen0 = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen0[0] = NSNumber(value: 0)

        let f16Hidden = try toFloat16(hiddenStates)
        let feat0 = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": f16Hidden,
            "cache_length": cacheLen0,
            "key_cache": mcdKey,
            "value_cache": mcdVal,
            "key_padding_mask": mask0,
            "kv_cache_update_mask": umask0,
        ])
        let out0 = try await model.compatPrediction(from: feat0, options: MLPredictionOptions())
        mcdKey = out0.featureValue(for: "new_key_cache")!.multiArrayValue!
        mcdVal = out0.featureValue(for: "new_value_cache")!.multiArrayValue!

        // Position 1: feed CB0 embedding → lm_head[0] → CB1
        let cb0Emb = try runCodeEmbedder(codeEmbedder, tokenId: cb0Token)
        let (mask1, umask1) = try makeMcdMasks(pos: 1, kvLen: kvLen)
        let cacheLen1 = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen1[0] = NSNumber(value: 1)

        let f16Cb0 = try toFloat16(cb0Emb)
        let feat1 = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": f16Cb0,
            "cache_length": cacheLen1,
            "key_cache": mcdKey,
            "value_cache": mcdVal,
            "key_padding_mask": mask1,
            "kv_cache_update_mask": umask1,
        ])
        let out1 = try await model.compatPrediction(from: feat1, options: MLPredictionOptions())
        mcdKey = out1.featureValue(for: "new_key_cache")!.multiArrayValue!
        mcdVal = out1.featureValue(for: "new_value_cache")!.multiArrayValue!

        // CB1 from lm_head[0]
        let allLogits1 = out1.featureValue(for: "all_logits")!.multiArrayValue!
        var cb1Logits = extractSliceLogits(allLogits1, sliceIndex: 0)

        let cb1 = sampleTopKTopP(
            logits: &cb1Logits,
            temperature: config.codeTemperature,
            topK: config.codeTopK,
            topP: config.codeTopP
        )
        var cbTokens = [cb1]

        // Positions 2-15: autoregressive decode for CB2-CB15
        for cbStep in 1..<15 {
            let prevCb = cbTokens.last!
            let linIdx = (cbStep - 1) * Qwen3TtsConstants.codecVocabSize + prevCb
            let cbEmb = try runMultiCodeEmbedder(multiCodeEmbedder, linearizedId: linIdx)

            let mcdPos = cbStep + 1
            let (mask, umask) = try makeMcdMasks(pos: mcdPos, kvLen: kvLen)
            let cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
            cacheLen[0] = NSNumber(value: mcdPos)

            let f16Emb = try toFloat16(cbEmb)
            let feat = try MLDictionaryFeatureProvider(dictionary: [
                "input_embeds": f16Emb,
                "cache_length": cacheLen,
                "key_cache": mcdKey,
                "value_cache": mcdVal,
                "key_padding_mask": mask,
                "kv_cache_update_mask": umask,
            ])
            let out = try await model.compatPrediction(from: feat, options: MLPredictionOptions())
            mcdKey = out.featureValue(for: "new_key_cache")!.multiArrayValue!
            mcdVal = out.featureValue(for: "new_value_cache")!.multiArrayValue!

            let allLogits = out.featureValue(for: "all_logits")!.multiArrayValue!
            var cbLogits = extractSliceLogits(allLogits, sliceIndex: cbStep)
            cbTokens.append(
                sampleTopKTopP(
                    logits: &cbLogits,
                    temperature: config.codeTemperature,
                    topK: config.codeTopK,
                    topP: config.codeTopP
                )
            )
        }

        // Return CB tokens AND final KV caches (for reuse as templates in next frame)
        return (cbTokens, mcdKey, mcdVal)
    }

    /// Create key_padding_mask and kv_cache_update_mask for MultiCodeDecoder.
    private static func makeMcdMasks(
        pos: Int, kvLen: Int
    ) throws -> (MLMultiArray, MLMultiArray) {
        let mask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        let umask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)

        // Use direct memory access for faster initialization
        mask.dataPointer.withMemoryRebound(to: Float16.self, capacity: kvLen) { maskPtr in
            // Fill [0...pos] with 0.0
            for i in 0...pos {
                maskPtr[i] = Float16(0.0)
            }
            // Fill [pos+1...kvLen] with -10000.0
            let negVal = Float16(-10000.0)
            for i in (pos + 1)..<kvLen {
                maskPtr[i] = negVal
            }
        }

        umask.dataPointer.withMemoryRebound(to: Float16.self, capacity: kvLen) { umaskPtr in
            // Zero all using memset
            memset(umaskPtr, 0, kvLen * MemoryLayout<Float16>.size)
            // Set position to 1.0
            umaskPtr[pos] = Float16(1.0)
        }

        return (mask, umask)
    }

    /// Get zero-initialized KV caches with the model's expected stride layout.
    ///
    /// CoreML compiled models use specific non-contiguous memory layouts.
    /// The only reliable way to get properly-strided arrays is to run a
    /// prediction and use the output KV caches, then zero them for reuse.
    private static func getModelStridedKVCaches(
        model: MLModel, kvLen: Int
    ) async throws -> (MLMultiArray, MLMultiArray) {
        // Create minimal inputs for a warmup prediction
        let kvDim = Qwen3TtsConstants.mcdKvDim
        let shape: [NSNumber] = [1, NSNumber(value: kvDim), 1, NSNumber(value: kvLen)]

        // Use zero inputs — the output stride layout is what matters
        let dummyInput = try MLMultiArray(shape: [1, 1024, 1, 1], dataType: .float16)
        let dummyKey = try MLMultiArray(shape: shape, dataType: .float16)
        let dummyVal = try MLMultiArray(shape: shape, dataType: .float16)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        for i in 0..<kvLen {
            mask[i] = NSNumber(value: Float(-10000.0))
        }
        let umask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        let cacheLen = try MLMultiArray(shape: [1], dataType: .int32)

        let feat = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": dummyInput,
            "cache_length": cacheLen,
            "key_cache": dummyKey,
            "value_cache": dummyVal,
            "key_padding_mask": mask,
            "kv_cache_update_mask": umask,
        ])

        let out = try await model.compatPrediction(from: feat, options: MLPredictionOptions())
        let outKey = out.featureValue(for: "new_key_cache")!.multiArrayValue!
        let outVal = out.featureValue(for: "new_value_cache")!.multiArrayValue!

        // Zero the caches while preserving their stride layout using memset for performance
        memset(outKey.dataPointer, 0, outKey.count * MemoryLayout<Float16>.size)
        memset(outVal.dataPointer, 0, outVal.count * MemoryLayout<Float16>.size)

        return (outKey, outVal)
    }

    // MARK: - SpeechDecoder

    /// Run the SpeechDecoder on all codec frames.
    private static func runSpeechDecoder(
        allFrames: [[Int]],
        store: Qwen3TtsModelStore
    ) async throws -> [Float] {
        let model = try await store.speechDecoder()
        let fixedLen = Qwen3TtsConstants.speechDecoderFrames  // 125
        let numCb = Qwen3TtsConstants.numCodebooks  // 16

        // Build codes tensor [1, 16, 125] int32
        let codes = try MLMultiArray(
            shape: [1, NSNumber(value: numCb), NSNumber(value: fixedLen)],
            dataType: .int32
        )

        // Initialize to zero (pad) using subscript for stride safety
        for i in 0..<(numCb * fixedLen) {
            codes[i] = NSNumber(value: Int32(0))
        }

        // Fill: codes[0, cb, t] = allFrames[t][cb]
        for t in 0..<min(allFrames.count, fixedLen) {
            let frame = allFrames[t]
            for cb in 0..<min(frame.count, numCb) {
                codes[cb * fixedLen + t] = NSNumber(value: Int32(frame[cb]))
            }
        }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": codes
        ])

        let output = try await model.compatPrediction(from: features, options: MLPredictionOptions())

        guard let audioArray = output.featureValue(for: "audio")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing SpeechDecoder output")
        }

        return extractFloatArray(from: audioArray)
    }

    // MARK: - Model Runners

    /// TextProjector: text_token → embedding [1, 1024, 1, 1].
    private static func runTextProjector(_ model: MLModel, tokenId: Int) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = NSNumber(value: tokenId)

        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let output = try model.prediction(from: features, options: MLPredictionOptions())

        guard let embeds = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing TextProjector output")
        }
        return embeds
    }

    /// CodeEmbedder: codec_token → embedding [1, 1024, 1, 1].
    private static func runCodeEmbedder(_ model: MLModel, tokenId: Int) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = NSNumber(value: tokenId)

        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let output = try model.prediction(from: features, options: MLPredictionOptions())

        guard let embeds = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing CodeEmbedder output")
        }
        return embeds
    }

    /// MultiCodeEmbedder: linearized CB index → embedding [1, 1024, 1, 1].
    private static func runMultiCodeEmbedder(
        _ model: MLModel, linearizedId: Int
    ) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = NSNumber(value: linearizedId)

        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let output = try model.prediction(from: features, options: MLPredictionOptions())

        guard let embeds = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing MultiCodeEmbedder output")
        }
        return embeds
    }

    // MARK: - MLTensor Fast Path (macOS 15+)

    @available(macOS 15.0, iOS 18.0, *)
    private static func materializeTensorToArray(_ tensor: MLTensor) async throws -> MLMultiArray {
        // Convert MLTensor -> MLShapedArray -> MLMultiArray
        let shapedArray = await tensor.shapedArray(of: Float.self)
        return MLMultiArray(shapedArray)
    }

    @available(macOS 15.0, iOS 18.0, *)
    private static func runTextProjectorTensor(_ model: MLModel, tokenId: Int) async throws -> MLTensor {
        let inputTensor = MLTensor(shape: [1], scalars: [Int32(tokenId)])
        let outputs = try await model.prediction(from: ["input_ids": inputTensor])

        guard let embedTensor = outputs["input_embeds"] else {
            throw TTSError.processingFailed("Missing TextProjector tensor output")
        }
        // Convert to float32 for consistent arithmetic
        return embedTensor.cast(to: Float.self)
    }

    @available(macOS 15.0, iOS 18.0, *)
    private static func runCodeEmbedderTensor(_ model: MLModel, tokenId: Int) async throws -> MLTensor {
        let inputTensor = MLTensor(shape: [1], scalars: [Int32(tokenId)])
        let outputs = try await model.prediction(from: ["input_ids": inputTensor])

        guard let embedTensor = outputs["input_embeds"] else {
            throw TTSError.processingFailed("Missing CodeEmbedder tensor output")
        }
        // Convert to float32 for consistent arithmetic
        return embedTensor.cast(to: Float.self)
    }

    @available(macOS 15.0, iOS 18.0, *)
    private static func runMultiCodeEmbedderTensor(_ model: MLModel, linearizedId: Int) async throws -> MLTensor {
        let inputTensor = MLTensor(shape: [1], scalars: [Int32(linearizedId)])
        let outputs = try await model.prediction(from: ["input_ids": inputTensor])

        guard let embedTensor = outputs["input_embeds"] else {
            throw TTSError.processingFailed("Missing MultiCodeEmbedder tensor output")
        }
        // Convert to float32 for consistent arithmetic
        return embedTensor.cast(to: Float.self)
    }

    // MARK: - Sampling

    /// Suppress control tokens [2048, 3072) except EOS (2150).
    private static func suppressControlTokens(_ logits: inout [Float]) {
        let eosToken = Qwen3TtsConstants.codecEosId
        let vocabSize = Qwen3TtsConstants.codecVocabSize

        // Save EOS logit before suppression
        let eosLogit = eosToken < logits.count ? logits[eosToken] : -Float.infinity

        // Suppress [2048, 3072)
        for i in vocabSize..<min(3072, logits.count) {
            logits[i] = -.infinity
        }

        // Restore EOS
        if eosToken < logits.count {
            logits[eosToken] = eosLogit
        }
    }

    /// Suppress EOS token (for min_new_tokens enforcement).
    private static func suppressEos(_ logits: inout [Float]) {
        let eosToken = Qwen3TtsConstants.codecEosId
        if eosToken < logits.count {
            logits[eosToken] = -.infinity
        }
    }

    /// Apply enhanced penalties to CB0 logits: repetition, frequency, and presence.
    ///
    /// - Repetition penalty: divides/multiplies logits based on sign (1.15 default)
    /// - Frequency penalty: subtracts penalty proportional to token count
    /// - Presence penalty: subtracts fixed penalty if token appeared at least once
    private static func applyEnhancedPenalties(
        _ logits: inout [Float],
        generatedIds: [Int],
        repetitionPenalty: Float = Qwen3TtsConstants.cb0RepetitionPenalty,
        frequencyPenalty: Float = Qwen3TtsConstants.cb0FrequencyPenalty,
        presencePenalty: Float = Qwen3TtsConstants.cb0PresencePenalty
    ) {
        guard !generatedIds.isEmpty else { return }

        // Count token occurrences for frequency penalty
        var tokenCounts: [Int: Int] = [:]
        for tokenId in generatedIds {
            tokenCounts[tokenId, default: 0] += 1
        }

        // Apply all penalties
        for (tokenId, count) in tokenCounts {
            guard tokenId < logits.count else { continue }

            // 1. Repetition penalty (original logic)
            if repetitionPenalty != 1.0 {
                if logits[tokenId] > 0 {
                    logits[tokenId] /= repetitionPenalty
                } else {
                    logits[tokenId] *= repetitionPenalty
                }
            }

            // 2. Frequency penalty (proportional to count)
            if frequencyPenalty != 0.0 {
                logits[tokenId] -= frequencyPenalty * Float(count)
            }

            // 3. Presence penalty (fixed penalty for any appearance)
            if presencePenalty != 0.0 {
                logits[tokenId] -= presencePenalty
            }
        }
    }

    /// Greedy sampling: return argmax of logits (for CB0 main codebook).
    private static func sampleGreedy(logits: [Float]) -> Int {
        guard !logits.isEmpty else { return 0 }

        var maxIdx = 0
        var maxVal = logits[0]

        for i in 1..<logits.count {
            if logits[i] > maxVal {
                maxVal = logits[i]
                maxIdx = i
            }
        }

        return maxIdx
    }

    /// Sample from logits with temperature + top-k + top-p (nucleus sampling).
    private static func sampleTopKTopP(
        logits: inout [Float],
        temperature: Float = Qwen3TtsConstants.codeTemperature,
        topK: Int = Qwen3TtsConstants.codeTopK,
        topP: Float = Qwen3TtsConstants.codeTopP
    ) -> Int {
        let count = logits.count
        guard count > 0 else { return 0 }

        // Apply temperature
        for i in 0..<count {
            logits[i] /= temperature
        }

        // Top-k filtering
        if topK > 0 && topK < count {
            var sorted = logits
            sorted.sort(by: >)
            let threshold = sorted[topK - 1]
            for i in 0..<count where logits[i] < threshold {
                logits[i] = -.infinity
            }
        }

        // Softmax to get probabilities
        let maxLogit = logits.max() ?? 0
        var expSum: Float = 0
        var expLogits = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let e = exp(logits[i] - maxLogit)
            expLogits[i] = e
            expSum += e
        }

        // Normalize to probabilities
        var probs = [Float](repeating: 0, count: count)
        for i in 0..<count {
            probs[i] = expLogits[i] / expSum
        }

        // Top-p (nucleus) filtering
        if topP < 1.0 {
            // Sort indices by probability (descending)
            let sortedIndices = probs.indices.sorted { probs[$0] > probs[$1] }

            // Find cutoff index where cumulative probability >= topP
            var cumulative: Float = 0
            var cutoffIndex = 0
            for idx in sortedIndices {
                cumulative += probs[idx]
                cutoffIndex = idx
                if cumulative >= topP {
                    break
                }
            }

            // Zero out probabilities below threshold
            let threshold = probs[cutoffIndex]
            for i in 0..<count where probs[i] < threshold {
                probs[i] = 0
            }

            // Renormalize
            let newSum = probs.reduce(0, +)
            if newSum > 0 {
                for i in 0..<count {
                    probs[i] /= newSum
                }
            }
        }

        // Multinomial sampling
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<count {
            cumulative += probs[i]
            if cumulative >= r {
                return i
            }
        }

        return count - 1
    }

    /// Extract logits for a specific lm_head slice from all_logits.
    ///
    /// all_logits shape from MultiCodeDecoder: [1, 15, 2048].
    /// We extract [0, sliceIndex, :] and return as [Float].
    private static func extractSliceLogits(
        _ allLogits: MLMultiArray, sliceIndex: Int
    ) -> [Float] {
        let vocabSize = Qwen3TtsConstants.codecVocabSize
        let offset = sliceIndex * vocabSize

        var result = [Float](repeating: 0, count: vocabSize)

        // Use direct memory access for performance
        if allLogits.dataType == .float16 {
            allLogits.dataPointer.withMemoryRebound(to: Float16.self, capacity: allLogits.count) { ptr in
                result.withUnsafeMutableBufferPointer { resultPtr in
                    var sourceBuffer = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: ptr + offset),
                        height: 1,
                        width: vImagePixelCount(vocabSize),
                        rowBytes: vocabSize * MemoryLayout<Float16>.size
                    )
                    var destBuffer = vImage_Buffer(
                        data: resultPtr.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(vocabSize),
                        rowBytes: vocabSize * MemoryLayout<Float>.size
                    )
                    vImageConvert_Planar16FtoPlanarF(&sourceBuffer, &destBuffer, 0)
                }
            }
        } else {
            // Fallback for other types
            for i in 0..<vocabSize {
                result[i] = allLogits[offset + i].floatValue
            }
        }

        return result
    }

    // MARK: - Audio Post-Processing

    /// Trim leading and trailing silence from audio samples.
    private static func trimSilence(
        _ samples: [Float],
        sampleRate: Int,
        threshold: Float = 0.005,
        windowMs: Int = 10,
        padMs: Int = 20
    ) -> [Float] {
        let windowSize = sampleRate * windowMs / 1000
        let padSize = sampleRate * padMs / 1000
        guard samples.count > windowSize else { return samples }

        // Find first non-silent window
        var start = 0
        for i in stride(from: 0, to: samples.count - windowSize, by: windowSize) {
            var sum: Float = 0
            for j in i..<(i + windowSize) {
                sum += samples[j] * samples[j]
            }
            let rms = (sum / Float(windowSize)).squareRoot()
            if rms > threshold {
                start = max(0, i - padSize)
                break
            }
        }

        // Find last non-silent window
        let bigWindow = sampleRate / 5
        var end = samples.count
        for i in stride(from: samples.count - bigWindow, through: 0, by: -windowSize) {
            let windowEnd = min(i + bigWindow, samples.count)
            var sum: Float = 0
            for j in i..<windowEnd {
                sum += samples[j] * samples[j]
            }
            let rms = (sum / Float(windowEnd - i)).squareRoot()
            if rms > threshold {
                end = min(samples.count, windowEnd + padSize)
                break
            }
        }

        guard start < end else { return samples }
        return Array(samples[start..<end])
    }

    /// Apply comprehensive audio post-processing pipeline.
    ///
    /// Steps:
    /// 1. De-esser: reduces harsh sibilance (5-10 kHz)
    /// 2. Low-pass filter: removes harsh high frequencies above 16 kHz
    /// 3. Normalization: adjust loudness to target LUFS
    private static func applyPostProcessing(
        _ samples: [Float],
        sampleRate: Int,
        config: Qwen3TtsSamplingConfig
    ) -> [Float] {
        guard config.enablePostProcessing else { return samples }
        guard !samples.isEmpty else { return samples }

        var processed = samples

        // 1. De-esser
        processed = applyDeEsser(
            processed,
            sampleRate: sampleRate,
            lowFreq: Qwen3TtsConstants.deEsserLowFreq,
            highFreq: Qwen3TtsConstants.deEsserHighFreq,
            thresholdDb: config.deEsserThresholdDb,
            ratio: config.deEsserRatio
        )

        // 2. Low-pass filter
        processed = applyLowPassFilter(
            processed,
            sampleRate: sampleRate,
            cutoffHz: config.lowPassCutoffHz
        )

        // 3. Normalization
        processed = normalize(
            processed,
            targetLufs: config.targetLufs
        )

        return processed
    }

    /// De-esser: reduces harsh sibilance in the 5-10 kHz range.
    ///
    /// Applies multiband compression to sibilance frequencies:
    /// - Isolates sibilance band (5-10 kHz) via bandpass
    /// - Detects envelope (RMS)
    /// - Applies compression when above threshold
    /// - Mixes back with original signal
    private static func applyDeEsser(
        _ samples: [Float],
        sampleRate: Int,
        lowFreq: Float,
        highFreq: Float,
        thresholdDb: Float,
        ratio: Float
    ) -> [Float] {
        guard samples.count > 0 else { return samples }

        // Simple bandpass filter for sibilance detection (5-10 kHz)
        let filtered = applyBandPassFilter(
            samples,
            sampleRate: sampleRate,
            lowHz: lowFreq,
            highHz: highFreq
        )

        // Detect sibilance envelope (RMS with 5ms window)
        let windowSize = max(1, sampleRate / 200)  // 5ms window
        var envelope = [Float](repeating: 0, count: samples.count)
        for i in 0..<samples.count {
            let start = max(0, i - windowSize / 2)
            let end = min(samples.count, i + windowSize / 2)
            var sum: Float = 0
            for j in start..<end {
                sum += filtered[j] * filtered[j]
            }
            envelope[i] = (sum / Float(end - start)).squareRoot()
        }

        // Apply compression to sibilance
        let thresholdLinear = pow(10, thresholdDb / 20)
        var result = samples
        for i in 0..<samples.count {
            let env = envelope[i]
            if env > thresholdLinear {
                // Calculate gain reduction
                let excess = env / thresholdLinear
                let gainReduction = 1.0 / pow(excess, (ratio - 1) / ratio)
                result[i] *= gainReduction
            }
        }

        return result
    }

    /// Simple bandpass filter using two first-order filters (high-pass then low-pass).
    private static func applyBandPassFilter(
        _ samples: [Float],
        sampleRate: Int,
        lowHz: Float,
        highHz: Float
    ) -> [Float] {
        // High-pass to remove < lowHz
        let highPassed = applyHighPassFilter(samples, sampleRate: sampleRate, cutoffHz: lowHz)
        // Low-pass to remove > highHz
        return applyLowPassFilter(highPassed, sampleRate: sampleRate, cutoffHz: highHz)
    }

    /// Simple first-order high-pass filter (removes frequencies below cutoff).
    private static func applyHighPassFilter(
        _ samples: [Float],
        sampleRate: Int,
        cutoffHz: Float
    ) -> [Float] {
        guard samples.count > 1 else { return samples }

        // RC = 1 / (2π * cutoff)
        let rc = 1.0 / (2.0 * Float.pi * cutoffHz)
        let dt = 1.0 / Float(sampleRate)
        let alpha = rc / (rc + dt)

        var result = [Float](repeating: 0, count: samples.count)
        result[0] = samples[0]

        for i in 1..<samples.count {
            result[i] = alpha * (result[i - 1] + samples[i] - samples[i - 1])
        }

        return result
    }

    /// Simple first-order low-pass filter (removes frequencies above cutoff).
    private static func applyLowPassFilter(
        _ samples: [Float],
        sampleRate: Int,
        cutoffHz: Float
    ) -> [Float] {
        guard samples.count > 1 else { return samples }

        // alpha = dt / (RC + dt), where RC = 1/(2π*cutoff)
        let rc = 1.0 / (2.0 * Float.pi * cutoffHz)
        let dt = 1.0 / Float(sampleRate)
        let alpha = dt / (rc + dt)

        var result = [Float](repeating: 0, count: samples.count)
        result[0] = samples[0]

        for i in 1..<samples.count {
            result[i] = alpha * samples[i] + (1 - alpha) * result[i - 1]
        }

        return result
    }

    /// Normalize audio to target loudness (LUFS).
    ///
    /// Uses simplified loudness measurement (integrated RMS) and applies gain.
    private static func normalize(
        _ samples: [Float],
        targetLufs: Float
    ) -> [Float] {
        guard !samples.isEmpty else { return samples }

        // Calculate RMS (simplified loudness)
        var sumSquares: Float = 0
        for sample in samples {
            sumSquares += sample * sample
        }
        let rms = (sumSquares / Float(samples.count)).squareRoot()

        // Convert target LUFS to linear gain
        // LUFS ≈ -0.691 + 10*log10(RMS²) (simplified)
        // target_rms = 10^((targetLUFS + 0.691) / 20)
        let targetRms = pow(10, (targetLufs + 0.691) / 20)

        // Calculate gain needed
        let gain = rms > 0 ? targetRms / rms : 1.0

        // Apply gain with peak limiting at ±0.99
        var result = [Float](repeating: 0, count: samples.count)
        for i in 0..<samples.count {
            let scaled = samples[i] * gain
            result[i] = max(-0.99, min(0.99, scaled))  // Hard limiter
        }

        return result
    }

    // MARK: - MLMultiArray Helpers

    /// Extract Float array from MLMultiArray using subscript access (stride-safe).
    private static func extractFloatArray(from array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)

        // Direct memory copy for performance
        if array.dataType == .float32 {
            array.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { ptr in
                result.withUnsafeMutableBufferPointer { resultPtr in
                    memcpy(resultPtr.baseAddress!, ptr, count * MemoryLayout<Float>.size)
                }
            }
        } else if array.dataType == .float16 {
            // Convert float16 to float32
            array.dataPointer.withMemoryRebound(to: Float16.self, capacity: count) { ptr in
                result.withUnsafeMutableBufferPointer { resultPtr in
                    var sourceBuffer = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: ptr),
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float16>.size
                    )
                    var destBuffer = vImage_Buffer(
                        data: resultPtr.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.size
                    )
                    vImageConvert_Planar16FtoPlanarF(&sourceBuffer, &destBuffer, 0)
                }
            }
        } else {
            // Fallback for other types
            for i in 0..<count {
                result[i] = array[i].floatValue
            }
        }

        return result
    }

    /// Create [1, 1024, 1, 1] float32 embedding from Float array.
    private static func createEmbedding(from data: [Float]) throws -> MLMultiArray {
        let dim = data.count
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: dim), 1, 1], dataType: .float32)

        // Use direct memory copy for performance
        data.withUnsafeBufferPointer { dataPtr in
            array.dataPointer.withMemoryRebound(to: Float.self, capacity: dim) { arrayPtr in
                memcpy(arrayPtr, dataPtr.baseAddress!, dim * MemoryLayout<Float>.size)
            }
        }

        return array
    }

    /// Add two embedding MLMultiArrays element-wise using Accelerate.
    private static func addEmbeddings(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let count = a.count
        let result = try MLMultiArray(shape: a.shape, dataType: .float32)

        // Use vDSP for vectorized addition
        a.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { aPtr in
            b.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { bPtr in
                result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                    vDSP_vadd(aPtr, 1, bPtr, 1, resultPtr, 1, vDSP_Length(count))
                }
            }
        }

        return result
    }

    /// Convert MLMultiArray to float16, preserving stride layout.
    ///
    /// If already float16, returns as-is. CoreML models expect their own output
    /// stride layout, so we must not make non-contiguous arrays contiguous.
    private static func toFloat16(_ array: MLMultiArray) throws -> MLMultiArray {
        if array.dataType == .float16 { return array }
        let count = array.count
        let result = try MLMultiArray(shape: array.shape, dataType: .float16)
        for i in 0..<count {
            result[i] = array[i]
        }
        return result
    }

}
