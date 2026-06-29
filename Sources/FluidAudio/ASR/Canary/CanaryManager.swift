@preconcurrency import CoreML
import Foundation

/// Single emitted decoder token with the chosen-token log-probability.
///
/// `piece` is the raw SentencePiece piece from the vocabulary (e.g. `"▁ramipril"`),
/// preserving the `▁` word-boundary marker. `logProb` is the natural log
/// probability of the chosen token under the (optionally boosted) logits at the
/// step that emitted it. Special / control tokens (`<|...|>`) are filtered out by
/// the caller (`transcribeRich`) — but boost-driven biasing operates on the raw
/// token stream before filtering, so `logProb` reflects the actual decode step.
public struct CanaryToken: Sendable {
    public let id: Int
    public let piece: String
    public let logProb: Float

    public init(id: Int, piece: String, logProb: Float) {
        self.id = id
        self.piece = piece
        self.logProb = logProb
    }
}

/// Rich transcription result: detokenized text plus per-emitted-token records.
///
/// `tokens` includes every non-special, non-EOS token in emission order (one per
/// decoder step that wasn't a control token), across all internal 15 s windows
/// after seam-merge deduplication. Use `tokens.map(\.logProb)` to compute
/// average / sequence confidence.
public struct CanaryTranscriptionResult: Sendable {
    public let text: String
    public let tokens: [CanaryToken]

    public init(text: String, tokens: [CanaryToken]) {
        self.text = text
        self.tokens = tokens
    }
}

/// Per-step logit modifier applied inside the autoregressive decode loop.
///
/// Called every step with the list of non-special tokens emitted so far (prompt
/// excluded; control tokens excluded). The returned `[Float]` of length
/// `CanaryConfig.vocabSize` is added directly to the projection logits before
/// `log_softmax` + `argmax`. Return `nil` to skip the step.
public typealias CanaryLogitBias = @Sendable (_ previousTokens: [Int]) -> [Float]?

/// Manager for NVIDIA Canary-1B-v2 transcription (attention encoder-decoder).
///
/// Pipeline: waveform → [Preprocessor fp32/CPU] mel → [Encoder int4/ANE] →
/// transpose to [1, T, D] → greedy autoregressive loop ([Decoder] → last hidden
/// → [Projection] → argmax until EOS) → SentencePiece detokenize.
///
/// The decoder carries no KV cache: each step re-runs the full `[1, 256]` token
/// sequence (matches the converted CoreML model). The 15 s window is fixed; audio
/// longer than 15 s is truncated (chunking is a future addition).
public actor CanaryManager {

    private let models: CanaryModels
    private let prompt: [Int32]
    private static let logger = AppLogger(category: "CanaryManager")

    public init(models: CanaryModels, prompt: [Int32] = CanaryConfig.promptEnTranscribePnc) {
        self.models = models
        self.prompt = prompt
    }

    /// Load models from the default cache (downloading if needed), then build a manager.
    public static func load(
        precision: CanaryPrecision = .int4,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CanaryManager {
        let models = try await CanaryModels.downloadAndLoad(precision: precision, progressHandler: progressHandler)
        return CanaryManager(models: models)
    }

    /// Transcribe a 16 kHz mono audio file.
    public func transcribe(audioURL: URL, logitBias: CanaryLogitBias? = nil) throws -> String {
        let converter = AudioConverter(sampleRate: Double(CanaryConfig.sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)
        return try transcribe(audio: samples, logitBias: logitBias)
    }

    /// Transcribe 16 kHz mono float samples (in [-1, 1]).
    ///
    /// Audio within the 15 s window is decoded in one pass. Longer audio is split
    /// into overlapping 15 s windows (hop = 15 s − `chunkOverlapSeconds`), decoded
    /// independently, and stitched at the seams via token-level
    /// longest-common-substring (`mergeTokenStreams`). No model change — each
    /// window still sees the fixed 15 s contract and the decoder is reset per window.
    ///
    /// `logitBias` is an optional per-step bias added to the projection logits
    /// before `argmax`. Used for word/keyword boosting; see `CanaryLogitBias`.
    public func transcribe(audio: [Float], logitBias: CanaryLogitBias? = nil) throws -> String {
        try transcribeRich(audio: audio, logitBias: logitBias).text
    }

    /// Rich variant: returns text + per-token records (id, piece, logProb).
    /// `logitBias` semantics match `transcribe(audio:logitBias:)`.
    public func transcribeRich(audio: [Float], logitBias: CanaryLogitBias? = nil)
        throws -> CanaryTranscriptionResult
    {
        let maxN = CanaryConfig.maxSamples
        var merged: [CanaryToken] = []
        if audio.count <= maxN {
            merged = try transcribeWindow(audio: audio, logitBias: logitBias)
        } else {
            let hop = maxN - CanaryConfig.chunkOverlapSamples
            var start = 0
            var chunkIndex = 0
            while start < audio.count {
                let end = min(start + maxN, audio.count)
                // Don't decode a final tail that is pure overlap — the previous window
                // already covered it.
                if chunkIndex > 0, (end - start) <= (maxN - hop) { break }

                let windowTokens = try transcribeWindow(
                    audio: Array(audio[start..<end]), logitBias: logitBias)
                merged = Self.mergeTokenStreamsRich(prefix: merged, suffix: windowTokens)

                chunkIndex += 1
                if end >= audio.count { break }
                start += hop
            }
        }

        let text = detokenize(merged.map(\.id))
        return CanaryTranscriptionResult(text: text, tokens: merged)
    }

    /// Run the 4-stage pipeline over a single ≤15 s window; returns generated
    /// rich tokens (prompt stripped, EOS excluded, control tokens excluded).
    private func transcribeWindow(audio: [Float], logitBias: CanaryLogitBias? = nil)
        throws -> [CanaryToken]
    {
        let (mel, melLength) = try runPreprocessor(audio: audio)
        let (encoder, encoderLength) = try runEncoder(mel: mel, melLength: melLength)
        let (embeddings, encoderMask) = try makeDecoderContext(encoder: encoder, encoderLength: encoderLength)
        return try greedyDecode(embeddings: embeddings, encoderMask: encoderMask, logitBias: logitBias)
    }

    /// Merge two adjacent window token streams using longest-common-substring.
    ///
    /// Both windows transcribe `chunkOverlapSeconds` of identical audio at their
    /// seam, so their token ids share a common substring near the prefix's tail /
    /// the suffix's head. Search a bounded window (`windowTokens` at the boundary)
    /// for the longest common substring of length ≥ `minMatch`. On a hit, drop the
    /// suffix's matched head so the seam is not duplicated; on a miss, concatenate
    /// plainly — better to duplicate a few tokens than to lose content.
    static func mergeTokenStreams(
        prefix: [Int],
        suffix: [Int],
        windowTokens: Int = 32,
        minMatch: Int = 4
    ) -> [Int] {
        if prefix.isEmpty { return suffix }
        if suffix.isEmpty { return prefix }

        let pTail = Array(prefix.suffix(windowTokens))
        let sHead = Array(suffix.prefix(windowTokens))
        let m = pTail.count
        let n = sHead.count
        if m == 0 || n == 0 { return prefix + suffix }

        // Classic LCS-substring DP (O(m·n), m,n ≤ windowTokens).
        var dp = [Int](repeating: 0, count: n + 1)
        var bestLen = 0
        var bestSEnd = 0  // index in sHead (exclusive) where the match ends
        for i in 1...m {
            var prev = 0
            for j in 1...n {
                let temp = dp[j]
                if pTail[i - 1] == sHead[j - 1] {
                    dp[j] = prev + 1
                    if dp[j] > bestLen {
                        bestLen = dp[j]
                        bestSEnd = j
                    }
                } else {
                    dp[j] = 0
                }
                prev = temp
            }
        }

        guard bestLen >= minMatch else { return prefix + suffix }
        return prefix + Array(suffix.dropFirst(bestSEnd))
    }

    // MARK: - Pipeline

    /// waveform → mel [1, 128, 1501]. Audio is padded/truncated to the fixed 15 s window.
    private func runPreprocessor(audio: [Float]) throws -> (MLMultiArray, MLMultiArray) {
        let maxN = CanaryConfig.maxSamples
        let validN = min(audio.count, maxN)
        if audio.count > maxN {
            Self.logger.warning("Audio \(audio.count) samples > 15 s window; truncating to \(maxN)")
        }

        let signal = try MLMultiArray(shape: [1, maxN as NSNumber], dataType: .float32)
        let sptr = signal.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sptr, 0, maxN * MemoryLayout<Float32>.size)
        audio.prefix(validN).withUnsafeBufferPointer { src in
            sptr.update(from: src.baseAddress!, count: validN)
        }

        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: validN)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: signal),
            "audio_length": MLFeatureValue(multiArray: length),
        ])
        let out = try models.preprocessor.prediction(from: input)
        guard let mel = out.featureValue(for: "processed")?.multiArrayValue,
            let melLen = out.featureValue(for: "processed_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Canary preprocessor produced no `processed`")
        }
        return (mel, melLen)
    }

    /// mel → encoder [1, 1024, 188].
    private func runEncoder(mel: MLMultiArray, melLength: MLMultiArray) throws -> (MLMultiArray, Int) {
        let featLen = try MLMultiArray(shape: [1], dataType: .int32)
        featLen[0] = NSNumber(value: melLength[0].intValue)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "features": MLFeatureValue(multiArray: mel),
            "features_length": MLFeatureValue(multiArray: featLen),
        ])
        let out = try models.encoder.prediction(from: input)
        guard let enc = out.featureValue(for: "encoder")?.multiArrayValue else {
            throw ASRError.processingFailed("Canary encoder produced no `encoder`")
        }
        let encLen = out.featureValue(for: "encoder_length")?.multiArrayValue?[0].intValue ?? CanaryConfig.encoderFrames
        return (enc, encLen)
    }

    /// encoder [1, D, T] → encoder_embeddings [1, T, D] + encoder_mask [1, T].
    ///
    /// CoreML pads the encoder's last dim to a 64-element boundary (T=188 →
    /// stride 192), so the transpose must use the array's real strides, not a
    /// dense linear read.
    private func makeDecoderContext(encoder: MLMultiArray, encoderLength: Int) throws -> (MLMultiArray, MLMultiArray) {
        let d = CanaryConfig.encoderHidden
        let t = CanaryConfig.encoderFrames
        let emb = try MLMultiArray(shape: [1, t as NSNumber, d as NSNumber], dataType: .float32)
        let eptr = emb.dataPointer.assumingMemoryBound(to: Float32.self)
        let strides = encoder.strides.map { $0.intValue }
        let sD = strides[1]
        let sT = strides[2]
        let read = floatReader(encoder)
        for ti in 0..<t {
            let dst = ti * d
            let tBase = ti * sT
            for di in 0..<d {
                eptr[dst + di] = read(di * sD + tBase)
            }
        }

        let mask = try MLMultiArray(shape: [1, t as NSNumber], dataType: .float32)
        let mptr = mask.dataPointer.assumingMemoryBound(to: Float32.self)
        let valid = min(max(encoderLength, 1), t)
        for i in 0..<t { mptr[i] = i < valid ? 1.0 : 0.0 }
        return (emb, mask)
    }

    /// Greedy autoregressive decode with optional per-step logit biasing.
    ///
    /// Returns `(id, piece, logProb)` per non-special non-EOS emitted token.
    /// `logProb` is the log-softmax value of the chosen token under the
    /// (optionally biased) logits at that step.
    ///
    /// Control/special tokens (`<|...|>`) are still emitted into the decoder's
    /// `input_ids` history (they're part of the canary prompt machinery) but
    /// are filtered from the returned list — and from the `previousTokens`
    /// passed to `logitBias` — so biasing operates on real content tokens.
    private func greedyDecode(
        embeddings: MLMultiArray, encoderMask: MLMultiArray, logitBias: CanaryLogitBias? = nil
    ) throws -> [CanaryToken] {
        // Use the decoder's actual sequence length (the exported `[1, S]` shape),
        // so a shorter decoder export (e.g. S=128) is picked up automatically.
        let s =
            models.decoder.modelDescription.inputDescriptionsByName["input_ids"]?
            .multiArrayConstraint?.shape.last?.intValue ?? CanaryConfig.maxDecoderSteps

        let inputIds = try MLMultiArray(shape: [1, s as NSNumber], dataType: .int32)
        let decoderMask = try MLMultiArray(shape: [1, s as NSNumber], dataType: .float32)
        let idptr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let mkptr = decoderMask.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<s {
            idptr[i] = 0
            mkptr[i] = 0
        }
        let promptLen = min(prompt.count, s)
        for i in 0..<promptLen {
            idptr[i] = prompt[i]
            mkptr[i] = 1
        }
        var pos = promptLen

        let hidden = try MLMultiArray(shape: [1, CanaryConfig.encoderHidden as NSNumber], dataType: .float32)
        let hptr = hidden.dataPointer.assumingMemoryBound(to: Float32.self)
        let d = CanaryConfig.encoderHidden

        // Scratch buffer for biased logits (only allocated if biasing is enabled).
        let vocab = CanaryConfig.vocabSize
        var biasedLogits: [Float] = logitBias != nil
            ? [Float](repeating: 0, count: vocab)
            : []
        // Content-only token history for the boost callback (control tokens stripped).
        var contentTokens: [Int] = []
        var generated: [CanaryToken] = []

        while pos < s {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "decoder_mask": MLFeatureValue(multiArray: decoderMask),
                "encoder_embeddings": MLFeatureValue(multiArray: embeddings),
                "encoder_mask": MLFeatureValue(multiArray: encoderMask),
            ])
            let out = try models.decoder.prediction(from: input)
            guard let dec = out.featureValue(for: "decoder")?.multiArrayValue else {
                throw ASRError.processingFailed("Canary decoder produced no `decoder`")
            }

            // hidden state at the last valid position (decoder output may be stride-padded)
            let decStrides = dec.strides.map { $0.intValue }
            let rowBase = (pos - 1) * decStrides[1]
            let elemStride = decStrides[2]
            let readDec = floatReader(dec)
            for h in 0..<d { hptr[h] = readDec(rowBase + h * elemStride) }

            let projInput = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden)
            ])
            let projOut = try models.projection.prediction(from: projInput)
            guard let logits = projOut.featureValue(for: "logits")?.multiArrayValue else {
                throw ASRError.processingFailed("Canary projection produced no `logits`")
            }

            // Materialise logits into a [Float] buffer (also handles fp16 → fp32).
            let n = min(logits.count, vocab)
            if biasedLogits.count != vocab {
                biasedLogits = [Float](repeating: 0, count: vocab)
            }
            let readLogit = floatReader(logits)
            for i in 0..<n { biasedLogits[i] = readLogit(i) }
            for i in n..<vocab { biasedLogits[i] = -Float.greatestFiniteMagnitude }

            // Apply optional per-step bias (decoded-content history only).
            if let bias = logitBias?(contentTokens), !bias.isEmpty {
                let m = min(bias.count, vocab)
                for i in 0..<m { biasedLogits[i] += bias[i] }
            }

            // argmax + log-softmax(chosen) on the biased logit vector.
            let (next, chosenLogProb) = Self.argmaxLogSoftmax(biasedLogits)
            if next == CanaryConfig.eosId { break }

            // Decoder history advances for ALL emitted tokens (incl. specials).
            idptr[pos] = Int32(next)
            mkptr[pos] = 1
            pos += 1

            // Filter specials/control tokens from the rich result and from
            // the history fed to the boost callback.
            let piece = models.tokenizer.piece(forId: next) ?? ""
            if !Self.isControlPiece(piece) {
                contentTokens.append(next)
                generated.append(CanaryToken(id: next, piece: piece, logProb: chosenLogProb))
            }
        }
        return generated
    }

    /// Argmax + log-softmax of the chosen index in a single pass.
    /// `log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))`.
    @inline(__always)
    private static func argmaxLogSoftmax(_ logits: [Float]) -> (Int, Float) {
        let n = logits.count
        if n == 0 { return (0, 0) }
        var maxIdx = 0
        var maxVal = logits[0]
        for i in 1..<n where logits[i] > maxVal {
            maxVal = logits[i]
            maxIdx = i
        }
        // logsumexp = max + log(sum exp(x_i - max))
        var sumExp: Float = 0
        for i in 0..<n { sumExp += expf(logits[i] - maxVal) }
        let lse = maxVal + logf(max(sumExp, .leastNormalMagnitude))
        let lp = logits[maxIdx] - lse
        return (maxIdx, lp)
    }

    /// SentencePiece pieces of the form `<|...|>` are decoder control tokens
    /// (start-of-transcript, language tags, pnc/notimestamp, etc). They get fed
    /// back into the decoder history but never carry content for boosting or
    /// for the rich-token output.
    @inline(__always)
    private static func isControlPiece(_ piece: String) -> Bool {
        piece.hasPrefix("<|") && piece.hasSuffix("|>")
    }

    /// Rich-stream variant of `mergeTokenStreams`. Runs the same LCS-based seam
    /// detector on the underlying token ids; on a hit, the suffix's matched head
    /// is dropped from the rich stream too.
    static func mergeTokenStreamsRich(
        prefix: [CanaryToken],
        suffix: [CanaryToken],
        windowTokens: Int = 32,
        minMatch: Int = 4
    ) -> [CanaryToken] {
        if prefix.isEmpty { return suffix }
        if suffix.isEmpty { return prefix }

        let pTail = Array(prefix.suffix(windowTokens)).map(\.id)
        let sHead = Array(suffix.prefix(windowTokens)).map(\.id)
        let m = pTail.count
        let n = sHead.count
        if m == 0 || n == 0 { return prefix + suffix }

        var dp = [Int](repeating: 0, count: n + 1)
        var bestLen = 0
        var bestSEnd = 0
        for i in 1...m {
            var prev = 0
            for j in 1...n {
                let temp = dp[j]
                if pTail[i - 1] == sHead[j - 1] {
                    dp[j] = prev + 1
                    if dp[j] > bestLen {
                        bestLen = dp[j]
                        bestSEnd = j
                    }
                } else {
                    dp[j] = 0
                }
                prev = temp
            }
        }
        guard bestLen >= minMatch else { return prefix + suffix }
        return prefix + Array(suffix.dropFirst(bestSEnd))
    }

    private func detokenize(_ tokens: [Int]) -> String {
        models.tokenizer.decode(ids: tokens)
            .replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    // MARK: - MLMultiArray helpers

    /// Returns a dtype-aware element reader for `arr` indexed by flat offset.
    /// The closure captures a pointer derived from `arr.dataPointer`; it is only
    /// valid while `arr` is alive (which it is for the duration of each use here).
    private func floatReader(_ arr: MLMultiArray) -> (Int) -> Float {
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float32.self)
            return { p[$0] }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            return { float16BitsToFloat(p[$0]) }
        default:
            return { arr[$0].floatValue }
        }
    }

    private func argmax(_ logits: MLMultiArray) -> Int {
        let n = logits.count
        var best = 0
        var bestVal = -Float.greatestFiniteMagnitude
        switch logits.dataType {
        case .float32:
            let p = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            for i in 0..<n where p[i] > bestVal {
                bestVal = p[i]
                best = i
            }
        case .float16:
            let p = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<n {
                let v = float16BitsToFloat(p[i])
                if v > bestVal {
                    bestVal = v
                    best = i
                }
            }
        default:
            for i in 0..<n {
                let v = logits[i].floatValue
                if v > bestVal {
                    bestVal = v
                    best = i
                }
            }
        }
        return best
    }
}

/// Decode an IEEE-754 half-precision bit pattern to Float (avoids a hard Float16 dependency).
@inline(__always)
private func float16BitsToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp = UInt32(h & 0x7C00) >> 10
    let mant = UInt32(h & 0x03FF)
    if exp == 0 {
        if mant == 0 { return Float(bitPattern: sign) }
        // subnormal
        var e: UInt32 = 127 - 15 + 1
        var m = mant
        while (m & 0x0400) == 0 {
            m <<= 1
            e -= 1
        }
        m &= 0x03FF
        return Float(bitPattern: sign | (e << 23) | (m << 13))
    }
    if exp == 0x1F {
        return Float(bitPattern: sign | 0x7F80_0000 | (mant << 13))
    }
    let e = exp - 15 + 127
    return Float(bitPattern: sign | (e << 23) | (mant << 13))
}
