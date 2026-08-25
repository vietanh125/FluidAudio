/// Pure dynamic programming algorithms for CTC keyword spotting.
///
/// Extracted from `CtcKeywordSpotter` so that the DP logic can be tested
/// independently of CoreML model loading. All methods are static and
/// take only primitive inputs (`[[Float]]` log-prob matrices and `[Int]`
/// token ID arrays).
///
/// Implements the CTC-WS dynamic program from NeMo's `ctc_word_spotter.py`
/// (arXiv:2406.07096). Unlike a naive token-only DP, this version operates
/// on the **blank-expanded symbol sequence** `[B, t1, B, t2, ..., tN, B]`
/// and accumulates blank emission log-probs along stay/within-token paths.
/// This is what makes the score probabilistically meaningful and what
/// correctly enforces a blank between repeated tokens.
///
/// Performance notes (parity-tested against the straightforward
/// implementation in `CtcDPAlgorithmParityTests`):
/// - Emission log-probs are gathered once per call into dense per-token
///   rows so the hot loop never touches the `[T, vocab]` matrix.
/// - The internal expanded-graph tables are flat buffers indexed manually
///   (`t * sLen + s`), removing array-of-array indirection from the O(T·N)
///   inner loop.
/// - `ctcWordSpotMultiple` first computes an exact score upper bound
///   (sum of per-token frame maxima; every other path term is ≤ 0) and
///   returns early when even that bound misses `minScore` — for broad
///   vocabulary lists this prunes almost every term.
enum CtcDPAlgorithm {

    /// Wildcard token ID: represents "*" that matches anything at zero cost.
    static let wildcardTokenId = ContextBiasingConstants.wildcardTokenId

    // MARK: - Expanded symbol helpers

    /// Per-state metadata for the blank-expanded sequence
    /// `[B, t1, B, t2, ..., tN, B]`, precomputed once per DP call.
    ///
    /// `emissionRowIndex` addresses the gathered emission rows:
    /// `-2` = blank row, `-1` = wildcard (zero cost), `>= 0` = index into
    /// the unique-token rows.
    private struct ExpandedGraph {
        let sLen: Int
        let emissionRowIndex: [Int]
        let isMatchState: [Bool]
        let canSkipBlank: [Bool]
        let uniqueTokenIds: [Int]
    }

    private static func buildExpandedGraph(_ keywordTokens: [Int]) -> ExpandedGraph {
        let n = keywordTokens.count
        let sLen = 2 * n + 1

        var uniqueTokenIds: [Int] = []
        var rowIndexById: [Int: Int] = [:]
        for id in keywordTokens where id != wildcardTokenId {
            if rowIndexById[id] == nil {
                rowIndexById[id] = uniqueTokenIds.count
                uniqueTokenIds.append(id)
            }
        }

        var emissionRowIndex = [Int](repeating: -2, count: sLen)
        var isMatchState = [Bool](repeating: false, count: sLen)
        for (i, id) in keywordTokens.enumerated() {
            let sIdx = 2 * i + 1
            isMatchState[sIdx] = true
            emissionRowIndex[sIdx] = id == wildcardTokenId ? -1 : rowIndexById[id]!
        }

        // Skip-blank from `s-2` is allowed only for a non-blank symbol distinct
        // from the one at `s-2` (wildcard ids compare like any other id).
        // Repeated tokens MUST pass through the blank — the CTC rule a naive
        // token-only DP violates.
        var canSkipBlank = [Bool](repeating: false, count: sLen)
        for i in 1..<keywordTokens.count {
            canSkipBlank[2 * i + 1] = keywordTokens[i] != keywordTokens[i - 1]
        }

        return ExpandedGraph(
            sLen: sLen,
            emissionRowIndex: emissionRowIndex,
            isMatchState: isMatchState,
            canSkipBlank: canSkipBlank,
            uniqueTokenIds: uniqueTokenIds
        )
    }

    /// Gather dense emission rows from the `[T, vocab]` matrix: one row for
    /// the blank symbol and one per unique keyword token. Out-of-vocabulary
    /// conventions match the previous per-cell lookup exactly: invalid blank
    /// ids emit `0`, invalid token ids emit `-Float.greatestFiniteMagnitude`.
    private static func gatherEmissionRows(
        logProbs: [[Float]],
        uniqueTokenIds: [Int],
        blankId: Int
    ) -> (blankRow: [Float], tokenRows: [Float]) {
        let frameCount = logProbs.count
        let vocabSize = logProbs.first?.count ?? 0
        let neg = -Float.greatestFiniteMagnitude

        var blankRow = [Float](repeating: 0, count: frameCount)
        if blankId >= 0 && blankId < vocabSize {
            for t in 0..<frameCount { blankRow[t] = logProbs[t][blankId] }
        }

        var tokenRows = [Float](repeating: neg, count: uniqueTokenIds.count * frameCount)
        for (r, id) in uniqueTokenIds.enumerated() where id >= 0 && id < vocabSize {
            let base = r * frameCount
            for t in 0..<frameCount { tokenRows[base + t] = logProbs[t][id] }
        }
        return (blankRow, tokenRows)
    }

    // MARK: - Core DP

    /// Core DP table construction shared by all CTC word spotting variants.
    ///
    /// The internal table is built on the blank-expanded symbol sequence of
    /// length `2N + 1`. Three transitions are evaluated per state:
    ///   - **stay** at `s`: adds `log p_t[symbol_s]` (blank emission cost
    ///     for stays in blank states; token emission for stays in token
    ///     states).
    ///   - **advance** from `s-1`: standard CTC step.
    ///   - **skip blank** from `s-2`: only when the symbols differ.
    ///
    /// The returned `dp[t][n]` is projected back to the public
    /// "n tokens consumed" view via
    /// `dp[t][n] = max(dpI[t][2n - 1], dpI[t][2n])`, i.e. the best of
    /// "ended on token n" or "ended on the blank after token n". Free start
    /// is preserved with `dp[t][0] = 0` for all `t`.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword (may include `wildcardTokenId`)
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: `(dp, backtrack, lastMatch)` with public `[T+1][N+1]` shape:
    ///   - `dp[t][n]` = best raw log-prob score for consuming the first `n`
    ///     tokens by frame `t` (sum of emission log-probs along the path,
    ///     **including** blank emissions).
    ///   - `backtrack[t][n]` = inferred keyword start frame (0-indexed) for
    ///     the best path ending at `dp[t][n]`.
    ///   - `lastMatch[t][n]` = frame at which the most recent non-blank
    ///     token was emitted along that path.
    ///
    /// > Note: Raw scores are **larger in magnitude** than a token-only DP
    /// > because blank emission costs are included. Callers using
    /// > token-count normalization (`/ N`) see systematically more negative
    /// > per-token averages; tune `defaultMinSpotterScore` and
    /// > `defaultMinVocabCtcScore` accordingly.
    static func fillDPTable(
        logProbs: [[Float]],
        keywordTokens: [Int],
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (dp: [[Float]], backtrack: [[Int]], lastMatch: [[Int]]) {
        let frameCount = logProbs.count
        let tokenCount = keywordTokens.count
        let neg = -Float.greatestFiniteMagnitude

        var dp = Array(repeating: Array(repeating: neg, count: tokenCount + 1), count: frameCount + 1)
        var backtrack = Array(repeating: Array(repeating: 0, count: tokenCount + 1), count: frameCount + 1)
        var lastMatch = Array(repeating: Array(repeating: 0, count: tokenCount + 1), count: frameCount + 1)

        // Free start: matching zero tokens has score 0 at any frame.
        for t in 0...frameCount { dp[t][0] = 0 }
        if tokenCount == 0 { return (dp, backtrack, lastMatch) }

        let graph = buildExpandedGraph(keywordTokens)
        let sLen = graph.sLen
        let (blankRow, tokenRows) = gatherEmissionRows(
            logProbs: logProbs, uniqueTokenIds: graph.uniqueTokenIds, blankId: blankId
        )

        // Flat internal tables on the expanded graph: index = t * sLen + s.
        var dpI = [Float](repeating: neg, count: (frameCount + 1) * sLen)
        var startI = [Int32](repeating: 0, count: (frameCount + 1) * sLen)
        var lastTokI = [Int32](repeating: 0, count: (frameCount + 1) * sLen)
        // s = 0 is the initial blank; free-start convention says any frame
        // can be the start of the keyword, so dp at the initial state is 0
        // and the candidate start frame is the current frame index.
        for t in 0...frameCount {
            dpI[t * sLen] = 0
            startI[t * sLen] = Int32(t)
        }

        graph.emissionRowIndex.withUnsafeBufferPointer { rowIdxBuf in
            graph.isMatchState.withUnsafeBufferPointer { matchBuf in
                graph.canSkipBlank.withUnsafeBufferPointer { skipBuf in
                    blankRow.withUnsafeBufferPointer { blankBuf in
                        tokenRows.withUnsafeBufferPointer { tokenBuf in
                            dpI.withUnsafeMutableBufferPointer { dpBuf in
                                startI.withUnsafeMutableBufferPointer { startBuf in
                                    lastTokI.withUnsafeMutableBufferPointer { lastBuf in
                                        runExpandedDP(
                                            frameCount: frameCount, sLen: sLen, neg: neg,
                                            rowIdx: rowIdxBuf, isMatch: matchBuf, canSkip: skipBuf,
                                            blankRow: blankBuf, tokenRows: tokenBuf,
                                            dpI: dpBuf, startI: startBuf, lastTokI: lastBuf
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Project the expanded states back to the public n-tokens-consumed view.
        for t in 0...frameCount {
            let base = t * sLen
            for n in 1...tokenCount {
                let sTok = 2 * n - 1
                let sBlank = 2 * n
                let scTok = sTok < sLen ? dpI[base + sTok] : neg
                let scBlank = sBlank < sLen ? dpI[base + sBlank] : neg
                if scTok >= scBlank {
                    dp[t][n] = scTok
                    backtrack[t][n] = Int(startI[base + sTok])
                    lastMatch[t][n] = Int(lastTokI[base + sTok])
                } else {
                    dp[t][n] = scBlank
                    backtrack[t][n] = Int(startI[base + sBlank])
                    lastMatch[t][n] = Int(lastTokI[base + sBlank])
                }
            }
        }

        return (dp, backtrack, lastMatch)
    }

    /// The O(T·sLen) inner recurrence over flat buffers. Semantics are
    /// identical to the straightforward nested-array implementation
    /// (see `CtcDPAlgorithmParityTests`); only the data layout differs.
    // swift-format-ignore: FunctionBodyLength
    private static func runExpandedDP(
        frameCount: Int, sLen: Int, neg: Float,
        rowIdx: UnsafeBufferPointer<Int>,
        isMatch: UnsafeBufferPointer<Bool>,
        canSkip: UnsafeBufferPointer<Bool>,
        blankRow: UnsafeBufferPointer<Float>,
        tokenRows: UnsafeBufferPointer<Float>,
        dpI: UnsafeMutableBufferPointer<Float>,
        startI: UnsafeMutableBufferPointer<Int32>,
        lastTokI: UnsafeMutableBufferPointer<Int32>
    ) {
        for t in 1...frameCount {
            let cur = t * sLen
            let prev = cur - sLen
            let frameIdx = t - 1
            for sIdx in 1..<sLen {
                let emissionRow = rowIdx[sIdx]
                let added: Float
                switch emissionRow {
                case -1: added = 0
                case -2: added = blankRow[frameIdx]
                default: added = tokenRows[emissionRow * frameCount + frameIdx]
                }

                let stay = dpI[prev + sIdx]
                let advance = dpI[prev + sIdx - 1]
                let skipBlank = canSkip[sIdx] ? dpI[prev + sIdx - 2] : neg

                var bestPred = stay
                var predKind = 0  // 0 = stay, 1 = advance, 2 = skip-blank
                if advance > bestPred {
                    bestPred = advance
                    predKind = 1
                }
                if skipBlank > bestPred {
                    bestPred = skipBlank
                    predKind = 2
                }

                if bestPred <= neg / 2 {
                    dpI[cur + sIdx] = neg
                    continue
                }

                dpI[cur + sIdx] = bestPred + added
                let isMatchFrame = isMatch[sIdx]

                switch predKind {
                case 0:
                    startI[cur + sIdx] = startI[prev + sIdx]
                    lastTokI[cur + sIdx] = isMatchFrame ? Int32(t) : lastTokI[prev + sIdx]
                case 1:
                    if sIdx == 1 {
                        // First non-blank symbol: keyword starts at this frame.
                        startI[cur + sIdx] = Int32(t - 1)
                    } else {
                        startI[cur + sIdx] = startI[prev + sIdx - 1]
                    }
                    lastTokI[cur + sIdx] = isMatchFrame ? Int32(t) : lastTokI[prev + sIdx - 1]
                default:
                    // Skip-blank: predecessor is at sIdx - 2. For sIdx = 2 we
                    // still inherit start from state 0, which records the
                    // candidate keyword-start frame.
                    startI[cur + sIdx] = startI[prev + sIdx - 2]
                    lastTokI[cur + sIdx] = isMatchFrame ? Int32(t) : lastTokI[prev + sIdx - 2]
                }
            }
        }
    }

    /// Count non-wildcard tokens for score normalization.
    static func nonWildcardCount(_ keywordTokens: [Int]) -> Int {
        keywordTokens.filter { $0 != wildcardTokenId }.count
    }

    /// Exact upper bound on the *normalized* score any alignment can reach.
    ///
    /// Every term the DP adds is a log-prob (≤ 0) except wildcard stays
    /// (= 0), and each token state contributes at least one emission. The
    /// sum of each token occurrence's best frame emission therefore bounds
    /// every path's raw score from above. A term whose bound misses the
    /// spotting threshold cannot produce a detection.
    static func normalizedScoreUpperBound(
        logProbs: [[Float]],
        keywordTokens: [Int],
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> Float {
        let frameCount = logProbs.count
        let vocabSize = logProbs.first?.count ?? 0
        guard frameCount > 0 else { return -Float.infinity }

        var maxById: [Int: Float] = [:]
        for id in Set(keywordTokens) where id != wildcardTokenId {
            guard id >= 0 && id < vocabSize else {
                maxById[id] = -Float.greatestFiniteMagnitude
                continue
            }
            var best = -Float.greatestFiniteMagnitude
            for t in 0..<frameCount where logProbs[t][id] > best {
                best = logProbs[t][id]
            }
            maxById[id] = best
        }

        var rawBound: Float = 0
        for id in keywordTokens where id != wildcardTokenId {
            rawBound += maxById[id] ?? -Float.greatestFiniteMagnitude
        }
        let normFactor = nonWildcardCount(keywordTokens)
        return normFactor > 0 ? rawBound / Float(normFactor) : rawBound
    }

    // MARK: - Word Spotting

    /// Constrained CTC word spotting within a temporal window.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword
    ///   - searchStartFrame: Start of search window (inclusive)
    ///   - searchEndFrame: End of search window (exclusive)
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: `(score, startFrame, endFrame)` in global frame coordinates.
    ///   `score` is normalized by the number of non-wildcard tokens — i.e.
    ///   the *per-token* average log-probability of the best alignment,
    ///   which includes blank-emission costs along stay paths.
    static func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int,
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        let T = logProbs.count
        let N = keywordTokens.count

        let clampedStart = max(0, searchStartFrame)
        let clampedEnd = min(T, searchEndFrame)

        if N == 0 || clampedEnd <= clampedStart {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let windowLogProbs = Array(logProbs[clampedStart..<clampedEnd])
        let windowT = windowLogProbs.count

        if windowT < N {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: windowLogProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        var bestEnd = 0
        var bestScore = -Float.greatestFiniteMagnitude

        for t in N...windowT {
            if dp[t][N] > bestScore {
                bestScore = dp[t][N]
                bestEnd = t
            }
        }

        let bestStart = backtrack[bestEnd][N]
        let actualEndFrame = lastMatch[bestEnd][N]

        let normFactor = nonWildcardCount(keywordTokens)
        let normalizedScore = normFactor > 0 ? bestScore / Float(normFactor) : bestScore

        let globalStart = clampedStart + bestStart
        let globalEnd = clampedStart + actualEndFrame

        return (normalizedScore, globalStart, globalEnd)
    }

    /// Find ALL occurrences of a keyword in the log-probabilities.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword
    ///   - minScore: Minimum normalized score threshold
    ///   - mergeOverlap: Whether to merge overlapping detections
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: Array of `(score, startFrame, endFrame)` tuples
    static func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = ContextBiasingConstants.defaultMinSpotterScore,
        mergeOverlap: Bool = true,
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 {
            return []
        }

        guard T >= N else { return [] }

        // Entry gate: skip the DP entirely when even the exact upper bound
        // on the normalized score cannot reach the threshold. For broad
        // vocabulary lists this prunes the vast majority of terms.
        let upperBound = normalizedScoreUpperBound(
            logProbs: logProbs, keywordTokens: keywordTokens, blankId: blankId
        )
        if upperBound < minScore {
            return []
        }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        let wildcardFreeCount = nonWildcardCount(keywordTokens)
        let normFactor = wildcardFreeCount > 0 ? Float(wildcardFreeCount) : 1.0

        var candidates: [(score: Float, startFrame: Int, endFrame: Int)] = []

        for t in N...T {
            let rawScore = dp[t][N]
            let normalizedScore = rawScore / normFactor

            let prevScore = t > N ? dp[t - 1][N] / normFactor : -Float.greatestFiniteMagnitude
            let nextScore = t < T ? dp[t + 1][N] / normFactor : -Float.greatestFiniteMagnitude

            let isLocalMax = normalizedScore >= prevScore && normalizedScore > nextScore
            let meetsThreshold = normalizedScore >= minScore

            if isLocalMax && meetsThreshold {
                let startFrame = backtrack[t][N]
                let actualEndFrame = lastMatch[t][N]
                candidates.append((score: normalizedScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        if candidates.isEmpty {
            var bestEnd = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for t in N...T {
                let normalizedScore = dp[t][N] / normFactor
                if normalizedScore > bestScore {
                    bestScore = normalizedScore
                    bestEnd = t
                }
            }
            if bestScore >= minScore {
                let startFrame = backtrack[bestEnd][N]
                let actualEndFrame = lastMatch[bestEnd][N]
                candidates.append((score: bestScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        guard mergeOverlap else { return candidates }

        let sorted = candidates.sorted { $0.startFrame < $1.startFrame }
        var merged: [(score: Float, startFrame: Int, endFrame: Int)] = []

        for candidate in sorted {
            if let last = merged.last {
                if candidate.startFrame <= last.endFrame {
                    var best = candidate.score > last.score ? candidate : last
                    best.endFrame = max(last.endFrame, candidate.endFrame)
                    merged[merged.count - 1] = best
                } else {
                    merged.append(candidate)
                }
            } else {
                merged.append(candidate)
            }
        }

        return merged
    }
}
