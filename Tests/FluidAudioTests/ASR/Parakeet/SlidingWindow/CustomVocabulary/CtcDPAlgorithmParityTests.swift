import XCTest

@testable import FluidAudio

/// Parity tests for the optimized `CtcDPAlgorithm` (flat buffers, gathered
/// emission rows, upper-bound entry gating) against a verbatim copy of the
/// straightforward nested-array implementation it replaced. Both must be
/// bit-identical on every input: same float operations in the same order.
final class CtcDPAlgorithmParityTests: XCTestCase {

    // MARK: - Seeded RNG (SplitMix64) for reproducible fuzzing

    private struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private func randomLogProbs(
        frames: Int, vocab: Int, rng: inout SplitMix64
    ) -> [[Float]] {
        (0..<frames).map { _ in
            (0..<vocab).map { _ in Float.random(in: -12.0 ... -0.05, using: &rng) }
        }
    }

    private func randomTokens(
        count: Int, vocab: Int, rng: inout SplitMix64
    ) -> [Int] {
        (0..<count).map { _ in
            let roll = Int.random(in: 0..<20, using: &rng)
            if roll == 0 { return CtcDPAlgorithm.wildcardTokenId }
            if roll == 1 { return vocab + Int.random(in: 0..<5, using: &rng) }
            let id = Int.random(in: 0..<vocab, using: &rng)
            return id
        }
    }

    // MARK: - Parity: fillDPTable

    func testFillDPTableMatchesReferenceOnRandomInputs() {
        var rng = SplitMix64(state: 0xDEAD_BEEF)
        for round in 0..<60 {
            let frames = Int.random(in: 1...40, using: &rng)
            let vocab = Int.random(in: 8...50, using: &rng)
            let blankId = round % 7 == 0 ? vocab + 3 : vocab - 1
            let tokenCount = Int.random(in: 1...8, using: &rng)
            var tokens = randomTokens(count: tokenCount, vocab: vocab, rng: &rng)
            if round % 5 == 0 && tokens.count >= 2 {
                tokens[1] = tokens[0]  // force a repeated token (no skip-blank)
            }
            let logProbs = randomLogProbs(frames: frames, vocab: vocab, rng: &rng)

            let got = CtcDPAlgorithm.fillDPTable(
                logProbs: logProbs, keywordTokens: tokens, blankId: blankId)
            let want = ReferenceCtcDP.fillDPTable(
                logProbs: logProbs, keywordTokens: tokens, blankId: blankId)

            XCTAssertEqual(got.dp, want.dp, "dp mismatch in round \(round)")
            XCTAssertEqual(got.backtrack, want.backtrack, "backtrack mismatch in round \(round)")
            XCTAssertEqual(got.lastMatch, want.lastMatch, "lastMatch mismatch in round \(round)")
        }
    }

    // MARK: - Parity: ctcWordSpotMultiple (including the entry gate)

    func testSpotMultipleMatchesReferenceAcrossThresholds() {
        var rng = SplitMix64(state: 0x5EED_50DA)
        for round in 0..<40 {
            let frames = Int.random(in: 4...60, using: &rng)
            let vocab = 40
            let blankId = vocab - 1
            let tokens = randomTokens(
                count: Int.random(in: 1...6, using: &rng), vocab: vocab, rng: &rng)
            var logProbs = randomLogProbs(frames: frames, vocab: vocab, rng: &rng)
            // Plant a strong occurrence so some rounds detect above threshold.
            if tokens.count <= frames {
                for (i, id) in tokens.enumerated() where id >= 0 && id < vocab {
                    logProbs[i][id] = -0.01
                }
            }

            for minScore in [Float(-20), -8, -4, -1] {
                for mergeOverlap in [true, false] {
                    let got = CtcDPAlgorithm.ctcWordSpotMultiple(
                        logProbs: logProbs, keywordTokens: tokens,
                        minScore: minScore, mergeOverlap: mergeOverlap, blankId: blankId)
                    let want = ReferenceCtcDP.ctcWordSpotMultiple(
                        logProbs: logProbs, keywordTokens: tokens,
                        minScore: minScore, mergeOverlap: mergeOverlap, blankId: blankId)
                    XCTAssertEqual(got.count, want.count, "count mismatch r\(round) ms\(minScore)")
                    for (g, w) in zip(got, want) {
                        XCTAssertEqual(g.score, w.score, "score mismatch r\(round) ms\(minScore)")
                        XCTAssertEqual(g.startFrame, w.startFrame)
                        XCTAssertEqual(g.endFrame, w.endFrame)
                    }
                }
            }
        }
    }

    func testUpperBoundNeverPrunesAReachableDetection() {
        var rng = SplitMix64(state: 0xB00_57ED)
        for _ in 0..<200 {
            let frames = Int.random(in: 2...30, using: &rng)
            let vocab = 20
            let tokens = randomTokens(
                count: Int.random(in: 1...5, using: &rng), vocab: vocab, rng: &rng)
            let logProbs = randomLogProbs(frames: frames, vocab: vocab, rng: &rng)
            let bound = CtcDPAlgorithm.normalizedScoreUpperBound(
                logProbs: logProbs, keywordTokens: tokens, blankId: vocab - 1)
            let detections = CtcDPAlgorithm.ctcWordSpotMultiple(
                logProbs: logProbs, keywordTokens: tokens,
                minScore: -Float.greatestFiniteMagnitude / 4,
                mergeOverlap: false, blankId: vocab - 1)
            for detection in detections {
                XCTAssertLessThanOrEqual(
                    detection.score, bound,
                    "upper bound must dominate every achievable score")
            }
        }
    }
}

/// Verbatim copy of the pre-optimization `CtcDPAlgorithm` core (nested-array
/// tables, per-cell emission lookups, no gating) used as the parity oracle.
private enum ReferenceCtcDP {

    static let wildcardTokenId = CtcDPAlgorithm.wildcardTokenId

    private enum ExpandedSymbol {
        case blank
        case token(Int)
        case wildcard
    }

    private static func buildExpandedSequence(_ keywordTokens: [Int]) -> [ExpandedSymbol] {
        var s: [ExpandedSymbol] = []
        s.reserveCapacity(2 * keywordTokens.count + 1)
        for id in keywordTokens {
            s.append(.blank)
            s.append(id == wildcardTokenId ? .wildcard : .token(id))
        }
        s.append(.blank)
        return s
    }

    private static func emissionLogProb(
        symbol: ExpandedSymbol, frame: [Float], blankId: Int
    ) -> Float {
        switch symbol {
        case .blank:
            return blankId >= 0 && blankId < frame.count ? frame[blankId] : 0
        case .token(let id):
            return id >= 0 && id < frame.count ? frame[id] : -Float.greatestFiniteMagnitude
        case .wildcard:
            return 0
        }
    }

    private static func canSkipBlank(_ s: [ExpandedSymbol], at idx: Int) -> Bool {
        guard idx >= 2 else { return false }
        switch s[idx] {
        case .blank:
            return false
        case .token(let cur):
            if case .token(let prev) = s[idx - 2], prev == cur { return false }
            return true
        case .wildcard:
            if case .wildcard = s[idx - 2] { return false }
            return true
        }
    }

    static func fillDPTable(
        logProbs: [[Float]], keywordTokens: [Int], blankId: Int
    ) -> (dp: [[Float]], backtrack: [[Int]], lastMatch: [[Int]]) {
        let T = logProbs.count
        let N = keywordTokens.count
        let neg = -Float.greatestFiniteMagnitude

        var dp = Array(repeating: Array(repeating: neg, count: N + 1), count: T + 1)
        var backtrack = Array(repeating: Array(repeating: 0, count: N + 1), count: T + 1)
        var lastMatch = Array(repeating: Array(repeating: 0, count: N + 1), count: T + 1)

        for t in 0...T { dp[t][0] = 0 }
        if N == 0 { return (dp, backtrack, lastMatch) }

        let s = buildExpandedSequence(keywordTokens)
        let sLen = s.count

        var dpI = Array(repeating: Array(repeating: neg, count: sLen), count: T + 1)
        var startI = Array(repeating: Array(repeating: 0, count: sLen), count: T + 1)
        var lastTokI = Array(repeating: Array(repeating: 0, count: sLen), count: T + 1)
        for t in 0...T {
            dpI[t][0] = 0
            startI[t][0] = t
        }

        for t in 1...T {
            let frame = logProbs[t - 1]
            for sIdx in 1..<sLen {
                let sym = s[sIdx]
                let emitLogProb = emissionLogProb(symbol: sym, frame: frame, blankId: blankId)
                let isWildcard: Bool = { if case .wildcard = sym { return true } else { return false } }()
                let isToken: Bool = { if case .token = sym { return true } else { return false } }()
                let added: Float = isWildcard ? 0 : emitLogProb

                let stay = dpI[t - 1][sIdx]
                let advance = dpI[t - 1][sIdx - 1]
                let skipBlank = canSkipBlank(s, at: sIdx) ? dpI[t - 1][sIdx - 2] : neg

                var bestPred = stay
                var predKind = 0
                if advance > bestPred {
                    bestPred = advance
                    predKind = 1
                }
                if skipBlank > bestPred {
                    bestPred = skipBlank
                    predKind = 2
                }

                if bestPred <= neg / 2 {
                    dpI[t][sIdx] = neg
                    continue
                }

                dpI[t][sIdx] = bestPred + added
                let isMatchFrame = isToken || isWildcard

                switch predKind {
                case 0:
                    startI[t][sIdx] = startI[t - 1][sIdx]
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx]
                case 1:
                    if sIdx == 1 {
                        startI[t][sIdx] = t - 1
                    } else {
                        startI[t][sIdx] = startI[t - 1][sIdx - 1]
                    }
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx - 1]
                default:
                    startI[t][sIdx] = startI[t - 1][sIdx - 2]
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx - 2]
                }
            }
        }

        for t in 0...T {
            for n in 1...N {
                let sTok = 2 * n - 1
                let sBlank = 2 * n
                let scTok = sTok < sLen ? dpI[t][sTok] : neg
                let scBlank = sBlank < sLen ? dpI[t][sBlank] : neg
                if scTok >= scBlank {
                    dp[t][n] = scTok
                    backtrack[t][n] = startI[t][sTok]
                    lastMatch[t][n] = lastTokI[t][sTok]
                } else {
                    dp[t][n] = scBlank
                    backtrack[t][n] = startI[t][sBlank]
                    lastMatch[t][n] = lastTokI[t][sBlank]
                }
            }
        }

        return (dp, backtrack, lastMatch)
    }

    static func nonWildcardCount(_ keywordTokens: [Int]) -> Int {
        keywordTokens.filter { $0 != wildcardTokenId }.count
    }

    static func ctcWordSpotMultiple(
        logProbs: [[Float]], keywordTokens: [Int], minScore: Float,
        mergeOverlap: Bool, blankId: Int
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 { return [] }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: logProbs, keywordTokens: keywordTokens, blankId: blankId)

        let wildcardFreeCount = nonWildcardCount(keywordTokens)
        let normFactor = wildcardFreeCount > 0 ? Float(wildcardFreeCount) : 1.0

        var candidates: [(score: Float, startFrame: Int, endFrame: Int)] = []

        guard T >= N else { return [] }

        for t in N...T {
            let rawScore = dp[t][N]
            let normalizedScore = rawScore / normFactor

            let prevScore = t > N ? dp[t - 1][N] / normFactor : -Float.greatestFiniteMagnitude
            let nextScore = t < T ? dp[t + 1][N] / normFactor : -Float.greatestFiniteMagnitude

            let isLocalMax = normalizedScore >= prevScore && normalizedScore > nextScore
            let meetsThreshold = normalizedScore >= minScore

            if isLocalMax && meetsThreshold {
                candidates.append(
                    (score: normalizedScore, startFrame: backtrack[t][N], endFrame: lastMatch[t][N]))
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
                candidates.append(
                    (
                        score: bestScore, startFrame: backtrack[bestEnd][N],
                        endFrame: lastMatch[bestEnd][N]
                    ))
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
