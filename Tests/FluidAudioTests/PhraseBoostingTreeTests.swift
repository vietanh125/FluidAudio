import XCTest

@testable import FluidAudio

/// Validates the GPU-PB port against the golden vectors from NeMo's
/// `tests/collections/asr/test_boosting_tree.py` (phrases "abc"/"abd"/"c" as
/// token ids [[1,2,3],[1,2,4],[3]], context_score=1.0, depth_scaling=1.0,
/// vocab_size=5), plus the stateless-state equivalence the Swift port relies on.
final class PhraseBoostingTreeTests: XCTestCase {

    private func makeNeMoFixtureTree(finalEosScore: Float = 0.0) -> PhraseBoostingTree {
        PhraseBoostingTree(
            terms: [
                (tokens: [1, 2, 3], word: "abc"),
                (tokens: [1, 2, 4], word: "abd"),
                (tokens: [3], word: "c"),
            ],
            vocabSize: 5,
            config: PhraseBoostingTreeConfig(
                contextScore: 1.0, depthScaling: 1.0, unkScore: 0.0,
                finalEosScore: finalEosScore, alpha: 1.0
            )
        )
    }

    /// NeMo `test_building_context_graph`: node scores with depth scaling.
    func testNodeScoresMatchNeMo() {
        let tree = makeNeMoFixtureTree()
        // Walk states by transition: root -[1]-> a -[2]-> ab -[3]-> abc
        let a = tree.transition(from: 0, token: 1)
        let ab = tree.transition(from: a, token: 2)
        let abc = tree.transition(from: ab, token: 3)
        let abd = tree.transition(from: ab, token: 4)
        let c = tree.transition(from: 0, token: 3)

        // Arc scores: depth1 = 1.0; depth2 = 1*1 + ln2; depth3 = 1*1 + ln3.
        // node_score(abc) = 1 + (1+ln2) + (1+ln3) = 4.7918 (NeMo asserts 4.79)
        let abcScores = tree.scoreSentence([1, 2, 3])
        XCTAssertEqual(abcScores.reduce(0, +), 4.7918, accuracy: 1e-3)
        XCTAssertEqual(tree.scoreSentence([3])[0], 1.0, accuracy: 1e-6)

        XCTAssertTrue(tree.isFinal(state: abc))
        XCTAssertTrue(tree.isFinal(state: abd))
        XCTAssertTrue(tree.isFinal(state: c))
        XCTAssertFalse(tree.isFinal(state: a))
        XCTAssertFalse(tree.isFinal(state: ab))
        // Fail link of abc is the "c" node: from abc, token 3 -> c's continuation
        // behavior is encoded in transitions; abd fails to root.
    }

    /// NeMo `test_boosting_tree_inference`: exact per-step scores, including
    /// backoff retraction (-1.6931 when abandoning "ab" for "a").
    func testSentenceScoresMatchNeMoGoldenVectors() {
        let tree = makeNeMoFixtureTree()

        let expectations: [(sentence: [Int], scores: [Float])] = [
            ([1, 2, 3, 2, 1], [1.0000, 1.6931, 2.0986, 0.0000, 1.0000]),  // 'abcba'
            ([2, 2, 1, 2, 4], [0.0000, 0.0000, 1.0000, 1.6931, 2.0986]),  // 'bbabd'
            ([3, 1, 2, 1], [1.0000, 1.0000, 1.6931, -1.6931]),  // 'caba' (retraction)
            ([], []),
        ]

        for (sentence, expected) in expectations {
            let got = tree.scoreSentence(sentence)
            XCTAssertEqual(got.count, expected.count)
            for (g, e) in zip(got, expected) {
                XCTAssertEqual(g, e, accuracy: 1e-4, "sentence \(sentence)")
            }
        }
    }

    /// NeMo `test_eos_handling`: eos score = clamp(max boost, 0) + final weight.
    func testEosHandlingMatchesNeMo() {
        let tree = makeNeMoFixtureTree(finalEosScore: 1.0)

        // State "a" (history [1]): max boost is the "ab" continuation 1.6931,
        // not a final state -> eos = 1.6931 + 0.
        let scoresA = tree.boostLogprobs(previousTokens: [1][...], eosId: 0)
        XCTAssertEqual(scoresA[0], 1.6931, accuracy: 1e-3)

        // State "c" (history [3]): max boost is 1.0 (phrase starts from root),
        // final state -> eos = 1.0 + 1.0 = 2.0.
        let scoresC = tree.boostLogprobs(previousTokens: [3][...], eosId: 0)
        XCTAssertEqual(scoresC[0], 2.0, accuracy: 1e-4)
    }

    /// The completed-phrase rule: leaving a final node costs nothing
    /// (backoff weight 0), so the earned boost is kept.
    func testNoRetractionAfterCompletedPhrase() {
        let tree = makeNeMoFixtureTree()
        // After 'abc' (final node), emitting 'b' (token 2) must score 0
        // (root unk), NOT -(node score).
        let scores = tree.scoreSentence([1, 2, 3, 2])
        XCTAssertEqual(scores[3], 0.0, accuracy: 1e-4)
    }

    /// Stateless-by-history equivalence: folding transitions over the full
    /// history equals recovering the state from only the last maxDepth tokens.
    func testStatelessStateEqualsStreamingState() {
        let tree = PhraseBoostingTree(
            terms: [
                (tokens: [1, 2, 3], word: "abc"),
                (tokens: [2, 3, 4, 5], word: "wxyz"),
                (tokens: [3], word: "c"),
                (tokens: [5, 1], word: "ea"),
            ],
            vocabSize: 8,
            config: PhraseBoostingTreeConfig(contextScore: 1.0, depthScaling: 2.0)
        )

        // Deterministic pseudo-random token stream over the small vocab.
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func nextToken() -> Int {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Int((seed >> 33) % 8)
        }

        var history: [Int] = []
        var streamingState = 0
        for _ in 0..<500 {
            let tok = nextToken()
            history.append(tok)
            streamingState = tree.transition(from: streamingState, token: tok)
            let statelessState = tree.state(after: history[...])
            XCTAssertEqual(statelessState, streamingState, "diverged after \(history.count) tokens")
            // Score vectors must therefore agree too.
            let a = tree.rawScores(state: streamingState)
            let b = tree.rawScores(state: statelessState)
            XCTAssertEqual(a, b)
        }
    }

    /// depth_scaling=2 formula: arc into depth-d node = c0*beta + ln(d).
    func testDepthScalingFormula() {
        let tree = PhraseBoostingTree(
            terms: [(tokens: [1, 2, 3], word: "abc")],
            vocabSize: 5,
            config: PhraseBoostingTreeConfig(contextScore: 1.0, depthScaling: 2.0)
        )
        let scores = tree.scoreSentence([1, 2, 3])
        XCTAssertEqual(scores[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(scores[1], 2.0 + Foundation.log(Float(2)), accuracy: 1e-5)
        XCTAssertEqual(scores[2], 2.0 + Foundation.log(Float(3)), accuracy: 1e-5)
    }

    /// unk_score applies to tokens that start no phrase, and alpha scales
    /// the whole vector.
    func testUnkScoreAndAlpha() {
        let tree = PhraseBoostingTree(
            terms: [(tokens: [1, 2], word: "ab")],
            vocabSize: 4,
            config: PhraseBoostingTreeConfig(
                contextScore: 1.0, depthScaling: 1.0, unkScore: 0.5, alpha: 2.0)
        )
        let scores = tree.boostLogprobs(previousTokens: [][...])
        XCTAssertEqual(scores[1], 2.0, accuracy: 1e-6)  // alpha * c0
        XCTAssertEqual(scores[0], 1.0, accuracy: 1e-6)  // alpha * unk
        XCTAssertEqual(scores[3], 1.0, accuracy: 1e-6)
    }

    /// GPU-PB boosters flag the two-stage blank requirement; heuristic ones don't.
    func testBoosterPreservesBlankCategoryFlag() {
        let gpuPB = ParakeetBooster.gpuPB(
            tokenMap: ["ab": [1, 2]], vocabSize: 8193,
            config: PhraseBoostingTreeConfig())
        XCTAssertTrue(gpuPB.preservesBlankCategory)
        XCTAssertEqual(gpuPB.wordCount, 1)
        XCTAssertEqual(gpuPB.vocabSize, 8193)

        let heuristic = ParakeetBooster.fromTokenMap(["ab": [1, 2]], vocabSize: 8193)
        XCTAssertFalse(heuristic.preservesBlankCategory)
        XCTAssertEqual(heuristic.wordCount, 1)
        XCTAssertEqual(heuristic.vocabSize, 8193)
    }

    /// Shared-prefix phrases share nodes; scores on shared arcs are stable.
    func testSharedPrefixMaxScore() {
        let tree = PhraseBoostingTree(
            terms: [
                (tokens: [1, 2, 3], word: "abc"),
                (tokens: [1, 2], word: "ab"),
            ],
            vocabSize: 5,
            config: PhraseBoostingTreeConfig(contextScore: 1.0, depthScaling: 1.0)
        )
        // "ab" node is final (phrase "ab") AND on the "abc" path.
        let ab = tree.state(after: [1, 2][...])
        XCTAssertTrue(tree.isFinal(state: ab))
        // Continuing to "abc" still scores the depth-3 arc.
        let scores = tree.rawScores(state: ab)
        XCTAssertEqual(scores[3], 1.0 + Foundation.log(Float(3)), accuracy: 1e-5)
        // Abandoning after final "ab" costs nothing.
        XCTAssertEqual(scores[0], 0.0, accuracy: 1e-6)
    }
}
