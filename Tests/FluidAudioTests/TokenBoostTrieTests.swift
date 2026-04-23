import XCTest

@testable import FluidAudio

final class TokenBoostTrieTests: XCTestCase {

    private let vocabSize = 100

    func testFirstTokenBaseBoost() {
        let trie = TokenBoostTrie(
            terms: [(tokens: [10, 20, 30], word: "alpha")],
            vocabSize: vocabSize
        )

        let boost = trie.boostLogprobs(
            previousTokens: ArraySlice<Int>(),
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 10
        )

        XCTAssertEqual(boost.count, vocabSize)
        XCTAssertEqual(boost[10], 0.5, accuracy: 1e-6, "first token gets baseBoost")
        XCTAssertEqual(boost[20], 0.0, "non-first token gets no bonus without context")
        XCTAssertEqual(boost[0], 0.0)
    }

    func testPrefixExtensionMatchesLookback1() {
        let trie = TokenBoostTrie(
            terms: [(tokens: [10, 20, 30], word: "alpha")],
            vocabSize: vocabSize
        )

        // Emitted last token = 10 → token 20 should now extend the trie.
        let boost = trie.boostLogprobs(
            previousTokens: [10][...],
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 10
        )

        // lookback=1 → strength = 3.0 * (1 + 1*0.25) = 3.75
        XCTAssertEqual(boost[20], 3.75, accuracy: 1e-6)
        // 10 is still a first-token so it keeps baseBoost
        XCTAssertEqual(boost[10], 0.5, accuracy: 1e-6)
    }

    func testLongerLookbackWinsViaMax() {
        // Two terms sharing a prefix; longer match should give higher bonus.
        let trie = TokenBoostTrie(
            terms: [
                (tokens: [10, 20, 30], word: "alpha"),
                (tokens: [10, 20, 40], word: "beta"),
            ],
            vocabSize: vocabSize
        )

        // After emitting [10, 20] both 30 and 40 are valid continuations at
        // lookback=2 (strength 3.0 * 1.5 = 4.5). They are NOT continuations at
        // lookback=1 via [20] alone because [20] isn't a live trie prefix.
        let boost = trie.boostLogprobs(
            previousTokens: [10, 20][...],
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 10
        )

        XCTAssertEqual(boost[30], 4.5, accuracy: 1e-6)
        XCTAssertEqual(boost[40], 4.5, accuracy: 1e-6)
    }

    func testUnrelatedPreviousTokensDoNotTriggerBonuses() {
        let trie = TokenBoostTrie(
            terms: [(tokens: [10, 20, 30], word: "alpha")],
            vocabSize: vocabSize
        )

        let boost = trie.boostLogprobs(
            previousTokens: [99, 98][...],
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 10
        )

        // Only firstTokens slot (10) is non-zero.
        XCTAssertEqual(boost[10], 0.5, accuracy: 1e-6)
        for i in 0..<vocabSize where i != 10 {
            XCTAssertEqual(boost[i], 0.0, "index \(i) should be zero")
        }
    }

    func testMaxPrefixLenBoundsLookback() {
        let trie = TokenBoostTrie(
            terms: [(tokens: [1, 2, 3, 4, 5], word: "term")],
            vocabSize: vocabSize
        )

        // With maxPrefixLen=2, emitting [1,2,3,4] means lookback 1,2 only.
        // Lookback 1 = [4] — not in trie. Lookback 2 = [3,4] — not in trie.
        // So no sequence bonus, only firstToken.
        let tight = trie.boostLogprobs(
            previousTokens: [1, 2, 3, 4][...],
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 2
        )
        XCTAssertEqual(tight[5], 0.0, accuracy: 1e-6)

        // With maxPrefixLen=4 the walk [1,2,3,4] matches; 5 gets a sequence bonus.
        let wide = trie.boostLogprobs(
            previousTokens: [1, 2, 3, 4][...],
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 4
        )
        // strength = 3.0 * (1 + 4*0.25) = 6.0
        XCTAssertEqual(wide[5], 6.0, accuracy: 1e-6)
    }

    func testOutOfVocabTokensAreDroppedSafely() {
        // A term whose tokens reference ids >= vocabSize should not crash or
        // write out of bounds.
        let trie = TokenBoostTrie(
            terms: [(tokens: [10, 200, 30], word: "weird")],
            vocabSize: vocabSize
        )
        let boost = trie.boostLogprobs(
            previousTokens: [10][...],
            baseBoost: 0.5,
            sequenceBoost: 3.0,
            maxPrefixLen: 10
        )
        XCTAssertEqual(boost.count, vocabSize)
        // Token 200 is the continuation of 10 but is >= vocabSize, so it's
        // silently skipped. firstToken 10 still gets baseBoost.
        XCTAssertEqual(boost[10], 0.5, accuracy: 1e-6)
    }

    func testEmptyTermsYieldZeroVector() {
        let trie = TokenBoostTrie(terms: [], vocabSize: vocabSize)
        let boost = trie.boostLogprobs(previousTokens: [1, 2, 3][...])
        XCTAssertEqual(boost.count, vocabSize)
        XCTAssertEqual(boost.reduce(0, +), 0, accuracy: 1e-6)
        XCTAssertEqual(trie.wordCount, 0)
        XCTAssertEqual(trie.firstTokenCount, 0)
    }
}
