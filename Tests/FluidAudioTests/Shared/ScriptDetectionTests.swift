import XCTest
@testable import FluidAudio

final class ScriptDetectionTests: XCTestCase {

    // MARK: - Script Property Tests

    func testLatinScriptLanguages() {
        let latinLanguages: [Language] = [
            .english, .polish, .spanish, .french, .german, .italian, .portuguese
        ]

        for language in latinLanguages {
            XCTAssertEqual(
                language.script, .latin,
                "\(language.rawValue) should use Latin script")
        }
    }

    func testCyrillicScriptLanguages() {
        let cyrillicLanguages: [Language] = [
            .russian, .ukrainian, .belarusian, .bulgarian, .serbian
        ]

        for language in cyrillicLanguages {
            XCTAssertEqual(
                language.script, .cyrillic,
                "\(language.rawValue) should use Cyrillic script")
        }
    }

    // MARK: - Basic Script Matching Tests

    func testMatchesLatinText() {
        XCTAssertTrue(ScriptDetection.matches("hello", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("world", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("Hello World!", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("123 abc", script: .latin))
    }

    func testMatchesCyrillicText() {
        XCTAssertTrue(ScriptDetection.matches("привет", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("мир", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("Привет мир!", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("123 абв", script: .cyrillic))
    }

    func testDoesNotMatchMixedScripts() {
        XCTAssertFalse(ScriptDetection.matches("hello мир", script: .latin))
        XCTAssertFalse(ScriptDetection.matches("hello мир", script: .cyrillic))
        XCTAssertFalse(ScriptDetection.matches("привет world", script: .latin))
        XCTAssertFalse(ScriptDetection.matches("привет world", script: .cyrillic))
    }

    // MARK: - SentencePiece Boundary Marker Tests

    func testStripsSentencePieceBoundaryMarker() {
        // U+2581 (LOWER ONE EIGHTH BLOCK) is SentencePiece word boundary marker
        XCTAssertTrue(ScriptDetection.matches("\u{2581}hello", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("\u{2581}world", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("\u{2581}привет", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("\u{2581}мир", script: .cyrillic))
    }

    func testMultipleBoundaryMarkers() {
        XCTAssertTrue(ScriptDetection.matches("\u{2581}\u{2581}hello", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("\u{2581}\u{2581}привет", script: .cyrillic))
    }

    func testBoundaryMarkerOnly() {
        // Boundary marker alone should return false (empty after stripping)
        XCTAssertFalse(ScriptDetection.matches("\u{2581}", script: .latin))
        XCTAssertFalse(ScriptDetection.matches("\u{2581}", script: .cyrillic))
        XCTAssertFalse(ScriptDetection.matches("\u{2581}\u{2581}", script: .latin))
    }

    // MARK: - Polish Language Tests (Issue #512)

    func testPolishLatinCharacters() {
        // Polish uses Latin Extended-A for special characters
        XCTAssertTrue(ScriptDetection.matches("ą", script: .latin))  // U+0105
        XCTAssertTrue(ScriptDetection.matches("ć", script: .latin))  // U+0107
        XCTAssertTrue(ScriptDetection.matches("ę", script: .latin))  // U+0119
        XCTAssertTrue(ScriptDetection.matches("ł", script: .latin))  // U+0142
        XCTAssertTrue(ScriptDetection.matches("ń", script: .latin))  // U+0144
        XCTAssertTrue(ScriptDetection.matches("ó", script: .latin))  // U+00F3
        XCTAssertTrue(ScriptDetection.matches("ś", script: .latin))  // U+015B
        XCTAssertTrue(ScriptDetection.matches("ź", script: .latin))  // U+017A
        XCTAssertTrue(ScriptDetection.matches("ż", script: .latin))  // U+017C
    }

    func testPolishWords() {
        XCTAssertTrue(ScriptDetection.matches("cześć", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("świat", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("Polska", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("zażółć", script: .latin))
    }

    func testPolishWordsWithBoundaryMarker() {
        XCTAssertTrue(ScriptDetection.matches("\u{2581}cześć", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("\u{2581}świat", script: .latin))
    }

    func testRejectsPolishTextAsCyrillic() {
        XCTAssertFalse(ScriptDetection.matches("cześć", script: .cyrillic))
        XCTAssertFalse(ScriptDetection.matches("świat", script: .cyrillic))
    }

    // MARK: - Punctuation and Special Characters

    func testPunctuationWithLatin() {
        XCTAssertTrue(ScriptDetection.matches("hello!", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("world?", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("test.", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("hello, world!", script: .latin))
    }

    func testPunctuationWithCyrillic() {
        XCTAssertTrue(ScriptDetection.matches("привет!", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("мир?", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("тест.", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("привет, мир!", script: .cyrillic))
    }

    func testSpacesAndWhitespace() {
        XCTAssertTrue(ScriptDetection.matches("hello world", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("  hello  ", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("привет мир", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("  привет  ", script: .cyrillic))
    }

    // MARK: - Edge Cases

    func testEmptyString() {
        XCTAssertFalse(ScriptDetection.matches("", script: .latin))
        XCTAssertFalse(ScriptDetection.matches("", script: .cyrillic))
    }

    func testWhitespaceOnly() {
        XCTAssertTrue(ScriptDetection.matches(" ", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("   ", script: .latin))
        XCTAssertTrue(ScriptDetection.matches(" ", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("   ", script: .cyrillic))
    }

    func testNumbers() {
        XCTAssertTrue(ScriptDetection.matches("123", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("123", script: .cyrillic))
        XCTAssertTrue(ScriptDetection.matches("456 789", script: .latin))
        XCTAssertTrue(ScriptDetection.matches("456 789", script: .cyrillic))
    }

    // MARK: - Filter Top-K Tests

    func testFilterTopKReturnsFirstMatchingToken() {
        let topKIds = [1, 2, 3, 4]
        let topKLogits: [Float] = [0.9, 0.7, 0.5, 0.3]
        let vocabulary = [
            1: "привет",      // Cyrillic
            2: "hello",       // Latin
            3: "мир",         // Cyrillic
            4: "world",       // Latin
        ]

        // Should return first Latin match (ID=2, "hello")
        let result = ScriptDetection.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)
        if let logit = result?.logit {
            XCTAssertEqual(logit, 0.7, accuracy: Float(0.001))
        }
    }

    func testFilterTopKWithSentencePieceBoundaryMarker() {
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.7, 0.5]
        let vocabulary = [
            1: "\u{2581}привет",  // Cyrillic with boundary marker
            2: "\u{2581}hello",   // Latin with boundary marker
            3: "\u{2581}мир",     // Cyrillic with boundary marker
        ]

        let result = ScriptDetection.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)
        if let logit = result?.logit {
            XCTAssertEqual(logit, 0.7, accuracy: Float(0.001))
        }
    }

    func testFilterTopKReturnsNilWhenNoMatch() {
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.7, 0.5]
        let vocabulary = [
            1: "привет",
            2: "мир",
            3: "тест",
        ]

        // All tokens are Cyrillic, should return nil for Latin
        let result = ScriptDetection.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNil(result)
    }

    func testFilterTopKSkipsMissingVocabularyEntries() {
        let topKIds = [1, 2, 3, 4]
        let topKLogits: [Float] = [0.9, 0.7, 0.5, 0.3]
        let vocabulary = [
            1: "привет",
            // 2 is missing
            3: "мир",
            4: "world",  // Latin
        ]

        // Should skip missing ID=2 and return ID=4 ("world")
        let result = ScriptDetection.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 4)
        if let logit = result?.logit {
            XCTAssertEqual(logit, 0.3, accuracy: Float(0.001))
        }
    }

    func testFilterTopKEmptyArrays() {
        let result = ScriptDetection.filterTopK(
            topKIds: [],
            topKLogits: [],
            vocabulary: [:],
            preferredScript: .latin
        )

        XCTAssertNil(result)
    }

    func testFilterTopKPolishScenario() {
        // Real-world scenario from issue #512
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.6, 0.4]
        let vocabulary = [
            1: "\u{2581}при",     // Cyrillic (top-1, wrong script)
            2: "\u{2581}prz",     // Polish/Latin (top-2, correct script)
            3: "\u{2581}прі",     // Cyrillic
        ]

        let result = ScriptDetection.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)  // Should select Polish token
        if let logit = result?.logit {
            XCTAssertEqual(logit, 0.6, accuracy: Float(0.001))
        }
    }

    // MARK: - Language Enum Tests

    func testAllLanguagesHaveScript() {
        // Ensure all languages have a defined script
        for language in Language.allCases {
            let script = language.script
            XCTAssertTrue(
                script == .latin || script == .cyrillic,
                "\(language.rawValue) must have a valid script")
        }
    }

    func testLanguageRawValues() {
        XCTAssertEqual(Language.english.rawValue, "en")
        XCTAssertEqual(Language.polish.rawValue, "pl")
        XCTAssertEqual(Language.russian.rawValue, "ru")
        XCTAssertEqual(Language.ukrainian.rawValue, "uk")
    }

    // MARK: - Unicode Range Tests

    func testLatinExtendedARange() {
        // Test characters in Latin Extended-A (U+0100 to U+017F)
        XCTAssertTrue(ScriptDetection.matches("Ā", script: .latin))  // U+0100
        XCTAssertTrue(ScriptDetection.matches("ž", script: .latin))  // U+017E
        XCTAssertTrue(ScriptDetection.matches("ſ", script: .latin))  // U+017F
    }

    func testCyrillicRange() {
        // Test characters in Cyrillic (U+0400 to U+04FF)
        XCTAssertTrue(ScriptDetection.matches("Ѐ", script: .cyrillic))  // U+0400
        XCTAssertTrue(ScriptDetection.matches("ӿ", script: .cyrillic))  // U+04FF
        XCTAssertTrue(ScriptDetection.matches("Ӏ", script: .cyrillic))  // U+04C0
    }
}
