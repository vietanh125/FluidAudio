import Foundation
import XCTest

@testable import FluidAudio

final class CohereAsrConfigTests: XCTestCase {

    // MARK: - Config Constants

    func testSpecialTokenIdsAreInRange() {
        let vocabSize = CohereAsrConfig.vocabSize
        let tokenIds = [
            CohereAsrConfig.decoderStartTokenId,
            CohereAsrConfig.eosTokenId,
            CohereAsrConfig.padTokenId,
        ]

        for tokenId in tokenIds {
            XCTAssertGreaterThanOrEqual(tokenId, 0, "Token ID \(tokenId) should be non-negative")
            XCTAssertLessThan(
                tokenId, vocabSize, "Token ID \(tokenId) should be < vocabSize (\(vocabSize))")
        }
    }

    func testFixedAudioLengthMatchesMaxDuration() {
        // 30 seconds at 16kHz = 480,000 samples
        // Mel spectrogram has 100 frames per second
        // So 30 seconds = 3000 frames
        let expectedFrames = Int(CohereAsrConfig.maxAudioSeconds * 100)
        XCTAssertEqual(CohereAsrConfig.fixedAudioLength, expectedFrames)
        XCTAssertEqual(CohereAsrConfig.fixedAudioLength, 3000)
    }

    func testSampleRateIs16kHz() {
        XCTAssertEqual(CohereAsrConfig.sampleRate, 16000)
    }

    func testNumMelBinsIs80() {
        XCTAssertEqual(CohereAsrConfig.numMelBins, 80)
    }

    // MARK: - Language

    func testLanguageFromIsoCode() {
        XCTAssertEqual(CohereAsrConfig.Language(from: "en"), .english)
        XCTAssertEqual(CohereAsrConfig.Language(from: "fr"), .french)
        XCTAssertEqual(CohereAsrConfig.Language(from: "zh"), .chinese)
        XCTAssertEqual(CohereAsrConfig.Language(from: "ja"), .japanese)
        XCTAssertEqual(CohereAsrConfig.Language(from: "ko"), .korean)
    }

    func testLanguageFromIsoCodeCaseInsensitive() {
        XCTAssertEqual(CohereAsrConfig.Language(from: "EN"), .english)
        XCTAssertEqual(CohereAsrConfig.Language(from: "Fr"), .french)
        XCTAssertEqual(CohereAsrConfig.Language(from: "ZH"), .chinese)
    }

    func testLanguageFromEnglishName() {
        XCTAssertEqual(CohereAsrConfig.Language(from: "French"), .french)
        XCTAssertEqual(CohereAsrConfig.Language(from: "English"), .english)
        XCTAssertEqual(CohereAsrConfig.Language(from: "japanese"), .japanese)
        XCTAssertEqual(CohereAsrConfig.Language(from: "Chinese"), .chinese)
    }

    func testLanguageFromInvalidStringReturnsNil() {
        XCTAssertNil(CohereAsrConfig.Language(from: "klingon"))
        XCTAssertNil(CohereAsrConfig.Language(from: ""))
        XCTAssertNil(CohereAsrConfig.Language(from: "xx"))
    }

    func testAllLanguagesHaveEnglishNames() {
        for language in CohereAsrConfig.Language.allCases {
            XCTAssertFalse(
                language.englishName.isEmpty, "\(language) should have a non-empty English name")
        }
    }

    func testLanguageRawValueIsIsoCode() {
        XCTAssertEqual(CohereAsrConfig.Language.english.rawValue, "en")
        XCTAssertEqual(CohereAsrConfig.Language.french.rawValue, "fr")
        XCTAssertEqual(CohereAsrConfig.Language.chinese.rawValue, "zh")
        XCTAssertEqual(CohereAsrConfig.Language.japanese.rawValue, "ja")
    }

    func testLanguageRoundTrip() {
        for language in CohereAsrConfig.Language.allCases {
            let fromIso = CohereAsrConfig.Language(from: language.rawValue)
            XCTAssertEqual(
                fromIso, language, "Round-trip via ISO code should work for \(language)")

            let fromName = CohereAsrConfig.Language(from: language.englishName)
            XCTAssertEqual(
                fromName, language, "Round-trip via English name should work for \(language)")
        }
    }

    func testLanguageHasCorrectFleursCode() {
        XCTAssertEqual(CohereAsrConfig.Language.english.fleursCode, "en_us")
        XCTAssertEqual(CohereAsrConfig.Language.french.fleursCode, "fr_fr")
        XCTAssertEqual(CohereAsrConfig.Language.chinese.fleursCode, "cmn_hans_cn")
        XCTAssertEqual(CohereAsrConfig.Language.japanese.fleursCode, "ja_jp")
        XCTAssertEqual(CohereAsrConfig.Language.korean.fleursCode, "ko_kr")
        XCTAssertEqual(CohereAsrConfig.Language.spanish.fleursCode, "es_419")
    }

    func testAsianLanguagesUseCER() {
        XCTAssertTrue(CohereAsrConfig.Language.chinese.usesCER)
        XCTAssertTrue(CohereAsrConfig.Language.japanese.usesCER)
        XCTAssertTrue(CohereAsrConfig.Language.korean.usesCER)

        XCTAssertFalse(CohereAsrConfig.Language.english.usesCER)
        XCTAssertFalse(CohereAsrConfig.Language.french.usesCER)
        XCTAssertFalse(CohereAsrConfig.Language.german.usesCER)
    }

    func testSupports14Languages() {
        XCTAssertEqual(CohereAsrConfig.Language.allCases.count, 14)
    }

    func testWesternLanguagesCount() {
        let westernLanguages: [CohereAsrConfig.Language] = [
            .english, .french, .german, .italian, .spanish,
            .portuguese, .greek, .dutch, .polish, .arabic,
        ]
        XCTAssertEqual(westernLanguages.count, 10)
        for lang in westernLanguages {
            XCTAssertFalse(lang.usesCER, "\(lang) should use WER, not CER")
        }
    }

    func testAsianLanguagesCount() {
        let asianLanguages: [CohereAsrConfig.Language] = [
            .chinese, .japanese, .korean, .vietnamese,
        ]
        XCTAssertEqual(asianLanguages.count, 4)
    }
}
