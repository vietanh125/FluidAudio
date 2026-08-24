import Foundation

/// BPE tokenizer for CTC vocabulary boosting.
/// Only implements encoding - no decoding, chat templates, or other features.
/// Supports the specific tokenizer.json format used by Parakeet models.
///
/// Text normalization: Applies NFKC normalization before BPE encoding, plus
/// lowercasing when the loaded vocabulary is all-lowercase (matching the standard
/// NeMo CTC pipeline for uncased models such as parakeet-ctc-110m). Vocabularies
/// containing cased pieces (e.g. the German v3 SentencePiece-BPE vocab) are
/// detected at load time and encoded case-preserving — lowercasing a cased vocab
/// drops encode fidelity from 100% to ~45%.
public final class BpeTokenizer: Sendable {
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let addedTokens: [String: Int]
    private let lowercasesInput: Bool

    public enum Error: Swift.Error, LocalizedError {
        case fileNotFound(URL)
        case invalidJSON(String)
        case missingField(String)
        case unsupportedTokenizerType(String)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let url):
                return "tokenizer.json not found at \(url.path)"
            case .invalidJSON(let message):
                return "Invalid JSON: \(message)"
            case .missingField(let field):
                return "Missing required field: \(field)"
            case .unsupportedTokenizerType(let type):
                return "Unsupported tokenizer type: \(type). Only 'BPE' is supported."
            }
        }
    }

    /// Load tokenizer from a folder containing tokenizer.json
    public static func load(from modelFolder: URL) throws -> BpeTokenizer {
        let tokenizerPath = modelFolder.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.fileNotFound(tokenizerPath)
        }

        let data = try Data(contentsOf: tokenizerPath)

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Error.invalidJSON("Root is not a dictionary")
        }

        // Parse model section
        guard let model = json["model"] as? [String: Any] else {
            throw Error.missingField("model")
        }

        guard let modelType = model["type"] as? String else {
            throw Error.missingField("model.type")
        }

        guard modelType == "BPE" else {
            throw Error.unsupportedTokenizerType(modelType)
        }

        // Parse vocabulary: {"token": id, ...}
        guard let vocabDict = model["vocab"] as? [String: Int] else {
            throw Error.missingField("model.vocab")
        }

        // Parse merges: ["a b", "c d", ...]
        guard let mergesArray = model["merges"] as? [String] else {
            throw Error.missingField("model.merges")
        }

        let merges = mergesArray.compactMap { mergeStr -> (String, String)? in
            let parts = mergeStr.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { return nil }
            return (String(parts[0]), String(parts[1]))
        }

        // Parse added_tokens (special tokens like <unk>, <pad>)
        var addedTokensDict: [String: Int] = [:]
        let addedTokensList = (json["added_tokens"] as? [[String: Any]]) ?? []
        for token in addedTokensList {
            guard let content = token["content"] as? String,
                let id = token["id"] as? Int
            else { continue }
            addedTokensDict[content] = id
        }

        let hasCasedPiece = vocabDict.keys.contains { $0.contains(where: \.isUppercase) }

        return BpeTokenizer(
            vocab: vocabDict,
            merges: merges,
            addedTokens: addedTokensDict,
            lowercasesInput: !hasCasedPiece
        )
    }

    private init(
        vocab: [String: Int], merges: [(String, String)], addedTokens: [String: Int],
        lowercasesInput: Bool
    ) {
        self.vocab = vocab
        self.merges = merges
        self.addedTokens = addedTokens
        self.lowercasesInput = lowercasesInput
    }

    /// Encode text to token IDs using BPE.
    ///
    /// - Parameters:
    ///   - text: Input text.
    ///   - addSpecialTokens: Currently unused; kept for API compatibility.
    ///   - prependWordBoundary: If `true` (default), prepend a SentencePiece
    ///     `▁` word-boundary marker before encoding, matching standard NeMo
    ///     CTC tokenization. Set to `false` to produce a "mid-utterance"
    ///     tokenization that omits the leading boundary, which is useful
    ///     for keyword-spotting against audio where the keyword does not
    ///     start at a word boundary (e.g. compound matches like
    ///     `Liv` + `marli` → `Livmarli`).
    public func encode(
        _ text: String,
        addSpecialTokens: Bool = false,
        prependWordBoundary: Bool = true
    ) -> [Int] {
        // Normalize: NFKC, lowercasing only for uncased vocabularies (see class docs)
        let normalized = (lowercasesInput ? text.lowercased() : text).precomposedStringWithCompatibilityMapping

        // Pre-tokenize: replace spaces with ▁ (sentencepiece style)
        let leading = prependWordBoundary ? ASRConstants.sentencePieceWordBoundary : ""
        let preprocessed =
            leading + normalized.replacingOccurrences(of: " ", with: ASRConstants.sentencePieceWordBoundary)

        // Split into characters
        var word = preprocessed.map { String($0) }

        // Apply BPE merges iteratively
        while word.count >= 2 {
            // Find the highest priority merge (earliest in merges list)
            var bestMergeIndex: Int? = nil
            var bestMergePair: (String, String)? = nil

            for i in 0..<word.count - 1 {
                let pair = (word[i], word[i + 1])

                // Check if this pair has a merge rule
                guard let mergeIndex = merges.firstIndex(where: { $0.0 == pair.0 && $0.1 == pair.1 }) else {
                    continue
                }

                // Update best merge if this is higher priority (lower index)
                if bestMergeIndex.map({ mergeIndex < $0 }) ?? true {
                    bestMergeIndex = mergeIndex
                    bestMergePair = pair
                }
            }

            // No more merges possible
            guard let (first, second) = bestMergePair else { break }

            // Apply the merge to ALL occurrences of the winning pair (standard BPE)
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2  // Skip the next token since we merged it
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        // Convert tokens to IDs
        return word.compactMap { token -> Int? in
            // Check added tokens first (special tokens)
            if let id = addedTokens[token] {
                return id
            }
            // Then check vocabulary
            if let id = vocab[token] {
                return id
            }
            // Unknown token - return <unk> ID if available
            return addedTokens["<unk>"] ?? vocab["<unk>"] ?? 0
        }
    }
}
