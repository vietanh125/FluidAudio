import Foundation

public final class Tokenizer: Sendable {
    private let vocab: [String: String]
    private let idToToken: [Int: String]

    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: String]

        var idToToken: [Int: String] = [:]
        for (key, value) in json {
            if let id = Int(key) {
                idToToken[id] = value
            }
        }
        self.vocab = json
        self.idToToken = idToToken
    }

    /// Raw vocabulary piece for a token id (`nil` if id is out of range).
    /// Does not apply the SentencePiece word-boundary substitution that `decode` does.
    public func piece(forId id: Int) -> String? {
        idToToken[id]
    }

    public func decode(ids: [Int]) -> String {
        var text = ""
        for id in ids {
            if let token = idToToken[id] {
                text += token
            }
        }
        // Replace SentencePiece word boundary marker with space, then trim
        return text.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
