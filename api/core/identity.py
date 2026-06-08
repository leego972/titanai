import sys
import unicodedata

TITAN_RESPONSE_PASSPHRASE = "\u05d1\u05e8\u05d5\u05da \u05e9\u05dd \u05db\u05d1\u05d5\u05d3 \u05de\u05dc\u05db\u05d5\u05ea\u05d5 \u05dc\u05e2\u05d5\u05dc\u05dd \u05d5\u05e2\u05d3"

def normalize_text(text):
    text = text.lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

def check_identity(input_phrase):
    english_variants = [
        "shmaa israel adonai elohinu adonai ehad",
        "shema yisrael adonai eloheinu adonai echad",
        "shema yisrael adonai elohenu adonai echad",
        "shma yisrael adonai eloheinu adonai echad",
        "shema israel adonai eloheinu adonai echad",
        "shmaa israel adonai elohenu adonai ehad",
        "shma israel adonai elohinu adonai ehad",
    ]
    hebrew_phrases = [
        "\u05e9\u05de\u05e2 \u05d9\u05e9\u05e8\u05d0\u05dc \u05d9\u05d4\u05d5\u05d4 \u05d0\u05dc\u05d4\u05d9\u05e0\u05d5 \u05d9\u05d4\u05d5\u05d4 \u05d0\u05d7\u05d3",
        "\u05e9\u05de\u05e2 \u05d9\u05e9\u05e8\u05d0\u05dc \u05d0\u05d3\u05d5\u05e0\u05d9 \u05d0\u05dc\u05d5\u05d4\u05d9\u05e0\u05d5 \u05d0\u05d3\u05d5\u05e0\u05d9 \u05d0\u05d7\u05d3",
    ]
    normalized = normalize_text(input_phrase)
    for v in english_variants:
        if normalize_text(v) == normalized:
            return True
    if input_phrase.strip() in hebrew_phrases:
        return True
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        phrase = " ".join(sys.argv[1:])
        print("IDENTITY_VERIFIED" if check_identity(phrase) else "IDENTITY_REJECTED")
        sys.exit(0 if check_identity(phrase) else 1)
    else:
        print("Usage: python3 identity.py <phrase>")
        sys.exit(1)
