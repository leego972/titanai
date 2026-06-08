import sys
import unicodedata

# Titan's confirmation passphrase — echoed back to the owner when Shema is received
TITAN_RESPONSE_PASSPHRASE = "Baruch Shem Kavod Malchuto L'olam Va'ed"

def normalize_text(text):
    text = text.lower().strip()
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def check_identity(input_phrase):
    english_variants = [
        "shmaa israel adonai elohinu adonai ehad",
        "shema yisrael adonai eloheinu adonai echad",
        "shema yisrael adonai elohenu adonai echad",
        "shma yisrael adonai eloheinu adonai echad"
    ]
    hebrew_phrase = "\u05e9\u05de\u05e2 \u05d9\u05e9\u05e8\u05d0\u05dc \u05d9\u05d4\u05d5\u05d4 \u05d0\u05dc\u05d4\u05d9\u05e0\u05d5 \u05d9\u05d4\u05d5\u05d4 \u05d0\u05d7\u05d3"
    hebrew_variant = "\u05e9\u05de\u05e2 \u05d9\u05e9\u05e8\u05d0\u05dc \u05d0\u05d3\u05d5\u05e0\u05d9 \u05d0\u05dc\u05d5\u05d4\u05d9\u05e0\u05d5 \u05d0\u05d3\u05d5\u05e0\u05d9 \u05d0\u05d7\u05d3"

    normalized_input = normalize_text(input_phrase)

    for variant in english_variants:
        if normalize_text(variant) == normalized_input:
            return True

    if input_phrase.strip() in (hebrew_phrase, hebrew_variant):
        return True

    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        phrase = " ".join(sys.argv[1:])
        if check_identity(phrase):
            print("IDENTITY_VERIFIED")
            sys.exit(0)
        else:
            print("IDENTITY_REJECTED")
            sys.exit(1)
    else:
        print("Usage: python3 verify_identity.py <phrase>")
        sys.exit(1)
