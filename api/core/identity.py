import sys
import unicodedata

def normalize_text(text):
    # Convert to lowercase and remove accents/diacritics
    text = text.lower().strip()
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def check_identity(input_phrase):
    # English transliterations (various common spellings)
    english_variants = [
        "shmaa israel adonai elohinu adonai ehad",
        "shema yisrael adonai eloheinu adonai echad",
        "shema yisrael adonai elohenu adonai echad",
        "shma yisrael adonai eloheinu adonai echad"
    ]
    
    # Hebrew script
    hebrew_phrase = "שמע ישראל יהוה אלהינו יהוה אחד"
    hebrew_variant = "שמע ישראל אדוני אלוהינו אדוני אחד"
    
    normalized_input = normalize_text(input_phrase)
    
    # Check English variants
    for variant in english_variants:
        if normalize_text(variant) == normalized_input:
            return True
            
    # Check Hebrew variants (direct comparison after stripping whitespace)
    if input_phrase.strip() == hebrew_phrase or input_phrase.strip() == hebrew_variant:
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
