import re
from parser import extract_text_from_pdf


def split_into_clauses(text):
    """
    Takes full contract text
    Returns list of clause chunks
    """

    # ── Step 1: Split on common legal section patterns ──────

    # Pattern explanation:
    # \n     = newline
    # \d+\.  = number followed by dot (1. 2. 3.)
    # [A-Z]  = capital letter
    # {2,}   = at least 2 times (catches ALL CAPS headings)

    # Split on numbered sections like "1." "2." "3."
    pattern_numbered = r'\n(?=\d+\.)\s*'

    # Split on lettered sections like "a." "b."
    pattern_lettered = r'\n(?=[a-z]\.\s[A-Z])'

    # Split on ALL CAPS headings (minimum 4 chars)
    pattern_caps = r'\n(?=[A-Z]{4,}[\s\n])'

    # Split on common legal keywords at start of line
    pattern_keywords = r'\n(?=(?:WHEREAS|NOW THEREFORE|IN WITNESS|PROVIDED THAT|ARTICLE|SECTION|SCHEDULE)\b)'

    # Combine all patterns
    combined_pattern = f'{pattern_numbered}|{pattern_lettered}|{pattern_caps}|{pattern_keywords}'

    # Split the text
    raw_chunks = re.split(combined_pattern, text)

    # ── Step 2: Clean each chunk ─────────────────────────────
    cleaned_chunks = []

    for chunk in raw_chunks:
        # Remove extra whitespace
        chunk = chunk.strip()

        # Skip if empty
        if not chunk:
            continue

        # Skip if too short (less than 20 words)
        # These are usually just headings or page numbers
        word_count = len(chunk.split())
        if word_count < 20:
            continue

        # Skip if too long (more than 400 words)
        # LEGAL-BERT can only handle ~400 words
        # Take first 400 words if too long
        if word_count > 400:
            words = chunk.split()
            chunk = ' '.join(words[:400])

        cleaned_chunks.append(chunk)

    return cleaned_chunks


def segment_pdf(pdf_path):
    """
    Takes PDF path
    Returns list of clause chunks
    (combines parser + segmenter in one function)
    """
    # Extract text using our parser
    print(f"Extracting text from: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    if not text:
        print("ERROR: Could not extract text")
        return []

    # Split into clauses
    clauses = split_into_clauses(text)

    return clauses


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        "data/sample_contract.pdf",
        "data/COE-Sample.pdf",
        "data/sample-service-agreement.pdf"
    ]

    for pdf_path in test_files:
        print("=" * 60)
        print(f"FILE: {pdf_path}")
        print("=" * 60)

        clauses = segment_pdf(pdf_path)

        print(f"\nTotal clauses extracted: {len(clauses)}")
        print()

        # Show each clause
        for i, clause in enumerate(clauses):
            print(f"--- Clause {i+1} ({len(clause.split())} words) ---")
            print(clause[:200])  # show first 200 chars
            print()

        print()