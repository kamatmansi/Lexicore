"""
segmenter.py
Splits contract text into clause-sized chunks.

Fixes over the original version:
  P11 - long chunks are SPLIT, never truncated (no silent data loss)
  P12 - multi-level numbering (3.4, 21.2.3) handled; fewer merged sections
  P13 - table of contents stripped before splitting
  P14 - minimum word floor lowered from 20 to 8
  P20 - sub-clauses merged back under their parent stem (NEW)

Pipeline order inside split_into_clauses():
  strip ToC -> structural split -> normalise whitespace
  -> merge sub-clauses (P20) -> split over-long (P11) -> drop tiny fragments
"""

import re
from parser import extract_text_from_pdf

# ── Config ──────────────────────────────────────────────────
MIN_WORDS = 8      # below this, almost certainly a heading or fragment
MAX_WORDS = 350    # LEGAL-BERT ~512 tokens; leave headroom for subwords


# ── Step 1: strip table of contents ─────────────────────────
def strip_table_of_contents(text):
    """
    Removes ToC lines. These are identifiable by dot leaders followed by a
    page number, e.g.:
        A  No employment relationship between the parties ............ 1
    """
    lines = text.split("\n")
    kept = []
    removed = 0

    toc_pattern = re.compile(r'\.{4,}\s*\d+\s*$')

    for line in lines:
        if toc_pattern.search(line):
            removed += 1
            continue
        kept.append(line)

    if removed:
        print(f"  Stripped {removed} table-of-contents lines")

    return "\n".join(kept)


# ── Step 2: split on structural boundaries ──────────────────
def _structural_split(text):
    """
    Splits on legal document structure markers.
    Returns raw chunks (may still be too long - handled later).
    """

    # Numbered sections, including multi-level: 1.  3.4  21.2.3
    pattern_numbered = r'\n(?=\d+(?:\.\d+)*\.?[ \t]*\n?[ \t]*[A-Za-z])'

    # Lettered sub-clauses: "a. Something"  "(b) Something"
    pattern_lettered = r'\n(?=\(?[a-z]\)?\.?[ \t]+[A-Z])'

    # ALL CAPS headings, at least 4 chars
    pattern_caps = r'\n(?=[A-Z]{4,}[\s\n])'

    # Common legal section openers
    pattern_keywords = (
        r'\n(?=(?:WHEREAS|NOW THEREFORE|IN WITNESS|PROVIDED THAT'
        r'|ARTICLE|SECTION|SCHEDULE|ANNEXURE|APPENDIX)\b)'
    )

    combined = '|'.join([
        pattern_numbered,
        pattern_lettered,
        pattern_caps,
        pattern_keywords,
    ])

    return re.split(combined, text)


# ── Step 3 (NEW - P20): merge sub-clauses under their parent ─
_NUM_PREFIX = re.compile(r'^(\d+(?:\.\d+)*)\.?[\s)]')
_LETTER_PREFIX = re.compile(r'^\(?([a-z]|[ivx]{1,4})\)?[.)]\s')


def _clause_number(chunk):
    """
    Returns the leading clause number as a string ('3', '21.2.3'),
    or None if the chunk does not start with one.
    """
    m = _NUM_PREFIX.match(chunk)
    return m.group(1) if m else None


def _is_descendant(child_num, parent_num):
    """
    True if child_num sits under parent_num in the numbering tree.
        '3.1'     under '3'      -> True
        '21.2.3'  under '21.2'   -> True
        '3.2'     under '3.1'    -> False (siblings)
        '4'       under '3'      -> False
    """
    if child_num is None or parent_num is None:
        return False
    return child_num.startswith(parent_num + ".")


def _merge_sub_clauses(chunks, max_words=MAX_WORDS):
    """
    P20 FIX.

    The structural split correctly separates '3', '3.1', '3.2'... but that
    orphans the sub-clauses from the stem that gives them meaning:

        "3 The service provider warrants that it and its support staff:"
        "3.1 are capable of providing the services to the principal;"
        "3.2 will provide the services to the principal;"

    Read alone, 3.2 has no subject - it is a fragment, not an obligation.
    This merges each sub-clause back into its parent, stopping if the
    merged group would exceed max_words.

    Also absorbs lettered sub-items ('a.', '(b)', '(i)') into whatever
    chunk precedes them, since those never stand alone either.
    """
    merged = []
    group_num = None      # clause number that the current group is rooted at

    for chunk in chunks:
        num = _clause_number(chunk)
        is_letter_item = bool(_LETTER_PREFIX.match(chunk))

        can_merge = (
            merged
            and (_is_descendant(num, group_num) or (num is None and is_letter_item))
            and len(merged[-1].split()) + len(chunk.split()) <= max_words
        )

        if can_merge:
            merged[-1] = merged[-1] + " " + chunk
            # group_num stays anchored to the parent so 3.1, 3.2, 3.3
            # all attach to the same stem
        else:
            merged.append(chunk)
            group_num = num

    return merged


# ── Step 4: split over-long chunks instead of truncating ────
def _split_long_chunk(chunk, max_words=MAX_WORDS):
    """
    Breaks a chunk that exceeds max_words into several chunks.

    THIS IS THE P11 FIX. The original did:
        chunk = ' '.join(words[:400])
    which silently deleted everything past word 400.
    """
    words = chunk.split()
    if len(words) <= max_words:
        return [chunk]

    sentences = re.split(r'(?<=[.;:])\s+', chunk)

    parts = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())

        if sent_len > max_words:
            if current:
                parts.append(" ".join(current))
                current, current_len = [], 0
            sent_words = sent.split()
            for i in range(0, len(sent_words), max_words):
                parts.append(" ".join(sent_words[i:i + max_words]))
            continue

        if current_len + sent_len > max_words:
            parts.append(" ".join(current))
            current, current_len = [sent], sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        parts.append(" ".join(current))

    return parts


# ── Main ────────────────────────────────────────────────────
def split_into_clauses(text, verbose=True):
    """
    Takes full contract text.
    Returns list of clause chunks.

    Guarantee: no source text is discarded except fragments shorter than
    MIN_WORDS. Total word count out should be close to word count in.
    """
    words_in = len(text.split())

    text = strip_table_of_contents(text)
    raw_chunks = _structural_split(text)

    # ── Normalise whitespace BEFORE merging, so number prefixes
    #    sit at the very start of each chunk and can be matched.
    normalised = []
    for chunk in raw_chunks:
        if chunk is None:          # re.split can emit None for empty groups
            continue
        chunk = re.sub(r'\s*\n\s*', ' ', chunk)
        chunk = re.sub(r'[ \t]{2,}', ' ', chunk).strip()
        if chunk:
            normalised.append(chunk)

    # ── P20: reattach sub-clauses to their parent stem
    before = len(normalised)
    normalised = _merge_sub_clauses(normalised)
    if verbose and before != len(normalised):
        print(f"  Merged {before - len(normalised)} sub-clauses into parents")

    # ── P11: split rather than truncate, then drop tiny fragments
    cleaned = []
    dropped_short = 0

    for chunk in normalised:
        for part in _split_long_chunk(chunk):
            part = part.strip()
            if not part:
                continue
            if len(part.split()) < MIN_WORDS:
                dropped_short += 1
                continue
            cleaned.append(part)

    if verbose:
        words_out = sum(len(c.split()) for c in cleaned)
        retention = (words_out / words_in * 100) if words_in else 0
        print(f"  {len(cleaned)} clauses | {words_out}/{words_in} words kept "
              f"({retention:.1f}%) | {dropped_short} short fragments dropped")

    return cleaned


def segment_pdf(pdf_path):
    """
    PDF path -> list of clause chunks.
    """
    print(f"Extracting text from: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    if not text:
        print("ERROR: Could not extract text "
              "(PDF may be a scanned image with no text layer)")
        return []

    return split_into_clauses(text)


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
        print()

        for i, clause in enumerate(clauses, 1):
            preview = clause[:150].replace("\n", " ")
            print(f"--- Clause {i} ({len(clause.split())} words) ---")
            print(preview)
            print()