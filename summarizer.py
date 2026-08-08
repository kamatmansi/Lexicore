# ============================================================
# LexiCore — summarizer.py
# Produces a short extractive summary of a clause using TextRank.
#
# NOTE: TextRank is EXTRACTIVE, not abstractive — it selects the most
# representative sentence(s) already present in the clause text; it
# does not rewrite legal language into simpler wording. Worth stating
# explicitly in your report: "plain-English" here means "shorter,"
# not "simplified vocabulary."
#
# Install first:
#   pip install spacy pytextrank
#   python -m spacy download en_core_web_sm
# ============================================================

import spacy
import pytextrank  # noqa: F401 — registers the "textrank" pipeline component

_nlp = None


def load_summarizer():
    """Load spaCy + TextRank pipeline once. Call at app startup, not per clause."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.add_pipe("textrank")
    return _nlp


def summarize_clause(text: str, sentence_count: int = 1) -> str:
    """Return the top `sentence_count` most representative sentence(s) from
    the clause. Falls back to the full clause text if it's already too short
    for TextRank to meaningfully rank (e.g. a single-sentence clause)."""
    nlp = load_summarizer()
    doc = nlp(text)

    num_sentences = len(list(doc.sents))
    if num_sentences <= sentence_count:
        # Nothing to summarize down to — the clause is already at or below
        # the target length, so just return it as-is.
        return text.strip()

    summary_sentences = [str(sent) for sent in doc._.textrank.summary(limit_sentences=sentence_count)]
    if not summary_sentences:
        return text.strip()

    return " ".join(summary_sentences).strip()


def summarize_clauses(clauses: list[dict], sentence_count: int = 1) -> list[dict]:
    """Takes the output of risk_scorer.score_clauses() (list of dicts with a
    'text' key) and adds a 'summary' key to each, in place."""
    for item in clauses:
        item["summary"] = summarize_clause(item["text"], sentence_count=sentence_count)
    return clauses


if __name__ == "__main__":
    # Quick manual test using a real multi-sentence clause (Clause 4 style,
    # NDA "Obligations of Receiving Party" from your sample_contract.pdf)
    sample = (
        "Receiving Party shall hold and maintain the Confidential Information in strictest "
        "confidence for the sole and exclusive benefit of the Disclosing Party. Receiving Party "
        "shall carefully restrict access to Confidential Information to employees, contractors "
        "and third parties as is reasonably required and shall require those persons to sign "
        "nondisclosure restrictions at least as protective as those in this Agreement. Receiving "
        "Party shall not, without prior written approval of Disclosing Party, use for Receiving "
        "Party's own benefit, publish, copy, or otherwise disclose to others, or permit the use "
        "by others for their benefit, any Confidential Information."
    )
    print("Original clause:\n", sample)
    print("\nSummary:\n", summarize_clause(sample, sentence_count=1))
