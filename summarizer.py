"""
summarizer.py
Generates a plain-language summary of an analysed contract.

Spec reference: FR6 - "The system must generate a plain-language summary of
the contract using TextRank-based NLP."

DESIGN NOTE (worth knowing for the viva):

TextRank is an *extractive* algorithm. It ranks text that already exists and
selects the most central items. It cannot make legal language plainer, because
it does not generate text - it only selects. So "TextRank" and "plain-language"
cannot both be satisfied by TextRank alone.

This module is therefore hybrid:
  1. TextRank ranks clauses by centrality - how representative each clause is
     of the document as a whole (the algorithm named in the spec)
  2. That centrality is combined with the computed risk score, so the clauses
     surfaced are both important to the contract AND consequential to the user
  3. The overview and risk areas are assembled from computed figures

Every line of output is either quoted verbatim from the contract or built from
a number the pipeline computed. Nothing is paraphrased or invented - a
summariser that fabricates an obligation is worse than no summariser.

TextRank reference: Mihalcea & Tarau (2004), "TextRank: Bringing Order into
Texts", EMNLP.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Ranking weights ─────────────────────────────────────────
# How much each signal contributes to a clause's final ranking.
# Risk dominates deliberately: this is a risk analyser, so a central but
# harmless clause should not outrank a peripheral but dangerous one.
# Centrality still matters - it separates substantive provisions from
# incidental ones at the same risk level.
W_RISK = 0.65
W_CENTRALITY = 0.35

# ── TextRank parameters ─────────────────────────────────────
DAMPING = 0.85          # standard PageRank damping factor
MAX_ITER = 100
TOLERANCE = 1e-6
SIM_THRESHOLD = 0.10    # ignore very weak edges

MIN_SENT_WORDS = 6
MAX_SENT_WORDS = 90

DEFAULT_PROVISIONS = 5
MAX_QUOTE_CHARS = 300


# ── Sentence splitting ──────────────────────────────────────
# Legal text is full of abbreviations and enumerations that break naive
# splitting on ".". These lookbehinds cover the cases in the sample contracts.
_ABBREV = (r"(?<!\bNo)(?<!\bInc)(?<!\bLtd)(?<!\bPty)(?<!\bCo)(?<!\bCorp)"
           r"(?<!\bcl)(?<!\bSch)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)")

_SENT_SPLIT = re.compile(_ABBREV + r"(?<=[.;])\s+(?=[A-Z(])")


def split_sentences(text):
    """
    Splits contract text into sentences.

    Legal drafting uses ';' as a sentence-level separator inside enumerated
    lists as often as '.', so both are treated as boundaries.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    out = []
    for part in _SENT_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        n = len(part.split())
        if n < MIN_SENT_WORDS or n > MAX_SENT_WORDS:
            continue
        out.append(part)
    return out


# ── PageRank ────────────────────────────────────────────────
def _pagerank(similarity, damping=DAMPING, max_iter=MAX_ITER, tol=TOLERANCE):
    """
    Power-iteration PageRank over a similarity matrix.

    Each item (clause or sentence) is a node; edge weight is the cosine
    similarity of their TF-IDF vectors. An item scores highly when it is
    similar to many other high-scoring items - i.e. when it is central to
    what the document is about.

    Implemented directly rather than via networkx: ~15 lines, no extra
    dependency, and the mechanism is worth being able to explain.
    """
    n = similarity.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    col_sums = similarity.sum(axis=0)
    col_sums[col_sums == 0] = 1.0          # isolated nodes
    transition = similarity / col_sums

    scores = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        prev = scores
        scores = teleport + damping * (transition @ scores)
        if np.abs(scores - prev).sum() < tol:
            break

    return scores


def _centrality(texts):
    """
    TextRank centrality for a list of texts, normalised to 0-1.
    """
    if not texts:
        return np.array([])
    if len(texts) == 1:
        return np.array([1.0])

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        return np.full(len(texts), 1.0 / len(texts))

    # TF-IDF rows are L2-normalised, so X @ X.T is cosine similarity directly.
    similarity = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < SIM_THRESHOLD] = 0.0

    scores = _pagerank(similarity)

    span = scores.max() - scores.min()
    if span == 0:
        return np.full(len(texts), 0.5)
    return (scores - scores.min()) / span


# Words that indicate a sentence carries the substantive obligation.
# Used to pick the quoted line, since TextRank centrality is unreliable
# within a short clause - it picked "waives all moral rights" over the
# actual assignment of pre-existing IP.
_SUBSTANTIVE = re.compile(
    r"\b(shall|must|will|agrees?|assigns?|indemnif\w+|grants?|warrants?"
    r"|undertakes?|may not|shall not|is liable|are liable)\b", re.I)


def _key_sentence(clause_text, reasons=None):
    """
    The sentence most likely to carry the clause's substance.

    Priority:
      1. A sentence matching a keyword from a fired risk rule
      2. The longest sentence containing an obligation verb
      3. TextRank centrality
    """
    sentences = split_sentences(clause_text)
    if not sentences:
        text = clause_text.strip()
    elif len(sentences) == 1:
        text = sentences[0]
    else:
        # 1. Prefer a sentence that shows why this clause was flagged
        trigger = None
        for reason in (reasons or []):
            key = re.sub(r"\(.*?\)", "", reason).lower()
            for word in ("indemnif", "discretion", "perpetu", "irrevocabl",
                         "assign", "terminat", "compete", "solicit", "waive"):
                if word in key:
                    for s in sentences:
                        if word in s.lower():
                            trigger = s
                            break
                if trigger:
                    break
            if trigger:
                break

        if trigger:
            text = trigger
        else:
            # 2. Longest sentence carrying an obligation verb
            obligated = [s for s in sentences if _SUBSTANTIVE.search(s)]
            if obligated:
                text = max(obligated, key=len)
            else:
                scores = _centrality(sentences)
                text = sentences[int(np.argmax(scores))]

    text = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text)   # drop clause number
    if len(text) > MAX_QUOTE_CHARS:
        text = text[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0] + "..."
    return text


# ── Risk area grouping ──────────────────────────────────────
# Maps the specific reasons produced by risk_scorer into broader themes, so
# the summary reports "Liability and indemnity (3 clauses)" rather than
# repeating every individual rule that fired.
RISK_AREAS = [
    ("Liability and indemnity", [
        "indemnity", "liability", "uncapped", "cap on liability",
    ]),
    ("Restrictions on future activity", [
        "non-compete", "restrictive covenant", "exclusivity", "no-solicit",
        "volume restriction", "price restrictions", "minimum commitment",
    ]),
    ("Intellectual property", [
        "ip ownership", "joint ip", "licence", "license",
    ]),
    ("Termination and exit", [
        "termination", "terminate", "post-termination", "survives termination",
    ]),
    ("Unilateral control", [
        "discretion", "waiver of rights", "irrevocable", "perpetual",
    ]),
    ("Transfer and assignment", [
        "assignment", "anti-assignment", "change of control", "rofr",
    ]),
    ("Financial exposure", [
        "damages", "penalt", "liquidated",
    ]),
]


def _classify_reason(reason, clause_type):
    """Returns the risk area a reason belongs to, or None."""
    haystack = f"{reason} {clause_type}".lower()
    for area, keywords in RISK_AREAS:
        if any(k in haystack for k in keywords):
            return area
    return None


def _risk_areas(scored):
    """
    Groups flagged clauses into themed risk areas.
    Returns [(area_name, clause_count, [example reasons])] sorted by count.
    """
    areas = {}

    for c in scored:
        if c["risk_level"] not in ("HIGH", "MEDIUM"):
            continue

        seen_here = set()
        for reason in c["reasons"]:
            if reason == "No risk indicators found":
                continue
            area = _classify_reason(reason, c["clause_type"])
            if area is None or area in seen_here:
                continue
            seen_here.add(area)

            entry = areas.setdefault(area, {"count": 0, "reasons": set()})
            entry["count"] += 1
            # Strip trailing confidence figures for readability
            entry["reasons"].add(reason.split("(")[0].strip())

    out = []
    for area, data in areas.items():
        out.append((area, data["count"], sorted(data["reasons"])[:3]))

    out.sort(key=lambda t: -t[1])
    return out


# ── Overview ────────────────────────────────────────────────
def _overview(summary):
    """
    Plain-language overview built only from computed figures.
    """
    total = summary["total_clauses"]
    high = summary["high"]
    medium = summary["medium"]
    low = summary["low"]
    score = summary["overall_score"]
    level = summary["overall_level"]

    lines = [
        f"This contract contains {total} identifiable "
        f"{'clause' if total == 1 else 'clauses'}.",
        f"Overall risk assessment: {level} ({score} out of 100).",
    ]

    flagged = high + medium
    if flagged == 0:
        lines.append(
            "No clauses were flagged as carrying significant risk. This does "
            "not mean the contract is safe to sign unread - it means no "
            "recognised risk patterns were detected."
        )
    else:
        parts = []
        if high:
            parts.append(f"{high} high-risk")
        if medium:
            parts.append(f"{medium} medium-risk")
        if low:
            parts.append(f"{low} low-risk")
        lines.append(
            f"Flagged: {', '.join(parts)}."
        )

    return lines


# ── Public interface ────────────────────────────────────────
def summarize(scored, summary, top_provisions=DEFAULT_PROVISIONS):
    """
    Takes the output of risk_scorer.score_clauses() and risk_scorer.summarize().

    Returns:
        {
          "overview":    [str]   - plain-language figures
          "provisions":  [dict]  - ranked key provisions
          "risk_areas":  [tuple] - (area, clause_count, example reasons)
          "method":      str
        }

    Ranking combines TextRank centrality with the computed risk score
    (see W_RISK / W_CENTRALITY at the top of this file).
    """
    if not scored:
        return {
            "overview": ["No clauses could be extracted from this document."],
            "provisions": [],
            "risk_areas": [],
            "method": "none",
        }

    texts = [c["clause_text"] for c in scored]
    centrality = _centrality(texts)

    max_risk = max((c["risk_score"] for c in scored), default=0)
    risk_norm = [
        (c["risk_score"] / max_risk) if max_risk else 0.0 for c in scored
    ]

    ranked = []
    for i, c in enumerate(scored):
        combined = W_RISK * risk_norm[i] + W_CENTRALITY * float(centrality[i])
        ranked.append((combined, i, c))

    ranked.sort(key=lambda t: -t[0])

    provisions = []
    for combined, i, c in ranked[:top_provisions]:
        reasons = [r for r in c["reasons"] if r != "No risk indicators found"]
        provisions.append({
            "clause_type": c["clause_type"],
            "risk_level": c["risk_level"],
            "risk_score": c["risk_score"],
            "confidence": c.get("confidence", 0.0),
            "importance": round(combined, 3),
            "quote": _key_sentence(c["clause_text"], reasons),
            "reasons": reasons,
        })

    return {
        "overview": _overview(summary),
        "provisions": provisions,
        "risk_areas": _risk_areas(scored),
        "method": (f"TextRank centrality over {len(scored)} clauses "
                   f"(damping {DAMPING}) combined with computed risk score "
                   f"at {W_RISK:.2f}/{W_CENTRALITY:.2f}"),
    }


def format_text(result):
    """
    Renders the summary as plain text, for CLI output or export.
    """
    out = []

    out.append("CONTRACT OVERVIEW")
    out.append("=" * 62)
    for line in result["overview"]:
        out.append(line)

    if result["provisions"]:
        out.append("")
        out.append("KEY PROVISIONS")
        out.append("=" * 62)
        out.append("Ranked by importance to the contract and level of risk.")
        for i, p in enumerate(result["provisions"], 1):
            out.append("")
            out.append(f"{i}. {p['clause_type']}  -  "
                       f"{p['risk_level']} (risk {p['risk_score']}/10)")
            out.append(f"   \"{p['quote']}\"")
            if p["reasons"]:
                out.append(f"   Flagged for: {'; '.join(p['reasons'])}")

    if result["risk_areas"]:
        out.append("")
        out.append("MAIN RISK AREAS")
        out.append("=" * 62)
        for area, count, reasons in result["risk_areas"]:
            out.append(f"- {area}: {count} "
                       f"{'clause' if count == 1 else 'clauses'}")

    out.append("")
    out.append("-" * 62)
    out.append("Automated analysis. Not legal advice.")

    return "\n".join(out)


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from segmenter import segment_pdf
    from classifier import classify_clauses
    from risk_scorer import score_clauses, summarize as risk_summarize

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/adversarial_test.pdf"

    clauses = segment_pdf(pdf)
    scored = score_clauses(classify_clauses(clauses))
    stats = risk_summarize(scored)

    print()
    print(format_text(summarize(scored, stats)))