"""
risk_scorer.py
Assigns risk levels to classified clauses for LexiCore.

Hybrid approach:
  1. Clause-type risk weights  (uses classifier output)
  2. Keyword / pattern rules   (fires independently of the classifier)

Rule 2 exists because the baseline classifier returns
"General / Unclassified" for many real clauses (CUAD's label set does not
cover every contract type). Rules keep the tool useful in those cases.
"""

import re

# ── Risk weights by CUAD clause type ────────────────────────
# 0 = informational, 1 = low, 2 = medium, 3 = high
CLAUSE_RISK = {
    # High: one-sided obligations, hard to exit, uncapped exposure
    "Uncapped Liability":                 3,
    "Non-Compete":                        3,
    "Exclusivity":                        3,
    "Liquidated Damages":                 3,
    "IP Ownership Assignment":            3,
    "Irrevocable Or Perpetual License":   3,
    "Most Favored Nation":                3,
    "Covenant Not To Sue":                3,
    "Anti-Assignment":                    3,
    "Minimum Commitment":                 3,

    # Medium: negotiable but materially affects position
    "Cap On Liability":                   2,
    "Termination For Convenience":        2,
    "Change Of Control":                  2,
    "No-Solicit Of Employees":            2,
    "No-Solicit Of Customers":            2,
    "Non-Disparagement":                  2,
    "Price Restrictions":                 2,
    "Volume Restriction":                 2,
    "Revenue/Profit Sharing":             2,
    "ROFR/ROFO/ROFN":                     2,
    "Audit Rights":                       2,
    "Insurance":                          2,
    "Post-Termination Services":          2,
    "Source Code Escrow":                 2,
    "Warranty Duration":                  2,
    "Notice Period To Terminate Renewal": 2,
    "Third Party Beneficiary":            2,
    "Competitive Restriction Exception":  2,

    # Low: standard terms worth reading, rarely dangerous alone
    "Governing Law":                      1,
    "License Grant":                      1,
    "Non-Transferable License":           1,
    "Affiliate License-Licensee":         1,
    "Affiliate License-Licensor":         1,
    "Joint IP Ownership":                 1,
    "Renewal Term":                       1,
    "Expiration Date":                    1,
    "Unlimited/All-You-Can-Eat-License":  1,

    # Informational: metadata, not risk
    "Document Name":                      0,
    "Parties":                            0,
    "Agreement Date":                     0,
    "Effective Date":                     0,
}

DEFAULT_TYPE_RISK = 0   # unknown / unclassified contributes nothing on its own


# ── Keyword rules ───────────────────────────────────────────
# (regex, risk_bump, human-readable reason)
# These fire on the raw clause text regardless of predicted type.
KEYWORD_RULES = [
    (r"\bsole discretion\b|\babsolute discretion\b", 2,
     "Gives one party unilateral discretion"),

    (r"\bindemnif(y|ies|ication)\b", 2,
     "Indemnity obligation - potential open-ended liability"),

    (r"\bwithout limitation\b.{0,60}\bliab", 2,
     "Liability language without a stated cap"),

    (r"\bperpetual\b|\bin perpetuity\b", 2,
     "Perpetual commitment"),

    (r"\birrevocabl\w*\b(?!.{0,40}\bsubmits?\b)", 2,
     "Irrevocable commitment"),

    (r"\bwaive[sd]?\b.{0,40}\b(right|claim)", 2,
     "Waiver of rights"),

    (r"(?<!non-)(?<!non )\bexclusive\b.{0,40}\b(right|licen[cs]e|jurisdiction)", 2,
     "Exclusivity commitment"),

    (r"\bshall not\b.{0,60}\b(compete|solicit|engage)", 2,
     "Restrictive covenant on future activity"),

    (r"\bterminate\b.{0,50}\bat any time\b", 2,
     "Termination at will by one party"),

    (r"\bautomatically renew|\bauto-renew", 2,
     "Automatic renewal"),

    (r"\bliquidated damages\b|\bpenalt(y|ies)\b", 2,
     "Pre-agreed damages or penalties"),

    (r"\bassign\b.{0,50}\bwithout\b.{0,30}\bconsent\b", 1,
     "Assignment without consent"),

    (r"\bgoverned by the laws\b|\bgoverning law\b", 1,
     "Governing law / jurisdiction specified"),

    (r"\bconfidential", 1,
     "Confidentiality obligation"),

    (r"\bsurvive\b.{0,40}\btermination\b", 1,
     "Obligation survives termination"),

    (r"\bnotice\b.{0,40}\b(\d+|thirty|sixty|ninety)\s*(\(\d+\))?\s*days?\b", 1,
     "Time-bound notice requirement"),
]

# Compile once
_COMPILED_RULES = [(re.compile(p, re.IGNORECASE | re.DOTALL), bump, reason)
                   for p, bump, reason in KEYWORD_RULES]


# ── Scoring ─────────────────────────────────────────────────
def score_clause(clause):
    """
    Takes one dict from classifier.classify_clauses():
        {clause_text, clause_type, confidence}
    Returns the same dict plus:
        risk_score (int), risk_level (str), reasons (list of str)
    """
    text = clause["clause_text"]
    clause_type = clause["clause_type"]
    confidence = clause.get("confidence", 0.0)

    reasons = []

    # ── 1. Risk from predicted clause type ──────────────────
    type_risk = CLAUSE_RISK.get(clause_type, DEFAULT_TYPE_RISK)

    # Discount by classifier confidence so a shaky prediction
    # cannot single-handedly flag a clause as high risk.
    if type_risk > 0:
        weighted = type_risk * min(1.0, confidence / 0.35)
        if weighted >= 0.5:
            reasons.append(f"Clause type '{clause_type}' "
                           f"(confidence {confidence:.2f})")
    else:
        weighted = 0.0

    # ── 2. Risk from keyword rules ──────────────────────────
    rule_risk = 0
    for pattern, bump, reason in _COMPILED_RULES:
        if pattern.search(text):
            rule_risk += bump
            reasons.append(reason)

    # ── 3. Combine ──────────────────────────────────────────
    raw = weighted + rule_risk
    score = int(min(10, round(raw)))

    if score >= 5:
        level = "HIGH"
    elif score >= 2:
        level = "MEDIUM"
    elif score >= 1:
        level = "LOW"
    else:
        level = "NONE"

    result = dict(clause)
    result["risk_score"] = score
    result["risk_level"] = level
    result["reasons"] = reasons if reasons else ["No risk indicators found"]
    return result


def score_clauses(clauses):
    """
    Takes list of dicts from classifier.classify_clauses().
    Returns list of scored dicts. This is what dashboard.py consumes.
    """
    return [score_clause(c) for c in clauses]


def summarize(scored):
    """
    Contract-level rollup for the dashboard header.
    """
    if not scored:
        return {
            "total_clauses": 0, "high": 0, "medium": 0, "low": 0,
            "overall_score": 0, "overall_level": "NONE"
        }

    high = sum(1 for c in scored if c["risk_level"] == "HIGH")
    medium = sum(1 for c in scored if c["risk_level"] == "MEDIUM")
    low = sum(1 for c in scored if c["risk_level"] == "LOW")

   # Sum actual severity rather than counting level buckets (P24).
    # Level counting discards risk_score - a MEDIUM 4 counted the same
    # as a MEDIUM 2 - so merging related clauses (P20) lowered the total
    # even though the underlying risk was unchanged.
    total_words = sum(len(c["clause_text"].split()) for c in scored)
    total_risk = sum(c["risk_score"] for c in scored)
    density = total_risk / max(total_words, 1) * 1000
    overall = int(min(100, round(density * 3)))

    if overall >= 45:
        level = "HIGH"
    elif overall >= 25:
        level = "MEDIUM"
    elif overall > 0:
        level = "LOW"
    else:
        level = "NONE"

    return {
        "total_clauses": len(scored),
        "high": high,
        "medium": medium,
        "low": low,
        "overall_score": overall,
        "overall_level": level
    }


def analyze_pdf(pdf_path):
    """
    Full pipeline in one call: PDF -> scored clauses + summary.
    dashboard.py can call just this.
    """
    from segmenter import segment_pdf
    from classifier import classify_clauses

    clauses = segment_pdf(pdf_path)
    classified = classify_clauses(clauses)
    scored = score_clauses(classified)
    return scored, summarize(scored)


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        "data/sample_contract.pdf",
        "data/COE-Sample.pdf",
        "data/sample-service-agreement.pdf"
    ]

    for pdf in test_files:
        print("=" * 60)
        print(f"FILE: {pdf}")
        print("=" * 60)

        scored, summary = analyze_pdf(pdf)

        print(f"\nSUMMARY: {summary['total_clauses']} clauses | "
              f"HIGH {summary['high']} | MED {summary['medium']} | "
              f"LOW {summary['low']}")
        print(f"Overall risk: {summary['overall_score']}/100 "
              f"({summary['overall_level']})\n")

        for i, c in enumerate(scored, 1):
            print(f"--- Clause {i} [{c['risk_level']} {c['risk_score']}] ---")
            print(f"Type   : {c['clause_type']}")
            print(f"Reasons: {'; '.join(c['reasons'])}")
            print(f"Text   : {c['clause_text'][:120]}...")
            print()