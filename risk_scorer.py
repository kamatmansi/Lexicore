"""
risk_scorer.py
Assigns risk levels to classified clauses for LexiCore.
 
Hybrid approach:
  1. Clause-type risk weights  (uses classifier output)
  2. Keyword / pattern rules   (fires independently of the classifier)
 
Rule 2 exists because the classifier returns "General / Unclassified" for
clauses outside CUAD's domain (NDAs, employment contracts, leases). Rules
keep the tool useful in those cases.
 
Fixes in this version:
  P33 - most clause types now carry weight 0. Being *classified* is not risk.
        The previous weights were calibrated against a classifier that
        abstained on ~70% of clauses; LEGAL-BERT abstains on ~18%, so every
        dormant weight fired at once and routine clauses accumulated risk
        simply for being recognised.
  P35 - clause type keys did not match the classifier's label casing
        ("IP Ownership Assignment" vs "Ip Ownership Assignment"), so several
        high-risk types silently scored 0. Lookup is now case-insensitive.
"""
 
import re
 
# ── Risk weights by CUAD clause type ────────────────────────
# 0 = not a risk signal, 1 = low, 2 = medium, 3 = high
#
# Types absent from this table score 0. That is deliberate: a clause being
# identifiable as "Revenue/Profit Sharing" or "Governing Law" says what it
# IS, not whether it is dangerous. Only provisions that are genuinely
# one-sided, open-ended, or hard to exit carry weight.
CLAUSE_RISK = {
    # High: open-ended exposure or commitments that are hard to reverse
    "uncapped liability":                 3,
    "non-compete":                        3,
    "ip ownership assignment":            3,
    "irrevocable or perpetual license":   3,
    "liquidated damages":                 3,
    "covenant not to sue":                3,
 
    # Medium: materially restricts one party's freedom of action
    "exclusivity":                        2,
    "most favored nation":                2,
    "minimum commitment":                 2,
    "no-solicit of employees":            2,
    "no-solicit of customers":            2,
    "non-disparagement":                  2,
    "termination for convenience":        2,
    "volume restriction":                 2,
    "price restrictions":                 2,
 
    # Low: worth reading, rarely dangerous on its own
    "anti-assignment":                    1,
    "change of control":                  1,
    "cap on liability":                   1,
    "rofr/rofo/rofn":                     1,
    "post-termination services":          1,
    "source code escrow":                 1,
    "competitive restriction exception":  1,
    "joint ip ownership":                 1,
 
    # Everything else scores 0 by omission:
    #   Parties, Agreement Date, Effective Date, Expiration Date,
    #   Document Name, Governing Law, Renewal Term, Insurance, Audit Rights,
    #   Revenue/Profit Sharing, Warranty Duration, License Grant,
    #   Non-Transferable License, Affiliate License-Licensee,
    #   Affiliate License-Licensor, Third Party Beneficiary,
    #   Notice Period To Terminate Renewal, Unlimited/All-You-Can-Eat-License
}
 
DEFAULT_TYPE_RISK = 0    # unknown / unclassified contributes nothing
 
# P33: type risk only counts above this confidence. A 0.48 Anti-Assignment
# guess should not weigh the same as a 0.97 Insurance classification.
CONF_FLOOR = 0.60
 
# Confidence at which type risk reaches full weight.
CONF_FULL = 0.80
 
 
def _type_risk(clause_type):
    """Case-insensitive lookup (P35)."""
    return CLAUSE_RISK.get(clause_type.strip().lower(), DEFAULT_TYPE_RISK)
 
 
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
 
    # P22: was firing on any use of "penalties" (e.g. stamp duty clauses).
    # Now requires the penalty to be tied to breach or default.
    (r"\bliquidated damages\b|\bpenalt(?:y|ies)\b.{0,60}\b(breach|default|failure|terminat)", 2,
     "Pre-agreed damages or penalties"),
 
    (r"\bassign\b.{0,50}\bwithout\b.{0,30}\bconsent\b", 1,
     "Assignment without consent"),
 
    (r"\bgoverned by the laws\b|\bgoverning law\b", 1,
     "Governing law / jurisdiction specified"),
 
    (r"\bconfidential\w*\b.{0,80}\b(perpetu|indefinite|survive|no time limit|without limit)", 1,
     "Confidentiality obligation with no clear end date"),
 
    (r"\bsurvive\b.{0,40}\btermination\b", 1,
     "Obligation survives termination"),
 
    (r"\bnotice\b.{0,40}\b(\d+|thirty|sixty|ninety)\s*(\(\d+\))?\s*days?\b", 1,
     "Time-bound notice requirement"),
]
 
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
    type_risk = _type_risk(clause_type)
 
    if type_risk > 0 and confidence >= CONF_FLOOR:
        weighted = type_risk * min(1.0, confidence / CONF_FULL)
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
    # Level counting discards risk_score - a MEDIUM 4 counted the same as a
    # MEDIUM 2 - so merging related clauses (P20) lowered the total even
    # though the underlying risk was unchanged.
    #
    # Normalise by word count, not clause count (P17): clause count depends
    # on segmentation granularity, word count does not.
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
    dashboard.py calls this.
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
 