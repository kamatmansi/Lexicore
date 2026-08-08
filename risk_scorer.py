# ============================================================
# LexiCore — risk_scorer.py
# Maps a predicted clause type (from classifier.py) to a risk level.
# This is a manual/rule-based mapping, not ML — the reasoning behind
# each grouping is worth a paragraph in your report.
# ============================================================

# Covers the standard 41 CUAD clause categories. If your cleaned dataset
# uses a slightly different 36, any unmatched clause type will safely
# fall back to "Medium" (see get_risk() below) rather than crash.
RISK_MAP = {
    # --- High risk: significant financial, legal, or strategic exposure ---
    "Uncapped Liability": "High",
    "Liquidated Damages": "High",
    "Non-Compete": "High",
    "Exclusivity": "High",
    "Most Favored Nation": "High",
    "Ip Ownership Assignment": "High",  # matches model's exact casing (not "IP")
    "Joint Ip Ownership": "High",       # matches model's exact casing (not "IP")
    "Unlimited/All-You-Can-Eat-License": "High",
    "Irrevocable Or Perpetual License": "High",
    "Change Of Control": "High",
    "Minimum Commitment": "High",
    "Volume Restriction": "High",
    "Revenue/Profit Sharing": "High",
    "Price Restrictions": "High",
    "Covenant Not To Sue": "High",
    "Rofr/Rofo/Rofn": "High",  # right of first refusal/offer/negotiation — restricts future business freedom

    # --- Medium risk: meaningful obligations, but more standard/negotiable ---
    "Termination For Convenience": "Medium",
    "Anti-Assignment": "Medium",
    "Non-Transferable License": "Medium",
    "Affiliate License-Licensor": "Medium",
    "Affiliate License-Licensee": "Medium",
    "Cap On Liability": "Medium",  # limits exposure, but still worth reviewing the cap amount
    "Audit Rights": "Medium",
    "Post-Termination Services": "Medium",
    "Source Code Escrow": "Medium",
    "Warranty Duration": "Medium",
    "No-Solicit Of Employees": "Medium",
    "No-Solicit Of Customers": "Medium",
    "Competitive Restriction Exception": "Medium",
    "Governing Law": "Medium",
    "Renewal Term": "Medium",
    "Notice Period To Terminate Renewal": "Medium",
    "Third Party Beneficiary": "Medium",
    "Non-Disparagement": "Medium",
    "License Grant": "Medium",

    # --- Low risk: administrative / informational, rarely disputed ---
    "Document Name": "Low",
    "Parties": "Low",
    "Agreement Date": "Low",
    "Effective Date": "Low",
    "Expiration Date": "Low",
    "Insurance": "Low",
}


def get_risk(clause_type: str) -> str:
    """Return risk level for a clause type. Defaults to Medium if unseen —
    a safe middle ground rather than silently under- or over-flagging."""
    return RISK_MAP.get(clause_type, "Medium")


def score_clauses(classified_clauses: list[dict]) -> list[dict]:
    """Takes the output of classifier.classify_clauses() and adds a 'risk' key
    to each clause dict in place."""
    for item in classified_clauses:
        item["risk"] = get_risk(item["clause_type"])
    return classified_clauses


if __name__ == "__main__":
    # Quick manual test
    sample = {"clause_type": "Termination For Convenience", "confidence": 0.8371}
    sample["risk"] = get_risk(sample["clause_type"])
    print(sample)