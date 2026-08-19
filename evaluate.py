"""
evaluate.py
Measures the LexiCore risk layer against a contract with a known answer key.

Run:  python evaluate.py

The answer key is defined below. Ground truth is keyed by the clause NUMBER
printed in the contract (1-11), not by the segmenter's chunk index, because
segmentation may merge or split chunks.
"""

import re
from segmenter import segment_pdf
from classifier import classify_clauses
from risk_scorer import score_clauses, summarize

PDF = "data/adversarial_test.pdf"

# Clause number -> (short name, should_be_flagged)
ANSWER_KEY = {
    1:  ("Definitions",                False),
    2:  ("Indemnity",                  True),
    3:  ("Notices",                    False),
    4:  ("IP Assignment",              True),
    5:  ("Severability",               False),
    6:  ("Restraint of Trade",         True),
    7:  ("Entire Agreement",           False),
    8:  ("Fees and Payment",           True),
    9:  ("Governing Law",              False),
    10: ("Term and Termination",       True),
    11: ("Counterparts",               False),
}

FLAGGED_LEVELS = {"MEDIUM", "HIGH"}


def clause_number(text):
    """Reads the leading clause number, e.g. '4. Assignment of...' -> 4"""
    m = re.match(r"^\s*(\d{1,2})[.\s]", text)
    if m:
        n = int(m.group(1))
        return n if n in ANSWER_KEY else None
    return None


def main():
    clauses = segment_pdf(PDF)
    scored = score_clauses(classify_clauses(clauses))
    summary = summarize(scored)

    # Map each scored chunk back to a contract clause number.
    # If segmentation merged clauses, the highest risk in the chunk applies
    # to the clause the chunk starts with.
    by_clause = {}
    unmapped = []

    for c in scored:
        n = clause_number(c["clause_text"])
        if n is None:
            unmapped.append(c)
            continue
        prev = by_clause.get(n)
        if prev is None or c["risk_score"] > prev["risk_score"]:
            by_clause[n] = c

    # ── Per-clause table ────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'#':<3} {'CLAUSE':<22} {'EXPECT':<8} {'GOT':<8} {'SCORE':<6} {'':<4} TYPE")
    print("=" * 78)

    tp = fp = tn = fn = 0
    missing = []

    for n, (name, should_flag) in sorted(ANSWER_KEY.items()):
        c = by_clause.get(n)

        if c is None:
            missing.append(n)
            print(f"{n:<3} {name:<22} {'FLAG' if should_flag else 'clear':<8} "
                  f"{'NOT FOUND':<8}")
            if should_flag:
                fn += 1
            continue

        got_flag = c["risk_level"] in FLAGGED_LEVELS

        if should_flag and got_flag:
            tp += 1; mark = "OK"
        elif should_flag and not got_flag:
            fn += 1; mark = "MISS"
        elif not should_flag and got_flag:
            fp += 1; mark = "FALSE"
        else:
            tn += 1; mark = "OK"

        print(f"{n:<3} {name:<22} {'FLAG' if should_flag else 'clear':<8} "
              f"{c['risk_level']:<8} {c['risk_score']:<6} {mark:<6} "
              f"{c['clause_type'][:24]}")

    # ── Metrics ─────────────────────────────────────────────
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"True positives  : {tp}  (dangerous clauses correctly flagged)")
    print(f"False negatives : {fn}  (dangerous clauses MISSED)")
    print(f"False positives : {fp}  (safe clauses wrongly flagged)")
    print(f"True negatives  : {tn}  (safe clauses correctly left alone)")
    print()
    print(f"Precision : {precision:.2f}   of what was flagged, how much deserved it")
    print(f"Recall    : {recall:.2f}   of what was dangerous, how much was caught")
    print(f"F1        : {f1:.2f}")
    print()
    print(f"Document score: {summary['overall_score']}/100 "
          f"({summary['overall_level']})   -- expected HIGH")

    if missing:
        print(f"\nNOTE: clauses not matched to output: {missing}")
    if unmapped:
        print(f"NOTE: {len(unmapped)} chunks had no clause number "
              f"(headers/signature blocks)")

    # ── Regression checks ───────────────────────────────────
    print("\n" + "=" * 78)
    print("REGRESSION CHECKS")
    print("=" * 78)

    c9 = by_clause.get(9)
    if c9:
        bad = [r for r in c9["reasons"] if "xclusiv" in r]
        print(f"P16 (non-exclusive jurisdiction must not flag as exclusivity): "
              f"{'FAIL - ' + str(bad) if bad else 'pass'}")

    c4 = by_clause.get(4)
    if c4:
        hit = "ownership" in c4["clause_type"].lower()
        print(f"P35 (IP Ownership Assignment must now score): "
              f"{'pass' if hit or c4['risk_score'] > 0 else 'FAIL - scored 0'}")
        print(f"     clause 4 -> {c4['clause_type']} "
              f"({c4['confidence']}), score {c4['risk_score']}")

    # ── Detail on anything wrong ────────────────────────────
    wrong = [(n, by_clause[n]) for n, (nm, sf) in ANSWER_KEY.items()
             if n in by_clause
             and (by_clause[n]["risk_level"] in FLAGGED_LEVELS) != sf]

    if wrong:
        print("\n" + "=" * 78)
        print("INCORRECT CLASSIFICATIONS - detail")
        print("=" * 78)
        for n, c in wrong:
            name, should = ANSWER_KEY[n]
            print(f"\nClause {n} ({name}) - expected "
                  f"{'FLAG' if should else 'clear'}, got {c['risk_level']}")
            print(f"  type    : {c['clause_type']} ({c['confidence']})")
            print(f"  reasons : {'; '.join(c['reasons'])}")
            print(f"  text    : {c['clause_text'][:160]}...")


if __name__ == "__main__":
    main()