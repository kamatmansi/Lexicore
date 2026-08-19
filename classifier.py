"""
classifier.py
Clause type classification for LexiCore.

Primary model: LEGAL-BERT (nlpaueb/legal-bert-base-uncased) fine-tuned on
CUAD paragraph-level data. Checkpoint lives in models/legalbert-cuad/.

Fallback: TF-IDF + LogisticRegression baseline (models/baseline_clf.joblib),
kept so the two can be benchmarked against each other and so the pipeline
still runs if the transformer checkpoint is missing.

Public interface is unchanged - risk_scorer.py and dashboard.py call:
    classify_clause(text)     -> {clause_type, confidence}
    classify_clauses(clauses) -> [{clause_text, clause_type, confidence}]
"""

import os
import json

# ── Config ──────────────────────────────────────────────────
BERT_DIR = os.path.join("models", "legalbert-cuad")
BASELINE_PATH = os.path.join("models", "baseline_clf.joblib")

# 37 classes: random guessing is ~0.027, so a prediction at 0.30 is the
# model being fairly confident. Calibrate against 1/n_classes, not intuition.
MIN_CONFIDENCE = 0.30
UNKNOWN_LABEL = "General / Unclassified"

BATCH_SIZE = 16
MAX_LENGTH = 256          # matches the fine-tuning setting

_BERT = None              # lazy-loaded (tokenizer, model, device)
_BASELINE = None


# ── LEGAL-BERT ──────────────────────────────────────────────
def load_bert():
    """
    Loads the fine-tuned checkpoint once and caches it.
    First call takes ~15-20s on CPU while 110M parameters initialise.
    """
    global _BERT
    if _BERT is not None:
        return _BERT

    if not os.path.isdir(BERT_DIR):
        raise FileNotFoundError(
            f"No LEGAL-BERT checkpoint at {BERT_DIR}. "
            "Extract legalbert-cuad.zip there, or use backend='baseline'."
        )

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
    model.eval()                      # inference mode: no dropout, no grad

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _BERT = (tokenizer, model, device)
    return _BERT


def _classify_bert(texts):
    """
    Batched inference. Returns list of (label, confidence).
    """
    import torch

    tokenizer, model, device = load_bert()
    out = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)

        with torch.no_grad():         # no gradients needed for inference
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)

        conf, idx = probs.max(dim=-1)

        for c, j in zip(conf.tolist(), idx.tolist()):
            out.append((model.config.id2label[j], float(c)))

    return out


# ── TF-IDF baseline (kept for comparison) ───────────────────
def load_baseline():
    global _BASELINE
    if _BASELINE is None:
        import joblib
        if not os.path.exists(BASELINE_PATH):
            raise FileNotFoundError(
                f"No baseline model at {BASELINE_PATH}. "
                "Run: python classifier.py train"
            )
        _BASELINE = joblib.load(BASELINE_PATH)
    return _BASELINE


def _classify_baseline(texts):
    bundle = load_baseline()
    vecs = bundle["vectorizer"].transform(texts)
    probs = bundle["model"].predict_proba(vecs)

    out = []
    for row in probs:
        j = row.argmax()
        out.append((bundle["model"].classes_[j], float(row[j])))
    return out


# ── Public interface ────────────────────────────────────────
def classify_clauses(clauses, backend="bert"):
    """
    Takes list of clause strings (output of segmenter.split_into_clauses).
    Returns list of {clause_text, clause_type, confidence}.

    backend: 'bert' (default) or 'baseline'
    """
    if not clauses:
        return []

    if backend == "bert":
        preds = _classify_bert(clauses)
    elif backend == "baseline":
        preds = _classify_baseline(clauses)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    results = []
    for text, (label, conf) in zip(clauses, preds):
        if conf < MIN_CONFIDENCE:
            label = UNKNOWN_LABEL
        results.append({
            "clause_text": text,
            "clause_type": label,
            "confidence": round(conf, 3),
        })
    return results


def classify_clause(text, backend="bert"):
    """
    Single clause. Returns {clause_type, confidence}.
    Prefer classify_clauses() for multiple - batching is much faster.
    """
    r = classify_clauses([text], backend=backend)[0]
    return {"clause_type": r["clause_type"], "confidence": r["confidence"]}


# ── Baseline training (unchanged, for reproducing the comparison) ──
def train_baseline():
    """
    Trains the TF-IDF baseline on the paragraph-level dataset, using the
    same contract-level split as LEGAL-BERT so the two are comparable.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import joblib

    path = os.path.join("data", "cuad_paragraph.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it with the Colab extraction cell."
        )

    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    X_train = [r["clause_text"] for r in d["train"]]
    y_train = [r["clause_type"] for r in d["train"]]
    X_test = [r["clause_text"] for r in d["test"]]
    y_test = [r["clause_type"] for r in d["test"]]

    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=50000, sublinear_tf=True,
                                 strip_accents="unicode")
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=5.0, n_jobs=-1)
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    print(f"accuracy    : {accuracy_score(y_test, pred):.4f}")
    print(f"f1_macro    : {f1_score(y_test, pred, average='macro', zero_division=0):.4f}")
    print(f"f1_weighted : {f1_score(y_test, pred, average='weighted', zero_division=0):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model,
                 "labels": sorted(set(y_train))}, BASELINE_PATH)
    print(f"Saved -> {BASELINE_PATH}")


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_baseline()

    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Side-by-side on a real contract
        from segmenter import segment_pdf

        pdf = sys.argv[2] if len(sys.argv) > 2 else "data/COE-Sample.pdf"
        clauses = segment_pdf(pdf)

        bert = classify_clauses(clauses, backend="bert")
        base = classify_clauses(clauses, backend="baseline")

        print("\n" + "=" * 78)
        print(f"{'BERT':<34} | {'BASELINE':<34}")
        print("=" * 78)
        for b, s, text in zip(bert, base, clauses):
            mark = " " if b["clause_type"] == s["clause_type"] else "*"
            print(f"{mark}{b['clause_type'][:26]:<27}{b['confidence']:.2f} "
                  f"| {s['clause_type'][:26]:<27}{s['confidence']:.2f}")
            print(f"   {text[:70]}...")
        print("\n* = models disagree")

    else:
        from segmenter import segment_pdf

        pdf = sys.argv[1] if len(sys.argv) > 1 else "data/sample_contract.pdf"
        clauses = segment_pdf(pdf)
        results = classify_clauses(clauses)

        print("\n" + "=" * 60)
        print(f"CLASSIFIED {len(results)} CLAUSES FROM {pdf}")
        print("=" * 60)
        for i, r in enumerate(results, 1):
            print(f"\n--- Clause {i} ---")
            print(f"Type       : {r['clause_type']}")
            print(f"Confidence : {r['confidence']}")
            print(f"Text       : {r['clause_text'][:150]}...")