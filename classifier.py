"""
classifier.py
Clause type classification for LexiCore.

Baseline model: TF-IDF + Logistic Regression.
Trains on data/cuad_cleaned.json, saves to models/.
LEGAL-BERT can replace classify_clause() later without touching
risk_scorer.py or dashboard.py.
"""

import json
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Config ──────────────────────────────────────────────────
DATA_PATH = "data/cuad_cleaned.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_clf.joblib")

MIN_SAMPLES_PER_CLASS = 20   # drop clause types too rare to learn
MIN_CONFIDENCE = 0.25        # below this -> "General / Unclassified"
UNKNOWN_LABEL = "General / Unclassified"

_MODEL = None  # lazy-loaded cache


# ── Data loading ────────────────────────────────────────────
def load_training_data(path=DATA_PATH):
    """
    Returns (texts, labels) from cuad_cleaned.json.
    Drops empty text and very rare clause types.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    # Count how many samples each clause type has
    counts = {}
    for r in records:
        counts[r["clause_type"]] = counts.get(r["clause_type"], 0) + 1

    texts, labels = [], []
    for r in records:
        text = (r.get("clause_text") or "").strip()
        label = r["clause_type"]

        if len(text.split()) < 3:
            continue
        if counts[label] < MIN_SAMPLES_PER_CLASS:
            continue

        texts.append(text)
        labels.append(label)

    kept = sorted(set(labels))
    print(f"Loaded {len(records)} records -> {len(texts)} usable")
    print(f"Clause types kept: {len(kept)} (dropped types with "
          f"<{MIN_SAMPLES_PER_CLASS} samples)")

    return texts, labels


# ── Training ────────────────────────────────────────────────
def train(save=True):
    """
    Trains the baseline classifier and prints a report.
    Returns the fitted (vectorizer, model, labels) bundle.
    """
    texts, labels = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True
    )

    print("Vectorizing...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Feature count: {X_train_vec.shape[1]}")

    print("Training logistic regression...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # CUAD is heavily imbalanced
        C=5.0,
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)

    # ── Evaluate ────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = (y_pred == y_test).mean() if hasattr(y_pred, "mean") else \
        sum(p == t for p, t in zip(y_pred, y_test)) / len(y_test)
    print(f"Overall accuracy: {accuracy:.3f}")

    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "labels": sorted(set(labels))
    }

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        print(f"\nSaved model -> {MODEL_PATH}")

    return bundle


# ── Inference ───────────────────────────────────────────────
def load_model():
    """Loads the saved model once and caches it."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No model at {MODEL_PATH}. Run: python classifier.py train"
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def classify_clause(text):
    """
    Takes one clause string.
    Returns dict: {clause_type, confidence}
    """
    bundle = load_model()
    vec = bundle["vectorizer"].transform([text])
    probs = bundle["model"].predict_proba(vec)[0]

    best_idx = probs.argmax()
    label = bundle["model"].classes_[best_idx]
    confidence = float(probs[best_idx])

    if confidence < MIN_CONFIDENCE:
        label = UNKNOWN_LABEL

    return {"clause_type": label, "confidence": round(confidence, 3)}


def classify_clauses(clauses):
    """
    Takes list of clause strings (output of segmenter.split_into_clauses).
    Returns list of dicts: {clause_text, clause_type, confidence}
    This is the format risk_scorer.py will consume.
    """
    results = []
    for clause in clauses:
        pred = classify_clause(clause)
        results.append({
            "clause_text": clause,
            "clause_type": pred["clause_type"],
            "confidence": pred["confidence"]
        })
    return results


# ── TEST / CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        # Run the full pipeline on a sample PDF
        from segmenter import segment_pdf

        pdf = "data/sample_contract.pdf"
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