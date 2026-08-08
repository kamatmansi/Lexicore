# ============================================================
# LexiCore — classifier.py
# Loads the trained LEGAL-BERT model and classifies clause text.
# Used by dashboard.py (and anything else that needs predictions).
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Point this at wherever you place the downloaded model folder locally,
# e.g. "models/legal_bert_final" — or the Drive path if still in Colab.
MODEL_PATH = "models/legal_bert_final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = None
_model = None


def load_model(model_path: str = MODEL_PATH):
    """Load tokenizer + model once. Call this at app startup, not per clause —
    reloading a ~420MB model for every clause would be extremely slow."""
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _model.to(device)
    _model.eval()  # inference mode: disables dropout etc.
    return _tokenizer, _model


def classify_clause(text: str, max_length: int = 512) -> dict:
    """Classify a single clause. Returns predicted type + confidence."""
    if _model is None or _tokenizer is None:
        load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():  # no gradient tracking needed — this is inference, not training
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)

    predicted_id = predicted_id.item()
    confidence = confidence.item()
    label = _model.config.id2label[predicted_id]

    return {
        "clause_type": label,
        "confidence": round(confidence, 4),
    }


def classify_clauses(clauses: list[str]) -> list[dict]:
    """Classify a list of clause texts (e.g. output of segmenter.py)."""
    load_model()  # ensures model is loaded once before the loop
    results = []
    for clause in clauses:
        result = classify_clause(clause)
        result["text"] = clause
        results.append(result)
    return results


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check the model loads
    # and predicts something sensible before wiring it into the rest of the app.
    sample_clause = "Either party may terminate this Agreement upon 30 days written notice."
    load_model()
    print(classify_clause(sample_clause))