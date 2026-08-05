"""
dashboard.py
Streamlit UI for LexiCore.

Run with:  streamlit run dashboard.py
"""

import os
import json
import tempfile

import pandas as pd
import streamlit as st

from risk_scorer import score_clauses, summarize
from segmenter import segment_pdf
from classifier import classify_clauses

# ── Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="LexiCore", page_icon="§", layout="wide")

LEVEL_COLOR = {
    "HIGH":   "#c0392b",
    "MEDIUM": "#e67e22",
    "LOW":    "#f1c40f",
    "NONE":   "#95a5a6",
}


@st.cache_data(show_spinner=False)
def analyze(pdf_bytes, filename):
    """
    Cached so re-filtering the view does not re-run the pipeline.
    Takes raw bytes so Streamlit can hash the input.
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)

    clauses = segment_pdf(tmp_path)
    classified = classify_clauses(clauses)
    scored = score_clauses(classified)
    return scored, summarize(scored)


# ── Header ──────────────────────────────────────────────────
st.title("LexiCore")
st.caption("Legal contract risk analyzer - clause extraction, "
           "classification and risk flagging")

with st.expander("How this works / limitations", expanded=False):
    st.markdown("""
**Pipeline:** PDF text extraction (PyMuPDF) → clause segmentation (regex) →
clause type classification (TF-IDF + logistic regression, trained on CUAD) →
risk scoring (clause-type weights + keyword rules).

**Known limitations:**
- The classifier is trained on CUAD, which covers commercial/licensing
  agreements. Clause types outside that set are returned as
  *General / Unclassified*.
- Risk weights are heuristic, not learned - CUAD contains no risk labels.
- Clause segmentation is regex-based and may merge or split sections
  incorrectly on complex layouts.

**This is a decision-support tool, not legal advice.**
    """)

# ── Upload ──────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a contract (PDF)", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to begin. Sample files are in the `data/` folder.")
    st.stop()

with st.spinner("Analyzing contract..."):
    try:
        scored, summary = analyze(uploaded.getvalue(), uploaded.name)
    except FileNotFoundError as e:
        st.error(f"Model not found. Run `python classifier.py train` first.\n\n{e}")
        st.stop()

if not scored:
    st.warning("No clauses could be extracted from this PDF. "
               "It may be a scanned image rather than text.")
    st.stop()

# ── Summary ─────────────────────────────────────────────────
st.subheader("Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clauses", summary["total_clauses"])
c2.metric("High risk", summary["high"])
c3.metric("Medium risk", summary["medium"])
c4.metric("Low risk", summary["low"])
c5.metric("Overall", f"{summary['overall_score']}/100",
          summary["overall_level"])

st.progress(summary["overall_score"] / 100)

# ── Distribution chart ──────────────────────────────────────
df = pd.DataFrame(scored)

left, right = st.columns(2)

with left:
    st.markdown("**Risk distribution**")
    counts = df["risk_level"].value_counts().reindex(
        ["HIGH", "MEDIUM", "LOW", "NONE"], fill_value=0)
    st.bar_chart(counts)

with right:
    st.markdown("**Clause types detected**")
    types = df["clause_type"].value_counts().head(10)
    st.bar_chart(types)

# ── Filter ──────────────────────────────────────────────────
st.subheader("Clauses")

levels = st.multiselect(
    "Show risk levels",
    ["HIGH", "MEDIUM", "LOW", "NONE"],
    default=["HIGH", "MEDIUM", "LOW"]
)

visible = [c for c in scored if c["risk_level"] in levels]
st.caption(f"Showing {len(visible)} of {len(scored)} clauses")

# ── Clause cards ────────────────────────────────────────────
for i, c in enumerate(visible, 1):
    color = LEVEL_COLOR[c["risk_level"]]

    header = (f"{c['risk_level']}  ·  score {c['risk_score']}  ·  "
              f"{c['clause_type']}")

    with st.expander(header, expanded=(c["risk_level"] == "HIGH")):
        st.markdown(
            f"<div style='border-left:4px solid {color};padding-left:12px'>"
            f"<b>Why flagged</b><br>"
            + "<br>".join(f"• {r}" for r in c["reasons"]) +
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("**Clause text**")
        st.text(c["clause_text"])
        if c["clause_type"] != "General / Unclassified":
            st.caption(f"Classifier confidence: {c['confidence']}")

# ── Export ──────────────────────────────────────────────────
st.subheader("Export")

e1, e2 = st.columns(2)

with e1:
    st.download_button(
        "Download JSON",
        data=json.dumps({"summary": summary, "clauses": scored}, indent=2),
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_analysis.json",
        mime="application/json"
    )

with e2:
    export_df = df[["clause_type", "confidence", "risk_level",
                    "risk_score", "clause_text"]].copy()
    export_df["reasons"] = df["reasons"].apply(lambda r: "; ".join(r))
    st.download_button(
        "Download CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_analysis.csv",
        mime="text/csv"
    )