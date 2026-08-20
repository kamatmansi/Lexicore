"""
dashboard.py
Streamlit UI for LexiCore.

Run with:  streamlit run dashboard.py

Layout follows the order a user actually needs:
  Overview -> Summary -> Charts -> All clauses -> Export

Fixes in this version:
  FR6  - summary section wired in (summarizer.py)
  P18  - chart categories no longer reordered alphabetically by Streamlit
  P40  - CUAD acronyms rendered correctly in the UI ("Ip" -> "IP")
  NFR5 - uploaded files are now deleted after analysis rather than
         accumulating in the system temp directory
  NFR2 - analysis time measured and displayed
"""

import os
import json
import shutil
import tempfile
import time

import pandas as pd
import streamlit as st

from segmenter import segment_pdf
from classifier import classify_clauses
from risk_scorer import score_clauses, summarize as risk_summarize
from summarizer import summarize as build_summary, format_text

# ── Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="LexiCore", page_icon="§", layout="wide")

LEVEL_COLOR = {
    "HIGH":   "#c0392b",
    "MEDIUM": "#e67e22",
    "LOW":    "#f1c40f",
    "NONE":   "#95a5a6",
}

# P18: Streamlit's bar_chart sorts the index alphabetically, which rendered
# the risk distribution as HIGH, LOW, MEDIUM, NONE. Numeric prefixes force
# the intended severity order.
LEVEL_ORDER = {
    "HIGH":   "1 HIGH",
    "MEDIUM": "2 MEDIUM",
    "LOW":    "3 LOW",
    "NONE":   "4 NONE",
}


def pretty_type(t):
    """
    P40: CUAD labels are title-cased, so acronyms render as
    'Ip Ownership Assignment' and 'Rofr/Rofo/Rofn'. Restore them for
    display only - the underlying labels are left untouched so
    CLAUSE_RISK lookups and CSV exports still match.
    """
    for a, b in [("Ip ", "IP "), ("Rofr", "ROFR"), ("Rofo", "ROFO"),
                 ("Rofn", "ROFN"), ("Gst", "GST")]:
        t = t.replace(a, b)
    return t


@st.cache_data(show_spinner=False)
def analyze(pdf_bytes, filename):
    """
    Runs the full pipeline. Cached on file contents so re-filtering the
    view does not re-run analysis.

    NFR5: the uploaded file is written to a temporary directory, read, and
    the directory removed in a finally block. Previously these accumulated
    for the life of the machine.
    """
    tmp_dir = tempfile.mkdtemp(prefix="lexicore_")
    tmp_path = os.path.join(tmp_dir, filename)

    try:
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)

        started = time.perf_counter()

        clauses = segment_pdf(tmp_path)
        classified = classify_clauses(clauses)
        scored = score_clauses(classified)
        stats = risk_summarize(scored)
        summary = build_summary(scored, stats)

        elapsed = time.perf_counter() - started

        return scored, stats, summary, elapsed

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Header ──────────────────────────────────────────────────
st.title("LexiCore")
st.caption("Legal contract risk analyser - clause extraction, "
           "classification, risk scoring and summarisation")

with st.expander("How this works, and what it cannot do"):
    st.markdown("""
**Pipeline:** PDF text extraction (PyMuPDF) → rule-based clause segmentation →
clause classification (LEGAL-BERT fine-tuned on CUAD, 0.752 accuracy on a
contract-level split) → hybrid risk scoring (clause-type weights + keyword
rules) → TextRank summarisation.

**Known limitations:**
- The classifier is trained on CUAD, which covers commercial, distribution and
  licensing agreements. Employment contracts, NDAs and leases fall outside that
  domain and are classified less reliably.
- Risk weights are heuristic. CUAD contains no risk labels, and no public
  dataset annotates contract clauses by risk, because risk depends on which
  party you are.
- The system cannot detect what is **absent**. A contract with no liability cap
  is more dangerous than one with a poor cap, but only clauses that exist can
  be classified.
- It cannot assess **enforceability**. A restraint that is routine in one
  jurisdiction may be void in another.

**This is a triage tool. It tells you which clauses to read. It is not legal
advice.**
    """)

# ── Upload ──────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a contract (PDF)", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to begin. Sample contracts are in the `data/` folder.")
    st.stop()

# Load the model before timing, so the ~18s one-off initialisation of
# LEGAL-BERT's 110M parameters is not counted against per-contract
# analysis time (NFR2). In a deployed service the model is loaded once
# at startup, not per request.
from classifier import load_bert

if "model_loaded" not in st.session_state:
    with st.spinner("Loading LEGAL-BERT (one-time, ~20s)..."):
        load_bert()
    st.session_state.model_loaded = True

with st.spinner("Analysing contract..."):
    try:
        scored, stats, summary, elapsed = analyze(uploaded.getvalue(),
                                                  uploaded.name)
    except FileNotFoundError as e:
        st.error(f"Model not found. Extract the LEGAL-BERT checkpoint into "
                 f"`models/legalbert-cuad/` first.\n\n{e}")
        st.stop()

if not scored:
    st.warning("No clauses could be extracted. This PDF may be a scanned "
               "image with no text layer.")
    st.stop()

# ── Overview ────────────────────────────────────────────────
st.subheader("Contract overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clauses", stats["total_clauses"])
c2.metric("High risk", stats["high"])
c3.metric("Medium risk", stats["medium"])
c4.metric("Low risk", stats["low"])
c5.metric("Overall", f"{stats['overall_score']}/100",
          stats["overall_level"])

st.progress(stats["overall_score"] / 100)
st.caption(f"Analysed in {elapsed:.1f}s  ·  "
           f"{stats['total_clauses']} clauses")

# ── Summary ─────────────────────────────────────────────────
st.subheader("Summary")

for line in summary["overview"]:
    st.write(line)

if summary["provisions"]:
    st.markdown("**Key provisions** — ranked by importance to the contract "
                "and level of risk")

    for i, p in enumerate(summary["provisions"], 1):
        color = LEVEL_COLOR[p["risk_level"]]
        st.markdown(
            f"<div style='border-left:4px solid {color};"
            f"padding:6px 0 6px 12px;margin:10px 0'>"
            f"<b>{i}. {pretty_type(p['clause_type'])}</b> &nbsp;·&nbsp; "
            f"<span style='color:{color}'>{p['risk_level']}</span> "
            f"(risk {p['risk_score']}/10)<br>"
            f"<i>\"{p['quote']}\"</i>"
            + (f"<br><small>Flagged for: {'; '.join(p['reasons'])}</small>"
               if p["reasons"] else "")
            + "</div>",
            unsafe_allow_html=True
        )

if summary["risk_areas"]:
    st.markdown("**Main risk areas**")
    area_df = pd.DataFrame(
        [(a, n) for a, n, _ in summary["risk_areas"]],
        columns=["Risk area", "Clauses"]
    )
    st.dataframe(area_df, hide_index=True, use_container_width=True)

# ── Charts ──────────────────────────────────────────────────
st.subheader("Distribution")

df = pd.DataFrame(scored)
left, right = st.columns(2)

with left:
    st.markdown("**Risk levels**")
    counts = df["risk_level"].value_counts()
    ordered = pd.Series(
        {LEVEL_ORDER[k]: counts.get(k, 0) for k in
         ["HIGH", "MEDIUM", "LOW", "NONE"]}
    )
    st.bar_chart(ordered)

with right:
    st.markdown("**Clause types detected**")
    types = df[df["clause_type"] != "General / Unclassified"]["clause_type"]
    if len(types):
        named = types.map(pretty_type).value_counts().head(10)
        st.bar_chart(named)
    else:
        st.caption("No clauses were confidently classified.")

# ── All clauses ─────────────────────────────────────────────
st.subheader("All clauses")

levels = st.multiselect(
    "Show risk levels",
    ["HIGH", "MEDIUM", "LOW", "NONE"],
    default=["HIGH", "MEDIUM", "LOW"]
)

visible = [c for c in scored if c["risk_level"] in levels]
st.caption(f"Showing {len(visible)} of {len(scored)} clauses")

for c in visible:
    color = LEVEL_COLOR[c["risk_level"]]
    header = (f"{c['risk_level']}  ·  score {c['risk_score']}  ·  "
              f"{pretty_type(c['clause_type'])}")

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
stem = uploaded.name.rsplit(".", 1)[0]

e1, e2, e3 = st.columns(3)

with e1:
    st.download_button(
        "Summary (text)",
        data=format_text(summary),
        file_name=f"{stem}_summary.txt",
        mime="text/plain"
    )

with e2:
    st.download_button(
        "Full analysis (JSON)",
        data=json.dumps({"stats": stats, "summary": summary,
                         "clauses": scored}, indent=2),
        file_name=f"{stem}_analysis.json",
        mime="application/json"
    )

with e3:
    export_df = df[["clause_type", "confidence", "risk_level",
                    "risk_score", "clause_text"]].copy()
    export_df["reasons"] = df["reasons"].apply("; ".join)
    st.download_button(
        "Clauses (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_clauses.csv",
        mime="text/csv"
    )