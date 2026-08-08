# ============================================================
# LexiCore — dashboard.py
# Streamlit UI. Upload a PDF -> runs the full pipeline
# (parse -> segment -> classify -> risk score -> summarize) ->
# saves to database -> displays results. Also shows a sidebar
# history of previously analyzed contracts.
#
# Run with:  streamlit run dashboard.py
# (NOT python dashboard.py — Streamlit needs its own runner)
# ============================================================

import streamlit as st
import tempfile
import os

from segmenter import segment_pdf
from classifier import classify_clauses
from risk_scorer import score_clauses
from summarizer import summarize_clauses
from database import init_db, save_contract, get_all_contracts, get_clauses_for_contract

# ── Page setup ────────────────────────────────────────────
st.set_page_config(page_title="LexiCore", page_icon="⚖️", layout="wide")
init_db()  # make sure tables exist before anything else runs

RISK_COLORS = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}


def display_clauses(clauses):
    """Renders a list of clause dicts: text, clause_type, confidence, risk, summary."""
    for i, c in enumerate(clauses):
        color = RISK_COLORS.get(c["risk"], "#888888")
        with st.expander(f"Clause {i+1} — {c['clause_type']} ({c['risk']} risk)"):
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{c['risk']} RISK</span> "
                f"&nbsp;·&nbsp; confidence: {c['confidence']:.2%}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Summary:** {c['summary']}")
            st.caption("Full clause text")
            st.write(c["text"])


# ── Sidebar: history ──────────────────────────────────────
st.sidebar.title("📁 Analysis History")
contracts = get_all_contracts()

selected_contract_id = None
if contracts:
    options = {f"{c['filename']} ({c['upload_date'][:10]})": c["id"] for c in contracts}
    choice = st.sidebar.selectbox("View a past analysis", ["-- New Analysis --"] + list(options.keys()))
    if choice != "-- New Analysis --":
        selected_contract_id = options[choice]
else:
    st.sidebar.write("No contracts analyzed yet.")

# ── Main area ─────────────────────────────────────────────
st.title("⚖️ LexiCore — Legal Contract Risk Analyzer")

if selected_contract_id:
    # ---- Viewing a saved past analysis (no re-processing needed) ----
    contract = next(c for c in contracts if c["id"] == selected_contract_id)
    st.subheader(f"📄 {contract['filename']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clauses", contract["total_clauses"])
    col2.metric("High Risk", contract["high_risk_count"])
    col3.metric("Medium Risk", contract["medium_risk_count"])
    col4.metric("Low Risk", contract["low_risk_count"])

    display_clauses(get_clauses_for_contract(selected_contract_id))

else:
    # ---- New analysis ----
    uploaded_file = st.file_uploader("Upload a contract (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Analyze Contract"):
            # Save uploaded file to a temp path — parser.py needs a real file path, not raw bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Extracting and segmenting clauses..."):
                clause_texts = segment_pdf(tmp_path)

            if not clause_texts:
                st.error("Could not extract any clauses from this PDF. Try a different file.")
            else:
                with st.spinner(f"Classifying {len(clause_texts)} clauses with LEGAL-BERT..."):
                    classified = classify_clauses(clause_texts)

                with st.spinner("Scoring risk..."):
                    scored = score_clauses(classified)

                with st.spinner("Generating summaries..."):
                    final_clauses = summarize_clauses(scored)

                with st.spinner("Saving to database..."):
                    contract_id = save_contract(uploaded_file.name, final_clauses)

                os.remove(tmp_path)  # clean up the temp PDF

                st.success(f"Analysis complete — {len(final_clauses)} clauses processed.")

                high = sum(1 for c in final_clauses if c["risk"] == "High")
                medium = sum(1 for c in final_clauses if c["risk"] == "Medium")
                low = sum(1 for c in final_clauses if c["risk"] == "Low")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Clauses", len(final_clauses))
                col2.metric("High Risk", high)
                col3.metric("Medium Risk", medium)
                col4.metric("Low Risk", low)

                display_clauses(final_clauses)
    else:
        st.info("👆 Upload a PDF contract to get started.")
