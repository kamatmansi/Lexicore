"""
database.py
SQLite persistence for LexiCore.

Spec reference: FR9 - "Processed contracts, extracted clauses, risk scores,
and summaries must be stored persistently in the SQLite database."
Architecture section 7.1.3 specifies four tables.

WHY SQLITE
SQLite is a single file on disk (lexicore.db). No server process, no install,
no configuration - it ships with Python. That suits a single-user desktop tool.
Its one real limitation is that it serialises writes: many simultaneous readers
are fine, but only one writer at a time. Irrelevant here; worth naming as a
scaling limitation.

WHY FOUR TABLES RATHER THAN ONE
A contract has many clauses. Storing everything in one table would repeat the
filename and upload date once per clause - 70 times for the service agreement.
Changing one would mean updating 70 rows, and missing one would leave
contradictory data. Instead each fact is stored once, and rows point at each
other with foreign keys. This is normalisation.

    contracts (1) --< clauses (many) --< risk_scores (1 per clause)
              (1) --< summaries (1 per contract)

WHY THIS INTERFACE
Everything below the public functions is SQL. Everything above never sees it -
dashboard.py calls save_analysis() and gets an id back. Swapping SQLite for
another database later means rewriting the inside of these functions and
nothing else. Same principle that let LEGAL-BERT replace TF-IDF without
touching risk_scorer.py.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("LEXICORE_DB", "lexicore.db")


# ── Schema ──────────────────────────────────────────────────
# Column names follow the Phase 1 architecture diagram. Extra columns beyond
# the spec (confidence, reasons, content_hash) exist because they are computed
# by the pipeline and shown in the UI - without them, reloading a saved
# analysis would silently lose information the user had already seen.
SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    contract_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL,
    upload_date     TEXT    NOT NULL,
    raw_text        TEXT,
    total_clauses   INTEGER NOT NULL DEFAULT 0,
    overall_score   INTEGER NOT NULL DEFAULT 0,
    overall_level   TEXT    NOT NULL DEFAULT 'NONE',
    analysis_time   REAL,
    content_hash    TEXT
);

CREATE TABLE IF NOT EXISTS clauses (
    clause_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id     INTEGER NOT NULL,
    clause_index    INTEGER NOT NULL,
    clause_text     TEXT    NOT NULL,
    clause_type     TEXT    NOT NULL,
    confidence      REAL,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS risk_scores (
    score_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    clause_id       INTEGER NOT NULL,
    risk_level      TEXT    NOT NULL,
    score_value     INTEGER NOT NULL,
    reasons         TEXT,
    FOREIGN KEY (clause_id) REFERENCES clauses(clause_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id     INTEGER NOT NULL,
    summary_text    TEXT    NOT NULL,
    provisions      TEXT,
    risk_areas      TEXT,
    generated_at    TEXT    NOT NULL,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_clauses_contract
    ON clauses(contract_id);
CREATE INDEX IF NOT EXISTS idx_scores_clause
    ON risk_scores(clause_id);
CREATE INDEX IF NOT EXISTS idx_contracts_hash
    ON contracts(content_hash);
"""


def _connect():
    """
    Opens the database, creating the file on first use.

    row_factory=sqlite3.Row lets rows be read by column name
    (row["filename"]) rather than by position (row[1]), which is far less
    error-prone when the schema changes.

    Foreign keys are OFF by default in SQLite for backwards compatibility,
    so ON DELETE CASCADE would silently do nothing without this pragma.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates tables if they do not exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.executescript(SCHEMA)


def find_by_hash(content_hash):
    """
    Returns the contract_id of an already-saved identical file, or None.

    P41: this check must query the DATABASE, not Streamlit session state.
    Session state is forgotten on restart and - worse - is not cleared when
    a contract is deleted, so re-uploading a deleted file was silently
    skipped and never reappeared in the sidebar.
    """
    if not content_hash:
        return None
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT contract_id FROM contracts WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
    return row["contract_id"] if row else None


# ── Save ────────────────────────────────────────────────────
def save_analysis(filename, scored, stats, summary,
                  raw_text=None, analysis_time=None, content_hash=None):
    """
    Persists one complete analysis across all four tables.
    Returns the new contract_id.

    The whole save runs in one transaction: `with conn` commits if the block
    completes and rolls back if anything raises. Without that, a crash midway
    could leave a contract row with only some of its clauses attached.
    """
    init_db()
    now = datetime.now().isoformat(timespec="seconds")

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO contracts
               (filename, upload_date, raw_text, total_clauses,
                overall_score, overall_level, analysis_time, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, now, raw_text, stats["total_clauses"],
             stats["overall_score"], stats["overall_level"],
             analysis_time, content_hash)
        )
        contract_id = cur.lastrowid

        for i, c in enumerate(scored):
            cur = conn.execute(
                """INSERT INTO clauses
                   (contract_id, clause_index, clause_text,
                    clause_type, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (contract_id, i, c["clause_text"], c["clause_type"],
                 c.get("confidence"))
            )
            clause_id = cur.lastrowid

            # reasons is a Python list; SQLite has no list type, so it is
            # stored as JSON text and parsed back on load.
            conn.execute(
                """INSERT INTO risk_scores
                   (clause_id, risk_level, score_value, reasons)
                   VALUES (?, ?, ?, ?)""",
                (clause_id, c["risk_level"], c["risk_score"],
                 json.dumps(c.get("reasons", [])))
            )

        if summary:
            conn.execute(
                """INSERT INTO summaries
                   (contract_id, summary_text, provisions,
                    risk_areas, generated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (contract_id,
                 " ".join(summary.get("overview", [])),
                 json.dumps(summary.get("provisions", [])),
                 json.dumps(summary.get("risk_areas", [])),
                 now)
            )

    return contract_id


# ── Load ────────────────────────────────────────────────────
def load_analysis(contract_id):
    """
    Reads one saved analysis back in the same shape the pipeline produces,
    so dashboard.py renders a loaded contract with the same code that
    renders a freshly analysed one.

    Returns None if the contract_id does not exist.
    """
    init_db()

    with _connect() as conn:
        contract = conn.execute(
            "SELECT * FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()

        if contract is None:
            return None

        # JOIN combines rows from two tables where the foreign key matches,
        # so each clause arrives with its risk score attached.
        rows = conn.execute(
            """SELECT c.clause_text, c.clause_type, c.confidence,
                      r.risk_level, r.score_value, r.reasons
               FROM clauses c
               JOIN risk_scores r ON r.clause_id = c.clause_id
               WHERE c.contract_id = ?
               ORDER BY c.clause_index""",
            (contract_id,)
        ).fetchall()

        scored = [{
            "clause_text": r["clause_text"],
            "clause_type": r["clause_type"],
            "confidence": r["confidence"],
            "risk_level": r["risk_level"],
            "risk_score": r["score_value"],
            "reasons": json.loads(r["reasons"] or "[]"),
        } for r in rows]

        levels = [c["risk_level"] for c in scored]
        stats = {
            "total_clauses": contract["total_clauses"],
            "high": levels.count("HIGH"),
            "medium": levels.count("MEDIUM"),
            "low": levels.count("LOW"),
            "overall_score": contract["overall_score"],
            "overall_level": contract["overall_level"],
        }

        srow = conn.execute(
            "SELECT * FROM summaries WHERE contract_id = ?", (contract_id,)
        ).fetchone()

        summary = None
        if srow:
            summary = {
                "overview": [srow["summary_text"]],
                "provisions": json.loads(srow["provisions"] or "[]"),
                "risk_areas": [tuple(a) for a in
                               json.loads(srow["risk_areas"] or "[]")],
                "method": "loaded from database",
            }

    return {
        "contract_id": contract["contract_id"],
        "filename": contract["filename"],
        "upload_date": contract["upload_date"],
        "analysis_time": contract["analysis_time"],
        "scored": scored,
        "stats": stats,
        "summary": summary,
    }


# ── List / delete ───────────────────────────────────────────
def list_contracts(limit=50):
    """Recently analysed contracts, newest first. For the history sidebar."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT contract_id, filename, upload_date, total_clauses,
                      overall_score, overall_level
               FROM contracts
               ORDER BY contract_id DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_contract(contract_id):
    """
    Removes a contract. ON DELETE CASCADE means its clauses, risk scores
    and summary go with it - no orphaned rows left behind.
    """
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM contracts WHERE contract_id = ?",
                     (contract_id,))


def stats_overview():
    """Aggregate figures across everything stored. Useful in a demo."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS contracts,
                      COALESCE(SUM(total_clauses), 0) AS clauses,
                      COALESCE(AVG(overall_score), 0) AS avg_score
               FROM contracts"""
        ).fetchone()
    return dict(row)


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for c in list_contracts():
            print(f"[{c['contract_id']:3}] {c['filename']:<35} "
                  f"{c['upload_date']}  "
                  f"{c['overall_score']:3}/100 {c['overall_level']}")
        print()
        print(stats_overview())

    else:
        print("Round-trip test\n" + "=" * 50)

        scored = [
            {"clause_text": "The Contractor shall indemnify the Client "
                            "without limitation.",
             "clause_type": "Cap On Liability", "confidence": 0.89,
             "risk_level": "HIGH", "risk_score": 6,
             "reasons": ["Indemnity obligation", "Perpetual commitment"]},
            {"clause_text": "Any notice shall be in writing.",
             "clause_type": "General / Unclassified", "confidence": 0.12,
             "risk_level": "NONE", "risk_score": 0,
             "reasons": ["No risk indicators found"]},
        ]
        stats = {"total_clauses": 2, "high": 1, "medium": 0, "low": 0,
                 "overall_score": 71, "overall_level": "HIGH"}
        summary = {"overview": ["This contract contains 2 clauses."],
                   "provisions": [{"clause_type": "Cap On Liability",
                                   "risk_level": "HIGH", "risk_score": 6,
                                   "quote": "The Contractor shall indemnify...",
                                   "reasons": ["Indemnity obligation"]}],
                   "risk_areas": [("Liability and indemnity", 1, ["Indemnity"])]}

        cid = save_analysis("roundtrip_test.pdf", scored, stats, summary,
                            analysis_time=2.0, content_hash="testhash123")
        print(f"saved   -> contract_id {cid}")

        back = load_analysis(cid)
        print(f"loaded  -> {back['filename']}, "
              f"{len(back['scored'])} clauses, "
              f"{back['stats']['overall_score']}/100 "
              f"{back['stats']['overall_level']}")

        ok = (
            len(back["scored"]) == len(scored)
            and back["scored"][0]["clause_type"] == scored[0]["clause_type"]
            and back["scored"][0]["reasons"] == scored[0]["reasons"]
            and back["stats"]["high"] == stats["high"]
        )
        print(f"round-trip intact: {'PASS' if ok else 'FAIL'}")

        # P41 regression check
        found = find_by_hash("testhash123")
        print(f"hash lookup finds it: {'PASS' if found == cid else 'FAIL'}")

        delete_contract(cid)
        print(f"deleted -> {cid}")
        print(f"gone from load:      {load_analysis(cid) is None}")
        print(f"gone from hash:      {find_by_hash('testhash123') is None} "
              f"(P41: must be True, or re-upload would be skipped)")