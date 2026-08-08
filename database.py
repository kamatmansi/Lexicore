# ============================================================
# LexiCore — database.py
# SQLite storage layer. Keeps a full history of every analyzed
# contract (not just the latest), so the dashboard can list past
# analyses and let the user click back into any of them.
#
# Schema:
#   contracts(id, filename, upload_date, total_clauses,
#             high_risk_count, medium_risk_count, low_risk_count)
#   clauses(id, contract_id -> contracts.id, text, clause_type,
#           confidence, risk, summary)
# ============================================================

import sqlite3
from datetime import datetime

DB_PATH = "lexicore.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, e.g. row["clause_type"]
    return conn


def init_db():
    """Create tables if they don't exist yet. Safe to call every app startup."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            total_clauses INTEGER,
            high_risk_count INTEGER,
            medium_risk_count INTEGER,
            low_risk_count INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clauses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            clause_type TEXT,
            confidence REAL,
            risk TEXT,
            summary TEXT,
            FOREIGN KEY (contract_id) REFERENCES contracts (id)
        )
    """)

    conn.commit()
    conn.close()


def save_contract(filename: str, clauses: list[dict]) -> int:
    """Saves a contract and all its analyzed clauses. `clauses` should be the
    final list of dicts after classifier -> risk_scorer -> summarizer have
    all run, i.e. each dict has: text, clause_type, confidence, risk, summary.
    Returns the new contract's id."""
    conn = get_connection()
    cursor = conn.cursor()

    high = sum(1 for c in clauses if c["risk"] == "High")
    medium = sum(1 for c in clauses if c["risk"] == "Medium")
    low = sum(1 for c in clauses if c["risk"] == "Low")

    cursor.execute("""
        INSERT INTO contracts (filename, upload_date, total_clauses, high_risk_count, medium_risk_count, low_risk_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, datetime.now().isoformat(), len(clauses), high, medium, low))

    contract_id = cursor.lastrowid

    for c in clauses:
        cursor.execute("""
            INSERT INTO clauses (contract_id, text, clause_type, confidence, risk, summary)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (contract_id, c["text"], c["clause_type"], c["confidence"], c["risk"], c["summary"]))

    conn.commit()
    conn.close()
    return contract_id


def get_all_contracts() -> list[dict]:
    """Returns summary info for every analyzed contract, most recent first —
    used to populate a history list in the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM contracts ORDER BY upload_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_clauses_for_contract(contract_id: int) -> list[dict]:
    """Returns all clauses belonging to one contract — used when the user
    clicks into a specific contract in the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clauses WHERE contract_id = ?", (contract_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    # Quick manual test: create tables, insert a fake contract, read it back
    init_db()

    fake_clauses = [
        {
            "text": "Sample clause text here.",
            "clause_type": "Termination For Convenience",
            "confidence": 0.8371,
            "risk": "Medium",
            "summary": "Sample clause text here.",
        }
    ]

    new_id = save_contract("test_contract.pdf", fake_clauses)
    print(f"Saved contract with id: {new_id}")

    print("\nAll contracts:")
    for contract in get_all_contracts():
        print(dict(contract))

    print(f"\nClauses for contract {new_id}:")
    for clause in get_clauses_for_contract(new_id):
        print(dict(clause))
