import json

# ── Step 1: Load the CUAD dataset ──────────────────
print("Loading CUAD dataset...")

with open("data/CUAD_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset loaded successfully!")
print()

# ── Step 2: How many contracts are there? ──────────
contracts = data["data"]
print(f"Total contracts in CUAD: {len(contracts)}")
print()

# ── Step 3: Look at the first contract ─────────────
first_contract = contracts[0]
print(f"First contract name: {first_contract['title']}")
print()

# ── Step 4: Look at the text of first contract ─────
first_paragraph = first_contract["paragraphs"][0]
contract_text = first_paragraph["context"]

print("--- First 500 characters of contract text ---")
print(contract_text[:500])
print()

# ── Step 5: What clause types exist? ───────────────
print("--- Clause types found in first contract ---")
questions = first_paragraph["qas"]
print(f"Number of clause types checked: {len(questions)}")
print()

for i, qa in enumerate(questions[:10]):  # show first 10
    clause_type = qa["question"]
    has_answer = len(qa["answers"]) > 0
    print(f"{i+1}. {clause_type} → {'FOUND ✓' if has_answer else 'NOT PRESENT'}")

print()
print("--- Sample clause text (first one found) ---")

# Find first clause that actually exists in this contract
for qa in questions:
    if len(qa["answers"]) > 0:
        print(f"Clause type: {qa['question']}")
        print(f"Clause text: {qa['answers'][0]['text'][:300]}")
        break

# ── Step 6: Extract clean clause type names ─────────
print()
print("--- All 41 Clause Types (clean names) ---")

import re

all_clause_types = []

for qa in questions:
    full_question = qa["question"]
    
    # Extract just the name inside quotes
    # e.g. "Document Name" from the long question
    match = re.search(r'"([^"]+)"', full_question)
    if match:
        clean_name = match.group(1)
    else:
        clean_name = full_question  # fallback
    
    all_clause_types.append(clean_name)
    
for i, clause in enumerate(all_clause_types):
    print(f"{i+1:2}. {clause}")

# ── Step 7: Count clause frequency across ALL contracts ─────
print()
print("--- Counting clause types across all 510 contracts ---")
print("(This may take a few seconds...)")

from collections import Counter

clause_counter = Counter()

for contract in contracts:
    for paragraph in contract["paragraphs"]:
        for qa in paragraph["qas"]:
            # Extract clean clause name
            match = re.search(r'"([^"]+)"', qa["question"])
            clean_name = match.group(1) if match else qa["question"]
            
            # Only count if clause actually EXISTS in contract
            if len(qa["answers"]) > 0:
                clause_counter[clean_name] += 1

print()
print(f"{'Clause Type':<45} {'Count':>6}")
print("-" * 55)

for clause, count in clause_counter.most_common():
    bar = "█" * (count // 10)  # visual bar
    print(f"{clause:<45} {count:>6}  {bar}")

print()
print(f"Total clause types found: {len(clause_counter)}")
print(f"Most common: {clause_counter.most_common(1)[0]}")
print(f"Least common: {clause_counter.most_common()[-1]}")