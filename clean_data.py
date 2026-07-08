import json
import re
from collections import Counter

print("=" * 60)
print("LEXICORE — DATA CLEANING SCRIPT")
print("=" * 60)

# ── Step 1: Load raw CUAD data ──────────────────────────────
print("\n[1/6] Loading raw CUAD dataset...")
with open("data/CUAD_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
contracts = data["data"]
print(f"     Loaded {len(contracts)} contracts")

# ── Step 2: Define rare clause types to SKIP ───────────────
# These have too few examples for model to learn properly
RARE_CLAUSES_TO_SKIP = [
    "Source Code Escrow",           # 13 examples
    "Price Restrictions",           # 15 examples
    "Unlimited/All-You-Can-Eat-License",  # 17 examples
    "Affiliate License-Licensor",   # 23 examples
    "Most Favored Nation",          # 28 examples
]

print(f"\n[2/6] Skipping {len(RARE_CLAUSES_TO_SKIP)} rare clause types")
for c in RARE_CLAUSES_TO_SKIP:
    print(f"     → {c}")

# ── Step 3: Helper — clean raw text ────────────────────────
def clean_text(text):
    # Fix broken words split across lines (e.g. "Agree-\nment")
    text = re.sub(r'-\n', '', text)
    
    # Remove page numbers like "Page 3 of 12"
    text = re.sub(r'page \d+ of \d+', '', text, flags=re.IGNORECASE)
    
    # Remove exhibit headers like "EXHIBIT 10.6"
    text = re.sub(r'EXHIBIT\s+[\d\.]+', '', text)
    
    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Collapse multiple newlines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# ── Step 4: Helper — extract clean clause name ─────────────
def get_clean_clause_name(question):
    match = re.search(r'"([^"]+)"', question)
    return match.group(1) if match else question

# ── Step 5: Helper — chunk long text ───────────────────────
# LEGAL-BERT can only handle 512 tokens (~400 words)
# If clause text is longer we take first 400 words only
def chunk_text(text, max_words=400):
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

# ── Step 6: Build cleaned dataset ──────────────────────────
print("\n[3/6] Cleaning and extracting clause samples...")

cleaned_samples = []
skipped_rare = 0
skipped_empty = 0
skipped_long = 0
kept = 0

for contract in contracts:
    contract_title = contract["title"]
    
    for paragraph in contract["paragraphs"]:
        raw_text = paragraph["context"]
        clean_contract_text = clean_text(raw_text)
        
        for qa in paragraph["qas"]:
            # Get clean clause type name
            clause_type = get_clean_clause_name(qa["question"])
            
            # Skip rare clause types
            if clause_type in RARE_CLAUSES_TO_SKIP:
                skipped_rare += 1
                continue
            
            # Skip if clause not present in this contract
            if len(qa["answers"]) == 0:
                skipped_empty += 1
                continue
            
            # Get the clause text
            clause_text = qa["answers"][0]["text"]
            clause_text = clean_text(clause_text)
            
            # Skip if clause text is empty after cleaning
            if len(clause_text.strip()) == 0:
                skipped_empty += 1
                continue
            
            # Chunk if too long
            original_len = len(clause_text.split())
            clause_text = chunk_text(clause_text)
            if original_len > 400:
                skipped_long += 1
            
            # Add to cleaned samples
            cleaned_samples.append({
                "contract": contract_title,
                "clause_type": clause_type,
                "clause_text": clause_text
            })
            kept += 1

# ── Step 7: Show cleaning summary ──────────────────────────
print(f"\n[4/6] Cleaning Summary:")
print(f"     Total samples processed : {kept + skipped_rare + skipped_empty}")
print(f"     ✅ Kept                  : {kept}")
print(f"     ❌ Skipped (rare types)  : {skipped_rare}")
print(f"     ❌ Skipped (empty/absent): {skipped_empty}")
print(f"     ✂️  Chunked (too long)   : {skipped_long}")

# ── Step 8: Show clause distribution after cleaning ────────
print(f"\n[5/6] Clause distribution after cleaning:")
type_counter = Counter(s["clause_type"] for s in cleaned_samples)
print(f"\n{'Clause Type':<45} {'Count':>6}")
print("-" * 55)
for clause, count in type_counter.most_common():
    print(f"{clause:<45} {count:>6}")

# ── Step 9: Show a sample cleaned entry ────────────────────
print(f"\n[6/6] Sample cleaned entry:")
print("-" * 55)
sample = cleaned_samples[0]
print(f"Contract  : {sample['contract']}")
print(f"Clause Type: {sample['clause_type']}")
print(f"Clause Text: {sample['clause_text'][:300]}")

# ── Step 10: Save cleaned dataset ──────────────────────────
output_path = "data/cuad_cleaned.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_samples, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print(f"✅ CLEANING COMPLETE!")
print(f"   Saved {len(cleaned_samples)} clean samples")
print(f"   File: {output_path}")
print(f"{'=' * 60}")