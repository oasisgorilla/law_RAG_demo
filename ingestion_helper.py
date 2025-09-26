# RAG ingestion helper: Convert the uploaded CSV into RAG-ready JSONL with
# (1) short-chunk merging, (2) section number extraction, and (3) enriched metadata.
#
# Input  : /mnt/data/canada_acts_page1_canonical_chunks.csv
# Outputs: /mnt/data/canada_acts_rag_chunks.jsonl
#          /mnt/data/canada_acts_rag_chunks.parquet (optional, for analysis)
#
# The script also prints quick stats and shows a small preview.

import pandas as pd
import re
import hashlib
import json
from pathlib import Path

SRC = Path("canada_acts_page1_canonical_chunks.csv")
OUT_JSONL = Path("canada_acts_rag_chunks.jsonl")
OUT_PARQUET = Path("canada_acts_rag_chunks.parquet")

assert SRC.exists(), f"Input file not found: {SRC}"

# 1) Load
df = pd.read_csv(SRC)
df.columns = [c.strip().lower() for c in df.columns]

# Basic sanity columns
expected_cols = {"title", "identifier", "source_url", "chunk_index", "text"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

# Clean & sort
df["text"] = df["text"].fillna("").astype(str)
df["chunk_index"] = pd.to_numeric(df["chunk_index"], errors="coerce").fillna(0).astype(int)
df["source_url"] = df["source_url"].astype(str)
df = df.sort_values(["source_url", "chunk_index"]).reset_index(drop=True)

# Helper: SHA1
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# Helper: section number extractor (best-effort, Canada-friendly generic patterns)
SECTION_PATTERNS = [
    r"\b[Ss]ection\s+(\d+[A-Za-z\-]*)",   # "Section 5" / "section 5.1"
    r"\b[Ss]\.\s*(\d+[A-Za-z\-]*)",       # "s. 5" / "s.5"
    r"^\s*(\d+[A-Za-z\-]*)\.\s",          # line begins "5. ..."  (avoid lists later)
    r"\b[Aa]rticle\s+(\d+[A-Za-z\-]*)",   # "Article 5" (rare in CA, but keep)
]

def extract_section_no(text: str) -> str | None:
    # Try strict first: section/s.
    for pat in SECTION_PATTERNS[:2]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    # Fallback: line-start numeric, but try to avoid enumerated lists by length heuristic
    m = re.search(SECTION_PATTERNS[2], text, flags=re.MULTILINE)
    if m:
        # If it's too short and contains list indicators (a), (b) right away, skip
        head = text[:120]
        if re.search(r"\([a-z]\)", head):
            return None
        return m.group(1)
    # Last resort
    m = re.search(SECTION_PATTERNS[3], text)
    if m:
        return m.group(1)
    return None

# 2) Short-chunk merge (threshold in characters)
MIN_CHARS = 200

merged_rows = []
for url, g in df.groupby("source_url", sort=False):
    rows = g.to_dict(orient="records")
    i, n = 0, len(rows)
    while i < n:
        base = rows[i]
        acc_text = (base["text"] or "").strip()
        j = i + 1
        # Merge forward until length >= MIN_CHARS or end
        while len(acc_text) < MIN_CHARS and j < n:
            nxt = rows[j]
            acc_text = (acc_text + "\n\n" + (nxt["text"] or "").strip()).strip()
            j += 1

        merged_rows.append({
            "title": base.get("title"),
            "identifier": base.get("identifier"),
            "source_url": base.get("source_url"),
            "text": acc_text,
        })
        i = j

# 3) Enrich metadata & reindex chunk_index per document
out_records = []
for url, g in pd.DataFrame(merged_rows).groupby("source_url", sort=False):
    doc_id = sha1(url)
    for new_idx, row in enumerate(g.to_dict(orient="records")):
        text = row["text"]
        sec = extract_section_no(text)
        md = {
            "title": row.get("title"),
            "identifier": row.get("identifier"),
            "url": row.get("source_url"),
            "chunk_index": int(new_idx),
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}-{new_idx:04d}",
            # Enriched fixed meta
            "language": "en",
            "jurisdiction": "CA-Federal",
        }
        out_records.append({
            "text": text,
            "metadata": md,
            "text_hash": sha1(text),
            "section_no": sec,
            "char_len": len(text),
        })

out_df = pd.DataFrame(out_records)

# 4) Save JSONL & Parquet
OUT_JSONL.write_text(
    "\n".join(out_df.apply(lambda r: json.dumps({"text": r["text"], "metadata": r["metadata"]}, ensure_ascii=False), axis=1).tolist()),
    encoding="utf-8"
)
try:
    out_df.to_parquet(OUT_PARQUET, index=False)
except Exception as e:
    # Parquet is optional; if pyarrow missing, ignore.
    pass

# 5) Quick stats
n_docs = out_df["metadata"].apply(lambda m: m["doc_id"]).nunique()
n_chunks = len(out_df)
avg_len = int(out_df["char_len"].mean())
min_len = int(out_df["char_len"].min())
max_len = int(out_df["char_len"].max())
with_section = int(out_df["section_no"].notna().sum())

stats = pd.DataFrame([{
    "docs": n_docs,
    "chunks": n_chunks,
    "avg_chars": avg_len,
    "min_chars": min_len,
    "max_chars": max_len,
    "section_no_coverage": f"{with_section}/{n_chunks} ({with_section*100.0/n_chunks:.1f}%)"
}])

print("WROTE:", OUT_JSONL, "and", OUT_PARQUET)
