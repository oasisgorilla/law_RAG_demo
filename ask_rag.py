import os, argparse, json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from pathlib import Path

SYSTEM_PROMPT = """You are a legal RAG assistant. Use ONLY the given context.
If you cannot find evidence, say you cannot find it.
Format:
- Summary (<=3 sentences)
- Cited Sections (up to 3 with section numbers)
- Sources (up to 3 URLs)"""

def load_jsonl(path):
    recs=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj=json.loads(line)
        recs.append({"id": obj["metadata"].get("chunk_id"),
                     "text": obj["text"],
                     "meta": obj["metadata"]})
    return recs

def build_bm25(records):
    return BM25Okapi([r["text"].split() for r in records])

def bm25_topk(bm25, records, q, k=5):
    scores = bm25.get_scores(q.split())
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [records[i] for i in idx]

def vector_topk(col, emb, q, k=5):
    qv = emb.encode([q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=qv, n_results=k, include=["metadatas","documents"])
    out=[]
    for i in range(len(res["ids"][0])):
        out.append({"id":res["ids"][0][i], "text":res["documents"][0][i], "meta":res["metadatas"][0][i]})
    return out

def rrf(dense, sparse, k=6, k_rrf=60):
    scores={}
    def add(lst):
        for r, it in enumerate(lst):
            key=it["id"]
            scores.setdefault(key, {"it":it, "s":0.0})
            scores[key]["s"] += 1.0/(k_rrf + r + 1)
    add(dense); add(sparse)
    return [v["it"] for v in sorted(scores.values(), key=lambda x:x["s"], reverse=True)[:k]]

def build_context(chunks):
    lines=[]
    for i, c in enumerate(chunks, 1):
        url=c["meta"].get("url")
        sec=c["meta"].get("section_no")
        head=f"[{i}] (section: {sec}) {url}"
        body=c["text"].strip()
        lines.append(head+"\n"+body)
    return "\n\n".join(lines)

def main(jsonl, db_dir, collection, emb_model, query, model, base_url, api_key):
    # LLM 클라이언트 (OpenRouter or LM Studio)
    client = OpenAI(base_url=base_url, api_key=api_key)

    # 데이터/리트리버
    recs = load_jsonl(jsonl)
    bm25 = build_bm25(recs)

    chroma = chromadb.PersistentClient(path=db_dir, settings=Settings())
    col = chroma.get_collection(name=collection)
    emb = SentenceTransformer(emb_model)

    dense = vector_topk(col, emb, query, k=5)
    sparse = bm25_topk(bm25, recs, query, k=5)
    ctx_chunks = rrf(dense, sparse, k=3)

    ctx = build_context(ctx_chunks)
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"Question: {query}\n\nContext:\n{ctx}"}
    ]
    res = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=600)
    print("\n=== Answer ===\n")
    print(res.choices[0].message.content)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--db_dir", default="rag_chroma")
    ap.add_argument("--collection", default="laws")
    ap.add_argument("--emb_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--query", required=True)
    # LLM 백엔드 스위치
    ap.add_argument("--model", default="deepseek/deepseek-chat-v3.1:free")  # OpenRouter 예시 또는 LM Studio의 로컬 모델명
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1")  # LM Studio면 http://localhost:1234/v1
    ap.add_argument("--api_key", required=True)
    args = ap.parse_args()
    main(**vars(args))
