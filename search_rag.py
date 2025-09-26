import argparse, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

def load_jsonl(path):
    records=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        # 검색용 간단 구조
        records.append({
            "id": obj["metadata"].get("chunk_id"),
            "doc_id": obj["metadata"].get("doc_id"),
            "url": obj["metadata"].get("url"),
            "section_no": obj["metadata"].get("section_no") if "section_no" in obj else None,
            "text": obj["text"],
            "meta": obj["metadata"],
        })
    return records

def build_bm25(records):
    # 간단 토크나이저(공백 기준); 한글/다국어면 개선 가능
    corpus = [r["text"].split() for r in records]
    return BM25Okapi(corpus)

def bm25_topk(bm25, records, query, k=5):
    scores = bm25.get_scores(query.split())
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"id": records[i]["id"], "text": records[i]["text"], "url": records[i]["url"], "meta": records[i]["meta"], "bm25": float(scores[i])} for i in idx]

def vector_topk(client, collection, emb_model, query, k=5):
    qv = emb_model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qv, n_results=k, include=["distances","documents","metadatas"])
    out=[]
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "url": res["metadatas"][0][i].get("url"),
            "meta": res["metadatas"][0][i],
            "vec": 1.0 - res["distances"][0][i]  # cosine similarity approx
        })
    return out

def rrf_fusion(dense, sparse, k=10, k_rrf=60):
    # Reciprocal Rank Fusion
    # 입력은 각기 랭킹된 리스트
    scores={}
    def add(list_, tag):
        for rank, item in enumerate(list_):
            key=item["id"]
            scores.setdefault(key, {"item": item, "score":0.0})
            scores[key]["score"] += 1.0/(k_rrf + rank + 1)
    add(dense, "vec")
    add(sparse, "bm25")
    fused = sorted(scores.values(), key=lambda x:x["score"], reverse=True)[:k]
    return [x["item"] for x in fused]

def main(jsonl, db_dir, collection, model, query, k=5):
    # Load BM25 corpus
    records = load_jsonl(jsonl)
    bm25 = build_bm25(records)

    # Vector store
    client = chromadb.PersistentClient(path=db_dir, settings=Settings())
    col = client.get_collection(collection_name=collection)
    emb = SentenceTransformer(model)

    dense = vector_topk(client, col, emb, query, k=k)
    sparse = bm25_topk(bm25, records, query, k=k)
    fused = rrf_fusion(dense, sparse, k=8)

    print("\n=== Query ===")
    print(query)
    print("\n=== Fused Top ===")
    for i, item in enumerate(fused, 1):
        sec = item["meta"].get("section_no")
        print(f"{i:02d}. id={item['id']} sec={sec} url={item['url']}")
        print(item["text"][:300].replace("\n"," ") + ("..." if len(item["text"])>300 else ""))
        print("-"*80)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--db_dir", default="rag_chroma")
    ap.add_argument("--collection", default="laws")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    main(args.jsonl, args.db_dir, args.collection, args.model, args.query, args.k)
