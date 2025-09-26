import os, json, argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def load_jsonl(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        yield obj["text"], obj["metadata"]

def main(jsonl, db_dir="rag_chroma", collection="laws", model="intfloat/multilingual-e5-base"):
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    # 1) 임베딩 모델
    emb = SentenceTransformer(model)

    # 2) Chroma DB
    client = chromadb.PersistentClient(path=db_dir, settings=Settings(allow_reset=True))
    col = client.get_or_create_collection(collection, metadata={"hnsw:space":"cosine"})

    # 3) 데이터 적재
    texts, metas, ids = [], [], []
    for i, (text, meta) in enumerate(load_jsonl(jsonl)):
        texts.append(text)
        metas.append(meta)
        ids.append(meta.get("chunk_id") or f"{meta.get('doc_id','doc')}-{meta.get('chunk_index',i)}")

    # 4) 임베딩 후 upsert
    print(f"Embedding {len(texts)} chunks with {model} ...")
    vectors = emb.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print("Upserting to Chroma ...")
    col.upsert(ids=ids, embeddings=vectors.tolist(), metadatas=metas, documents=texts)
    print("Done. Collection size:", col.count())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--db_dir", default="rag_chroma")
    ap.add_argument("--collection", default="laws")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")  # bge-m3도 OK
    args = ap.parse_args()
    main(args.jsonl, args.db_dir, args.collection, args.model)
