import os, json, time
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

SYSTEM_PROMPT = """You are a legal RAG assistant. Use ONLY the given context.
If you cannot find evidence, say you cannot find it.
Format:
- Summary (<=3 sentences)
- Cited Sections (up to 3 with section numbers)
- Sources (up to 3 URLs)"""

# ---------- Config (환경변수로 제어) ----------
JSONL_PATH = os.getenv("RAG_JSONL", "./canada_acts_rag_chunks.jsonl")
CHROMA_DIR = os.getenv("RAG_DB_DIR", "./rag_chroma")
COLLECTION = os.getenv("RAG_COLLECTION", "laws")
EMB_MODEL  = os.getenv("RAG_EMB_MODEL", "intfloat/multilingual-e5-base")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")  # 또는 런타임에 요청마다 받아도 됨

TOPK_DENSE = int(os.getenv("RAG_TOPK_DENSE", "5"))
TOPK_SPARSE = int(os.getenv("RAG_TOPK_SPARSE", "5"))
TOPK_FINAL = int(os.getenv("RAG_TOPK_FINAL", "3"))

# ---------- Data models ----------
class AskRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")
    model: Optional[str] = Field(default=None, description="LLM 모델 ID (미지정 시 서버 기본값)")
    base_url: Optional[str] = Field(default=None, description="OpenAI 호환 엔드포인트")
    api_key: Optional[str] = Field(default=None, description="요청별 API 키(없으면 서버 기본값 사용)")
    title_filter: Optional[str] = Field(default=None, description="법령명 필터(있으면 해당 타이틀만 검색)")
    k_dense: Optional[int] = Field(default=None, ge=1, le=20)
    k_sparse: Optional[int] = Field(default=None, ge=1, le=20)
    k_final: Optional[int] = Field(default=None, ge=1, le=10)

class SourceItem(BaseModel):
    id: str
    url: Optional[str] = None
    section_no: Optional[str] = None
    title: Optional[str] = None

class AskResponse(BaseModel):
    answer: Optional[str] = None
    sources: List[SourceItem] = []
    clarify_needed: bool = False
    clarify_message: Optional[str] = None
    latency_ms: int

# ---------- App & global singletons ----------
app = FastAPI(title="Legal RAG Service", version="0.1.0")

# 전역 리소스 (모델/인덱스/코퍼스)는 한 번만 로드
EMB: SentenceTransformer = None
CHROMA = None
COL = None
BM25 = None
RECORDS: List[Dict[str, Any]] = []

def load_jsonl(path: str):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        meta = obj["metadata"]
        out.append({
            "id": meta.get("chunk_id"),
            "text": obj["text"],
            "meta": meta,
            "title": meta.get("title") or meta.get("law_title"),
            "url": meta.get("url"),
            "section_no": meta.get("section_no")
        })
    return out

def build_bm25(records, title_filter=None):
    pool = [r for r in records if (not title_filter or (title_filter.lower() in (r["title"] or "").lower()))]
    if not pool:
        return None, []
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([r["text"].split() for r in pool])
    return bm, pool

def vector_topk(col, emb, q: str, k=5, title_filter: Optional[str]=None):
    qv = emb.encode([q], normalize_embeddings=True).tolist()
    where = {"title": {"$eq": title_filter}} if title_filter else None
    res = col.query(query_embeddings=qv, n_results=k, include=["metadatas","documents"], where=where)
    out=[]
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i]
        })
    return out

def bm25_topk(bm25, pool, q: str, k=5):
    if not pool or bm25 is None:
        return []
    scores = bm25.get_scores(q.split())
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [pool[i] for i in idx]

def rrf(dense, sparse, k=6, k_rrf=60):
    scores={}
    def add(lst):
        for rank, it in enumerate(lst):
            key = it["id"]
            scores.setdefault(key, {"it":it, "s":0.0})
            scores[key]["s"] += 1.0 / (k_rrf + rank + 1)
    add(dense); add(sparse)
    fused = sorted(scores.values(), key=lambda x:x["s"], reverse=True)[:k]
    return [v["it"] for v in fused]

def build_context(chunks: List[Dict[str,Any]]) -> str:
    parts=[]
    for i, c in enumerate(chunks, 1):
        url = c.get("url") or c["meta"].get("url")
        meta = c.get("meta", {})
        sec = meta.get("section_no") or c.get("section_no")
        title = meta.get("title") or c.get("title")
        head = f"[{i}] (title: {title} | section: {sec}) {url}"
        body = (c["text"] if "text" in c else c.get("document") or "").strip()
        parts.append(head+"\n"+body)
    return "\n\n".join(parts)

def expand_query(q: str) -> str:
    q2 = q
    if ("근로자" in q) or ("노동자" in q):
        q2 += " employee worker"
    if ("노동" in q and "권리" in q) or ("labour" in q and "rights" in q):
        q2 += " labour rights employment"
    return q2

def need_clarify(user_q: str, ctx_chunks: List[Dict[str,Any]]) -> Optional[str]:
    # 질문에 특정 법령명 단서가 있고, 컨텍스트가 그 법을 포함하지 않으면 역질문
    hints = ["Canada Labour Code", "Canada Elections Act", "Competition Act",
             "Access to Information Act", "Contraventions Act", "Canada Emergency Response Benefit Act"]
    mentioned = [h for h in hints if h.lower() in user_q.lower()]
    if mentioned:
        have = any((c.get("meta", {}).get("title","") or c.get("title","")).lower() in [m.lower() for m in mentioned]
                   for c in ctx_chunks)
        if not have:
            return f"Your question mentions {mentioned[0]}, but the retrieved context is from other acts. Do you want me to restrict results to '{mentioned[0]}'?"
    # 또는, 상위 청크의 서로 다른 title이 2개 이상이면 모호
    titles = list({(c.get("meta", {}).get("title") or c.get("title") or "") for c in ctx_chunks})
    if len([t for t in titles if t]) >= 3:
        return "Multiple candidate acts match your query. Would you like to choose one (e.g., Canada Labour Code, Competition Act)?"
    return None

@app.on_event("startup")
def startup():
    global EMB, CHROMA, COL, BM25, RECORDS
    # 임베딩/벡터스토어/코퍼스 로드
    EMB = SentenceTransformer(EMB_MODEL)
    CHROMA = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())
    COL = CHROMA.get_collection(name=COLLECTION)
    RECORDS = load_jsonl(JSONL_PATH)
    # BM25는 전체 코퍼스 기준 기본 인덱스
    global BM25_DEFAULT, BM25_POOL_DEFAULT
    BM25_DEFAULT, BM25_POOL_DEFAULT = build_bm25(RECORDS)

@app.get("/health")
def health():
    return {"status":"ok","records":len(RECORDS),"collection":COLLECTION,"emb_model":EMB_MODEL}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()
    if not req.query.strip():
        raise HTTPException(400, "query is required")

    # 파라미터
    k_dense = req.k_dense or TOPK_DENSE
    k_sparse = req.k_sparse or TOPK_SPARSE
    k_final = req.k_final or TOPK_FINAL

    qx = expand_query(req.query)
    # Dense
    dense = vector_topk(COL, EMB, qx, k=k_dense, title_filter=req.title_filter)
    # Sparse
    if req.title_filter:
        bm25, pool = build_bm25(RECORDS, title_filter=req.title_filter)
    else:
        bm25, pool = BM25_DEFAULT, BM25_POOL_DEFAULT
    sparse = bm25_topk(bm25, pool, qx, k=k_sparse)

    # FUSE
    fused = rrf(dense, sparse, k=k_final*2)
    if not fused:
        raise HTTPException(404, detail=f"No results for title_filter='{req.title_filter}' and query.")
    # 상위 k_final 추림
    # (sparse에서 온 아이템은 key 이름이 다를 수 있으니 정규화)
    norm = []
    ids_seen = set()
    for it in fused:
        id_ = it.get("id")
        if id_ in ids_seen: 
            continue
        ids_seen.add(id_)
        if "meta" not in it:  # sparse쪽
            it = {"id": it["id"], "text": it["text"], "meta": it["meta"] if "meta" in it else {"title": it.get("title"), "url": it.get("url"), "section_no": it.get("section_no")}}
        norm.append(it)
        if len(norm) >= k_final: break

    # 역질문 필요 여부
    clarify_msg = need_clarify(req.query, norm)

    # LLM 호출 전에, 역질문이 필요하면 바로 반환
    if clarify_msg:
        return AskResponse(
            answer=None,
            sources=[
                SourceItem(id=it["id"], url=it["meta"].get("url"), section_no=it["meta"].get("section_no"), title=it["meta"].get("title"))
                for it in norm
            ],
            clarify_needed=True,
            clarify_message=clarify_msg,
            latency_ms=int((time.time()-t0)*1000)
        )

    # 컨텍스트 → LLM
    ctx = build_context(norm)
    client = OpenAI(base_url=req.base_url or LLM_BASE_URL, api_key=(req.api_key or LLM_API_KEY))
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"Question: {req.query}\n\nContext:\n{ctx}"}
    ]
    resp = client.chat.completions.create(
        model=(req.model or LLM_MODEL), 
        messages=messages,
        temperature=0.2,
        max_tokens=600
    )
    answer = resp.choices[0].message.content

    return AskResponse(
        answer=answer,
        sources=[
            SourceItem(id=it["id"], url=it["meta"].get("url"), section_no=it["meta"].get("section_no"), title=it["meta"].get("title"))
            for it in norm
        ],
        clarify_needed=False,
        latency_ms=int((time.time()-t0)*1000)
    )
