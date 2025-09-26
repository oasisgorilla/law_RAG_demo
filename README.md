# Legal RAG Demo

이 프로젝트는 캐나다 법령에 적용된 **검색 증강 생성(RAG)**의 데모입니다. 
법률 텍스트를 수집하고, 벡터 데이터베이스를 구축하며, FastAPI 서비스를 통해 LLM으로 쿼리하는 간단한 과정을 연습한 결과물입니다.

---

## 📂 Project Structure
```
LAW_RAG_DEMO/
├─ app.py # FastAPI server (REST API for RAG)
├─ ask_rag.py # CLI tool to query the RAG pipeline
├─ index_rag.py # Embedding + indexing script
├─ ingestion_helper.py # Data ingestion helpers
├─ search_rag.py # Standalone search testing
├─ acts_scraper.py # Example scraper for Acts
├─ requirements.txt # Python dependencies
├─ rag_chroma/ # ChromaDB persistent storage (ignored in git)
├─ canada_acts_page1_canonical.csv # Sample raw data (ignored in git)
├─ canada_acts_page1_canonical_chunks.csv # Chunked sample (ignored in git)
├─ canada_acts_rag_chunks.jsonl # JSONL for indexing (ignored in git)
├─ canada_acts_rag_chunks.parquet # Parquet version (ignored in git)
└─ venv/ # Virtual environment (ignored in git)
```
---

## ⚙️ Setup

### 1. 가상환경 설정 및 필요한 라이브러리 설치
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. 데이터 수집(Scraping) 및 전처리
acts_scraper.py 실행시켜 https://laws-lois.justice.gc.ca/eng/acts/에서 자주 찾아본 법령정보를 수집한다.
<img width="838" height="838" alt="image" src="https://github.com/user-attachments/assets/87950b5b-8094-4490-9127-10216aab7ac6" />
<br/> CHUNK_OUTPUT을 True로 설정하면 수집된 법령정보는 자동으로 1800 chunk에 맞게 분해되며,
<br/> 이후 ingestion_helper.py를 실행시켜 200 chunk 이하로 잘린 텍스트를 앞, 뒤로 다시 붙인다(문맥 유지)

### 2. 인덱싱(벡터DB 저장)
전처리 결과(jsonl파일)를 불러와서 임베딩 모델로 벡터화.
<br/> 임베딩 모델: intfloat/multilingual-e5-base

```bash
python index_rag.py \
  --jsonl ./canada_acts_rag_chunks.jsonl \
  --db_dir ./rag_chroma \
  --collection laws
```

### 3. Query via CLI
openrouter에서 API키를 생성한 뒤 아래 api_key 매개변수로 전달한다.
<br/> 기본 설정돼있는 모델은 무료 모델인 deepseek-chat-v3.1:free 이다.
```bash
python ask_rag.py \
  --jsonl ./canada_acts_rag_chunks.jsonl \
  --db_dir ./rag_chroma --collection laws \
  --query "What is the Short Title of the Access to Information Act?" \
  --model deepseek/deepseek-chat-v3.1:free \
  --base_url https://openrouter.ai/api/v1 \
  --api_key sk-or-...<openrouter API 키 입력>
```

### 4. Run FastAPI server
```bash
uvicorn app:app --reload --port 8000
```
- Health check: http://localhost:8000/health
- Ask endpoint: POST → http://localhost:8000/ask

Exanple request body:
```bash
{
  "query": "근로자의 정의 알려줘",
  "title_filter": "Canada Labour Code"
}
```

## 🧩 Features

- Hybrid search: BM25 + dense embeddings + Reciprocal Rank Fusion (RRF)

- Multilingual embeddings (intfloat/multilingual-e5-base)

- Backends: OpenRouter or LM Studio (OpenAI-compatible APIs)

- Clarification questions when retrieved context mismatches query law

- REST API with FastAPI + CLI support

---
