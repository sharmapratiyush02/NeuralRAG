# 🔧 Backend — NeuralRAG API Server

This folder contains the entire backend logic for NeuralRAG.

## Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI server — all REST endpoints, session store, routing |
| `embeddings.py` | BGE-Large-EN-v1.5 dense embedder + BM25 sparse embedder |
| `retriever.py` | Hybrid retrieval — dense + BM25 + Reciprocal Rank Fusion |
| `reranker.py` | BGE cross-encoder neural reranker (560M params) |
| `generator.py` | LLM answer generation — Llama-3 / Groq / Anthropic |
| `agent.py` | ReAct agentic reasoning loop with tool use |
| `multimodal.py` | Image ingestion — CLIP embeddings + LLaVA description |

## Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest/session` | Upload a file into session memory (no disk write) |
| `POST` | `/api/query` | Standard RAG query with session + source file filter |
| `POST` | `/api/query/agent` | Agentic ReAct query |
| `POST` | `/api/query/stream` | Streaming SSE response |
| `GET` | `/api/session/files` | List files in current session |
| `DELETE` | `/api/session` | Clear session on page close |
| `POST` | `/api/translate` | Translate text to target language via LLM |
| `GET` | `/api/stats` | System stats (queries, sessions, uptime) |
| `GET` | `/health` | Health check |

## Run

```bash
python backend/api.py
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Session Storage Design

Documents are **never written to disk**. Each session gets a UUID.  
On upload → chunks are embedded and stored in RAM under that UUID.  
On page close → `DELETE /api/session` frees all memory.

```
Upload → Parse → Chunk → Embed → SessionStore[session_id][filename]
Query  → Embed query → Cosine search → Rerank → Generate
Close  → SessionStore.clear_session(session_id)
```
