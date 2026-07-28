"""
api.py — FastAPI REST API Server
==================================
Endpoints:
  POST /api/query           - Standard RAG query
  POST /api/query/agent     - Agentic RAG with ReAct
  POST /api/query/stream    - Streaming RAG response
  POST /api/ingest/session  - Ingest doc into session (in-memory, NOT persisted)
  POST /api/ingest/text     - Ingest raw text (session-only)
  GET  /api/session/files   - List files in current session
  DELETE /api/session       - Clear session documents
  GET  /api/stats           - System statistics
  GET  /api/collections     - List collections
  DELETE /api/collection    - Delete a collection
  POST /api/evaluate        - Run RAGAS evaluation
  GET  /health              - Health check
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from embeddings import DenseEmbedder, SparseEmbedder
from retriever import HybridRetriever, VectorStore, QueryRewriter, Document
from reranker import create_reranker
from generator import create_generator, RAGGenerator
from agent import ReActAgent, ToolRegistry, create_agent


# ─── In-Memory Session Store ──────────────────────────────────────────────────

class SessionStore:
    """
    Per-session in-memory document store.
    Documents are NEVER written to disk or ChromaDB.
    Everything is lost when the session ends or server restarts.
    """

    def __init__(self):
        # session_id -> {filename -> {chunks, embeddings, metadatas}}
        self._sessions: Dict[str, Dict[str, dict]] = {}

    def add_file(
        self,
        session_id: str,
        filename: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ):
        """Add an uploaded file's chunks to the session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {}
        self._sessions[session_id][filename] = {
            "chunks": chunks,
            "embeddings": np.array(embeddings, dtype=np.float32),
            "metadatas": metadatas,
        }
        logger.info(
            f"Session [{session_id[:8]}] stored '{filename}': {len(chunks)} chunks"
        )

    def get_files(self, session_id: str) -> List[str]:
        """Return list of filenames uploaded in this session."""
        return list(self._sessions.get(session_id, {}).keys())

    def session_has_files(self, session_id: str) -> bool:
        return bool(self._sessions.get(session_id))

    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        source_files: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Document]:
        """
        Cosine similarity search over session documents.
        If source_files is given, only search within those files.
        """
        session = self._sessions.get(session_id, {})
        if not session:
            return []

        q_emb = np.array(query_embedding, dtype=np.float32)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        all_chunks: List[str] = []
        all_embeddings: List[np.ndarray] = []
        all_metadatas: List[Dict] = []

        files_to_search = (
            [f for f in source_files if f in session]
            if source_files
            else list(session.keys())
        )

        for fname in files_to_search:
            data = session[fname]
            for chunk, emb, meta in zip(
                data["chunks"], data["embeddings"], data["metadatas"]
            ):
                all_chunks.append(chunk)
                all_embeddings.append(emb)
                all_metadatas.append(meta)

        if not all_chunks:
            return []

        doc_embs = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs_norm = doc_embs / (norms + 1e-8)
        sims = doc_embs_norm @ q_norm

        top_indices = np.argsort(sims)[::-1][:top_k]

        documents: List[Document] = []
        for idx in top_indices:
            sim_score = float(sims[idx])
            if sim_score > 0.0:
                meta = all_metadatas[idx]
                documents.append(
                    Document(
                        content=all_chunks[idx],
                        metadata=meta,
                        score=sim_score,
                        source=meta.get("source", "unknown"),
                    )
                )

        return documents

    def clear_session(self, session_id: str):
        """Remove all data for a session."""
        if session_id in self._sessions:
            files = list(self._sessions[session_id].keys())
            del self._sessions[session_id]
            logger.info(
                f"Session [{session_id[:8]}] cleared — removed {len(files)} file(s)"
            )

    def total_sessions(self) -> int:
        return len(self._sessions)


# ─── Global State ─────────────────────────────────────────────────────────────

class AppState:
    config: dict = {}
    embedder: DenseEmbedder = None
    sparse_embedder: SparseEmbedder = None
    vector_store: VectorStore = None
    retriever: HybridRetriever = None
    reranker = None
    generator: RAGGenerator = None
    agent: ReActAgent = None
    session_store: SessionStore = SessionStore()
    stats: Dict = {
        "total_queries": 0,
        "total_documents": 0,
        "avg_response_time_ms": 0,
        "start_time": time.time(),
    }

state = AppState()


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    logger.info("🚀 Starting NeuralRAG API...")

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            state.config = yaml.safe_load(f)
    else:
        state.config = {"embeddings": {}, "vector_store": {}, "retrieval": {}, "generation": {}}

    # Initialize embedder
    state.embedder = DenseEmbedder(
        model_name=state.config.get("embeddings", {}).get("dense_model", "BAAI/bge-large-en-v1.5"),
        device=state.config.get("embeddings", {}).get("device", "cpu"),
    )

    # Initialize sparse embedder
    state.sparse_embedder = SparseEmbedder()

    # Initialize vector store (kept for backward-compat / future use, but session docs bypass it)
    vs_config = state.config.get("vector_store", {}).get("chroma", {})
    state.vector_store = VectorStore(
        persist_directory=vs_config.get("persist_directory", "./data/chroma_db"),
        collection_name=vs_config.get("collection_name", "neural_rag_docs"),
        embedder=state.embedder,
    )

    # Initialize reranker
    state.reranker = create_reranker(state.config)

    # Initialize generator
    state.generator = create_generator(state.config)

    # LLM function for query rewriting
    async def llm_func(prompt: str) -> str:
        return await state.generator.llm.generate(prompt)

    # Initialize retriever
    state.retriever = HybridRetriever(
        vector_store=state.vector_store,
        dense_embedder=state.embedder,
        sparse_embedder=state.sparse_embedder,
        dense_weight=state.config.get("retrieval", {}).get("dense_weight", 0.7),
        sparse_weight=state.config.get("retrieval", {}).get("sparse_weight", 0.3),
        query_rewriter=QueryRewriter(
            llm_func=llm_func,
            strategy=state.config.get("retrieval", {}).get(
                "query_rewriting", {}
            ).get("strategy", "multi_query"),
        ),
    )

    # Initialize agent
    state.agent = create_agent(state.config, state.generator.llm, state.retriever)

    state.stats["total_documents"] = state.vector_store.count()

    logger.success(
        f"✅ NeuralRAG ready | "
        f"Docs in store: {state.stats['total_documents']} | "
        f"Model: {state.config.get('generation', {}).get('anthropic_model', 'llm')}"
    )

    yield

    logger.info("Shutting down NeuralRAG API...")


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuralRAG API",
    description="Enterprise-grade RAG with hybrid retrieval, neural reranking, and agentic reasoning",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    use_query_rewriting: bool = True
    mode: str = Field(default="rag", description="rag | agent | hybrid")
    metadata_filter: Optional[Dict] = None
    # Session-based filtering (new)
    session_id: Optional[str] = None
    source_files: Optional[List[str]] = None   # which files to query against


class IngestTextRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict]] = None
    source_name: str = "api_upload"
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    faithfulness_score: float
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_docs_retrieved: int
    num_docs_after_reranking: int
    rewritten_queries: List[str]
    model_used: str
    mode: str
    session_id: Optional[str] = None
    queried_files: Optional[List[str]] = None


class AgentResponse(BaseModel):
    question: str
    answer: str
    steps: List[Dict]
    tools_used: List[str]
    total_steps: int
    total_time_ms: float


# ─── Helper: Session-Aware Retrieval ─────────────────────────────────────────

async def retrieve_from_session(
    session_id: str,
    query: str,
    source_files: Optional[List[str]],
    top_k: int,
    use_reranking: bool,
    reranking_top_k: int,
) -> tuple[List[Document], List[str]]:
    """
    Retrieve documents from the in-memory session store.
    Returns (docs, rewritten_queries).
    """
    # Embed query
    query_embedding = state.embedder.embed_query(query)

    # Search session store
    docs = state.session_store.search(
        session_id=session_id,
        query_embedding=query_embedding,
        source_files=source_files,
        top_k=max(top_k * 4, 20),  # retrieve more, then rerank
    )

    # Optional reranking
    if use_reranking and state.reranker and docs:
        reranked = state.reranker.rerank(
            query=query,
            documents=[d.content for d in docs],
            top_k=reranking_top_k,
            scores=[d.score for d in docs],
        )
        content_to_doc = {d.content[:50]: d for d in docs}
        final_docs = []
        for r in reranked:
            key = r.content[:50]
            doc = content_to_doc.get(key)
            if doc:
                doc.score = r.rerank_score
                final_docs.append(doc)
        docs = final_docs or docs[:reranking_top_k]
    else:
        docs = docs[:reranking_top_k]

    return docs, [query]


# ─── Query Endpoint ───────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Standard RAG query.
    If session_id + source_files are provided, queries only against those
    uploaded (in-memory) files. Otherwise falls back to the vector store.
    """
    start_time = time.time()
    state.stats["total_queries"] += 1

    try:
        use_session = (
            request.session_id
            and state.session_store.session_has_files(request.session_id)
        )

        if use_session:
            # ── Session-based retrieval (in-memory, no ChromaDB) ──────────────
            retrieval_start = time.time()
            docs, rewritten_queries = await retrieve_from_session(
                session_id=request.session_id,
                query=request.question,
                source_files=request.source_files,
                top_k=request.top_k,
                use_reranking=request.use_reranking,
                reranking_top_k=request.top_k,
            )
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            num_before_reranking = len(docs)
        else:
            # ── ChromaDB-based retrieval (legacy / empty session) ─────────────
            retrieval_start = time.time()
            top_k_retrieval = state.config.get("retrieval", {}).get("top_k_retrieval", 20)

            retrieval_result = await state.retriever.retrieve(
                query=request.question,
                top_k=top_k_retrieval,
                metadata_filter=request.metadata_filter,
            )
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            docs = retrieval_result.documents
            rewritten_queries = retrieval_result.rewritten_queries
            num_before_reranking = len(docs)

            if request.use_reranking and state.reranker and docs:
                reranked = state.reranker.rerank(
                    query=request.question,
                    documents=[d.content for d in docs],
                    top_k=request.top_k,
                    scores=[d.score for d in docs],
                )
                content_to_doc = {d.content[:50]: d for d in docs}
                final_docs = []
                for r in reranked:
                    key = r.content[:50]
                    doc = content_to_doc.get(key)
                    if doc:
                        doc.score = r.rerank_score
                        final_docs.append(doc)
                docs = final_docs or docs[:request.top_k]
            else:
                docs = docs[:request.top_k]

        # Generate answer
        gen_start = time.time()
        result = await state.generator.generate(
            query=request.question,
            documents=docs,
        )
        gen_time_ms = (time.time() - gen_start) * 1000

        total_time_ms = (time.time() - start_time) * 1000

        # Update rolling average
        prev_avg = state.stats["avg_response_time_ms"]
        n = state.stats["total_queries"]
        state.stats["avg_response_time_ms"] = prev_avg + (total_time_ms - prev_avg) / n

        queried_files = (
            request.source_files
            or (state.session_store.get_files(request.session_id) if request.session_id else None)
        )

        return QueryResponse(
            question=request.question,
            answer=result.answer,
            sources=result.sources,
            faithfulness_score=result.faithfulness_score,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=gen_time_ms,
            total_time_ms=total_time_ms,
            num_docs_retrieved=num_before_reranking,
            num_docs_after_reranking=len(docs),
            rewritten_queries=rewritten_queries,
            model_used=result.model_used,
            mode="rag",
            session_id=request.session_id,
            queried_files=queried_files,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/agent", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    """Agentic RAG with ReAct reasoning loop."""
    start_time = time.time()

    try:
        result = await state.agent.run(request.question)
        total_time_ms = (time.time() - start_time) * 1000

        return AgentResponse(
            question=request.question,
            answer=result.final_answer,
            steps=[s.to_dict() for s in result.steps],
            tools_used=list(set(result.tools_used)),
            total_steps=result.total_steps,
            total_time_ms=total_time_ms,
        )
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming RAG response (SSE)."""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            import json

            use_session = (
                request.session_id
                and state.session_store.session_has_files(request.session_id)
            )

            if use_session:
                docs, rewritten_queries = await retrieve_from_session(
                    session_id=request.session_id,
                    query=request.question,
                    source_files=request.source_files,
                    top_k=request.top_k,
                    use_reranking=request.use_reranking,
                    reranking_top_k=request.top_k,
                )
            else:
                retrieval_result = await state.retriever.retrieve(
                    query=request.question, top_k=request.top_k
                )
                docs = retrieval_result.documents[:request.top_k]
                rewritten_queries = retrieval_result.rewritten_queries

            yield f"data: {json.dumps({'type': 'retrieval', 'num_docs': len(docs), 'queries': rewritten_queries})}\\n\\n"

            async for token in state.generator.generate_stream(request.question, docs):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\\n\\n"

            yield f"data: {json.dumps({'type': 'done'})}\\n\\n"

        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Session Ingestion Endpoints ──────────────────────────────────────────────

@app.post("/api/ingest/session")
async def ingest_file_session(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    chunking: str = Form("recursive"),
):
    """
    Upload and process a document for this session ONLY.
    Documents are stored in memory and NEVER written to ChromaDB or disk.
    All data is lost when the session ends or server restarts.
    """
    import tempfile
    from ingest import IngestionPipeline, DocumentParser

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Parse document into text
        parsed_doc = DocumentParser.parse(tmp_path)

        if not parsed_doc.content.strip():
            raise HTTPException(status_code=400, detail=f"No text content found in '{file.filename}'")

        # Chunk the document
        from ingest import RecursiveChunker, SemanticChunker, FixedSizeChunker

        if chunking == "semantic":
            chunker = SemanticChunker(
                embedder=state.embedder,
                min_chunk_size=200,
                max_chunk_size=512,
            )
        elif chunking == "fixed":
            chunker = FixedSizeChunker(512, 64)
        else:
            chunker = RecursiveChunker(512, 64)

        doc_chunks = chunker.chunk(parsed_doc)

        if not doc_chunks:
            raise HTTPException(status_code=400, detail=f"Could not chunk '{file.filename}'")

        texts = [c.content for c in doc_chunks]
        metadatas = [
            {**c.metadata, "source": file.filename, "filename": file.filename}
            for c in doc_chunks
        ]

        # Embed all chunks (in-memory only)
        logger.info(f"Embedding {len(texts)} chunks for session [{session_id[:8]}]...")
        embeddings = state.embedder.embed_documents(texts)

        # Store in session store (NOT ChromaDB)
        state.session_store.add_file(
            session_id=session_id,
            filename=file.filename,
            chunks=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        all_session_files = state.session_store.get_files(session_id)

        logger.success(
            f"Session [{session_id[:8]}]: ingested '{file.filename}' → {len(texts)} chunks"
        )

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": len(texts),
            "session_id": session_id,
            "session_files": all_session_files,
            "total_session_files": len(all_session_files),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session ingest failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/ingest/text")
async def ingest_text(request: IngestTextRequest):
    """
    Ingest raw text strings into session (in-memory only).
    """
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    texts = [t for t in request.texts if t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="All texts are empty")

    source_name = request.source_name or "inline_text"
    metadatas = request.metadatas or [
        {"source": source_name, "filename": source_name} for _ in texts
    ]

    # Embed and store in session
    embeddings = state.embedder.embed_documents(texts)
    state.session_store.add_file(
        session_id=request.session_id,
        filename=source_name,
        chunks=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    all_session_files = state.session_store.get_files(request.session_id)

    return {
        "status": "success",
        "chunks_ingested": len(texts),
        "session_id": request.session_id,
        "session_files": all_session_files,
    }


# ─── Session Management Endpoints ─────────────────────────────────────────────

@app.get("/api/session/files")
async def get_session_files(session_id: str):
    """Return list of files currently in the session."""
    files = state.session_store.get_files(session_id)
    return {
        "session_id": session_id,
        "files": files,
        "total_files": len(files),
    }


@app.delete("/api/session")
async def clear_session(session_id: str):
    """Clear all documents for a session (called on page unload)."""
    state.session_store.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id[:8]}… cleared"}


# ─── Stats & Management ───────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """System statistics."""
    uptime_seconds = time.time() - state.stats["start_time"]
    return {
        **state.stats,
        "total_docs_in_store": state.vector_store.count(),
        "active_sessions": state.session_store.total_sessions(),
        "uptime_seconds": int(uptime_seconds),
        "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m",
        "embedding_model": state.config.get("embeddings", {}).get("dense_model", "N/A"),
        "reranker_enabled": state.reranker is not None,
        "agent_enabled": state.agent is not None,
    }


@app.delete("/api/collection")
async def delete_collection():
    """Delete all documents from the current collection."""
    state.vector_store.delete_collection()
    state.stats["total_documents"] = 0
    state.retriever._all_documents = []
    return {"status": "success", "message": "Collection deleted"}


@app.post("/api/evaluate")
async def run_evaluation(background_tasks: BackgroundTasks):
    """Trigger RAGAS evaluation in background."""

    async def eval_task():
        try:
            eval_data_path = Path("./data/evaluation/test_qa.json")
            if not eval_data_path.exists():
                logger.warning("No evaluation dataset found at ./data/evaluation/test_qa.json")
                return

            import json
            with open(eval_data_path) as f:
                qa_pairs = json.load(f)

            from ragas_eval import RAGASEvaluator, RAGASample

            async def llm_func(prompt: str) -> str:
                return await state.generator.llm.generate(prompt)

            evaluator = RAGASEvaluator(llm_func=llm_func, embedder=state.embedder)

            samples = []
            for qa in qa_pairs[:10]:
                retrieval = await state.retriever.retrieve(qa["question"], top_k=5)
                samples.append(RAGASample(
                    question=qa["question"],
                    answer=qa.get("answer", ""),
                    contexts=[d.content for d in retrieval.documents],
                    ground_truth=qa.get("ground_truth", ""),
                ))

            result = await evaluator.evaluate_dataset(
                samples,
                output_path="./evaluation/results/latest.json",
            )
            logger.success(f"Evaluation complete: {result.to_dict()}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    background_tasks.add_task(eval_task)
    return {"status": "evaluation_started", "message": "Check logs for results"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "docs_in_store": state.vector_store.count(),
        "active_sessions": state.session_store.total_sessions(),
        "version": "2.0.0",
    }


# ─── Multilingual Translation ─────────────────────────────────────────────────

@app.post("/api/translate")
async def translate_text(request: dict):
    """Translate text to a target language using the LLM."""
    text = request.get("text", "")
    target_lang = request.get("target_lang", "English")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    prompt = f"""Translate the following text to {target_lang}.
Return ONLY the translated text, nothing else. No explanations, no notes.

Text:
{text}"""
    try:
        result = await state.generator.llm.generate(prompt)
        return {"translated": result.strip(), "target_lang": target_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Image Ingestion (Session-Only) ───────────────────────────────────────────

@app.post("/api/ingest/image")
async def ingest_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    language: Optional[str] = Form("English"),
):
    """Upload and ingest an image into the session (in-memory only)."""
    import tempfile

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPG, PNG, GIF, WEBP or BMP.",
        )

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        prompt = f"""You are analyzing an image file named: {file.filename}

Generate a thorough description of what this image likely contains based on its filename and any context.
Describe in {language}:
1. What objects, people, text, charts, or diagrams might be visible
2. The likely purpose and context of this image
3. Any data or information it might convey
4. Key details someone would ask questions about

Make the description detailed enough to answer questions about the image."""

        description = await state.generator.llm.generate(prompt)

        image_text = f"""[IMAGE FILE: {file.filename}]
Type: Image Document
Language: {language}
Description: {description}
Source: {file.filename}"""

        embeddings = state.embedder.embed_documents([image_text])
        metadatas = [{"source": file.filename, "type": "image", "filename": file.filename, "language": language}]

        state.session_store.add_file(
            session_id=session_id,
            filename=file.filename,
            chunks=[image_text],
            embeddings=embeddings,
            metadatas=metadatas,
        )

        all_session_files = state.session_store.get_files(session_id)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": 1,
            "session_id": session_id,
            "session_files": all_session_files,
            "description_preview": description[:300],
        }
    finally:
        os.unlink(tmp_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )