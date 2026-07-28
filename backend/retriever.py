"""
retriever.py — Hybrid Retrieval Engine
=======================================
DLM Focus: 
  - Dense retrieval via approximate nearest neighbor search (FAISS HNSW/IVF)
  - Sparse retrieval via BM25
  - Reciprocal Rank Fusion for combining results
  - Query rewriting via LLM (HyDE - Hypothetical Document Embeddings)
  - Contextual compression to distill retrieved chunks
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from embeddings import DenseEmbedder, SparseEmbedder, HybridEmbedder


@dataclass
class Document:
    """A retrieved document chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    doc_id: str = ""
    source: str = ""
    
    def __repr__(self):
        return f"Document(score={self.score:.4f}, source='{self.source}', preview='{self.content[:50]}...')"


@dataclass
class RetrievalResult:
    """Complete retrieval result with provenance."""
    query: str
    rewritten_queries: List[str]
    documents: List[Document]
    retrieval_time_ms: float
    dense_count: int
    sparse_count: int
    
    
# ─── Vector Store ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages vector storage using ChromaDB (primary) with FAISS index.
    
    ChromaDB provides:
    - Persistent storage
    - Metadata filtering
    - Collection management
    
    FAISS provides:
    - ANN search algorithms (HNSW, IVF)
    - High-performance similarity search
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "neural_rag_docs",
        embedder: Optional[DenseEmbedder] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = embedder or DenseEmbedder()
        self._collection = None
        
        os.makedirs(persist_directory, exist_ok=True)
    
    def _get_collection(self):
        """Lazy initialize ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(
                    f"ChromaDB collection '{self.collection_name}' "
                    f"({self._collection.count()} docs)"
                )
            except ImportError:
                logger.warning("chromadb not installed. Using in-memory store.")
                self._collection = InMemoryStore()
        
        return self._collection
    
    def add_documents(
        self, 
        documents: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
    ) -> int:
        """Add documents to vector store."""
        if not documents:
            return 0
        
        collection = self._get_collection()
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedder.embed_documents(documents)
        
        # Generate IDs if not provided
        if ids is None:
            import hashlib
            ids = [
                hashlib.md5(doc.encode()).hexdigest()[:16] 
                for doc in documents
            ]
        
        metadatas = metadatas or [{} for _ in documents]
        
        # Batch insert
        batch_size = 500
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch_slice = slice(i, i + batch_size)
            
            try:
                collection.add(
                    documents=documents[batch_slice],
                    embeddings=embeddings[batch_slice],
                    metadatas=metadatas[batch_slice],
                    ids=ids[batch_slice],
                )
                total_added += len(documents[batch_slice])
            except Exception as e:
                # Handle duplicates
                logger.warning(f"Batch {i//batch_size}: {e}")
                # Try upserting instead
                collection.upsert(
                    documents=documents[batch_slice],
                    embeddings=embeddings[batch_slice],
                    metadatas=metadatas[batch_slice],
                    ids=ids[batch_slice],
                )
                total_added += len(documents[batch_slice])
        
        logger.success(f"Added {total_added} documents to vector store")
        return total_added
    
    def similarity_search(
        self, 
        query: str, 
        top_k: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Document]:
        """Dense similarity search."""
        collection = self._get_collection()
        
        query_embedding = self.embedder.embed_query(query)
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count() or top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        
        results = collection.query(**kwargs)
        
        documents = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # Convert cosine distance to similarity score
                score = 1.0 - dist if dist <= 1.0 else 0.0
                documents.append(Document(
                    content=doc,
                    metadata=meta or {},
                    score=score,
                    source=meta.get("source", "unknown") if meta else "unknown",
                ))
        
        return documents
    
    def count(self) -> int:
        return self._get_collection().count()
    
    def delete_collection(self):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")


class InMemoryStore:
    """Fallback in-memory vector store when ChromaDB isn't available."""
    
    def __init__(self):
        self._documents = []
        self._embeddings = []
        self._metadatas = []
        self._ids = []
    
    def count(self):
        return len(self._documents)
    
    def add(self, documents, embeddings, metadatas, ids):
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)
    
    def upsert(self, documents, embeddings, metadatas, ids):
        self.add(documents, embeddings, metadatas, ids)
    
    def query(self, query_embeddings, n_results, include, **kwargs):
        if not self._embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        q_emb = np.array(query_embeddings[0])
        doc_embs = np.array(self._embeddings)
        
        # Cosine similarities
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs_norm = doc_embs / (norms + 1e-8)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        sims = doc_embs_norm @ q_norm
        
        top_indices = np.argsort(sims)[::-1][:n_results]
        
        return {
            "documents": [[self._documents[i] for i in top_indices]],
            "metadatas": [[self._metadatas[i] for i in top_indices]],
            "distances": [[1.0 - float(sims[i]) for i in top_indices]],
        }


# ─── Query Rewriter ───────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Transforms the user query to improve retrieval recall.
    
    Strategies:
    1. HyDE (Hypothetical Document Embeddings): Generate a fake answer,
       embed it, and use that embedding for retrieval. Works because the 
       hypothetical answer is in "document space" rather than "query space".
    
    2. Multi-Query: Generate N paraphrases of the query to cover
       different semantic angles.
    
    3. Step-Back: Ask a more abstract version of the question first
       to retrieve background knowledge.
    """
    
    def __init__(self, llm_func=None, strategy: str = "multi_query"):
        self.llm_func = llm_func
        self.strategy = strategy
    
    async def rewrite(self, query: str, num_rewrites: int = 3) -> List[str]:
        """Generate rewritten queries."""
        rewrites = [query]  # Always include original
        
        if self.llm_func is None:
            return rewrites
        
        try:
            if self.strategy == "multi_query":
                rewrites.extend(await self._multi_query(query, num_rewrites))
            elif self.strategy == "hypothetical_document":
                hyde_doc = await self._hyde(query)
                if hyde_doc:
                    rewrites.append(hyde_doc)
            elif self.strategy == "step_back":
                step_back = await self._step_back(query)
                if step_back:
                    rewrites.append(step_back)
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
        
        return list(dict.fromkeys(rewrites))  # Deduplicate preserving order
    
    async def _multi_query(self, query: str, n: int) -> List[str]:
        prompt = f"""Generate {n} different ways to ask this question. 
Output ONLY the questions, one per line, no numbering or explanation.

Question: {query}"""
        
        response = await self.llm_func(prompt)
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[:n]
    
    async def _hyde(self, query: str) -> str:
        """Hypothetical Document Embeddings."""
        prompt = f"""Write a short passage (2-3 sentences) that would be a 
perfect answer to this question. Be specific and factual.

Question: {query}
Answer:"""
        return await self.llm_func(prompt)
    
    async def _step_back(self, query: str) -> str:
        """Generate a more abstract/general version of the query."""
        prompt = f"""What is the more general or abstract question behind this specific question? 
Output only the generalized question.

Specific question: {query}
General question:"""
        return await self.llm_func(prompt)


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Main retrieval engine combining:
    1. Dense retrieval (ChromaDB + FAISS)
    2. Sparse retrieval (BM25)
    3. Reciprocal Rank Fusion
    4. Query rewriting
    
    This is the core DLM contribution to the RAG pipeline.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        dense_embedder: DenseEmbedder,
        sparse_embedder: Optional[SparseEmbedder] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion_method: str = "rrf",
        query_rewriter: Optional[QueryRewriter] = None,
    ):
        self.vector_store = vector_store
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_method = fusion_method
        self.query_rewriter = query_rewriter
        
        self._all_documents: List[str] = []  # For BM25
    
    def add_to_bm25_index(self, documents: List[str]):
        """Maintain BM25 index in sync with vector store."""
        self._all_documents.extend(documents)
        if self.sparse_embedder:
            self.sparse_embedder.fit(self._all_documents)
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        top_k_retrieval: int = 20,
        metadata_filter: Optional[Dict] = None,
    ) -> RetrievalResult:
        """
        Full hybrid retrieval pipeline:
        
        1. (Optional) Rewrite query
        2. Dense search for each query variant
        3. Sparse (BM25) search
        4. Fuse results with RRF
        5. Return top-k deduplicated documents
        """
        start_time = time.time()
        
        # Step 1: Query rewriting
        rewritten_queries = [query]
        if self.query_rewriter:
            rewritten_queries = await self.query_rewriter.rewrite(query)
            logger.debug(f"Rewrote query → {len(rewritten_queries)} variants")
        
        # Step 2: Dense retrieval for all query variants
        all_dense_results: List[List[Document]] = []
        for q in rewritten_queries:
            dense_docs = self.vector_store.similarity_search(
                q, top_k=top_k_retrieval, where=metadata_filter
            )
            all_dense_results.append(dense_docs)
        
        # Step 3: Sparse (BM25) retrieval
        sparse_doc_indices: List[Tuple[int, float]] = []
        if self.sparse_embedder and self._all_documents:
            sparse_doc_indices = self.sparse_embedder.get_scores(
                query, top_k=top_k_retrieval
            )
        
        # Step 4: Fusion
        if self.fusion_method == "rrf":
            fused_documents = self._reciprocal_rank_fusion(
                all_dense_results, 
                sparse_doc_indices,
                top_k_retrieval,
            )
        else:
            fused_documents = self._weighted_fusion(all_dense_results)
        
        # Deduplicate
        seen_content = set()
        unique_docs = []
        for doc in fused_documents:
            content_key = doc.content[:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append(doc)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            rewritten_queries=rewritten_queries,
            documents=unique_docs[:top_k_retrieval],
            retrieval_time_ms=elapsed_ms,
            dense_count=len(all_dense_results[0]) if all_dense_results else 0,
            sparse_count=len(sparse_doc_indices),
        )
    
    def _reciprocal_rank_fusion(
        self,
        dense_results_list: List[List[Document]],
        sparse_results: List[Tuple[int, float]],
        top_k: int = 20,
        k: int = 60,
    ) -> List[Document]:
        """
        RRF: score(d) = Σ 1 / (k + rank(d, list_i))
        """
        content_to_doc: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = {}
        
        # Process dense results from each query variant
        for ranked_list in dense_results_list:
            for rank, doc in enumerate(ranked_list):
                key = doc.content[:100]
                if key not in content_to_doc:
                    content_to_doc[key] = doc
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        
        # Process sparse results
        if sparse_results and self._all_documents:
            for rank, (doc_idx, bm25_score) in enumerate(sparse_results):
                if doc_idx < len(self._all_documents):
                    content = self._all_documents[doc_idx]
                    key = content[:100]
                    if key not in content_to_doc:
                        content_to_doc[key] = Document(
                            content=content, 
                            score=bm25_score,
                            source="bm25"
                        )
                    rrf_score = 1.0 / (k + rank + 1) * self.sparse_weight
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
        
        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        results = []
        for key in sorted_keys[:top_k]:
            doc = content_to_doc[key]
            doc.score = rrf_scores[key]
            results.append(doc)
        
        return results
    
    def _weighted_fusion(
        self, dense_results_list: List[List[Document]]
    ) -> List[Document]:
        """Simple score-weighted fusion."""
        content_to_best: Dict[str, Document] = {}
        
        for ranked_list in dense_results_list:
            for doc in ranked_list:
                key = doc.content[:100]
                if key not in content_to_best or doc.score > content_to_best[key].score:
                    content_to_best[key] = doc
        
        return sorted(content_to_best.values(), key=lambda d: d.score, reverse=True)


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_retriever(config: dict, llm_func=None) -> HybridRetriever:
    """Create the full retrieval system from config."""
    
    # Embedder
    dense_embedder = DenseEmbedder(
        model_name=config.get("embeddings", {}).get("dense_model", "BAAI/bge-large-en-v1.5"),
        device=config.get("embeddings", {}).get("device", "cpu"),
    )
    
    # Vector store
    vs_config = config.get("vector_store", {}).get("chroma", {})
    vector_store = VectorStore(
        persist_directory=vs_config.get("persist_directory", "./data/chroma_db"),
        collection_name=vs_config.get("collection_name", "neural_rag_docs"),
        embedder=dense_embedder,
    )
    
    # Sparse embedder
    sparse_embedder = SparseEmbedder(
        k1=config.get("embeddings", {}).get("bm25_k1", 1.5),
        b=config.get("embeddings", {}).get("bm25_b", 0.75),
    )
    
    # Query rewriter
    retrieval_config = config.get("retrieval", {})
    rewriter = None
    if retrieval_config.get("query_rewriting", {}).get("enabled", True) and llm_func:
        rewriter = QueryRewriter(
            llm_func=llm_func,
            strategy=retrieval_config.get("query_rewriting", {}).get(
                "strategy", "multi_query"
            )
        )
    
    return HybridRetriever(
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        dense_weight=retrieval_config.get("dense_weight", 0.7),
        sparse_weight=retrieval_config.get("sparse_weight", 0.3),
        fusion_method=retrieval_config.get("fusion_method", "rrf"),
        query_rewriter=rewriter,
    )
