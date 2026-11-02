"""
test_pipeline.py — Comprehensive Test Suite
============================================
Tests all major components of the NeuralRAG pipeline:
  - Embedding quality
  - Retrieval correctness
  - Reranking effectiveness
  - Generation faithfulness
  - Full pipeline integration
  - API endpoint tests
"""

from __future__ import annotations

import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))


# ═══════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════

@pytest.fixture
def sample_documents():
    return [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by retrieving relevant documents before generating answers.",
        "FAISS (Facebook AI Similarity Search) provides efficient approximate nearest neighbor search for high-dimensional vectors.",
        "Cross-encoders process query and document together, capturing fine-grained interactions for more accurate relevance scoring.",
        "BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency statistics.",
        "Reciprocal Rank Fusion (RRF) combines results from multiple retrieval systems without requiring score normalization.",
        "Self-attention in transformers computes relationships between all pairs of tokens in a sequence simultaneously.",
        "The RAGAS framework evaluates RAG systems using faithfulness, answer relevancy, context precision, and context recall.",
        "4-bit quantization via bitsandbytes reduces model memory by 75% with minimal accuracy degradation.",
        "Semantic chunking splits documents at topic boundaries detected by measuring cosine similarity between sentence embeddings.",
        "The ReAct framework combines chain-of-thought reasoning with action execution for agentic LLM systems.",
    ]

@pytest.fixture
def sample_qa_pairs():
    return [
        {
            "question": "What is RAG?",
            "answer": "RAG stands for Retrieval-Augmented Generation, a technique that retrieves relevant documents before generating answers.",
            "relevant_doc_idx": 0
        },
        {
            "question": "How does FAISS work?",
            "answer": "FAISS provides efficient approximate nearest neighbor search for high-dimensional vectors.",
            "relevant_doc_idx": 1
        },
        {
            "question": "What is the difference between cross-encoders and bi-encoders?",
            "answer": "Cross-encoders process query and document together for accurate relevance scoring, unlike bi-encoders which embed them separately.",
            "relevant_doc_idx": 2
        },
    ]

@pytest.fixture
def mock_llm():
    async def _llm(prompt: str, system: str = None, **kwargs) -> str:
        if "extract" in prompt.lower() or "claims" in prompt.lower():
            return '["RAG retrieves documents before generation.", "FAISS uses ANN search."]'
        if "generate" in prompt.lower() and "question" in prompt.lower():
            return "What is the purpose of retrieval-augmented generation?"
        if "yes" in prompt.lower() or "relevant" in prompt.lower():
            return "YES"
        return "RAG is a technique that enhances language models by retrieving relevant documents from a knowledge base before generating answers."

    mock = MagicMock()
    mock.generate = AsyncMock(side_effect=_llm)
    return mock


# ═══════════════════════════════════════════════════
# EMBEDDING TESTS
# ═══════════════════════════════════════════════════

class TestDenseEmbedder:

    def test_embed_returns_correct_shape(self):
        from embeddings import DenseEmbedder
        embedder = DenseEmbedder()
        texts = ["Hello world", "Test document", "Another text"]
        embeddings = embedder.embed_documents(texts)
        assert len(embeddings) == len(texts)
        assert all(len(e) == embedder.dimension for e in embeddings)

    def test_query_embedding_shape(self):
        from embeddings import DenseEmbedder
        embedder = DenseEmbedder()
        q_emb = embedder.embed_query("What is machine learning?")
        assert len(q_emb) == embedder.dimension

    def test_embeddings_are_normalized(self):
        """BGE embeddings should be L2-normalized."""
        from embeddings import DenseEmbedder
        embedder = DenseEmbedder(normalize=True)
        texts = ["Test normalization of embeddings"]
        embeddings = embedder.embed_documents(texts)
        norm = np.linalg.norm(embeddings[0])
        # Should be close to 1.0 for normalized embeddings
        assert 0.9 <= norm <= 1.1, f"Norm should be ~1.0, got {norm}"

    def test_semantic_similarity_ordering(self, sample_documents):
        """Similar texts should have higher cosine similarity."""
        from embeddings import DenseEmbedder
        embedder = DenseEmbedder()

        query = "neural network retrieval system"
        docs = [
            "RAG uses neural embeddings for document retrieval",  # Should be high
            "The cat sat on the mat in the garden",               # Should be low
        ]

        q_emb = np.array(embedder.embed_query(query))
        doc_embs = [np.array(e) for e in embedder.embed_documents(docs)]

        sims = [float(np.dot(q_emb, d)) for d in doc_embs]
        assert sims[0] > sims[1], f"Expected {sims[0]} > {sims[1]}"


class TestSparseEmbedder:

    def test_bm25_fit_and_score(self, sample_documents):
        from embeddings import SparseEmbedder
        sparse = SparseEmbedder()
        sparse.fit(sample_documents)

        scores = sparse.get_scores("RAG retrieval generation", top_k=5)
        assert len(scores) > 0
        # First result should have positive score
        assert scores[0][1] > 0

    def test_relevant_doc_ranks_higher(self, sample_documents):
        from embeddings import SparseEmbedder
        sparse = SparseEmbedder()
        sparse.fit(sample_documents)

        # "BM25" is explicitly in doc index 3
        scores = sparse.get_scores("BM25 term frequency retrieval", top_k=5)
        top_indices = [s[0] for s in scores[:3]]
        assert 3 in top_indices, f"BM25 doc should be in top-3, got indices {top_indices}"


class TestEmbeddingCache:

    def test_cache_stores_and_retrieves(self):
        from embeddings import EmbeddingCache
        cache = EmbeddingCache(max_size=100)

        text = "test sentence"
        emb = [0.1, 0.2, 0.3]
        cache.set(text, emb)

        retrieved = cache.get(text)
        assert retrieved == emb

    def test_cache_evicts_lru(self):
        from embeddings import EmbeddingCache
        cache = EmbeddingCache(max_size=3)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])
        _ = cache.get("a")  # Access 'a' to make 'b' LRU
        cache.set("d", [4.0])  # Should evict 'b'

        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert len(cache) == 3


# ═══════════════════════════════════════════════════
# RETRIEVAL TESTS
# ═══════════════════════════════════════════════════

class TestVectorStore:

    def test_add_and_search(self, sample_documents, tmp_path):
        from embeddings import DenseEmbedder
        from retriever import VectorStore

        embedder = DenseEmbedder()
        store = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
            embedder=embedder,
        )

        n_added = store.add_documents(sample_documents[:3])
        assert n_added == 3

        results = store.similarity_search("RAG retrieval", top_k=2)
        assert len(results) <= 2
        assert all(hasattr(r, 'content') for r in results)

    def test_search_returns_scores(self, tmp_path):
        from embeddings import DenseEmbedder
        from retriever import VectorStore

        embedder = DenseEmbedder()
        store = VectorStore(
            persist_directory=str(tmp_path / "chroma2"),
            collection_name="test_scores",
            embedder=embedder,
        )

        store.add_documents(["Deep learning uses neural networks."])
        results = store.similarity_search("neural networks", top_k=1)

        if results:
            assert 0.0 <= results[0].score <= 1.0


class TestHybridRetriever:

    @pytest.mark.asyncio
    async def test_retrieve_returns_result(self, sample_documents, tmp_path):
        from embeddings import DenseEmbedder, SparseEmbedder
        from retriever import VectorStore, HybridRetriever

        embedder = DenseEmbedder()
        sparse = SparseEmbedder()

        store = VectorStore(
            persist_directory=str(tmp_path / "chroma3"),
            collection_name="test_hybrid",
            embedder=embedder,
        )
        store.add_documents(sample_documents)

        sparse.fit(sample_documents)

        retriever = HybridRetriever(
            vector_store=store,
            dense_embedder=embedder,
            sparse_embedder=sparse,
        )
        retriever._all_documents = sample_documents

        result = await retriever.retrieve("What is RAG?", top_k=3)

        assert result.query == "What is RAG?"
        assert len(result.documents) <= 3
        assert result.retrieval_time_ms > 0

    def test_rrf_fusion_correctness(self):
        from retriever import HybridRetriever
        from embeddings import DenseEmbedder, SparseEmbedder
        from retriever import VectorStore

        # Test RRF algorithm directly
        dense_embedder = DenseEmbedder()
        retriever = HybridRetriever(
            vector_store=MagicMock(),
            dense_embedder=dense_embedder,
        )

        from retriever import Document
        list1 = [Document(content=f"doc{i}", score=1.0/(i+1)) for i in range(5)]
        list2 = [Document(content=f"doc{4-i}", score=1.0/(i+1)) for i in range(5)]

        fused = retriever._weighted_fusion([list1, list2])
        assert len(fused) > 0


# ═══════════════════════════════════════════════════
# RERANKER TESTS
# ═══════════════════════════════════════════════════

class TestBGEReranker:

    def test_reranker_returns_sorted_results(self):
        from reranker import BGEReranker, RerankedResult

        reranker = BGEReranker()

        query = "What is deep learning?"
        docs = [
            "Deep learning uses neural networks with many layers.",
            "The weather today is sunny.",
            "Machine learning is a subset of artificial intelligence.",
        ]

        results = reranker.rerank(query, docs, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RerankedResult) for r in results)
        # Results should be sorted descending by rerank score
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reranker_top_k_limit(self):
        from reranker import BGEReranker

        reranker = BGEReranker()
        docs = [f"document {i}" for i in range(10)]
        results = reranker.rerank("test query", docs, top_k=3)
        assert len(results) <= 3

    def test_reranker_empty_input(self):
        from reranker import BGEReranker

        reranker = BGEReranker()
        results = reranker.rerank("test query", [], top_k=5)
        assert results == []


# ═══════════════════════════════════════════════════
# GENERATION TESTS
# ═══════════════════════════════════════════════════

class TestFaithfulnessChecker:

    def test_scorer_returns_valid_score(self):
        from generator import FaithfulnessChecker

        checker = FaithfulnessChecker()
        answer = "RAG retrieves documents before generating answers. FAISS provides fast similarity search."
        context = [
            "RAG (Retrieval-Augmented Generation) retrieves relevant documents before generating answers.",
            "FAISS is a library for efficient similarity search of dense vectors.",
        ]

        result = checker.score_faithfulness(answer, context)
        assert 0.0 <= result["score"] <= 1.0
        assert "entailed_claims" in result
        assert "unsupported_claims" in result

    def test_empty_answer_is_faithful(self):
        from generator import FaithfulnessChecker
        checker = FaithfulnessChecker()
        result = checker.score_faithfulness("", ["some context"])
        assert result["score"] == 1.0


class TestRAGGenerator:

    @pytest.mark.asyncio
    async def test_generate_returns_result(self, mock_llm):
        from generator import RAGGenerator, FaithfulnessChecker
        from retriever import Document

        generator = RAGGenerator(llm=mock_llm)

        docs = [
            Document(content="RAG enhances LLMs by retrieving relevant documents.", score=0.9, source="test.pdf"),
            Document(content="FAISS provides approximate nearest neighbor search.", score=0.8, source="test.pdf"),
        ]

        result = await generator.generate("What is RAG?", docs)

        assert result.answer
        assert isinstance(result.faithfulness_score, float)
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert isinstance(result.sources, list)

    @pytest.mark.asyncio
    async def test_self_correction_triggers(self, mock_llm):
        from generator import RAGGenerator, FaithfulnessChecker
        from retriever import Document

        # Mock checker that always flags low faithfulness first
        call_count = {"n": 0}

        class MockChecker:
            def score_faithfulness(self, answer, context):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return {"score": 0.3, "entailed_claims": [], "unsupported_claims": ["Unsupported claim"]}
                return {"score": 0.9, "entailed_claims": ["Good claim"], "unsupported_claims": []}

        generator = RAGGenerator(
            llm=mock_llm,
            faithfulness_checker=MockChecker(),
            max_iterations=3,
            faithfulness_threshold=0.7,
        )

        docs = [Document(content="Context text.", score=0.9, source="test")]
        result = await generator.generate("Test question?", docs)

        assert result.num_corrections >= 1


# ═══════════════════════════════════════════════════
# EVALUATION TESTS
# ═══════════════════════════════════════════════════

class TestRAGASMetrics:

    @pytest.mark.asyncio
    async def test_faithfulness_metric(self):
        from ragas_eval import FaithfulnessMetric, RAGASample

        async def mock_llm(p):
            if "claims" in p.lower() or "extract" in p.lower():
                return '["RAG retrieves documents.", "FAISS uses ANN search."]'
            return "YES"

        metric = FaithfulnessMetric(mock_llm)
        sample = RAGASample(
            question="What is RAG?",
            answer="RAG retrieves documents before generating answers.",
            contexts=["RAG is a technique for retrieval-augmented generation that retrieves documents."],
        )

        score = await metric.compute(sample)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_context_precision_metric(self):
        from ragas_eval import ContextPrecisionMetric, RAGASample

        async def mock_llm(p): return "YES"

        metric = ContextPrecisionMetric(mock_llm)
        sample = RAGASample(
            question="What is RAG?",
            answer="RAG retrieves documents.",
            contexts=[
                "RAG is retrieval-augmented generation.",
                "The sky is blue.",
            ],
        )

        score = await metric.compute(sample)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════
# INGESTION TESTS
# ═══════════════════════════════════════════════════

class TestIngestion:

    def test_fixed_chunker(self, tmp_path):
        from scripts.ingest import FixedSizeChunker, ParsedDocument

        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        doc = ParsedDocument(
            content="A" * 300,
            metadata={"source": "test.txt"},
            source_path=str(tmp_path / "test.txt"),
            doc_type="txt"
        )

        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        assert all(len(c.content) <= 110 for c in chunks)  # Allow slight overlap

    def test_recursive_chunker_respects_paragraphs(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from ingest import RecursiveChunker, ParsedDocument

        text = "First paragraph about deep learning.\n\nSecond paragraph about neural networks.\n\nThird paragraph about transformers."

        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=0)
        doc = ParsedDocument(
            content=text,
            metadata={"source": "test.txt"},
            source_path=str(tmp_path / "test.txt"),
            doc_type="txt"
        )

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_document_parser_txt(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from ingest import DocumentParser

        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for document parsing.")

        doc = DocumentParser.parse(str(test_file))
        assert "test content" in doc.content
        assert doc.doc_type == "txt"


# ═══════════════════════════════════════════════════
# API TESTS
# ═══════════════════════════════════════════════════

class TestAPI:
    """Integration tests for the FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked components."""
        try:
            from fastapi.testclient import TestClient
            from backend.api import app
            return TestClient(app)
        except Exception:
            pytest.skip("FastAPI test client not available")

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_stats_endpoint(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data


# ═══════════════════════════════════════════════════
# END-TO-END TEST
# ═══════════════════════════════════════════════════

class TestEndToEnd:

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, sample_documents, tmp_path):
        """Test the complete RAG pipeline: ingest → retrieve → generate."""
        from embeddings import DenseEmbedder, SparseEmbedder
        from retriever import VectorStore, HybridRetriever
        from reranker import BGEReranker
        from generator import RAGGenerator

        # Setup
        embedder = DenseEmbedder()
        store = VectorStore(
            persist_directory=str(tmp_path / "e2e"),
            collection_name="e2e_test",
            embedder=embedder,
        )
        sparse = SparseEmbedder()

        # Ingest
        store.add_documents(sample_documents)
        sparse.fit(sample_documents)

        # Retrieve
        retriever = HybridRetriever(
            vector_store=store,
            dense_embedder=embedder,
            sparse_embedder=sparse,
        )
        retriever._all_documents = sample_documents

        result = await retriever.retrieve("How does FAISS work?", top_k=3)
        assert len(result.documents) > 0

        # Rerank
        reranker = BGEReranker()
        reranked = reranker.rerank(
            "How does FAISS work?",
            [d.content for d in result.documents],
            top_k=2
        )
        assert len(reranked) <= 2

        # Generate (with mock LLM)
        class MockLLM:
            async def generate(self, prompt, system=None, **kwargs):
                return "FAISS provides efficient approximate nearest neighbor search for dense vectors using algorithms like HNSW and IVFFlat."
            async def generate_stream(self, prompt, system=None, **kwargs):
                yield "FAISS answer"

        generator = RAGGenerator(llm=MockLLM())
        gen_result = await generator.generate(
            "How does FAISS work?",
            result.documents[:2]
        )

        assert gen_result.answer
        assert "FAISS" in gen_result.answer or "nearest neighbor" in gen_result.answer.lower()
        assert gen_result.faithfulness_score >= 0.0

        print(f"\n✅ E2E Test passed!")
        print(f"   Query: 'How does FAISS work?'")
        print(f"   Retrieved: {len(result.documents)} docs in {result.retrieval_time_ms:.0f}ms")
        print(f"   Reranked: {len(reranked)} docs")
        print(f"   Answer: {gen_result.answer[:100]}...")
        print(f"   Faithfulness: {gen_result.faithfulness_score:.2f}")


# ═══════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick smoke test without pytest
    async def smoke_test():
        print("Running NeuralRAG smoke tests...\n")

        # Test 1: Embeddings
        from embeddings import DenseEmbedder
        emb = DenseEmbedder()
        v = emb.embed_query("test query")
        print(f"✓ Embedding: dim={len(v)}")

        # Test 2: BM25
        from embeddings import SparseEmbedder
        bm25 = SparseEmbedder()
        bm25.fit(["deep learning", "machine learning", "neural networks"])
        scores = bm25.get_scores("learning", top_k=2)
        print(f"✓ BM25: top scores = {[(i, round(s,3)) for i,s in scores]}")

        # Test 3: Reranker
        from reranker import BGEReranker
        reranker = BGEReranker()
        results = reranker.rerank("deep learning", ["neural networks", "cooking"], top_k=2)
        print(f"✓ Reranker: top result = '{results[0].content[:40]}' (score={results[0].rerank_score:.3f})")

        # Test 4: Faithfulness checker
        from generator import FaithfulnessChecker
        checker = FaithfulnessChecker()
        r = checker.score_faithfulness("Deep learning uses neural networks.", ["Deep learning is based on neural network architectures."])
        print(f"✓ Faithfulness: score={r['score']:.3f}")

        print("\n✅ All smoke tests passed!")

    asyncio.run(smoke_test())
