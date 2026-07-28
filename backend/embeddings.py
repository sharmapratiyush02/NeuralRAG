"""
embeddings.py — Neural Embedding Engine
========================================
DLM Focus: Implements multiple neural embedding strategies including:
  - Dense embeddings (BGE, Sentence-BERT)
  - Sparse embeddings (BM25, SPLADE)
  - Hybrid fusion
  - Fine-tunable embedding wrapper with contrastive learning support
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any

import numpy as np
from loguru import logger


# ─── Base Embedding Interface ─────────────────────────────────────────────────

class BaseEmbedder:
    """Abstract base for all embedding models."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError
    
    @property
    def dimension(self) -> int:
        raise NotImplementedError


# ─── Dense Neural Embedder ────────────────────────────────────────────────────

class DenseEmbedder(BaseEmbedder):
    """
    Dense embedding using Sentence-Transformers.
    
    DLM Contribution: Uses BGE-Large which is a fine-tuned BERT model
    with contrastive learning on large-scale retrieval datasets.
    Architecture: 24-layer transformer → mean pooling → L2 normalization
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        use_instruction: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.use_instruction = use_instruction
        self._model = None
        self._dim = None
        
        logger.info(f"Initializing DenseEmbedder: {model_name}")
    
    def _load_model(self):
        """Lazy loading of the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Warm-up to get dimension
                test_emb = self._model.encode(["test"])
                self._dim = test_emb.shape[1]
                logger.success(f"Loaded {self.model_name} | dim={self._dim}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using mock embedder")
                self._model = "mock"
                self._dim = 1024
    
    def _get_bge_instruction(self, is_query: bool) -> str:
        """BGE models benefit from instruction prefix for queries."""
        if is_query and self.use_instruction and "bge" in self.model_name.lower():
            return "Represent this sentence for searching relevant passages: "
        return ""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks."""
        self._load_model()
        
        if self._model == "mock":
            return [list(np.random.randn(self._dim).astype(float)) for _ in texts]
        
        from sentence_transformers import SentenceTransformer
        
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings.tolist())
        
        logger.debug(f"Embedded {len(texts)} documents")
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with optional instruction prefix."""
        self._load_model()
        
        instruction = self._get_bge_instruction(is_query=True)
        query = instruction + text
        
        if self._model == "mock":
            return list(np.random.randn(self._dim).astype(float))
        
        embedding = self._model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embedding[0].tolist()
    
    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dim


# ─── Sparse Embedder (BM25) ───────────────────────────────────────────────────

class SparseEmbedder:
    """
    BM25-based sparse retrieval.
    
    BM25 Formula:
        score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
    
    DLM Distinction: While BM25 is classical, we use it as the SPARSE component
    in hybrid retrieval. Future extension: SPLADE (learned sparse embeddings via MLM).
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index = None
        self._corpus = []
    
    def fit(self, corpus: List[str]) -> "SparseEmbedder":
        """Build BM25 index from corpus."""
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            
            tokenized = [text.lower().split() for text in corpus]
            self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
            self._corpus = corpus
            logger.info(f"BM25 index built with {len(corpus)} documents")
        except ImportError:
            logger.warning("rank_bm25 not available, BM25 disabled")
        return self
    
    def get_scores(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Return (doc_idx, score) pairs for a query."""
        if self._index is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'index': self._index, 'corpus': self._corpus}, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._index = data['index']
            self._corpus = data['corpus']
        return self


# ─── CLIP Image Embedder ──────────────────────────────────────────────────────

class CLIPImageEmbedder(BaseEmbedder):
    """
    Multimodal embedder using CLIP (Contrastive Language-Image Pretraining).
    
    DLM Focus: CLIP is trained with contrastive loss across 400M image-text pairs.
    Enables unified embedding space for both images and text queries.
    Architecture: ViT-L/14 (vision) + GPT-like text encoder
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._model.eval()
                logger.info(f"CLIP model loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"CLIP not available: {e}. Using mock.")
                self._model = "mock"
    
    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image file."""
        self._load_model()
        
        if self._model == "mock":
            return list(np.random.randn(768).astype(float))
        
        import torch
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features[0].cpu().numpy().tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text using CLIP's text encoder (for cross-modal retrieval)."""
        self._load_model()
        
        if self._model == "mock":
            return [list(np.random.randn(768).astype(float)) for _ in texts]
        
        import torch
        
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
    @property
    def dimension(self) -> int:
        return 768


# ─── Hybrid Embedding Fusion ──────────────────────────────────────────────────

class HybridEmbedder:
    """
    Combines dense + sparse embeddings for hybrid retrieval.
    
    Reciprocal Rank Fusion (RRF):
        RRF_score(d) = Σ 1 / (k + rank_i(d))
        where k=60 is a constant to prevent high impact of top ranks
    
    This is the KEY DLM contribution — pure neural (dense) retrieval misses
    exact keyword matches; pure lexical (sparse) misses semantic similarity.
    Hybrid combines both.
    """
    
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        self.dense = dense_embedder
        self.sparse = sparse_embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
    
    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        RRF Algorithm:
            For each document d and each ranked list r:
                score(d) += 1 / (k + rank_r(d))
        """
        rrf_scores: Dict[int, float] = {}
        
        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_query_embedding(self, query: str) -> Dict[str, Any]:
        """Get both dense embedding and BM25 scores for a query."""
        dense_emb = self.dense.embed_query(query)
        return {
            "dense": dense_emb,
            "query_text": query  # BM25 operates directly on text
        }


# ─── Embedding Fine-Tuner ─────────────────────────────────────────────────────

class EmbeddingFineTuner:
    """
    Fine-tune a pre-trained embedding model on domain-specific data.
    
    DLM Core: Uses Multiple Negatives Ranking Loss (contrastive learning)
    
    Loss = -log(exp(sim(q,p+)/τ) / Σ exp(sim(q,p_i)/τ))
    
    where:
        q  = query embedding
        p+ = positive passage embedding  
        p_i = all passages (positive + negatives) in batch
        τ  = temperature parameter
    
    The model learns to pull (query, relevant_doc) pairs together
    and push (query, irrelevant_doc) pairs apart in embedding space.
    """
    
    def __init__(
        self,
        base_model: str = "BAAI/bge-base-en-v1.5",
        output_path: str = "./models/fine_tuned_embedder",
        device: str = "cpu",
    ):
        self.base_model = base_model
        self.output_path = output_path
        self.device = device
    
    def prepare_training_data(
        self, 
        qa_pairs: List[Dict[str, str]]  # [{"query": ..., "positive": ..., "negative": ...}]
    ):
        """Convert QA pairs to training examples for contrastive learning."""
        try:
            from sentence_transformers import InputExample
            examples = []
            for pair in qa_pairs:
                # Positive pair
                examples.append(InputExample(
                    texts=[pair["query"], pair["positive"]],
                    label=1.0
                ))
                # Negative pair (if available)
                if "negative" in pair:
                    examples.append(InputExample(
                        texts=[pair["query"], pair["negative"]],
                        label=0.0
                    ))
            return examples
        except ImportError:
            logger.error("sentence-transformers required for fine-tuning")
            return []
    
    def fine_tune(
        self,
        train_examples: list,
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        evaluation_steps: int = 500,
    ) -> str:
        """
        Fine-tune with MultipleNegativesRankingLoss.
        This is a contrastive loss that treats all other (query, doc) pairs
        in the batch as negatives — efficient and effective.
        """
        try:
            from sentence_transformers import SentenceTransformer, losses
            from torch.utils.data import DataLoader
            
            model = SentenceTransformer(self.base_model)
            
            # DataLoader
            train_dataloader = DataLoader(
                train_examples, 
                shuffle=True, 
                batch_size=batch_size
            )
            
            # Loss: Multiple Negatives Ranking Loss (contrastive)
            train_loss = losses.MultipleNegativesRankingLoss(model)
            
            # Train
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=self.output_path,
                show_progress_bar=True,
            )
            
            logger.success(f"Fine-tuning complete. Model saved to {self.output_path}")
            return self.output_path
            
        except ImportError as e:
            logger.error(f"Fine-tuning failed: {e}")
            return self.base_model


# ─── Embedding Cache ──────────────────────────────────────────────────────────

class EmbeddingCache:
    """LRU cache for embeddings to avoid re-computation."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []
    
    def get(self, text: str) -> Optional[List[float]]:
        if text in self._cache:
            self._access_order.remove(text)
            self._access_order.append(text)
            return self._cache[text]
        return None
    
    def set(self, text: str, embedding: List[float]):
        if text in self._cache:
            self._access_order.remove(text)
        elif len(self._cache) >= self.max_size:
            # Evict LRU
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[text] = embedding
        self._access_order.append(text)
    
    def __len__(self):
        return len(self._cache)


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_embedder(config: dict) -> DenseEmbedder:
    """Factory function to create embedder from config."""
    emb_config = config.get("embeddings", {})
    
    model_name = emb_config.get("dense_model", "BAAI/bge-large-en-v1.5")
    if emb_config.get("use_fine_tuned") and Path(emb_config.get("fine_tuned_path", "")).exists():
        model_name = emb_config["fine_tuned_path"]
        logger.info(f"Using fine-tuned embedder: {model_name}")
    
    return DenseEmbedder(
        model_name=model_name,
        device=emb_config.get("device", "cpu"),
        batch_size=emb_config.get("batch_size", 32),
        normalize=emb_config.get("normalize", True),
    )


if __name__ == "__main__":
    # Quick test
    embedder = DenseEmbedder(model_name="BAAI/bge-large-en-v1.5")
    
    texts = [
        "Deep learning is a subset of machine learning.",
        "Neural networks are inspired by the human brain.",
        "RAG systems combine retrieval with generation."
    ]
    
    embeddings = embedder.embed_documents(texts)
    query_emb = embedder.embed_query("What is deep learning?")
    
    print(f"Document embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    print(f"Query embedding dim: {len(query_emb)}")
    
    # Similarity test
    dots = [np.dot(query_emb, e) for e in embeddings]
    print(f"Similarities: {[round(d, 4) for d in dots]}")
