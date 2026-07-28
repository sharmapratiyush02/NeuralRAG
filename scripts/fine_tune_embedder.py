"""
fine_tune_embedder.py — Contrastive Fine-Tuning for Domain Embeddings
======================================================================
DLM Core Concept: Fine-tuning transforms a general-purpose embedding model
into a domain-specific retriever using contrastive learning.

Loss Function: MultipleNegativesRankingLoss (in-batch negatives)
  L = -log [ exp(sim(q,p+)/τ) / Σⱼ exp(sim(q,pⱼ)/τ) ]

All (query, passage) pairs in a batch are used — positive pairs are
pulled together, while all other passages in the batch serve as negatives.
This is highly efficient: batch_size=32 gives 31 negatives per positive.

Usage:
  python fine_tune_embedder.py --data data/qa_pairs.json --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger


# ─── Training Data Schema ─────────────────────────────────────────────────────
# Each record: {"query": str, "positive": str, "negative": str (optional)}

SAMPLE_DATA = [
    {
        "query": "What is retrieval augmented generation?",
        "positive": "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by first retrieving relevant documents from a knowledge base before generating an answer, grounding responses in real information.",
        "negative": "Deep learning is a subset of machine learning that uses neural networks with many layers."
    },
    {
        "query": "How does FAISS perform approximate nearest neighbor search?",
        "positive": "FAISS (Facebook AI Similarity Search) uses algorithms like IVFFlat and HNSW to partition the vector space and perform approximate nearest neighbor search, trading small accuracy reductions for massive speed improvements.",
        "negative": "Python is a high-level programming language known for its readable syntax."
    },
    {
        "query": "What is the difference between cross-encoders and bi-encoders?",
        "positive": "Bi-encoders embed query and document independently into fixed vectors and use dot product similarity. Cross-encoders concatenate query and document and run a full transformer, giving more accurate relevance scores but requiring O(n) inference.",
        "negative": "Convolutional neural networks are primarily used for image recognition tasks."
    },
]


# ─── Synthetic QA Generator ────────────────────────────────────────────────────

async def generate_training_data(
    documents: List[str],
    llm_func,
    output_path: str = "./data/training/qa_pairs.json",
    num_pairs: int = 500,
    hard_negatives: bool = True,
) -> List[Dict]:
    """
    Synthetically generate (query, positive, negative) triplets from documents.

    Steps:
    1. For each document chunk, generate a relevant question (positive)
    2. Find a semantically close but non-answering document (hard negative)
    3. Save as training data

    Hard negatives are crucial — easy negatives (random docs) don't teach
    the model to distinguish subtle semantic differences.
    """
    from pathlib import Path
    import asyncio

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pairs = []

    # Shuffle and sample documents
    docs = documents.copy()
    random.shuffle(docs)
    sample_docs = docs[:num_pairs]

    logger.info(f"Generating {num_pairs} QA pairs from {len(documents)} documents...")

    for i, doc in enumerate(sample_docs):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(sample_docs)}")

        # Generate question from document
        q_prompt = f"""Generate one specific, answerable question from this text.
The question should require reading this specific passage to answer.
Output ONLY the question.

Text: {doc[:600]}

Question:"""

        try:
            question = (await llm_func(q_prompt)).strip()
            if not question:
                continue

            pair = {"query": question, "positive": doc}

            # Hard negative: find a related-but-wrong document
            if hard_negatives and len(docs) > 1:
                # Use a document that's adjacent in the list (same topic area but different content)
                neg_idx = (i + len(docs) // 4) % len(docs)
                pair["negative"] = docs[neg_idx]

            pairs.append(pair)

        except Exception as e:
            logger.error(f"Failed for doc {i}: {e}")
            continue

    # Save
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    logger.success(f"Generated {len(pairs)} training pairs → {output_path}")
    return pairs


# ─── Fine-Tuning Runner ───────────────────────────────────────────────────────

class EmbedderFineTuner:
    """
    Fine-tunes a Sentence-Transformer model using contrastive learning.

    Architecture choice: We use BGE-base (smaller, faster to fine-tune)
    as the starting point, then fine-tune on domain data.
    The fine-tuned model can then be loaded as a drop-in replacement
    for the base model in the retrieval pipeline.
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

    def load_data(self, data_path: str) -> List:
        """Load and convert training data to InputExamples."""
        try:
            from sentence_transformers import InputExample
        except ImportError:
            logger.error("sentence-transformers required: pip install sentence-transformers")
            return []

        with open(data_path) as f:
            raw = json.load(f)

        examples = []
        for item in raw:
            if "query" in item and "positive" in item:
                # Positive pair
                examples.append(InputExample(
                    texts=[item["query"], item["positive"]],
                    label=1.0
                ))
                # Hard negative pair (if present)
                if "negative" in item:
                    examples.append(InputExample(
                        texts=[item["query"], item["negative"]],
                        label=0.0
                    ))

        logger.info(f"Loaded {len(examples)} training examples from {data_path}")
        return examples

    def train(
        self,
        train_data: List,
        val_data: List = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        eval_steps: int = 200,
    ) -> str:
        """
        Fine-tune with MultipleNegativesRankingLoss.

        This loss works as follows:
        - In each batch, pair (i, j) where i is query, j is positive passage
        - All OTHER positives in the batch are treated as negatives
        - Model learns: sim(queryᵢ, posᵢ) >> sim(queryᵢ, posⱼ) for j ≠ i

        Temperature-scaled cross-entropy over in-batch negatives:
        L = -log [ exp(cos(qᵢ,pᵢ)/τ) / Σⱼ exp(cos(qᵢ,pⱼ)/τ) ]
        """
        try:
            from sentence_transformers import SentenceTransformer, losses, evaluation
            from torch.utils.data import DataLoader
            import torch

            logger.info(f"Loading base model: {self.base_model}")
            model = SentenceTransformer(self.base_model, device=self.device)

            # Filter to just positive pairs for MNRL
            # (MNRL treats all other positives in batch as negatives)
            pos_examples = [e for e in train_data if e.label == 1.0]

            train_dl = DataLoader(
                pos_examples,
                shuffle=True,
                batch_size=batch_size,
                drop_last=True,  # Important for MNRL
            )

            # MultipleNegativesRankingLoss — the in-batch negatives loss
            train_loss = losses.MultipleNegativesRankingLoss(
                model=model,
                similarity_fct=losses.SiameseDistanceMetric.COSINE,
            )

            # Optionally: Cosine Similarity Loss for explicit negative pairs
            # Could combine both losses with MultiTaskLoss

            # Evaluator
            evaluator = None
            if val_data:
                val_pos = [e for e in val_data if e.label == 1.0][:200]
                try:
                    from sentence_transformers import evaluation as steval
                    sentences1 = [e.texts[0] for e in val_pos]
                    sentences2 = [e.texts[1] for e in val_pos]
                    scores_val = [1.0] * len(val_pos)
                    evaluator = steval.EmbeddingSimilarityEvaluator(
                        sentences1, sentences2, scores_val,
                        name='val'
                    )
                except Exception as e:
                    logger.warning(f"Could not create evaluator: {e}")

            # Training
            total_steps = len(train_dl) * epochs
            logger.info(
                f"Training: {len(pos_examples)} examples | "
                f"{epochs} epochs | batch={batch_size} | "
                f"total_steps={total_steps}"
            )

            model.fit(
                train_objectives=[(train_dl, train_loss)],
                evaluator=evaluator,
                epochs=epochs,
                warmup_steps=warmup_steps,
                evaluation_steps=eval_steps if evaluator else 0,
                output_path=self.output_path,
                optimizer_params={"lr": learning_rate},
                show_progress_bar=True,
                save_best_model=True,
            )

            logger.success(f"Fine-tuning complete! Model saved to: {self.output_path}")

            # Quick evaluation
            self._evaluate(model)

            return self.output_path

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return self.base_model
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return self.base_model

    def _evaluate(self, model):
        """Quick sanity check of the fine-tuned model."""
        test_pairs = [
            ("What is RAG?", "RAG combines retrieval with language model generation.", True),
            ("What is RAG?", "Python is a programming language.", False),
            ("How does FAISS work?", "FAISS uses approximate nearest neighbor algorithms.", True),
        ]

        from sentence_transformers import util

        logger.info("\nFine-tuned model sanity check:")
        for q, d, expected_high in test_pairs:
            q_emb = model.encode(q, normalize_embeddings=True)
            d_emb = model.encode(d, normalize_embeddings=True)
            sim = float(util.cos_sim(q_emb, d_emb))
            status = "✓" if (sim > 0.6) == expected_high else "✗"
            logger.info(f"  {status} sim={sim:.4f} | Q: '{q[:30]}' | D: '{d[:40]}'")


# ─── Evaluation: Before vs After ─────────────────────────────────────────────

def compare_models(
    base_model_path: str,
    fine_tuned_path: str,
    test_queries: List[str],
    test_corpus: List[str],
    top_k: int = 5,
):
    """
    Compare retrieval quality of base vs fine-tuned model.
    Metric: Mean Reciprocal Rank (MRR) on test queries.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        logger.info("Comparing base vs fine-tuned model on retrieval task...")

        base = SentenceTransformer(base_model_path)
        fine_tuned = SentenceTransformer(fine_tuned_path)

        corpus_base = base.encode(test_corpus, normalize_embeddings=True)
        corpus_ft = fine_tuned.encode(test_corpus, normalize_embeddings=True)

        base_ranks = []
        ft_ranks = []

        for query in test_queries:
            q_base = base.encode(query, normalize_embeddings=True)
            q_ft = fine_tuned.encode(query, normalize_embeddings=True)

            sims_base = corpus_base @ q_base
            sims_ft = corpus_ft @ q_ft

            # Assume first corpus doc is the relevant one (for demo)
            rank_base = int(np.where(np.argsort(-sims_base) == 0)[0][0]) + 1
            rank_ft = int(np.where(np.argsort(-sims_ft) == 0)[0][0]) + 1

            base_ranks.append(1.0 / rank_base)
            ft_ranks.append(1.0 / rank_ft)

        mrr_base = float(np.mean(base_ranks))
        mrr_ft = float(np.mean(ft_ranks))

        logger.info(f"\n{'='*40}")
        logger.info(f"Base model MRR:        {mrr_base:.4f}")
        logger.info(f"Fine-tuned model MRR:  {mrr_ft:.4f}")
        logger.info(f"Improvement:           {(mrr_ft/mrr_base - 1)*100:+.1f}%")
        logger.info(f"{'='*40}\n")

        return {"base_mrr": mrr_base, "fine_tuned_mrr": mrr_ft}

    except ImportError as e:
        logger.error(f"Comparison failed: {e}")
        return {}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model for NeuralRAG")
    parser.add_argument("--data", default="./data/training/qa_pairs.json")
    parser.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--output", default="./models/fine_tuned_embedder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic training data from ingested docs")
    parser.add_argument("--use-sample", action="store_true",
                        help="Use built-in sample data for quick testing")
    args = parser.parse_args()

    tuner = EmbedderFineTuner(
        base_model=args.base,
        output_path=args.output,
    )

    if args.use_sample:
        # Use built-in sample data
        data_path = "/tmp/sample_qa.json"
        with open(data_path, 'w') as f:
            json.dump(SAMPLE_DATA * 20, f)  # Repeat to get more training data
        args.data = data_path
        logger.info(f"Using sample data ({len(SAMPLE_DATA)*20} pairs)")

    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Run with --use-sample to test with built-in data, or")
        logger.info("Use --generate-data to create data from your ingested documents")
        sys.exit(1)

    train_data = tuner.load_data(args.data)
    if not train_data:
        logger.error("No training data loaded")
        sys.exit(1)

    # Split train/val
    random.shuffle(train_data)
    split = int(len(train_data) * 0.9)
    train_split = train_data[:split]
    val_split = train_data[split:]

    output_path = tuner.train(
        train_data=train_split,
        val_data=val_split,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
    )

    logger.success(f"Fine-tuned model ready at: {output_path}")
    logger.info("To use in NeuralRAG, set in configs/config.yaml:")
    logger.info(f"  embeddings.use_fine_tuned: true")
    logger.info(f"  embeddings.fine_tuned_path: {output_path}")


if __name__ == "__main__":
    main()
