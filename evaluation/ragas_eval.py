"""
ragas_eval.py — RAGAS Evaluation Framework
==========================================
DLM Focus: Evaluation using neural models, not just string matching.

RAGAS Metrics (all computed using LLMs + embeddings):

1. Faithfulness: 
   - Claims extracted from answer via LLM
   - Each claim verified against context via NLI
   - score = |supported_claims| / |total_claims|

2. Answer Relevancy:
   - Generate N questions from the answer using LLM
   - Embed them + original question
   - score = mean cosine_sim(generated_Qi, original_Q)

3. Context Precision:
   - For each retrieved chunk, check if it's actually relevant
   - Uses LLM to judge relevance
   - score = |relevant_chunks| / |total_chunks|

4. Context Recall:
   - Given ground truth answer, check how much is covered by context
   - score = |GT_sentences_in_context| / |total_GT_sentences|
"""

from __future__ import annotations

import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class RAGASample:
    """One evaluation sample."""
    question: str
    answer: str
    contexts: List[str]          # Retrieved context chunks
    ground_truth: str = ""       # Reference answer (for recall)
    metadata: Dict = field(default_factory=dict)


@dataclass 
class RAGASResult:
    """Evaluation results for a dataset."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    per_sample_scores: List[Dict]
    num_samples: int
    
    @property
    def overall_score(self) -> float:
        """Harmonic mean of all metrics."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        # Harmonic mean
        scores = [s for s in scores if s > 0]
        if not scores:
            return 0.0
        return len(scores) / sum(1.0 / s for s in scores)
    
    def to_dict(self) -> Dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "overall_score": round(self.overall_score, 4),
            "num_samples": self.num_samples,
        }
    
    def __str__(self):
        return (
            f"\n{'='*50}\n"
            f"  RAGAS Evaluation Results ({self.num_samples} samples)\n"
            f"{'='*50}\n"
            f"  Faithfulness:      {self.faithfulness:.4f}\n"
            f"  Answer Relevancy:  {self.answer_relevancy:.4f}\n"
            f"  Context Precision: {self.context_precision:.4f}\n"
            f"  Context Recall:    {self.context_recall:.4f}\n"
            f"  ─────────────────────────────────\n"
            f"  Overall Score:     {self.overall_score:.4f}\n"
            f"{'='*50}"
        )


# ─── Metric Computers ─────────────────────────────────────────────────────────

class FaithfulnessMetric:
    """
    Faithfulness = % of answer claims supported by context.
    
    Algorithm:
    1. Use LLM to extract atomic claims from answer
    2. For each claim, use NLI or LLM to check if context supports it
    3. faithfulness = supported / total
    """
    
    def __init__(self, llm_func):
        self.llm = llm_func
    
    async def compute(self, sample: RAGASample) -> float:
        # Step 1: Extract claims
        claims = await self._extract_claims(sample.answer)
        if not claims:
            return 1.0  # No claims = vacuously faithful
        
        # Step 2: Verify each claim
        context_str = "\n".join(sample.contexts[:5])  # Use top 5 chunks
        supported = 0
        
        for claim in claims:
            is_supported = await self._verify_claim(claim, context_str)
            if is_supported:
                supported += 1
        
        score = supported / len(claims)
        logger.debug(f"Faithfulness: {supported}/{len(claims)} claims supported = {score:.3f}")
        return score
    
    async def _extract_claims(self, answer: str) -> List[str]:
        """Extract atomic factual claims from the answer."""
        prompt = f"""Break down this answer into individual factual claims.
Each claim should be a single verifiable statement.
Output ONLY a JSON array of strings.

Answer: {answer}

Claims:"""
        
        response = await self.llm(prompt)
        
        try:
            # Try to parse JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                return [c for c in claims if isinstance(c, str) and len(c) > 10]
        except Exception:
            pass
        
        # Fallback: split on periods
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    async def _verify_claim(self, claim: str, context: str) -> bool:
        """Check if context supports the claim."""
        prompt = f"""Does the following context support or entail this claim?
Output ONLY "YES" or "NO".

Context: {context[:1500]}

Claim: {claim}

Supported (YES/NO):"""
        
        response = await self.llm(prompt)
        return "yes" in response.lower()[:10]


class AnswerRelevancyMetric:
    """
    Answer Relevancy = how well the answer addresses the question.
    
    Algorithm:
    1. Generate N questions from the answer
    2. Embed all generated questions + original question
    3. relevancy = mean cosine_sim(generated_Qi, original_Q)
    
    Intuition: A relevant answer contains content that, if asked about,
    would generate questions similar to the original question.
    """
    
    def __init__(self, llm_func, embedder):
        self.llm = llm_func
        self.embedder = embedder
        self.n_questions = 3
    
    async def compute(self, sample: RAGASample) -> float:
        if not sample.answer.strip():
            return 0.0
        
        # Step 1: Generate candidate questions from the answer
        generated_questions = await self._generate_questions(sample.answer)
        
        if not generated_questions:
            return 0.5  # Can't evaluate
        
        # Step 2: Embed questions
        all_questions = [sample.question] + generated_questions
        try:
            embeddings = self.embedder.embed_documents(all_questions)
            
            original_emb = np.array(embeddings[0])
            generated_embs = [np.array(e) for e in embeddings[1:]]
            
            # Cosine similarity
            similarities = []
            for gen_emb in generated_embs:
                cos_sim = np.dot(original_emb, gen_emb) / (
                    np.linalg.norm(original_emb) * np.linalg.norm(gen_emb) + 1e-8
                )
                similarities.append(float(cos_sim))
            
            score = float(np.mean(similarities))
            logger.debug(f"Answer Relevancy: {score:.3f} (from {len(similarities)} question pairs)")
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Answer relevancy computation failed: {e}")
            return 0.5
    
    async def _generate_questions(self, answer: str) -> List[str]:
        """Generate questions that the answer would answer."""
        prompt = f"""Generate {self.n_questions} different questions that this text would be a good answer to.
Output ONLY the questions, one per line, no numbering.

Text: {answer[:800]}

Questions:"""
        
        response = await self.llm(prompt)
        questions = [
            q.strip().strip('?').strip() + '?'
            for q in response.strip().split('\n')
            if q.strip() and len(q.strip()) > 10
        ]
        return questions[:self.n_questions]


class ContextPrecisionMetric:
    """
    Context Precision = % of retrieved chunks that are actually relevant.
    
    High precision = we're not retrieving lots of irrelevant noise.
    
    Algorithm:
    1. For each retrieved chunk, ask LLM: "Is this relevant to answering Q?"
    2. precision = |relevant| / |total|
    
    Weighted variant (RAGAS default):
    precision@k = 1/|R| * Σ_k (P@k * rel(k))
    where rel(k) = 1 if chunk k is relevant, 0 otherwise
    """
    
    def __init__(self, llm_func):
        self.llm = llm_func
    
    async def compute(self, sample: RAGASample) -> float:
        if not sample.contexts:
            return 0.0
        
        relevance_flags = []
        for chunk in sample.contexts:
            is_relevant = await self._is_chunk_relevant(
                sample.question, chunk
            )
            relevance_flags.append(is_relevant)
        
        # Average precision (weighted by rank)
        precision = self._average_precision(relevance_flags)
        logger.debug(
            f"Context Precision: {sum(relevance_flags)}/{len(relevance_flags)} relevant "
            f"= AP@{len(relevance_flags)}: {precision:.3f}"
        )
        return precision
    
    def _average_precision(self, relevance_flags: List[bool]) -> float:
        """Compute average precision from relevance flags."""
        if not any(relevance_flags):
            return 0.0
        
        precision_at_k = []
        num_relevant = 0
        
        for k, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                num_relevant += 1
                precision_at_k.append(num_relevant / k)
        
        return float(np.mean(precision_at_k)) if precision_at_k else 0.0
    
    async def _is_chunk_relevant(self, question: str, chunk: str) -> bool:
        """Check if a chunk is relevant to the question."""
        prompt = f"""Is the following context useful for answering this question?
Output ONLY "YES" or "NO".

Question: {question}

Context: {chunk[:800]}

Useful (YES/NO):"""
        
        response = await self.llm(prompt)
        return "yes" in response.lower()[:10]


class ContextRecallMetric:
    """
    Context Recall = % of ground-truth answer covered by context.
    
    High recall = we retrieved everything needed to answer correctly.
    
    Algorithm:
    1. Break ground truth into sentences
    2. For each sentence, check if it can be attributed to any context chunk
    3. recall = |attributed_sentences| / |total_GT_sentences|
    """
    
    def __init__(self, llm_func):
        self.llm = llm_func
    
    async def compute(self, sample: RAGASample) -> float:
        if not sample.ground_truth or not sample.contexts:
            return 0.5  # Can't compute without reference
        
        # Split ground truth into sentences
        gt_sentences = re.split(r'(?<=[.!?])\s+', sample.ground_truth.strip())
        gt_sentences = [s.strip() for s in gt_sentences if len(s.strip()) > 15]
        
        if not gt_sentences:
            return 0.5
        
        context_combined = "\n".join(sample.contexts[:5])
        
        covered = 0
        for sentence in gt_sentences:
            is_covered = await self._is_sentence_in_context(sentence, context_combined)
            if is_covered:
                covered += 1
        
        score = covered / len(gt_sentences)
        logger.debug(f"Context Recall: {covered}/{len(gt_sentences)} = {score:.3f}")
        return score
    
    async def _is_sentence_in_context(self, sentence: str, context: str) -> bool:
        """Check if a sentence's information is present in context."""
        prompt = f"""Can the following statement be inferred from the context?
Output ONLY "YES" or "NO".

Context: {context[:1500]}

Statement: {sentence}

Inferable (YES/NO):"""
        
        response = await self.llm(prompt)
        return "yes" in response.lower()[:10]


# ─── RAGAS Evaluator ──────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Complete RAGAS evaluation suite.
    Orchestrates all four metrics across a dataset.
    """
    
    def __init__(
        self,
        llm_func,
        embedder,
        metrics: List[str] = None,
        use_ragas_library: bool = False,
    ):
        self.llm_func = llm_func
        self.embedder = embedder
        self.use_ragas_library = use_ragas_library
        
        self.metrics = metrics or [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        
        # Initialize metric computers
        self._faithfulness = FaithfulnessMetric(llm_func)
        self._relevancy = AnswerRelevancyMetric(llm_func, embedder)
        self._precision = ContextPrecisionMetric(llm_func)
        self._recall = ContextRecallMetric(llm_func)
    
    async def evaluate_sample(self, sample: RAGASample) -> Dict[str, float]:
        """Evaluate a single sample on all metrics."""
        scores = {}
        
        tasks = {}
        if "faithfulness" in self.metrics:
            tasks["faithfulness"] = self._faithfulness.compute(sample)
        if "answer_relevancy" in self.metrics:
            tasks["answer_relevancy"] = self._relevancy.compute(sample)
        if "context_precision" in self.metrics:
            tasks["context_precision"] = self._precision.compute(sample)
        if "context_recall" in self.metrics and sample.ground_truth:
            tasks["context_recall"] = self._recall.compute(sample)
        
        # Run all metrics concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for metric, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Metric {metric} failed: {result}")
                scores[metric] = 0.0
            else:
                scores[metric] = float(result)
        
        return scores
    
    async def evaluate_dataset(
        self,
        samples: List[RAGASample],
        output_path: Optional[str] = None,
    ) -> RAGASResult:
        """Evaluate a full dataset."""
        logger.info(f"Evaluating {len(samples)} samples with RAGAS...")
        
        per_sample_scores = []
        
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}: '{sample.question[:50]}...'")
            
            scores = await self.evaluate_sample(sample)
            scores["question"] = sample.question
            per_sample_scores.append(scores)
        
        # Aggregate scores
        metric_names = [m for m in self.metrics if m != "context_recall" or any(s.ground_truth for s in samples)]
        
        aggregated = {}
        for metric in self.metrics:
            values = [
                s[metric] for s in per_sample_scores 
                if metric in s
            ]
            aggregated[metric] = float(np.mean(values)) if values else 0.0
        
        result = RAGASResult(
            faithfulness=aggregated.get("faithfulness", 0.0),
            answer_relevancy=aggregated.get("answer_relevancy", 0.0),
            context_precision=aggregated.get("context_precision", 0.0),
            context_recall=aggregated.get("context_recall", 0.0),
            per_sample_scores=per_sample_scores,
            num_samples=len(samples),
        )
        
        print(result)
        
        # Save results
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    "summary": result.to_dict(),
                    "per_sample": per_sample_scores,
                }, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        return result
    
    async def evaluate_using_ragas_library(
        self, samples: List[RAGASample]
    ) -> Dict:
        """Use the official RAGAS library for evaluation."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            
            data = {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.contexts for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
            
            dataset = Dataset.from_dict(data)
            
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            
            return result
            
        except ImportError:
            logger.warning("RAGAS library not available. Using custom implementation.")
            result = await self.evaluate_dataset(samples)
            return result.to_dict()


# ─── Test Dataset Generator ───────────────────────────────────────────────────

async def generate_test_dataset(
    documents: List[str],
    llm_func,
    n_questions: int = 10,
    output_path: str = "./data/evaluation/test_qa.json",
) -> List[Dict]:
    """
    Synthetically generate QA pairs from documents for evaluation.
    
    This is the RAGAS "testset generation" approach:
    1. For each document chunk, generate a question that it answers
    2. Generate the reference answer
    3. Create evaluation samples
    """
    
    qa_pairs = []
    chunks_per_question = max(1, len(documents) // n_questions)
    
    for i in range(0, min(len(documents), n_questions * chunks_per_question), chunks_per_question):
        chunk = documents[i]
        
        # Generate question
        q_prompt = f"""Generate one specific question that can be answered using this text.
Output ONLY the question.

Text: {chunk[:600]}

Question:"""
        
        try:
            question = await llm_func(q_prompt)
            question = question.strip()
            
            # Generate reference answer
            a_prompt = f"""Answer this question based only on the provided text.

Text: {chunk[:600]}
Question: {question}

Answer:"""
            
            answer = await llm_func(a_prompt)
            
            qa_pairs.append({
                "question": question,
                "ground_truth": answer.strip(),
                "source_chunk": chunk[:200],
                "chunk_id": i,
            })
            
            logger.debug(f"Generated QA {len(qa_pairs)}/{n_questions}: {question[:60]}...")
            
        except Exception as e:
            logger.error(f"QA generation failed for chunk {i}: {e}")
        
        if len(qa_pairs) >= n_questions:
            break
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    logger.success(f"Generated {len(qa_pairs)} QA pairs → {output_path}")
    return qa_pairs


if __name__ == "__main__":
    # Quick demo with mock LLM
    async def mock_llm(prompt: str) -> str:
        return "YES"
    
    from embeddings import DenseEmbedder
    embedder = DenseEmbedder()
    
    evaluator = RAGASEvaluator(mock_llm, embedder)
    
    samples = [
        RAGASample(
            question="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation. It combines retrieval of relevant documents with text generation.",
            contexts=[
                "RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs by retrieving relevant documents before generation.",
                "The retrieval component uses vector similarity search to find relevant passages.",
            ],
            ground_truth="RAG is a method that retrieves relevant documents to augment language model generation."
        )
    ]
    
    result = asyncio.run(evaluator.evaluate_dataset(samples))
    print(result)
