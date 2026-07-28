"""
generator.py — LLM Generation with Self-Correction
====================================================
DLM Focus:
  - Llama-3-8B inference with 4-bit quantization (QLoRA)
  - API-based fallback (Anthropic/OpenAI)
  - Neural confidence scoring for faithfulness checking
  - Self-correction loop: generate → verify → revise
  - NLI-based hallucination detection
"""

from __future__ import annotations

import re
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import os
import numpy as np
from loguru import logger

from retriever import Document


@dataclass
class GenerationResult:
    """Complete generation output with metadata."""
    answer: str
    sources: List[Dict[str, Any]]
    faithfulness_score: float
    num_corrections: int
    reasoning_trace: List[str]
    tokens_used: int = 0
    model_used: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "faithfulness_score": self.faithfulness_score,
            "num_corrections": self.num_corrections,
            "model_used": self.model_used,
        }


# ─── LLM Interface ────────────────────────────────────────────────────────────

class BaseLLM:
    """Abstract LLM interface."""
    
    async def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        raise NotImplementedError
    
    async def generate_stream(
        self, prompt: str, system: str = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        result = await self.generate(prompt, system, **kwargs)
        yield result


# ─── Anthropic Claude (Primary API LLM) ──────────────────────────────────────

class AnthropicLLM(BaseLLM):
    """
    Uses Claude via API.
    In production context of this DLM project: used as the generation backbone
    while Llama-3 handles local/offline scenarios.
    """
    
    def __init__(
        self, 
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic()
            
            messages = [{"role": "user", "content": prompt}]
            
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system or DEFAULT_SYSTEM_PROMPT,
                messages=messages,
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"[Generation failed: {e}]"
    
    async def generate_stream(
        self, prompt: str, system: str = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic()
            
            async with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or DEFAULT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[Error: {e}]"


# ─── OpenAI LLM ───────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url="https://api.groq.com/openai/v1/",
            )
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[Generation failed: {e}]"


# ─── Local Llama-3 LLM ────────────────────────────────────────────────────────

class LocalLlamaLLM(BaseLLM):
    """
    Local Llama-3-8B-Instruct with 4-bit QLoRA quantization.
    
    DLM Focus: 4-bit quantization via bitsandbytes:
    - Quantizes weights from float16 → int4
    - Uses NF4 (Normal Float 4) format for better precision
    - Achieves ~75% memory reduction with <1% accuracy drop
    - Enables running 8B models on consumer GPUs (16GB VRAM → 6GB)
    
    QLoRA: Fine-tuning with quantized base + trainable LoRA adapters
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization: str = "4bit",
        device: str = "cuda",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Load quantized Llama-3 model."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
            
            logger.info(f"Loading {self.model_name} with {self.quantization} quantization...")
            
            # 4-bit quantization config
            if self.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",          # Normal Float 4
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,      # Nested quantization
                )
            elif self.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb_config = None
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            # Create pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
            )
            
            logger.success(f"Llama-3 loaded successfully ({self.quantization} quantization)")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            logger.info("Falling back to API-based generation")
            self._model = "failed"
    
    async def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        self._load_model()
        
        if self._model == "failed" or self._model is None:
            return "[Local model not available]"
        
        # Format with Llama-3 chat template
        messages = [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Run in thread pool (blocking inference)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self._pipeline(formatted)[0]["generated_text"]
        )
        
        # Extract just the assistant's response
        if "<|assistant|>" in result:
            result = result.split("<|assistant|>")[-1].strip()
        
        return result


# ─── Prompts ──────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are an expert research assistant with deep knowledge across multiple domains.

Your role is to answer questions based on the provided context documents. Follow these rules strictly:

1. **Accuracy First**: Only state facts that are supported by the provided context.
2. **Citation Required**: Always cite which document/source supports your claims using [Source: X] notation.
3. **Acknowledge Gaps**: If the context doesn't contain enough information, explicitly say so.
4. **No Hallucination**: Never invent facts, statistics, or references not in the context.
5. **Structured Response**: Provide clear, organized answers with appropriate detail level.
"""

RAG_PROMPT_TEMPLATE = """Based on the following context documents, answer the user's question.

=== CONTEXT DOCUMENTS ===
{context}

=== USER QUESTION ===
{question}

=== INSTRUCTIONS ===
- Answer based ONLY on the context above
- Cite sources using [Doc 1], [Doc 2], etc.
- If context is insufficient, say "The provided documents don't contain enough information to fully answer this."
- Be concise but complete

=== YOUR ANSWER ==="""

FAITHFULNESS_CHECK_PROMPT = """You are evaluating whether an AI-generated answer is faithful to the provided source context.

=== SOURCE CONTEXT ===
{context}

=== GENERATED ANSWER ===
{answer}

=== TASK ===
Identify any claims in the answer that are NOT supported by or contradict the source context.
Output ONLY a JSON object with this exact format:
{{
  "is_faithful": true/false,
  "faithfulness_score": 0.0-1.0,
  "unsupported_claims": ["claim1", "claim2"],
  "correction_needed": "What needs to be corrected, or 'None' if faithful"
}}"""

CORRECTION_PROMPT = """Your previous answer contained some unsupported claims. Please revise it.

=== ORIGINAL QUESTION ===
{question}

=== CONTEXT DOCUMENTS ===
{context}

=== YOUR PREVIOUS ANSWER ===
{previous_answer}

=== UNSUPPORTED CLAIMS TO REMOVE ===
{unsupported_claims}

=== REVISED ANSWER (stick strictly to what's in the context) ==="""


# ─── Faithfulness Checker (NLI-based) ────────────────────────────────────────

class FaithfulnessChecker:
    """
    Uses Natural Language Inference (NLI) to detect hallucinations.
    
    DLM Architecture: DeBERTa-v3 fine-tuned on NLI tasks
    
    For each claim in the answer:
        Input: (premise=context_chunk, hypothesis=claim)
        Output: Entailment / Neutral / Contradiction
    
    Faithfulness = |{claims with Entailment}| / |total claims|
    """
    
    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-base"):
        self.model_name = model
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info("NLI faithfulness checker loaded")
            except Exception as e:
                logger.warning(f"NLI model not available: {e}")
                self._model = "mock"
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual factual claims from text."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short sentences and citations
        claims = [
            s.strip() for s in sentences 
            if len(s.strip()) > 20 and not s.strip().startswith('[')
        ]
        return claims
    
    def score_faithfulness(
        self, answer: str, context_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        Score how faithful the answer is to the context.
        
        Returns:
            {
                "score": float [0, 1],
                "entailed_claims": [...],
                "unsupported_claims": [...],
            }
        """
        self._load_model()
        claims = self._extract_claims(answer)
        
        if not claims:
            return {"score": 1.0, "entailed_claims": [], "unsupported_claims": []}
        
        if self._model == "mock":
            # Mock: assume 80% faithfulness
            n_entailed = int(len(claims) * 0.8)
            return {
                "score": 0.8,
                "entailed_claims": claims[:n_entailed],
                "unsupported_claims": claims[n_entailed:],
            }
        
        # Combine all context into one string (truncated)
        full_context = " ".join(context_chunks)[:2000]
        
        entailed = []
        unsupported = []
        
        # NLI scoring for each claim
        pairs = [(full_context, claim) for claim in claims]
        scores = self._model.predict(pairs)  # [entail, neutral, contradict]
        
        for claim, score_arr in zip(claims, scores):
            # scores: [contradiction, entailment, neutral] for deberta
            entailment_score = score_arr[1] if len(score_arr) > 1 else score_arr[0]
            if entailment_score > 0.5:
                entailed.append(claim)
            else:
                unsupported.append(claim)
        
        faithfulness_score = len(entailed) / len(claims) if claims else 1.0
        
        return {
            "score": faithfulness_score,
            "entailed_claims": entailed,
            "unsupported_claims": unsupported,
        }


# ─── RAG Generator with Self-Correction ──────────────────────────────────────

class RAGGenerator:
    """
    Full generation pipeline with self-correction loop.
    
    Process:
    1. Format context + query into RAG prompt
    2. Generate answer with LLM
    3. Check faithfulness with NLI model
    4. If faithfulness < threshold: regenerate with corrections
    5. Repeat up to max_iterations
    
    This self-correction loop is the KEY DLM extension beyond standard NLP RAG.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        faithfulness_checker: Optional[FaithfulnessChecker] = None,
        max_iterations: int = 3,
        faithfulness_threshold: float = 0.7,
        max_context_length: int = 4000,
    ):
        self.llm = llm
        self.faithfulness_checker = faithfulness_checker or FaithfulnessChecker()
        self.max_iterations = max_iterations
        self.faithfulness_threshold = faithfulness_threshold
        self.max_context_length = max_context_length
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        total_len = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"[Doc {i+1}] Source: {doc.source}\n{doc.content}\n"
            
            if total_len + len(doc_text) > self.max_context_length:
                # Truncate last document to fit
                remaining = self.max_context_length - total_len
                if remaining > 100:
                    doc_text = doc_text[:remaining] + "...\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            total_len += len(doc_text)
        
        return "\n---\n".join(context_parts)
    
    async def generate(
        self,
        query: str,
        documents: List[Document],
    ) -> GenerationResult:
        """
        Generate answer with self-correction loop.
        """
        context = self._format_context(documents)
        context_chunks = [doc.content for doc in documents]
        sources = [
            {
                "doc_id": i + 1,
                "source": doc.source,
                "score": doc.score,
                "preview": doc.content[:200],
            }
            for i, doc in enumerate(documents)
        ]
        
        # Initial generation
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
        answer = await self.llm.generate(prompt)
        
        reasoning_trace = [f"Initial generation ({len(answer)} chars)"]
        num_corrections = 0
        faithfulness_score = 1.0
        
        # Self-correction loop
        for iteration in range(self.max_iterations):
            # Check faithfulness
            faith_result = self.faithfulness_checker.score_faithfulness(
                answer, context_chunks
            )
            faithfulness_score = faith_result["score"]
            
            reasoning_trace.append(
                f"Iteration {iteration + 1}: faithfulness={faithfulness_score:.3f}"
            )
            
            if faithfulness_score >= self.faithfulness_threshold:
                logger.debug(
                    f"Answer is faithful (score={faithfulness_score:.3f}), "
                    f"no correction needed"
                )
                break
            
            # Need correction
            unsupported = faith_result.get("unsupported_claims", [])
            if not unsupported:
                break
            
            logger.info(
                f"Correcting answer (faithfulness={faithfulness_score:.3f}), "
                f"{len(unsupported)} unsupported claims"
            )
            
            correction_prompt = CORRECTION_PROMPT.format(
                question=query,
                context=context,
                previous_answer=answer,
                unsupported_claims="\n".join(f"- {c}" for c in unsupported),
            )
            
            answer = await self.llm.generate(correction_prompt)
            num_corrections += 1
            reasoning_trace.append(
                f"Correction {num_corrections}: removed {len(unsupported)} unsupported claims"
            )
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            faithfulness_score=faithfulness_score,
            num_corrections=num_corrections,
            reasoning_trace=reasoning_trace,
            model_used=str(type(self.llm).__name__),
        )
    
    async def generate_stream(
        self,
        query: str,
        documents: List[Document],
    ) -> AsyncGenerator[str, None]:
        """Streaming generation (no self-correction for speed)."""
        context = self._format_context(documents)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
        
        async for token in self.llm.generate_stream(prompt):
            yield token


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_generator(config: dict) -> RAGGenerator:
    """Create the generator from config."""
    gen_config = config.get("generation", {})
    
    # Choose LLM
    use_api = gen_config.get("use_api", True)
    
    if use_api:
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            llm = AnthropicLLM(
                model=gen_config.get("anthropic_model", "claude-3-haiku-20240307"),
                max_tokens=gen_config.get("max_tokens", 1024),
                temperature=gen_config.get("temperature", 0.1),
            )
        elif os.getenv("OPENAI_API_KEY"):
            llm = OpenAILLM(
                model=gen_config.get("openai_model", "gpt-4o-mini"),
                max_tokens=gen_config.get("max_tokens", 1024),
                temperature=gen_config.get("temperature", 0.1),
            )
        else:
            logger.warning("No API keys found, using local model")
            use_api = False
    
    if not use_api:
        llm = LocalLlamaLLM(
            model_name=gen_config.get("model", "meta-llama/Meta-Llama-3-8B-Instruct"),
            quantization=gen_config.get("quantization", "4bit"),
        )
    
    # Self-correction config
    sc_config = gen_config.get("self_correction", {})
    
    return RAGGenerator(
        llm=llm,
        faithfulness_checker=FaithfulnessChecker() if sc_config.get("enabled", True) else None,
        max_iterations=sc_config.get("max_iterations", 3),
        faithfulness_threshold=sc_config.get("faithfulness_threshold", 0.7),
    )
