"""
multimodal.py — Multimodal RAG (Image + Text)
==============================================
DLM Focus: Extends RAG beyond text by adding:
  1. CLIP image embeddings — unified image/text embedding space
  2. Image captioning via BLIP-2 — converts images to searchable text
  3. LLaVA (Large Language and Vision Assistant) for image QA
  4. Mixed document retrieval: text chunks + image chunks in same index

Architecture: CLIP is trained with contrastive loss on 400M image-text pairs.
The resulting embedding space allows querying images with text and vice versa.

cos_sim(embed_text("a photo of a dog"), embed_image(dog.jpg)) ≈ 0.85
cos_sim(embed_text("a photo of a cat"), embed_image(dog.jpg)) ≈ 0.25
"""

from __future__ import annotations

import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class ImageChunk:
    """An image document chunk with embedding."""
    image_path: str
    caption: str
    embedding: List[float]
    metadata: Dict[str, Any]
    chunk_id: str


# ─── CLIP Embedder ────────────────────────────────────────────────────────────

class CLIPEmbedder:
    """
    CLIP (Contrastive Language-Image Pretraining) embeddings.

    Training objective: maximize cosine similarity between matched
    (image, caption) pairs and minimize it for unmatched pairs.

    Trained on 400M (image, text) pairs scraped from the internet.
    Result: images and text live in the SAME embedding space.

    This enables:
    - Text query → find relevant images
    - Image query → find relevant text
    - Cross-modal similarity ranking
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            logger.info(f"Loading CLIP: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
            logger.success("CLIP loaded")
        except Exception as e:
            logger.warning(f"CLIP not available: {e}")
            self._model = "mock"

    def embed_image(self, image_path: str) -> List[float]:
        """Embed an image file into CLIP space."""
        self._load()
        if self._model == "mock":
            return list(np.random.randn(768).astype(float))

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")

            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            return features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return list(np.zeros(768))

    def embed_image_from_bytes(self, image_bytes: bytes) -> List[float]:
        """Embed image from raw bytes."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(image_bytes)
            tmp = f.name
        emb = self.embed_image(tmp)
        os.unlink(tmp)
        return emb

    def embed_text(self, text: str) -> List[float]:
        """Embed text into CLIP space (same space as images!)."""
        self._load()
        if self._model == "mock":
            return list(np.random.randn(768).astype(float))

        try:
            import torch
            inputs = self._processor(
                text=[text], return_tensors="pt",
                padding=True, truncation=True, max_length=77
            )
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return list(np.zeros(768))

    def cross_modal_similarity(
        self, text: str, image_path: str
    ) -> float:
        """Compute cosine similarity between text and image."""
        t_emb = np.array(self.embed_text(text))
        i_emb = np.array(self.embed_image(image_path))
        return float(np.dot(t_emb, i_emb))


# ─── Image Captioner ─────────────────────────────────────────────────────────

class ImageCaptioner:
    """
    Generate natural language captions for images using BLIP-2.

    BLIP-2 Architecture:
    - Frozen image encoder (ViT-G/14)
    - Querying Transformer (Q-Former) — bridges vision and language
    - Frozen LLM (OPT-2.7B or FlanT5)

    Captions make images searchable via standard text retrieval.
    This is an alternative/complement to CLIP embeddings.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch

            logger.info(f"Loading BLIP-2: {self.model_name}")
            self._processor = Blip2Processor.from_pretrained(self.model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.success("BLIP-2 captioner loaded")
        except Exception as e:
            logger.warning(f"BLIP-2 not available: {e}")
            self._model = "mock"

    def caption(self, image_path: str, prompt: str = None) -> str:
        """Generate a caption for an image."""
        self._load()

        if self._model == "mock":
            return f"[Mock caption for {Path(image_path).name}]"

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            if prompt:
                inputs = self._processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                ).to("cuda" if torch.cuda.is_available() else "cpu", torch.float16)
            else:
                inputs = self._processor(
                    images=image,
                    return_tensors="pt",
                ).to("cuda" if torch.cuda.is_available() else "cpu", torch.float16)

            with torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=200)

            caption = self._processor.decode(generated[0], skip_special_tokens=True)
            return caption.strip()

        except Exception as e:
            logger.error(f"Captioning failed: {e}")
            return f"Image: {Path(image_path).name}"

    def detailed_description(self, image_path: str) -> str:
        """Generate detailed description for RAG indexing."""
        prompt = "Describe this image in detail, including all visible text, objects, charts, diagrams, and any technical content:"
        return self.caption(image_path, prompt=prompt)


# ─── Vision LLM (LLaVA) ──────────────────────────────────────────────────────

class VisionLLM:
    """
    LLaVA (Large Language and Vision Assistant) for multimodal QA.

    Architecture:
    - CLIP ViT-L/14 visual encoder (frozen)
    - Linear projection layer (trainable)
    - LLaMA-2 / Vicuna LLM decoder

    Enables: asking questions directly about image content.
    Used in the final answer generation step when relevant images
    are retrieved alongside text.
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            import torch

            logger.info(f"Loading LLaVA: {self.model_name}")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
            )
            logger.success("LLaVA loaded")
        except Exception as e:
            logger.warning(f"LLaVA not available: {e}")
            self._model = "mock"

    async def answer_with_image(
        self, question: str, image_path: str, context: str = ""
    ) -> str:
        """Answer a question given an image (and optional text context)."""
        self._load()

        if self._model == "mock":
            return f"[LLaVA mock answer] Based on the image at {Path(image_path).name}: The image appears to contain relevant information about '{question}'. In production, LLaVA would analyze the actual image content."

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            prompt = f"""<image>
{f'Context: {context}' if context else ''}

Question: {question}

Answer based on the image:"""

            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            text = self._processor.decode(output[0], skip_special_tokens=True)
            # Extract just the assistant response
            if "Answer based on the image:" in text:
                text = text.split("Answer based on the image:")[-1].strip()

            return text

        except Exception as e:
            logger.error(f"LLaVA inference failed: {e}")
            return f"Image analysis failed: {e}"


# ─── Multimodal Document Processor ───────────────────────────────────────────

class MultimodalProcessor:
    """
    Processes mixed (text + image) documents for indexing.

    Workflow for a PDF with embedded images:
    1. Extract text → text chunks (via DocumentParser)
    2. Extract images → save to disk
    3. For each image:
       a. Generate caption (BLIP-2)
       b. Create CLIP embedding
       c. Store in separate multimodal collection
    4. At query time, search BOTH text and image collections
    5. Use LLaVA for image-based answer generation
    """

    def __init__(
        self,
        clip_embedder: CLIPEmbedder,
        captioner: ImageCaptioner,
        vision_llm: Optional[VisionLLM] = None,
        image_output_dir: str = "./data/extracted_images",
    ):
        self.clip = clip_embedder
        self.captioner = captioner
        self.vision_llm = vision_llm
        self.image_dir = Path(image_output_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self._image_index: List[ImageChunk] = []

    def extract_images_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract all images from a PDF file."""
        image_paths = []

        try:
            import pypdf
            from PIL import Image
            import io

            reader = pypdf.PdfReader(pdf_path)
            pdf_name = Path(pdf_path).stem

            for page_num, page in enumerate(reader.pages):
                for img_idx, image_obj in enumerate(page.images):
                    try:
                        img_bytes = image_obj.data
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                        # Skip tiny images (likely icons/bullets)
                        if img.size[0] < 100 or img.size[1] < 100:
                            continue

                        save_path = self.image_dir / f"{pdf_name}_p{page_num+1}_img{img_idx+1}.jpg"
                        img.save(save_path, "JPEG", quality=85)
                        image_paths.append(str(save_path))

                    except Exception as e:
                        logger.debug(f"Could not extract image: {e}")

            logger.info(f"Extracted {len(image_paths)} images from {pdf_path}")

        except Exception as e:
            logger.error(f"Image extraction failed: {e}")

        return image_paths

    def index_image(self, image_path: str, metadata: Dict = None) -> ImageChunk:
        """Generate caption + embedding for one image and add to index."""
        import hashlib

        caption = self.captioner.detailed_description(image_path)
        embedding = self.clip.embed_image(image_path)

        chunk_id = hashlib.md5(image_path.encode()).hexdigest()[:16]

        chunk = ImageChunk(
            image_path=image_path,
            caption=caption,
            embedding=embedding,
            metadata={
                "source": Path(image_path).name,
                "type": "image",
                **(metadata or {}),
            },
            chunk_id=chunk_id,
        )

        self._image_index.append(chunk)
        logger.debug(f"Indexed image: {image_path} | Caption: {caption[:60]}...")
        return chunk

    def search_images(self, query: str, top_k: int = 3) -> List[ImageChunk]:
        """Search image index using text query via CLIP."""
        if not self._image_index:
            return []

        query_emb = np.array(self.clip.embed_text(query))

        scores = []
        for chunk in self._image_index:
            img_emb = np.array(chunk.embedding)
            score = float(np.dot(query_emb, img_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(img_emb) + 1e-8
            ))
            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scores[:top_k]]

    async def multimodal_answer(
        self,
        question: str,
        text_context: str,
        relevant_images: List[ImageChunk],
    ) -> str:
        """
        Generate answer using both text context and relevant images.
        If LLaVA is available, analyze images directly.
        Otherwise, use captions.
        """
        if not relevant_images:
            return None  # Fall back to text-only generation

        # Build image context
        image_context_parts = []
        for i, img_chunk in enumerate(relevant_images[:2]):  # Max 2 images
            if self.vision_llm and self.vision_llm._model != "mock":
                # Direct image analysis with LLaVA
                img_answer = await self.vision_llm.answer_with_image(
                    question, img_chunk.image_path, context=text_context[:500]
                )
                image_context_parts.append(
                    f"[Image {i+1} Analysis]: {img_answer}"
                )
            else:
                # Use caption as proxy
                image_context_parts.append(
                    f"[Image {i+1} Caption]: {img_chunk.caption}"
                )

        image_context = "\n\n".join(image_context_parts)

        return image_context


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_multimodal_processor(config: dict) -> MultimodalProcessor:
    """Create multimodal processor from config."""
    mm_config = config.get("multimodal", {})

    if not mm_config.get("enabled", False):
        return None

    clip = CLIPEmbedder(
        model_name=mm_config.get("image_embedding", {}).get(
            "model", "openai/clip-vit-large-patch14"
        )
    )

    captioner = ImageCaptioner(
        model_name=mm_config.get("image_captioning", {}).get(
            "model", "Salesforce/blip2-opt-2.7b"
        )
    )

    vision_llm = VisionLLM(
        model_name=mm_config.get("vision_llm", {}).get(
            "model", "llava-hf/llava-1.5-7b-hf"
        )
    )

    return MultimodalProcessor(
        clip_embedder=clip,
        captioner=captioner,
        vision_llm=vision_llm,
    )


if __name__ == "__main__":
    # Demo
    clip = CLIPEmbedder()
    captioner = ImageCaptioner()

    # Test text embedding
    text_emb = clip.embed_text("a neural network architecture diagram")
    print(f"Text embedding dim: {len(text_emb)}")

    # Test cross-modal similarity (with mock)
    print("\nCLIP Cross-Modal Similarity Demo:")
    queries = [
        "a diagram of a transformer architecture",
        "a photo of a dog",
        "bar chart showing model performance",
    ]
    for q in queries:
        emb = clip.embed_text(q)
        print(f"  '{q[:40]}' → dim={len(emb)}")
