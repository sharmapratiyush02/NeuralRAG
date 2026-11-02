"""
ingest.py — Document Ingestion Pipeline
=========================================
DLM Focus:
  - Semantic chunking (uses sentence embeddings to find optimal boundaries)
  - Multi-format document parsing (PDF, DOCX, TXT, images with OCR)
  - Metadata extraction and enrichment
  - Async parallel ingestion for large document sets
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@dataclass
class ParsedDocument:
    """A document parsed from a file."""
    content: str
    metadata: Dict[str, Any]
    source_path: str
    doc_type: str
    page_count: int = 0
    images: List[str] = field(default_factory=list)  # Paths to extracted images


@dataclass
class DocumentChunk:
    """A chunk of a document ready for embedding."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    chunk_index: int
    
    def to_chroma_format(self) -> Tuple[str, str, Dict]:
        return self.chunk_id, self.content, self.metadata


# ─── Document Parsers ─────────────────────────────────────────────────────────

class DocumentParser:
    """Multi-format document parser."""
    
    @staticmethod
    def parse_pdf(file_path: str) -> ParsedDocument:
        """Parse PDF with text extraction + OCR fallback."""
        try:
            import pypdf
            
            reader = pypdf.PdfReader(file_path)
            pages = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    pages.append(text)
                else:
                    # OCR fallback for scanned pages
                    logger.debug(f"Page {i+1} has minimal text, attempting OCR...")
                    ocr_text = DocumentParser._ocr_page(file_path, i)
                    if ocr_text:
                        pages.append(ocr_text)
            
            content = "\n\n".join(pages)
            
            return ParsedDocument(
                content=content,
                metadata={
                    "source": Path(file_path).name,
                    "file_path": file_path,
                    "page_count": len(reader.pages),
                    "file_type": "pdf",
                    "file_size": os.path.getsize(file_path),
                },
                source_path=file_path,
                doc_type="pdf",
                page_count=len(reader.pages),
            )
        except ImportError:
            logger.error("pypdf not installed: pip install pypdf")
            return DocumentParser._parse_text_fallback(file_path)
    
    @staticmethod
    def _ocr_page(pdf_path: str, page_num: int) -> str:
        """OCR a specific PDF page."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
            
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            if images:
                text = pytesseract.image_to_string(images[0])
                return text
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
        return ""
    
    @staticmethod
    def parse_docx(file_path: str) -> ParsedDocument:
        """Parse DOCX file."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Extract tables
            tables = []
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append(" | ".join(cells))
                if rows:
                    tables.append("\n".join(rows))
            
            content = "\n\n".join(paragraphs)
            if tables:
                content += "\n\n[TABLES]\n" + "\n\n".join(tables)
            
            return ParsedDocument(
                content=content,
                metadata={
                    "source": Path(file_path).name,
                    "file_path": file_path,
                    "file_type": "docx",
                    "paragraph_count": len(paragraphs),
                    "table_count": len(tables),
                },
                source_path=file_path,
                doc_type="docx",
            )
        except ImportError:
            logger.error("python-docx not installed: pip install python-docx")
            return DocumentParser._parse_text_fallback(file_path)
    
    @staticmethod
    def parse_text(file_path: str) -> ParsedDocument:
        """Parse plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return ParsedDocument(
            content=content,
            metadata={
                "source": Path(file_path).name,
                "file_path": file_path,
                "file_type": "txt",
                "char_count": len(content),
            },
            source_path=file_path,
            doc_type="txt",
        )
    
    @staticmethod
    def parse_image(file_path: str) -> ParsedDocument:
        """Parse image with OCR."""
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            return ParsedDocument(
                content=text,
                metadata={
                    "source": Path(file_path).name,
                    "file_path": file_path,
                    "file_type": "image",
                    "image_size": image.size,
                },
                source_path=file_path,
                doc_type="image",
            )
        except Exception as e:
            logger.error(f"Image parsing failed: {e}")
            return ParsedDocument(
                content="",
                metadata={"source": Path(file_path).name},
                source_path=file_path,
                doc_type="image",
            )
    
    @staticmethod
    def _parse_text_fallback(file_path: str) -> ParsedDocument:
        """Read file as plain text."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            content = ""
        
        return ParsedDocument(
            content=content,
            metadata={"source": Path(file_path).name, "file_path": file_path},
            source_path=file_path,
            doc_type="text",
        )
    
    @classmethod
    def parse(cls, file_path: str) -> ParsedDocument:
        """Auto-detect format and parse."""
        suffix = Path(file_path).suffix.lower()
        
        parsers = {
            ".pdf": cls.parse_pdf,
            ".docx": cls.parse_docx,
            ".doc": cls.parse_docx,
            ".txt": cls.parse_text,
            ".md": cls.parse_text,
            ".rst": cls.parse_text,
            ".png": cls.parse_image,
            ".jpg": cls.parse_image,
            ".jpeg": cls.parse_image,
        }
        
        parser = parsers.get(suffix, cls.parse_text)
        logger.info(f"Parsing {Path(file_path).name} as {suffix}")
        return parser(file_path)


# ─── Chunking Strategies ──────────────────────────────────────────────────────

class ChunkingStrategy:
    """Base chunking strategy."""
    
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """Simple fixed-size character chunking with overlap."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        text = document.content
        chunks = []
        
        i = 0
        chunk_idx = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end].strip()
            
            if chunk_text:
                chunk_id = hashlib.md5(
                    f"{document.source_path}_{chunk_idx}".encode()
                ).hexdigest()[:16]
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={**document.metadata, "chunk_index": chunk_idx},
                    chunk_id=chunk_id,
                    doc_id=hashlib.md5(document.source_path.encode()).hexdigest()[:16],
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
            
            i += self.chunk_size - self.chunk_overlap
        
        return chunks


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character text splitter.
    Tries to split on paragraph boundaries, then sentences, then words.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        current = ""
        
        for split in splits:
            if len(current) + len(split) + len(separator) <= self.chunk_size:
                current += (separator if current else "") + split
            else:
                if current:
                    good_splits.append(current)
                if len(split) > self.chunk_size:
                    # Recursively split this piece
                    sub_splits = self._split_text(split, separators[1:])
                    good_splits.extend(sub_splits)
                    current = ""
                else:
                    current = split
        
        if current:
            good_splits.append(current)
        
        return good_splits
    
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        text_chunks = self._split_text(document.content, self.separators)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 20:  # Skip very short chunks
                continue
            
            chunk_id = hashlib.md5(
                f"{document.source_path}_{i}".encode()
            ).hexdigest()[:16]
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata={**document.metadata, "chunk_index": i},
                chunk_id=chunk_id,
                doc_id=hashlib.md5(document.source_path.encode()).hexdigest()[:16],
                chunk_index=i,
            ))
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking using embedding similarity.
    
    DLM Core Algorithm:
    1. Split text into sentences
    2. Embed each sentence using a neural model
    3. Compute cosine similarity between consecutive sentences
    4. Break at points of LOW similarity (semantic shifts)
    5. Merge nearby sentences into chunks respecting size limits
    
    This produces chunks that are semantically coherent rather than
    arbitrarily cut at fixed character positions.
    
    Reference: "Propositionizer" approach (Chen et al., 2023)
    """
    
    def __init__(
        self,
        embedder,
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,
        similarity_threshold: float = 0.8,
    ):
        self.embedder = embedder
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download('punkt', quiet=True)
                sentences = nltk.sent_tokenize(text)
            return sentences
        except ImportError:
            # Fallback: split on period
            return re.split(r'(?<=[.!?])\s+', text)
    
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        import re
        
        sentences = self._split_sentences(document.content)
        if len(sentences) < 3:
            # Fallback to fixed chunker for very short docs
            return FixedSizeChunker(self.max_chunk_size).chunk(document)
        
        # Embed all sentences
        logger.debug(f"Embedding {len(sentences)} sentences for semantic chunking...")
        
        try:
            embeddings = self.embedder.embed_documents(sentences)
            embeddings = [np.array(e) for e in embeddings]
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to recursive chunker: {e}")
            return RecursiveChunker(self.max_chunk_size).chunk(document)
        
        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(embeddings)
        
        # Build chunks from sentences
        chunks = []
        chunk_sentences = []
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            chunk_sentences.append(sentence)
            current_text = " ".join(chunk_sentences)
            
            is_break = i in breakpoints
            is_too_long = len(current_text) > self.max_chunk_size
            is_last = i == len(sentences) - 1
            
            if (is_break and len(current_text) >= self.min_chunk_size) or is_too_long or is_last:
                # Emit chunk
                chunk_text = current_text.strip()
                if chunk_text:
                    chunk_id = hashlib.md5(
                        f"{document.source_path}_{len(chunks)}".encode()
                    ).hexdigest()[:16]
                    
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_index": len(chunks),
                            "sentence_count": len(chunk_sentences),
                        },
                        chunk_id=chunk_id,
                        doc_id=hashlib.md5(document.source_path.encode()).hexdigest()[:16],
                        chunk_index=len(chunks),
                    ))
                
                chunk_sentences = []
        
        logger.debug(
            f"Semantic chunker: {len(sentences)} sentences → "
            f"{len(chunks)} chunks (avg {len(document.content)//max(len(chunks),1)} chars)"
        )
        return chunks
    
    def _find_breakpoints(self, embeddings: List[np.ndarray]) -> set:
        """
        Find indices where semantic content shifts significantly.
        
        Uses a sliding window to compute local similarity and finds
        points where similarity drops below threshold.
        """
        breakpoints = set()
        
        for i in range(1, len(embeddings) - 1):
            # Cosine similarity with next sentence
            e1, e2 = embeddings[i], embeddings[i + 1]
            
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = float(np.dot(e1, e2) / (norm1 * norm2))
                
                if similarity < self.similarity_threshold:
                    breakpoints.add(i + 1)
        
        return breakpoints


# ─── Ingestion Pipeline ───────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Complete document ingestion pipeline.
    
    Flow:
    1. Discover documents in input directory
    2. Parse each document (PDF → text, images → OCR)
    3. Chunk documents (semantic/recursive/fixed)
    4. Generate embeddings
    5. Store in vector store + BM25 index
    """
    
    def __init__(
        self,
        vector_store,
        embedder,
        chunking_strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._all_chunk_texts: List[str] = []  # For BM25
    
    def _get_chunker(self) -> ChunkingStrategy:
        if self.chunking_strategy == "semantic":
            return SemanticChunker(
                embedder=self.embedder,
                min_chunk_size=200,
                max_chunk_size=self.chunk_size,
            )
        elif self.chunking_strategy == "recursive":
            return RecursiveChunker(self.chunk_size, self.chunk_overlap)
        else:
            return FixedSizeChunker(self.chunk_size, self.chunk_overlap)
    
    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file. Returns number of chunks added."""
        try:
            # Parse
            doc = DocumentParser.parse(file_path)
            if not doc.content.strip():
                logger.warning(f"Empty content: {file_path}")
                return 0
            
            # Chunk
            chunker = self._get_chunker()
            chunks = chunker.chunk(doc)
            
            if not chunks:
                logger.warning(f"No chunks generated for: {file_path}")
                return 0
            
            # Add to vector store
            texts = [c.content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            ids = [c.chunk_id for c in chunks]
            
            added = self.vector_store.add_documents(texts, metadatas, ids)
            self._all_chunk_texts.extend(texts)
            
            logger.success(
                f"Ingested '{Path(file_path).name}': "
                f"{len(chunks)} chunks, {sum(len(t) for t in texts):,} chars"
            )
            return added
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return 0
    
    def ingest_directory(
        self,
        directory: str,
        patterns: List[str] = None,
        recursive: bool = True,
    ) -> Dict[str, int]:
        """Ingest all documents in a directory."""
        patterns = patterns or ["*.pdf", "*.docx", "*.txt", "*.md"]
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return {}
        
        # Collect files
        files = []
        for pattern in patterns:
            if recursive:
                files.extend(dir_path.rglob(pattern))
            else:
                files.extend(dir_path.glob(pattern))
        
        files = list(set(files))  # Deduplicate
        
        if not files:
            logger.warning(f"No files found in {directory} matching {patterns}")
            return {}
        
        logger.info(f"Found {len(files)} files to ingest")
        
        results = {}
        total_chunks = 0
        
        for file_path in tqdm(files, desc="Ingesting documents"):
            chunks_added = self.ingest_file(str(file_path))
            results[str(file_path)] = chunks_added
            total_chunks += chunks_added
        
        logger.success(
            f"Ingestion complete: {len(files)} files → {total_chunks} total chunks"
        )
        return results
    
    def ingest_texts(
        self,
        texts: List[str],
        metadatas: List[Dict] = None,
        source_name: str = "inline",
    ) -> int:
        """Ingest raw text strings directly."""
        metadatas = metadatas or [{"source": source_name} for _ in texts]
        
        # Generate IDs
        ids = [
            hashlib.md5(f"{source_name}_{i}_{t[:50]}".encode()).hexdigest()[:16]
            for i, t in enumerate(texts)
        ]
        
        added = self.vector_store.add_documents(texts, metadatas, ids)
        self._all_chunk_texts.extend(texts)
        return added
    
    @property
    def all_texts(self) -> List[str]:
        """All ingested text chunks (for BM25 index)."""
        return self._all_chunk_texts


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="NeuralRAG Document Ingestion")
    parser.add_argument("--input", "-i", required=True, help="Input directory or file")
    parser.add_argument("--config", "-c", default="./configs/config.yaml")
    parser.add_argument("--collection", default=None, help="Override collection name")
    parser.add_argument("--chunking", default="recursive", 
                        choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.collection:
        config["vector_store"]["chroma"]["collection_name"] = args.collection
    
    # Initialize components
    from embeddings import DenseEmbedder
    from retriever import VectorStore
    
    embedder = DenseEmbedder(
        model_name=config["embeddings"]["dense_model"],
        device=config["embeddings"]["device"],
    )
    
    vector_store = VectorStore(
        persist_directory=config["vector_store"]["chroma"]["persist_directory"],
        collection_name=config["vector_store"]["chroma"]["collection_name"],
        embedder=embedder,
    )
    
    # Run ingestion
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedder=embedder,
        chunking_strategy=args.chunking,
        chunk_size=args.chunk_size,
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        n = pipeline.ingest_file(str(input_path))
        print(f"\n✅ Ingested 1 file → {n} chunks")
    elif input_path.is_dir():
        results = pipeline.ingest_directory(str(input_path))
        total = sum(results.values())
        print(f"\n✅ Ingested {len(results)} files → {total} total chunks")
    else:
        print(f"❌ Input not found: {args.input}")


if __name__ == "__main__":
    main()
