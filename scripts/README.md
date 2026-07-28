# 📜 Scripts — Ingestion & Utility Tools

This folder contains document ingestion and data processing scripts.

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Full document ingestion pipeline — parse, chunk, embed, store |
| `fine_tune_embedder.py` | Fine-tune the BGE embedding model on custom data |

## Ingestion Pipeline (`ingest.py`)

Handles: **PDF · DOCX · TXT · MD · Images**

```bash
# Ingest a single file
python scripts/ingest.py --input data/documents/paper.pdf

# Ingest an entire folder
python scripts/ingest.py --input data/documents/

# Choose chunking strategy
python scripts/ingest.py --input data/documents/ --chunking semantic
python scripts/ingest.py --input data/documents/ --chunking fixed --chunk-size 512
python scripts/ingest.py --input data/documents/ --chunking recursive
```

## Chunking Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| `recursive` | Splits on `\n\n` → `\n` → `.` → words | General documents (default) |
| `semantic` | Uses embedding similarity to find boundaries | Research papers, dense text |
| `fixed` | Fixed token count with overlap | Code, tables, structured data |

## Fine-Tuning Embeddings (`fine_tune_embedder.py`)

Fine-tunes BGE-Large on domain-specific QA pairs using contrastive learning.

```bash
python scripts/fine_tune_embedder.py \
  --data data/finetune/qa_pairs.json \
  --output models/fine_tuned_embedder \
  --epochs 3
```

## Contributor
**Krishna** — Data Ingestion Pipeline & Chunking Strategies
