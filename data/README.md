# 📁 Data — Storage Directory

This folder holds all runtime data for NeuralRAG.

## Structure

```
data/
├── chroma_db/          # ChromaDB vector database (auto-generated, gitignored)
├── documents/          # Raw source documents for ingestion (gitignored)
└── evaluation/
    └── test_qa.json    # QA pairs for RAGAS evaluation (gitignored)
```

## Important

> ⚠️ The contents of this folder are **gitignored** — nothing here is committed to GitHub.  
> Each team member generates their own data locally by running the ingestion pipeline.

## Generate Data Locally

```bash
# Place your documents in data/documents/
# Then run ingestion:
python scripts/ingest.py --input data/documents/
```

## Evaluation Dataset Format

To run RAGAS evaluation, create `data/evaluation/test_qa.json`:

```json
[
  {
    "question": "What is the main topic of this document?",
    "ground_truth": "The document discusses..."
  }
]
```
