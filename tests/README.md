# 🧪 Tests — Unit & Integration Test Suite

This folder contains all tests for the NeuralRAG pipeline.

## Files

| File | Purpose |
|------|---------|
| `test_pipeline.py` | Unit + integration tests for the full RAG pipeline |

## Run Tests

```bash
# Run all tests with pytest (recommended)
pip install pytest
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=backend --cov-report=term-missing

# Run standalone without pytest
python tests/test_pipeline.py
```

## What Is Tested

| Test Area | What It Covers |
|-----------|---------------|
| **Embeddings** | Dimensionality (1024-dim), normalization, cosine similarity |
| **Session Store** | Add files, retrieve files, source filtering, session isolation, clear |
| **Chunking** | Fixed-size chunks, chunk count, content hash deduplication |
| **API Schema** | Required response fields, faithfulness score range, timing consistency |
| **Retrieval** | Top-K results, source file filtering, empty session handling |

## Expected Results

All tests should pass with 0 failures:

```
TestEmbeddingProperties   .... OK
TestSessionStore          ....... OK
TestChunkingLogic         ... OK
TestAPIResponseFormat     .... OK

Completed — 18 tests, 0 failures, 0 errors
```

## Adding New Tests

Add test classes inheriting from `unittest.TestCase` in `test_pipeline.py`.  
Follow the naming convention: `test_<what_you_are_testing>`.

## Contributor
**Adit** — Testing, Integration & Quality Assurance
