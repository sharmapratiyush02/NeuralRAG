# 📊 Evaluation — RAGAS Metrics & Benchmarking

This folder contains evaluation tools to measure NeuralRAG pipeline quality.

## Files

| File | Purpose |
|------|---------|
| `ragas_eval.py` | RAGAS evaluator — faithfulness, relevancy, precision, recall |

## RAGAS Metrics Explained

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| **Faithfulness** | Are answers grounded in retrieved context? (no hallucination) | > 0.85 |
| **Answer Relevancy** | Does the answer actually address the question? | > 0.80 |
| **Context Precision** | Are the retrieved chunks truly relevant to the query? | > 0.75 |
| **Context Recall** | Did we retrieve all the information needed to answer? | > 0.70 |

## How to Run Evaluation

### From the UI
1. Go to the **Evaluate** tab in the web app
2. Click **Run RAGAS Evaluation Suite**
3. Results appear with scores and a visual breakdown

### From the Terminal
```bash
# Requires a test QA dataset at data/evaluation/test_qa.json
python evaluation/ragas_eval.py
```

## Test Dataset Format

Place your QA pairs at `data/evaluation/test_qa.json`:

```json
[
  {
    "question": "What is retrieval-augmented generation?",
    "ground_truth": "RAG combines a retrieval system with a language model..."
  }
]
```

## Contributor
**Neeraj** — Evaluation Pipeline & Benchmarking
