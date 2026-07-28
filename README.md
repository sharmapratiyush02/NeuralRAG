# 🧠 NeuralRAG — Enterprise Deep Learning RAG System

> **Course**: Deep Learning Models (DLM) | Advanced RAG with Neural Architectures  
> **Team**: Pratiyush · Neeraj · Krishna · Adit · Soham

---

## 🎯 What This Project Does

NeuralRAG is a **production-grade document question-answering system** powered by deep learning. Upload any PDF, DOCX, or TXT file — ask questions about it in **any language** — get accurate, cited answers.

**Key highlights:**
- 🔒 **Session-only storage** — uploaded files stay in memory, never saved to disk
- 📂 **Multi-document selection** — upload multiple files, choose which ones to query
- 🌐 **15+ languages** — ask in Hindi, Arabic, Spanish, Japanese, etc.
- 🤖 **Agentic RAG** — ReAct reasoning loop for complex multi-step questions
- 🏅 **Neural reranking** — BGE cross-encoder for precise result ranking
- 📊 **RAGAS evaluation** — faithfulness, relevancy, precision, recall metrics

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      NeuralRAG Pipeline                       │
├──────────────────────────────────────────────────────────────┤
│  UPLOAD  →  Parse  →  Chunk  →  Embed  →  RAM (session)     │
├──────────────────────────────────────────────────────────────┤
│  QUERY   →  Embed Query  →  Cosine Search  →  BGE Rerank    │
├──────────────────────────────────────────────────────────────┤
│  GENERATE →  LLM (Llama-3 / Groq)  →  Answer + Sources     │
├──────────────────────────────────────────────────────────────┤
│  TRANSLATE →  Any language  →  Return in user's language    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Windows
```bat
start.bat
```
Double-click `start.bat` — it sets up everything and opens the app automatically.

### Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate          # Windows
source venv/bin/activate        # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up config (add your API keys)
copy configs\config.example.yaml configs\config.yaml

# 5. Start the backend
python backend\api.py

# 6. Start the frontend (new terminal)
cd frontend
python -m http.server 3000
```

Open browser: **http://localhost:3000**

---

## 📁 Project Structure

```
NeuralRAG/
├── backend/
│   ├── api.py              # FastAPI server — all REST endpoints
│   ├── embeddings.py       # BGE-Large neural embeddings
│   ├── retriever.py        # Hybrid dense + BM25 retrieval + RRF fusion
│   ├── reranker.py         # BGE cross-encoder neural reranker
│   ├── generator.py        # Llama-3 / Groq LLM generation
│   ├── agent.py            # ReAct agentic reasoning loop
│   └── multimodal.py       # Image understanding (CLIP + LLaVA)
├── scripts/
│   ├── ingest.py           # Document ingestion pipeline
│   └── fine_tune_embedder.py  # Fine-tune embedding model
├── evaluation/
│   └── ragas_eval.py       # RAGAS evaluation metrics
├── tests/
│   └── test_pipeline.py    # Unit + integration tests
├── configs/
│   ├── config.yaml         # Your config (gitignored — has API keys)
│   └── config.example.yaml # Template for teammates
├── frontend/
│   └── index.html          # Full single-page web app
├── docs/
│   └── PROJECT_REPORT.md   # Detailed project report
├── requirements.txt
├── Dockerfile
├── start.bat               # Windows one-click launcher
└── start.sh                # Linux/macOS launcher
```

---

## 🧠 DLM Core Techniques

| Component | Deep Learning Technique |
|-----------|------------------------|
| **Embeddings** | BGE-Large-EN-v1.5 — 1024-dim bi-encoder, fine-tuned on MS-MARCO |
| **Reranking** | BGE-Reranker-Large — 560M cross-encoder, query×doc joint attention |
| **Retrieval** | Reciprocal Rank Fusion — combines dense + BM25 sparse rankings |
| **Generation** | Llama-3-8B-Instruct with 4-bit QLoRA quantization |
| **Agent** | ReAct loop — Reason → Act → Observe → repeat |
| **Multimodal** | CLIP embeddings + LLaVA for image question answering |
| **Evaluation** | RAGAS — NLI-based faithfulness scoring, no ground truth needed |

---

## 🔒 Session-Only Storage (Privacy Design)

```
Upload → Parse → Chunk → Embed → Store in RAM (keyed by session UUID)
Query  → Cosine similarity on session vectors → Rerank → Generate
Close  → DELETE /api/session → RAM freed instantly
```

Documents are **never written to ChromaDB or disk**. Each browser tab gets a unique session ID. Closing the tab automatically clears all uploaded data from memory.

---

## 🌐 Multilingual Pipeline

```
User query (any language)
    ↓ Translate to English (via LLM)
    ↓ Retrieve relevant chunks
    ↓ Generate answer in English
    ↓ Translate answer to user's language
    → Show answer in chosen language
```

Supported: Hindi, Marathi, Spanish, French, German, Arabic, Japanese, Chinese, Bengali, Tamil, Telugu, Urdu, Portuguese, Russian + more.

---

## 📊 Evaluation (RAGAS)

Run from the UI: **Evaluate tab → Run RAGAS Evaluation Suite**

| Metric | Measures |
|--------|---------|
| **Faithfulness** | Are answers grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are retrieved chunks actually relevant? |
| **Context Recall** | Did we retrieve all needed information? |

---

## ⚙️ Configuration

Copy `configs/config.example.yaml` → `configs/config.yaml` and fill in:

```yaml
api_keys:
  openai: "your-groq-or-openai-key"   # Used for LLM generation
  cohere: "your-cohere-key"            # Optional: for neural reranking
```

> ⚠️ `config.yaml` is gitignored — never commit API keys.

---

## 🔬 Research Extensions

1. **Adaptive Chunking** — semantic boundary detection for optimal chunk sizes
2. **Query Routing** — classify queries to route to specialized sub-indexes
3. **Confidence Calibration** — neural calibration of retrieval confidence
4. **Continual Learning** — incremental embedding updates without re-indexing

---

## 📝 Contributors

| Name | Role |
|------|------|
| Pratiyush | Project Lead · Backend Architecture · Session Storage System |
| Neeraj | Evaluation Pipeline · RAGAS Metrics · Benchmarking |
| Krishna | Frontend UI · Query Interface · Multilingual Support |
| Adit | Document Ingestion · Chunking Strategies · Data Pipeline |
| Soham | Testing · Integration · Deployment · Documentation |
