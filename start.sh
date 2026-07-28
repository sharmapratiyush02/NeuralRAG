#!/bin/bash
# ================================================================
# NeuralRAG — Quick Start Script
# ================================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "  ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗     ██████╗  █████╗  ██████╗"
echo "  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔══██╗██╔══██╗██╔════╝"
echo "  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ██████╔╝███████║██║  ███╗"
echo "  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██╔══██╗██╔══██║██║   ██║"
echo "  ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗██║  ██║██║  ██║╚██████╔╝"
echo "  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝"
echo -e "${NC}"
echo -e "  ${BLUE}Enterprise Deep Learning RAG System${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✓ Python ${PYTHON_VER}${NC}"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
mkdir -p data/chroma_db data/documents data/evaluation/results models/fine_tuned_embedder

# Check for API keys
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠ No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY${NC}"
    echo -e "  The system will use local Llama-3 model (requires GPU) or demo mode"
fi

# Ingest sample documents if data/ is empty
if [ -z "$(ls -A data/documents 2>/dev/null)" ]; then
    echo -e "${YELLOW}No documents found. Creating sample document...${NC}"
    cat > data/documents/sample_rag_overview.txt << 'EOF'
# Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
by retrieving relevant documents from a knowledge base before generating responses.

## Core Components

### 1. Document Ingestion
Documents are parsed, chunked, embedded using neural models, and stored in a vector database.
Semantic chunking uses sentence embeddings to find topically coherent boundaries.

### 2. Hybrid Retrieval
Dense retrieval uses BGE-Large embeddings and cosine similarity search via FAISS.
Sparse retrieval uses BM25 with TF-IDF weighting for exact keyword matching.
Reciprocal Rank Fusion (RRF) combines both ranked lists: score(d) = Σ 1/(k + rank_i(d)).

### 3. Neural Reranking
A cross-encoder (BGE-Reranker-Large) processes query and document together:
Input: [CLS] query [SEP] document [SEP]
Output: sigmoid(W · h_[CLS]) giving a relevance score in [0,1].

### 4. Generation with Self-Correction
The LLM generates an answer grounded in retrieved context.
A DeBERTa-based NLI model checks faithfulness claim by claim.
Unsupported claims trigger a targeted revision prompt.

## Deep Learning Contributions

Fine-tuned embeddings use MultipleNegativesRankingLoss (contrastive learning).
The agentic ReAct framework enables multi-hop retrieval and tool use.
RAGAS evaluation uses LLMs and embeddings to measure quality without human labels.
EOF
    echo -e "${GREEN}✓ Sample document created${NC}"
fi

# Ingest sample documents
echo -e "${YELLOW}Ingesting documents...${NC}"
python scripts/ingest.py --input data/documents/ --chunking recursive 2>/dev/null || \
    echo -e "${YELLOW}  (Ingestion will complete on first API startup)${NC}"

# Start API server
echo ""
echo -e "${GREEN}🚀 Starting NeuralRAG API...${NC}"
echo -e "   API:      ${BLUE}http://localhost:8000${NC}"
echo -e "   Docs:     ${BLUE}http://localhost:8000/docs${NC}"
echo -e "   Frontend: Open ${BLUE}frontend/index.html${NC} in your browser"
echo ""
echo -e "${YELLOW}Tip: For the UI, run in another terminal:${NC}"
echo -e "   cd frontend && python -m http.server 3000"
echo -e "   Then open ${BLUE}http://localhost:3000${NC}"
echo ""

cd backend && python api.py
