# NeuralRAG — Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies for OCR, PDF, etc.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Data directories
RUN mkdir -p data/chroma_db data/documents data/evaluation models

EXPOSE 8000

CMD ["python", "backend/api.py"]
