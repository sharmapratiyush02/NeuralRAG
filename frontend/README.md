# 🌐 Frontend — NeuralRAG Web Interface

This folder contains the complete single-page web application for NeuralRAG.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Full SPA — HTML + CSS + JavaScript in one file |

## Features

- **Query Tab** — Ask questions, switch between Standard RAG / Agent / Stream modes
- **Upload Tab** — Drag-and-drop PDF, DOCX, TXT, MD, and image files
- **Session Files Panel** — Sidebar shows uploaded files with checkboxes to select which to query
- **Evaluate Tab** — Run RAGAS evaluation suite and view metric scores
- **Architecture Tab** — Visual explanation of the DLM techniques used

## Multilingual Support

Select any of 15+ languages from the sidebar dropdown:
- Hindi, Marathi, Bengali, Tamil, Telugu, Urdu
- Spanish, French, German, Portuguese, Russian
- Arabic, Japanese, Chinese

Questions are auto-translated to English for retrieval, then answers are translated back.

## Session-Only Storage Notice

Uploaded files are stored in browser memory only.  
They are cleared when the page is refreshed or closed.  
This is by design — no user data is ever saved to disk.

## Run Locally

```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

> The backend API must be running at `http://localhost:8000` for full functionality.  
> If the API is offline, the UI enters **Demo Mode** automatically.
