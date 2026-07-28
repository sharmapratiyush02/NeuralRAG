# ⚙️ Configs — System Configuration

This folder contains all configuration files for NeuralRAG.

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | **Your actual config** — contains API keys. **Gitignored — never committed.** |
| `config.example.yaml` | Safe template — copy this to create your `config.yaml` |

## Setup

```bash
# Copy the example (do this once)
copy configs\config.example.yaml configs\config.yaml

# Then open config.yaml and fill in your API keys
```

## Key Settings

```yaml
embeddings:
  dense_model: "BAAI/bge-large-en-v1.5"   # embedding model
  device: "cpu"                              # or "cuda" if you have a GPU

generation:
  use_api: true                              # true = use Groq/OpenAI API
  openai_model: "llama-3.3-70b-versatile"  # model name on Groq

api_keys:
  openai: "your-groq-api-key-here"         # get free key at console.groq.com
  cohere: "your-cohere-key-here"           # optional, for neural reranking
```

## Get Free API Keys

| Service | URL | Used For |
|---------|-----|---------|
| **Groq** | https://console.groq.com | LLM generation (free, fast) |
| **Cohere** | https://dashboard.cohere.com | Neural reranking (optional) |
| **HuggingFace** | https://huggingface.co/settings/tokens | Model downloads |

> ⚠️ `config.yaml` is listed in `.gitignore` — it will **never** be committed to GitHub.  
> Share API keys only via WhatsApp/private message, **never** through GitHub.
