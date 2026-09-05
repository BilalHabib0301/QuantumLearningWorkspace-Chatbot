# RAG Engine Setup Guide

This guide describes how to set up the StudyMind RAG Engine from scratch on a
clean machine. It was tested end-to-end in a **fresh virtual environment**
(Python 3.11.9), following these exact steps — including the pitfalls below.

## Prerequisites

- **Python 3.11 or 3.12** — this is critical. Python 3.13/3.14 will fail to
  build `chroma-hnswlib` (no prebuilt wheel, native build requires a compiler).
  Verify with `python --version` before starting.
- **Git**
- **A Groq API key** (`GROQ_API_KEY`) for LLM generation and re-ranking.
- **A JWT secret** (`JWT_SECRET_KEY`) — the server refuses to start without it
  (see `auth.py`).

## Setup Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/QuantumLogicsLabs/QuantumLearningWorkspace-Chatbot.git
   cd QuantumLearningWorkspace-Chatbot
   ```

2. **Create a virtual environment with a compatible Python:**

   Use 3.11 or 3.12 explicitly — do **not** rely on the default `python`:

   ```bash
   # Windows (choose the matching available version):
   py -3.11 -m venv venv
   # macOS/Linux:
   python3.11 -m venv venv
   ```

   Activate it:

   ```bash
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies — use `requirements.txt`, not editable installs:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Why not `pip install -e .`?** The repo root is a *flat layout* containing
   > multiple top-level packages (`memory/`, `rag_venv/`), so setuptools refuses
   > to build an editable wheel. The Dockerfile also uses `requirements.txt`.

4. **Create your environment file:**

   ```bash
   cp rag-engine/.env.example .env
   ```

   Open `.env` and set at minimum:

   ```ini
   GROQ_API_KEY=your_groq_api_key
   JWT_SECRET_KEY=a_long_random_string
   ```

   `load_env()` (rag-service) looks for `.env` in the repo root, `chatbot/`,
   and `rag-engine/`, so placing it at the repo root works.

   Optional but recommended — set a ChromaDB persistence path you can
   recreate (see Troubleshooting):

   ```ini
   CHROMA_DB_PATH=C:/Users/<you>/shared_chroma_data
   ```

   If unset, the default is `C:\Dev\QuantumLearningWorkspace\shared_chroma_data`
   on Windows (see `vector_store.py`), which may not exist on a fresh machine.

5. **Start the server:**

   You must run from the `rag-engine/` directory so `uvicorn` can resolve
   `main:app`:

   ```bash
   cd rag-engine
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

   The first startup takes **30–60 seconds** while the embedding model
   (`all-MiniLM-L6-v2`) loads and ChromaDB connects. Be patient.

6. **Verify it's healthy:**

   ```bash
   curl http://127.0.0.1:8000/health
   ```

   Expected output (values may vary):

   ```json
   {"status":"ok","ready":true,"chunks_indexed":0,"embedding_model":"all-MiniLM-L6-v2","default_top_k":4,"max_distance":1.2,"cache_entries":0,"cache_hits":0,"cache_backend":"memory","rate_limit_backend":"memory","groq_configured":true}
   ```

   Discard the ChromaDB **telemetry warnings** (`Failed to send telemetry
   event ... capture()`): they are harmless and can be disabled via Chroma's
   `anonymized_telemetry=False`.

## Troubleshooting

### `KeyError: '_type'` on startup (ChromaDB corruption)

If the server crashes on startup with `KeyError: '_type'` (or other opaque
errors), the persistent ChromaDB directory is corrupted — this happened
multiple times during development.

- **Fix:** delete the directory set by `CHROMA_DB_PATH` (or the default path)
  and restart. A fresh collection is re-created automatically.
- **Note:** this wipes embedded documents, so re-ingest content afterwards.

### `pip install -r requirements.txt` fails to build `chroma-hnswlib`

- Check your Python version: must be **3.11 or 3.12**.
- On Windows, a compiler toolchain may be required; using 3.11/3.12 avoids
  the need for it because prebuilt wheels exist.

### Server runs but `/health` never returns

- Give it longer: first boot loads the sentence-transformer model.
- Confirm you started uvicorn from the `rag-engine/` directory.
- If the port is already in use, pick a different one (`--port 8010`).

### `ModuleNotFoundError: No module named 'main'`

- You are not in the `rag-engine/` directory, or the venv is not activated.
  Run the uvicorn command from `rag-engine/`.