# SDAIA RAG Assistant

Ask questions (Arabic or English) about SDAIA's AI documents. Runs fully local and free:
Ollama for the LLM and embeddings, Chroma as the vector store, FastAPI for the API.

```
PDFs ──► ingest.py ──► chunks ──► bge-m3 embeddings ──► Chroma (chroma_db/)
                                                            │
question ──► bge-m3 ──► MMR top-5 chunks ──► prompt ──► qwen2.5:7b ──► streamed answer + sources
```

| File | Role |
|---|---|
| `config.py` | Models, chunk size, top-k. Every value can be overridden with an env var. |
| `download_docs.py` | Downloads the SDAIA PDFs into `data/`. |
| `ingest.py` | Loads PDFs, cleans text (Arabic-safe), splits, embeds, saves the index. |
| `rag.py` | Retrieval + prompt + LLM chain. `python rag.py` gives a terminal chat. |
| `api.py` | `/ask`, `/ask/stream`, `/health`, and a tiny chat page at `/`. |

## Run it

1. Install [Ollama](https://ollama.com/download), then pull the two models:
   ```
   ollama pull qwen2.5:7b
   ollama pull bge-m3
   ```
2. Install Python deps (Python 3.10+):
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Get the documents and build the index:
   ```
   python download_docs.py
   python ingest.py
   ```
   Any extra PDF you drop into `data/` is indexed too; re-run `ingest.py` after adding files.
4. Start the API:
   ```
   uvicorn api:app --port 8000
   ```
   Open http://localhost:8000 for the chat page, or http://localhost:8000/docs to try the endpoints.

   ```
   curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
        -d '{"question": "ما هي مبادئ أخلاقيات الذكاء الاصطناعي؟"}'
   ```

## Why these models (Arabic)

- **LLM: `qwen2.5:7b`** instead of the tutorial's `llama3:8b`. Llama 3 8B is weak at Arabic
  (it often answers in English or produces broken Arabic); Qwen 2.5 was trained on Arabic and
  follows "answer in the question's language" reliably. Alternatives to try by setting
  `LLM_MODEL`: `aya-expanse:8b` (built for Arabic and 22 other languages, non-commercial
  licence) or `qwen2.5:14b` if you have 16 GB+ of RAM/VRAM.
- **Embeddings: `bge-m3`**, a multilingual model. This matters more than the LLM: most SDAIA
  PDFs are in English, and bge-m3 lets an Arabic question find the right English passage.
  English-only embedders like `nomic-embed-text` would not.

## Tuning knobs worth experimenting with

- `CHUNK_SIZE` / `CHUNK_OVERLAP` (default 1000 / 150 characters).
- `TOP_K` (default 5) and the retriever type in `rag.py` (MMR vs plain similarity).
- `TEMPERATURE` (default 0.1, low so answers stay close to the documents).
- The system prompt in `rag.py`.

Example: `TOP_K=8 LLM_MODEL=aya-expanse:8b uvicorn api:app --port 8000`
