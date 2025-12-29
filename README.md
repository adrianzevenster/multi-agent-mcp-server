
# Multi Agent RAG MCP

Multi Agent RAG MCP is a **local-first agent + RAG stack** built around:

- **Ollama** for running LLMs locally
- **Qdrant** as the vector database for retrieval
- A lightweight API/UI layer to test, iterate, and ship agent-driven retrieval (not just “UI over prompts”)

This repo is focused on **agent-driven retrieval that routes context dynamically** and keeps the system modular enough to swap models, embeddings, chunking, and retrieval strategies without rewriting the whole app.

---

## What this project does

- Ingests documents into a **vector index** (Qdrant)
- Runs **semantic search + retrieval** for a user prompt
- Builds a **grounded context block** (RAG context)
- Sends the prompt + retrieved context to an **Ollama-hosted model**
- Returns an answer with a bias toward **faithfulness** (answering from retrieved sources)

---

## High-level architecture

```
User/UI
|
v
API (chat endpoint)
|
+--> Retriever (Qdrant) ----> Top-k chunks
|
+--> Prompt builder --------> Grounded context prompt
|
+--> LLM runtime (Ollama) --> Final response
```

## Key components

### 1) Qdrant (Vector DB)

Qdrant stores:
- embeddings
- chunk text
- metadata (source, doc name, page/section, timestamps, tags)

Used for:
- Top-k nearest neighbor retrieval
- Filtering/routing by metadata (when enabled)

### 2) Ollama (LLM runtime)

Ollama provides:
- local model serving
- predictable latency and costs
- easy experimentation across models (e.g., llama3.x, mistral, etc.)

### 3) RAG formatting utilities

There is a small formatting layer to build a consistent “retrieved context” block used by the model.

## How retrieval is grounded

The system is designed so the model:
- answers **from the retrieved chunks first**
- avoids inventing details when context is missing
- can explicitly say “not found in retrieved context” when appropriate

This is the difference between “RAG as a UI feature” vs **retrieval as a first-class system behavior**.

---

## Running locally

### Prerequisites

- Docker + Docker Compose
- Ollama installed (or running in a container)
- Python 3.10+ (if running services locally outside Docker)

### Start Qdrant

If using Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Start Ollama

If running locally:
```bash
ollama serve
```

Pull a model:
```bash
ollama pull llama3.1
```

### Run the app

Depending on how your repo is wired, one of these will apply:
```bash
python app.py
```

or (if FastAPI / Uvicorn):
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment variables

Create a `.env` file for local config. Typical variables:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=Multi Agent RAG MCP

# Retrieval
TOP_K=8
MIN_SCORE=0.2

# Chunking / Embeddings (if applicable)
EMBEDDING_MODEL=...
CHUNK_SIZE=...
CHUNK_OVERLAP=...
```

> Keep secrets and local paths out of git. Use `.gitignore` for `.env`.

---

## Ingestion pipeline (typical flow)

1. Extract text from PDFs/docs
2. Chunk into semantic units
3. Embed each chunk
4. Upsert into Qdrant with metadata

Example structure stored in Qdrant payload:
```json
{
  "text": "chunk content...",
  "doc_id": "medical_ai.pdf",
  "source": "local",
  "page": 12,
  "section": "intro",
  "created_at": "2025-12-23T10:00:00Z"
}
```

---

## Query pipeline (typical flow)

1. User prompt arrives at `/chat`
2. Compute embedding for prompt (or use Ollama embedding model if configured)
3. Query Qdrant: `top_k`
4. Build RAG context block
5. Call Ollama `/api/chat`
6. Return response + optionally retrieved sources

---

## What makes this “agentic” (not just RAG)

- Retrieval isn’t a fixed step — it can be routed:
   - different collections per intent
   - different top-k per query type
   - metadata filters when the user specifies a domain
- The “agent” can decide:
   - when to retrieve vs answer directly
   - when to ask a follow-up (if missing context)
   - how to ground claims vs hedge

---

## Project goals

- Local-first, cost-controlled RAG stack
- Modular components (swap model, embeddings, chunker, retriever)
- Grounded outputs with low hallucination risk
- Ready to scale into multi-agent routing + evaluation loops later

---

## Troubleshooting

### Ollama port already in use

If you see:
`bind: address already in use`

Check what is using the port:
```bash
lsof -i :11434
```

Stop the process or change the port.

### Qdrant connection issues

Confirm Qdrant is reachable:
```bash
curl http://localhost:6333/collections
```

---

## Next improvements (practical roadmap)

- Add metadata-based routing (filters by doc/type)
- Add citations in responses (source chunk IDs)
- Add chunk-level reranking
- Add eval harness (faithfulness + relevance)
- Add streaming responses from Ollama
- Add multi-collection indexing per domain

---
```
