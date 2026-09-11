# Wintermute

A self-hosted, agentic **documentary assistant**: it ingests your documents, builds a knowledge base from them, and answers questions — strictly from what it has read, or with its own voice when the question falls outside the corpus.

> *Wintermute was hive mind, with the ultimate goal of becoming self-aware. For now, it mostly ingests PDFs.*

## Status

| Capability | State |
|---|---|
| General questions (model's own knowledge, in persona) | ✅ Working |
| Document ingestion (PDF extraction → validation → summarization) | 🚧 In progress — pipeline runs, storage/indexing still being built |
| Retrieval (question answering over the ingested corpus) | 🚧 Legacy prototype only — new agent not started |
| Knowledge base (characters, claims, locations…) | 📐 Designed & validated (models + unit tests), not yet wired to storage |

Honesty rule of the project: components say what they are. Working parts are used; unfinished parts announce themselves instead of failing silently.

## Overview

Wintermute is one of the components of a larger documentary toolchain about a fiction world: source documents (rulebooks, gazettes, stories…) are ingested, their content is extracted, validated, summarized, and — in later gates — distilled into structured knowledge (characters, relations, claims, events) that can be queried and browsed.

Everything runs locally: a personal Ollama instance supplies the language models, the orchestration is pure Python, and no data leaves the machine.

## Architecture

```
User front-end (Ollama-compatible chat UI, e.g. Open WebUI)
        │   OpenAI / Ollama wire protocols
        ▼
FastAPI gateway (app/api.py)  — one endpoint family, streaming traces included
        │   routes every user message through…
        ▼
Routing orchestrator (src/routing) — request analyzer (LLM) → structured requests
        │   dispatches, in order, to task agents…
        ▼
Task agents (src/agents)      — IngestionTask / GeneralTask / (RetrievalTask — soon)
        │
        ▼
Ingestion orchestrator (src/ingestion) — its own routing, then the ingestion graph
        │   extraction → validation → summarization → storage/indexing (in build)
        ▼
Local Ollama instance         — every LLM call (routing, analysis, summarization, answering)
```

In short: **front-end → FastAPI → Python agents → Ollama models**. The gateway speaks both the OpenAI and Ollama chat protocols, so any Ollama-compatible client treats Wintermute as just another model. While a request is processed, its internal steps stream to the client's "thinking" panel and are mirrored to `data/logs/wintermute.log`.

## Agentic structure

Requests are never interpreted by regex alone: an analyzer model turns each user message into an ordered list of structured requests, which a routing graph dispatches to task agents sequentially. Agents are small, single-purpose, and behind declared protocols (structural contracts), so implementations can be swapped without touching the graph.

Current agents:

- **GeneralTaskAgent** — answers anything outside the corpus with the model's own knowledge, in the voice of Wintermute (roleplay is a feature, not a bug).
- **IngestionTaskAgent** — hands file orders to the ingestion orchestrator.
- **RetrievalTaskAgent** — planned; retrieval requests currently fall back to the legacy RAG chain or an honest "not available yet".

Within ingestion, the graph chains placeholder-ready steps (extraction, extraction validation, hierarchical summarization, then knowledge extraction, validation, indexing) with per-step agents to be plugged in one by one. Each agent carries its prompt (markdown files under `prompts/`) and its LLM role from `config/llm.yaml`.

## Technologies

**In place:** Python · FastAPI · Ollama (all LLM calls) · Pydantic (models & validation) · unittest (462 tests) · YAML configuration · markdown prompt files.

**In the pipeline, at their respective gates:** ChromaDB (vector store) · PostgreSQL (structured knowledge) · markdown exports (human-browsable knowledge wiki, built from the extracted data).

The legacy prototype already exercises LangChain + ChromaDB for retrieval; the new pipeline will absorb it once RetrievalTaskAgent lands.

## User-facing surfaces

- **Chat** — any Ollama-compatible front-end (Open WebUI, and Ollama itself) pointed at the gateway. Streaming shows the routing/agent traces live, then the answer.
- **Knowledge wiki** — the long-term read surface: markdown documents generated from the knowledge base (characters, relations, claims), browsable like a wiki of the fiction world. Awaits the knowledge-storage gate.

## Tests

The suite is hermetic: no live Ollama call, no real corpus — LLM clients and stores are stubbed, so it runs identically on any machine.

```bash
# from the repository root
PYTHONPATH=. venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected output: `OK` — 460+ tests across routing, ingestion, extraction, knowledge, validation, summarization, helpers and API layers.

A syntax/compile sanity check:

```bash
venv/Scripts/python.exe -m compileall -q src app tests
```

## Configuration

- `config/llm.yaml` — LLM roles (routing, analysis, summarization, answering…), validated strictly at load: a missing role is an error, never a silent default.
- `config/ingestion.yaml` — document folders, extraction settings, job-file paths.
- `config/setup.yaml` — logging (console + rotating file under `data/logs/`).

## Running

```bash
python -m app.api          # http://127.0.0.1:8000
```

Then point an Ollama-compatible client at it (Open WebUI → add a connection to `http://127.0.0.1:8000`). `GET /health` reports the state of the RAG stack; ingestible documents go under `data/sources/`.
