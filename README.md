# Wintermute

A self-hosted, agentic **documentary assistant**: it ingests your documents, builds a knowledge base from them, and answers questions — strictly from what it has read, or with its own voice when the question falls outside the corpus.

> *Wintermute was hive mind, with the ultimate goal of becoming self-aware. For now, it mostly ingests PDFs.*

## Status

| Capability | State |
|---|---|
| General questions (model's own knowledge, in persona) | ✅ Working |
| Document ingestion (PDF extraction → validation → summarization → indexing) | ✅ Working — via the CLI (`scripts/ingest.py`), not the chat |
| Corpus removal (vectors, checkpoints, JSONs) | ✅ Working — `scripts/remove.py` |
| Retrieval (question answering over the ingested corpus) | ✅ Working — semantic search + AnswerAgent (relation/index lookups await the SQL gate) |
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
Task agents (src/agents)      — RetrievalTask / GeneralTask
        │
        ▼
Retrieval pipeline (src/retrieval) — decision table → retrieval graph → AnswerAgent

CLI (scripts/ingest.py, scripts/remove.py) — deterministic ingestion & removal,
        never reachable from the chat (deliberate design)
        │
        ▼
Ingestion orchestrator (src/ingestion) — the ingestion graph
        extraction → validation → summarization → indexing
        ▼
Local Ollama instance         — every LLM call (analysis, summarization, answering)
```

In short: **front-end → FastAPI → Python agents → Ollama models**. The gateway speaks both the OpenAI and Ollama chat protocols, so any Ollama-compatible client treats Wintermute as just another model. While a request is processed, its internal steps stream to the client's "thinking" panel and are mirrored to `data/logs/wintermute.log`.

## Agentic structure

Requests are never interpreted by regex alone: an analyzer model turns each user message into an ordered list of structured requests, which a routing graph dispatches to task agents sequentially. Agents are small, single-purpose, and behind declared protocols (structural contracts), so implementations can be swapped without touching the graph.

Current agents:

- **GeneralTaskAgent** — answers anything outside the corpus with the model's own knowledge, in the voice of Wintermute (roleplay is a feature, not a bug).
- **RetrievalTaskAgent** — hands corpus questions to the deterministic retrieval pipeline (semantic search → AnswerAgent, a grounded cited reply).

**Ingestion is deliberately not an agent**: a paradigm change moved it out of the chat. The analyzer cannot emit ingestion orders, so the router can never again hallucinate a file name into a phantom ingestion; instead `scripts/ingest.py` (and `scripts/remove.py`) run the strictly deterministic ingestion/removal flow, with a durable log (`data/logs/ingestion.log`) and explicit exit codes.

## Technologies

**In place:** Python · FastAPI · Ollama (all LLM calls) · Pydantic (models & validation) · unittest (600+ tests) · ChromaDB (vector store) · YAML configuration · markdown prompt files.

**In the pipeline, at their respective gates:** PostgreSQL (structured knowledge) · markdown exports (human-browsable knowledge wiki, built from the extracted data) · relation/index retrievals over the SQL layer.

## User-facing surfaces

- **Chat** — any Ollama-compatible front-end (Open WebUI, and Ollama itself) pointed at the gateway. Streaming shows the routing/agent traces live, then the answer.
- **Knowledge wiki** — the long-term read surface: markdown documents generated from the knowledge base (characters, relations, claims), browsable like a wiki of the fiction world. Awaits the knowledge-storage gate.

## Tests

The suite is hermetic: no live Ollama call, no real corpus — LLM clients and stores are stubbed, so it runs identically on any machine.

```bash
# from the repository root
PYTHONPATH=. venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected output: `OK` — 600+ tests across routing, retrieval, ingestion, extraction, indexing, knowledge, validation, summarization, helpers and API layers.

A syntax/compile sanity check:

```bash
venv/Scripts/python.exe -m compileall -q src app tests
```

## Configuration

- `config/llm.yaml` — LLM roles (routing, analysis, summarization, answering…), validated strictly at load: a missing role is an error, never a silent default.
- `config/ingestion.yaml` — document folders, extraction settings, job-file paths.
- `config/retrieval.yaml` — semantic-search tuning (top-k, min score, embedding role, query instruction).
- `config/setup.yaml` — vector-store paths & collection names, routing caps, meta-request switch, logging (console + rotating file under `data/logs/`).

## Running

```bash
python -m app.api          # http://127.0.0.1:8000
```

Then point an Ollama-compatible client at it (Open WebUI → add a connection to `http://127.0.0.1:8000`). `GET /health` reports the state of the RAG stack.

## Ingesting & removing documents

```bash
# ingest (origin is mandatory)
PYTHONPATH=. venv/Scripts/python.exe scripts/ingest.py -i "meow.pdf" -o canon
# force a full re-ingestion / force re-summarization
PYTHONPATH=. venv/Scripts/python.exe scripts/ingest.py -i "meow.pdf" -o community -f -s
# remove from the corpus (source file kept)
PYTHONPATH=. venv/Scripts/python.exe scripts/remove.py -i "meow.pdf"
```

Documents go under `data/sources/<ext>/` (the subfolder per extension mapping is set in `config/ingestion.yaml`); removal never touches the source file — it cleans the vector store, the job checkpoints, the extracted/summarized JSONs and the MinerU sandbox.

The `-o/--origin` label is **user-defined governance metadata**: the vocabulary lives in `config/setup.yaml` (`documents.origins`), and you adapt it to your own usage context (the shipped default defines `canon`, `community`, `rpg`; the first entry is the default for ingestions run without `-o`). The label is stored verbatim in the extraction/summarized JSONs, the vector metadata, and later the SQL and markdown layers.
