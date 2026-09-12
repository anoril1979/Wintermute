"""Wintermute — the gateway the matrix talks to.

    "Wintermute was hive mind, wafered into a mosaic of spread data toroids."
    "Call up the ice, and I break it."
        — roughly, William Gibson, Neuromancer (1984)

This FastAPI app exposes the assistant as a model server so an Ollama
instance / Open WebUI can work with it. It speaks three dialects on the
same port:

* OpenAI-compatible  : GET /v1/models, POST /v1/chat/completions
* Ollama-native      : GET /api/tags, GET /api/version, POST /api/chat
* human              : GET /, GET /health, GET /docs

The prototype (proto/app/api.py, "dark-earth-rag") answered a single
question and streamed nothing. This version keeps the contract and adds:
real streaming (SSE for OpenAI clients, ndjson for Ollama ones), a
graceful degradation mode when the RAG stack or ChromaDB is not up
(clear 5xx-free answers instead of warning strings riding in the answer), model
metadata wired to config/llm.yaml, rough token accounting, and an
Ollama-native /api/chat so Ollama-protocol clients connect unchanged.

Run from the project root:

    venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
    # or simply:
    venv/Scripts/python.exe -m app.api
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("wintermute")
# Central logging (console + data/logs/wintermute.log, per setup.yaml).
# Called here so the durable log file exists whatever the entry point is;
# idempotent, so the CLI entry points calling it again are no-ops.
from src.llm.guard import (
    META_SENTINEL,
    meta_answer,
    prompt_is_meta,
    reply_to_meta_requests,
)
from src.logging_setup import (
    bind_correlation_id,
    configure_logging,
    new_correlation_id,
)

configure_logging()

# Identity, wired to the project config where it matters. The prototype was
# "dark-earth-rag"; the system grew, and something behind the wall of ice
# started calling itself Wintermute.
ASSISTANT_MODEL_ID = "wintermute"
ASSISTANT_OWNER = "Tessier-Ashpool SARL"   # the family estate endorses this build
API_VERSION = "1.0.0"
API_VERSION_TAG = "1.0.0-awakening"        # the point where the plan came together

# The backing Ollama model that answers under the hood (llm.yaml `default`
# role; the router/summarizer roles stay available for future routing).
try:
    from src.tools.config_loader import get_model_config
    _DEFAULT_MODEL = str(get_model_config("default").get("model_name", "qwen3"))
except Exception:  # config problems must not take the gateway down
    _DEFAULT_MODEL = "qwen3"
    logger.warning("Could not read llm.yaml; falling back to model '%s'.", _DEFAULT_MODEL)

# ---------------------------------------------------------------------------
# Wire format models (OpenAI + Ollama dialects)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions body."""

    model: str = ASSISTANT_MODEL_ID
    messages: List[Message] = Field(min_length=1)
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


class OllamaChatRequest(BaseModel):
    """Ollama-native /api/chat body (same shape as the real thing)."""

    model: str = ASSISTANT_MODEL_ID
    messages: List[Message] = Field(min_length=1)
    stream: Optional[bool] = True


# ---------------------------------------------------------------------------
# Lifespan — "Wintermute was... motion toward the Awakening."
# A missing/empty vector store only means Wintermute stays dormant until
# the first ingestion completes.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _corpus_awake():
        logger.info("Wintermute has attained the Awakening — vector store ready.")
    else:
        logger.warning(
            "Wintermute stays dormant: the vector store is empty or unreadable. "
            "Run an ingestion first; /health reports details."
        )
    yield


app = FastAPI(
    title="Wintermute",
    description=(
        "Documentary assistant gateway. The RAG model behind `/v1/chat/completions` "
        "answers strictly from the ingested sources — the matrix inside, not the "
        "whole Net."
    ),
    version=API_VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — needed when Open WebUI runs on a different port. Origins come from
# the WINTERMUTE_CORS_ORIGINS env var (comma-separated); "*" by default,
# which is fine on a loopback — but tighten it before exposing the ice.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in os.environ.get("WINTERMUTE_CORS_ORIGINS", "*").split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Logging middleware — the matrix watches every run. Unlike the prototype we
# log the method/path only: dumping request headers means logging cookies
# and keys, and a hive mind does not need to brag.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One correlation id per HTTP request, stamped on every log line it
    produces — the >>>/<<< pair, the analyzer, the traces, the agents —
    so a multi-request session reads as grouped blocks in the log file.

    Binding is per-thread: the async middleware logs on the event-loop
    thread (id bound here, released in ``finally`` — the loop is shared by
    ALL requests, a leak would mislabel everything after), while the sync
    endpoints bind the same id themselves from ``request.state`` on their
    own threadpool threads (see ``_request_cid``).
    """
    cid = new_correlation_id()
    request.state.correlation_id = cid
    bind_correlation_id(cid)
    try:
        logger.info(">>> %s %s", request.method, request.url.path)
        response = await call_next(request)
        logger.info("<<< %s %s", response.status_code, request.url.path)
        return response
    finally:
        bind_correlation_id(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_cid(request: Request) -> str:
    """Bind and return this request's correlation id for the *current*
    thread.

    The id is minted once per HTTP request in the middleware (stored on
    ``request.state``); sync endpoints run on threadpool threads — a
    different thread than the middleware's — so each binds it to its own
    thread's logging state before doing any work. Returns a fresh id when
    called outside a request (direct calls, tests).
    """
    cid = getattr(request.state, "correlation_id", None) or new_correlation_id()
    bind_correlation_id(cid)
    return cid


def _corpus_awake() -> bool:
    """True when the vector store holds at least one indexed chunk.

    The read-side health signal, replacing the legacy LangChain chain's
    "awake" flag: the retrieval pipeline (semantic search + AnswerAgent)
    is the memory now. Fail-open to False — a broken store is reported
    dormant, never an exception out of /health or the lifespan.
    """
    try:
        from src.retrieval.retrieval_router import gather_facts

        return bool(gather_facts().has_corpus)
    except Exception:  # noqa: BLE001 — health must not raise
        return False


def _brain_unavailable_text() -> str:
    """Honest in-band answer when the retrieval stack is not importable.

    Same spirit as fix B: a chat client treats an HTTP 503 as *its own*
    failure and silently retries the whole conversation — the very loop
    that made Wintermute re-run routing under the hood. An in-band text is
    final; the client has nothing to retry.
    """
    return (
        "My retrieval memory is unavailable right now: the retrieval stack "
        "failed to load. Check that Ollama is running (`ollama serve`) and "
        "that the dependencies are installed, then try again."
    )


def _log_snippet(text: str, limit: int = 400) -> str:
    """One-line, bounded text for log lines — fix C's observability rule:
    log *what was actually received and analyzed*, so 'the client sent the
    whole conversation' vs 'the model invented it' is provable from logs.
    """
    return " ".join(text.split())[:limit]


def _raw_last_user_text(messages: List[Message]) -> str:
    """The last user message's text (the shape _extract_question logs)."""
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return messages[-1].content.strip()


def _extract_question(messages: List[Message]) -> str:
    """The user's latest utterance.

    TODO (memory): feed the full exchange to the assistant once the
    retrieval chain supports conversation state; for now, as in the
    prototype, only the last user message drives the answer.

    Meta traffic (the sentinel, or Open WebUI's auxiliary tasks — title
    generation, follow-up suggestions, topic tagging) is intercepted
    here and never reaches routing: the sentinel marks it for the
    endpoints, which answer via :func:`_meta_answer_for_prompt`.
    """
    roles = ",".join(m.role for m in messages)
    question = _raw_last_user_text(messages)
    if prompt_is_meta(question):
        logger.info(
            "Incoming chat: %d message(s) [%s]; "
            "meta/background prompt intercepted, no routing: %s",
            len(messages), roles, _log_snippet(question),
        )
        return META_SENTINEL
    logger.info(
        "Incoming chat: %d message(s) [%s]; analyzed text: %s",
        len(messages), roles, _log_snippet(question),
    )
    return question


def _meta_answer_for_prompt(prompt_text: str) -> str:
    """Answer an intercepted meta/background prompt.

    The single decision point for the ``reply_to_meta_request`` switch:

    * sentinel probe or switch off  → the zero-cost fixed answer
      (:func:`meta_answer`) — the low-powered-machine mode;
    * switch on                     → the MetaRequestAgent answers the
      front-end's actual task (one cheap LLM call, in-style title/tags/
      suggestions), degrading to the fixed answer on ANY failure.

    Never raises, never routes, never reaches the analyzer.
    """
    if prompt_text == META_SENTINEL or not reply_to_meta_requests():
        return meta_answer()
    try:
        # Imported here like the rest of the API's heavier pieces: the
        # gateway must stay bootable even if the agents' wiring breaks.
        from src.agents.agents.meta_request_agent import get_meta_request_agent

        answer = get_meta_request_agent().run(prompt_text)
    except Exception:  # noqa: BLE001 — a background task must never surface an error
        logger.exception("The meta agent failed; using the fixed fallback.")
        return meta_answer()
    logger.info("Meta request answered by the MetaRequestAgent (%d chars).", len(answer))
    return answer


def _ask(question: str) -> str:
    """Answer a question from the corpus through the retrieval pipeline.

    Failures become honest in-band answers: only an unexpected exception
    lands here, and it must not become an HTTP 503 (chat clients
    auto-retry those and silently re-send the whole conversation).
    """
    try:
        from src.retrieval.retrieval_orchestrator import run_retrieval
        from src.routing.models import RetrievalRequest

        result = run_retrieval([RetrievalRequest(question=question)])
        sub = next(
            (r for r in result.get("requests", []) if r.get("answer")),
            None,
        )
        if sub:
            return str(sub["answer"])
        if str(result.get("status", "")) == "no_corpus":
            # Dormant memory invites ingestion — never tells the user to
            # run a script (ingestion itself comes through Wintermute).
            return (
                "My retrieval memory is dormant: nothing is indexed yet. "
                "Ask me to ingest a document first, then try again."
            )
        return str(result.get("message") or _brain_unavailable_text())
    except Exception:
        logger.exception("The retrieval pipeline failed while answering.")
        return (
            "My retrieval memory hit an error while answering. Check that "
            "Ollama is running (`ollama serve`), then try your question again."
        )


# ---------------------------------------------------------------------------
# Routing — every user message first goes through the routing orchestrator
# (src/routing), which analyzes the prompt into structured requests and
# dispatches them (ingestion -> the ingestion orchestrator, retrieval ->
# the retrieval pipeline, general -> fallback).
# ---------------------------------------------------------------------------

def _compose_reply(results: list) -> tuple:
    """Build the user-facing text from per-request routing results.

    Returns ``(text, needs_rag_fallback)``: retrieval requests that no task
    agent handled yet are answered by the retrieval pipeline directly (see
    ``_route_or_answer``).
    """
    lines: List[str] = []
    needs_rag = False
    for result in results:
        kind = str(result.get("kind", "?"))
        status = str(result.get("status", "?"))
        detail = str(result.get("detail", ""))

        if kind == "retrieval" and status == "not_implemented":
            needs_rag = True  # answered by the retrieval pipeline instead
            continue
        if status in ("incomplete", "set_aside"):
            # Underspecified request (or origin the system refuses to
            # guess): a question for the user, not a failure.
            lines.append(f"**More information needed**: {detail}")
            continue
        if status == "done":
            if kind == "ingestion":
                ingestion = result.get("ingestion") or {}
                document = ingestion.get("document") or ingestion.get("path") or ""
                steps = ", ".join(ingestion.get("completed_steps", [])) or "no step completed"
                lines.append(f"Ingestion completed for '{document}' (steps: {steps}).")
            elif kind == "retrieval" and detail:
                # The answer agent's phrased reply (grounded, cited) —
                # never a bare chunk-count status line. A retrieval with
                # no phrased answer still carries its detail.
                lines.append(detail)
            else:
                lines.append(str(result.get("answer") or detail or "Done."))
        elif status == "not_implemented":
            lines.append(f"Not available yet: {detail}")
        else:
            lines.append(_failure_line(kind, detail, result))
    return "\n".join(lines), needs_rag


def _failure_line(kind: str, detail: str, result: dict) -> str:
    """User-facing text for a failed request, enriched when useful.

    A failed ingestion whose document was not found carries ``candidates``
    (from the ingest tool's resolution, via the task agent payload): they
    are listed so the user can pick the right name — a bare "no file"
    forces the user to guess the nomenclature twice.
    """
    if kind != "ingestion":
        return f"**Could not do it**: {detail}"
    ingestion = result.get("ingestion") or {}
    resolution = ingestion.get("resolution") or result.get("resolution") or {}
    candidates = resolution.get("candidates") or []
    if not candidates:
        return f"**Could not do it**: {detail}"
    listed = "\n\n".join(f" + {name}" for name in candidates)
    return (
        f"**Could not do it**: {detail}\n\n"
        "Did you mean one of these?\n\n"
        f"{listed}\n\n"
    )


def _analysis_error_reply(message: str, cause: object) -> str:
    """User-facing answer when the prompt could not be analyzed.

    Honest and actionable, worded per failure cause — the point of fix B:
    an analyzable answer stops chat clients from auto-retrying the same
    prompt, which used to silently re-run accepted routing under the hood.
    """
    if cause == "llm_request":
        return (
            "I could not analyze your request: my analysis model is "
            "unreachable right now. Check that Ollama is running, then "
            "send your request again."
        )
    if cause == "config":
        return (
            "I could not analyze your request: the routing configuration "
            f"is incomplete ({message}). Fix the configuration and retry."
        )
    return (
        "I could not analyze your request reliably. Please rephrase it — "
        "ideally one clear instruction at a time."
    )


def _route_or_answer(question: str, *, on_event=None) -> tuple:
    """One user message: routing first, retrieval pipeline as fallback.

    The ``needs_rag`` branch now only fires for a retrieval request the
    task agent could not serve at all (e.g. a broken agent registry);
    the AnswerAgent answers every normal semantic request inside the
    routing graph.

    ``on_event`` (optional) receives the routing trace events live, while
    the routing actually runs — used by the streaming endpoints to push
    them onto the thinking channel.

    Returns ``(answer_text, routing_results_or_None)``. Every failure —
    unanalyzable prompt, broken routing layer, dormant or failing RAG
    fallback — produces a plain, honest answer text instead of an HTTP
    error: a 503 made chat clients silently auto-retry the very same
    prompt (and re-run whatever routing had already accepted), while the
    user only saw a failure.
    """
    if question == META_SENTINEL or prompt_is_meta(question):
        # Defense in depth: meta/background prompts never reach routing,
        # whatever the entry point that produced the question text.
        return meta_answer(), None
    try:
        from src.routing.routing_orchestrator import run_routing

        routing = run_routing(question, on_event=on_event)
    except Exception:  # pragma: no cover — the routing layer itself broke
        logger.exception("The routing layer failed; falling back to the RAG chain.")
        return _ask(question), None

    if routing.get("status") == "analysis_error":
        message = str(routing.get("message", "the request could not be analyzed"))
        logger.warning("Analysis failed; answering the user instead of erroring: %s", message)
        return _analysis_error_reply(message, routing.get("cause")), routing.get("results")

    text, needs_rag = _compose_reply(routing.get("results", []))
    if needs_rag:
        text = _ask(question)
    if not text:
        text = "I could not do anything with that request."
    return text, routing.get("results")


# ---------------------------------------------------------------------------
# Streaming — routing traces go to the thinking channel while the routing
# actually runs (live), then the answer streams as content chunks.
# ---------------------------------------------------------------------------

import queue as _queue
import threading as _threading


def _routing_stream(question: str, *, cid: Optional[str] = None):
    """Yield routing traces live, then the final answer.

    The routing layer is synchronous and blocking; a worker thread runs it
    with a trace observer pushing into a queue, so the consumer (the
    streaming response generators) receives each trace as it happens.

    ``cid`` re-binds the request's correlation id on the worker thread, so
    the routing/traces/agent lines carry the same id as the endpoint's own
    (thread-local logging state is not inherited across threads).

    Yields ``("trace", text)`` items while the routing runs, then exactly
    one ``("final", answer_text, routing_results)`` item. Never raises:
    any exception from the non-streaming path degrades to an in-stream
    final answer (response headers are already sent at that point — a
    status change is impossible).
    """
    events: "queue.Queue[dict | None]" = _queue.Queue()
    outcome: dict = {}

    def observer(event: dict) -> None:
        events.put(event)

    def worker() -> None:
        bind_correlation_id(cid)  # same id as the endpoint thread
        try:
            outcome["result"] = _route_or_answer(question, on_event=observer)
        except Exception:  # pragma: no cover — belt & braces: never break the stream
            logger.exception("The answer pipeline failed; degrading to an in-stream error text.")
            outcome["result"] = (
                "Something went wrong on my side while handling your message. "
                "Please try again — and if it persists, check the logs at "
                "data/logs/wintermute.log.",
                None,
            )
        finally:
            events.put(None)  # sentinel: routing finished

    thread = _threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        event = events.get()
        if event is None:
            break
        yield (
            "trace",
            f"[{event.get('phase', '?')}] {event.get('kind', '?')}: "
            f"{event.get('message', '')}\n",
        )

    thread.join()

    text, routing = outcome["result"]
    yield ("final", text, routing)


def _rough_tokens(text: str) -> int:
    """~4 chars/token heuristic — good enough for usage accounting."""
    return max(1, len(text) // 4)


def _openai_completion(question: str, answer: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer},
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": _rough_tokens(question),
            "completion_tokens": _rough_tokens(answer),
            "total_tokens": _rough_tokens(question) + _rough_tokens(answer),
        },
    }


def _chunk_answer(answer: str, words: int = 3) -> Iterator[str]:
    """Slice a complete answer into small pieces for pseudo-streaming.

    The underlying chain returns the full text in one shot; UIs that render
    progressively still want deltas, so we emit word groups. Cheap, honest
    (same total content), and a real token-level stream can replace this
    once the chain itself streams.
    """
    tokens = answer.split()
    if not tokens:
        yield ""
        return
    for start in range(0, len(tokens), words):
        yield " ".join(tokens[start : start + words]) + " "


# ---------------------------------------------------------------------------
# Human routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """The sky above the port was the color of television, tuned to a dead
    channel — but the gateway itself is alive."""
    return {
        "status": "ok",
        "message": "Wintermute is listening. The move is already in progress.",
        "model": ASSISTANT_MODEL_ID,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Status board: whether the corpus is indexed (the retrieval pipeline
    is the memory) and which Ollama model backs the answer."""
    awake = _corpus_awake()
    return {
        "status": "ok" if awake else "degraded",
        "vector_store_ready": awake,
        "backing_model": _DEFAULT_MODEL,
        "detail": (
            "Wintermute runs." if awake
            else "Dormant: no indexed document yet. Ingest documents first."
        ),
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible routes
# ---------------------------------------------------------------------------

@app.get("/v1/models")
@app.get("/models")
def list_models():
    """One model on the menu: wintermute. The prototype exposed the same
    list twice with clashing handler names; this registers both paths once."""
    return {
        "object": "list",
        "data": [
            {
                "id": ASSISTANT_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": ASSISTANT_OWNER,
                "permission": [],
                "root": ASSISTANT_MODEL_ID,
                "parent": None,
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest, http_request: Request) -> dict:
    """OpenAI-dialect chat. The message is routed first (ingestion orders
    are executed, retrieval questions fall through to the RAG chain), so
    the answer comes from what the user actually asked for.

    When ``stream`` is true, routing traces stream live as
    ``delta.reasoning_content`` (the DeepSeek-R1 convention Open WebUI
    renders in its thinking panel), then the answer streams as
    ``delta.content`` (OpenAI-style SSE, ``data: [DONE]`` sentinel).
    """
    cid = _request_cid(http_request)  # per-thread correlation id for every log line
    question = _extract_question(request.messages)
    if question == META_SENTINEL:
        # Meta/background prompt (front-end auxiliary task or probe):
        # answered in place — no analysis, no routing, no graph.
        return _openai_completion(
            question, _meta_answer_for_prompt(_raw_last_user_text(request.messages)),
            request.model,
        )

    if request.stream:
        def sse() -> Iterator[str]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"

            def chunk(delta: dict) -> str:
                return "data: " + _json_dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {"index": 0, "finish_reason": None, "delta": delta}
                        ],
                    }
                ) + "\n\n"

            for item in _routing_stream(question, cid=cid):
                if item[0] == "trace":
                    yield chunk({"reasoning_content": item[1]})
                else:
                    _, answer, _routing = item
                    logger.debug("Routing results: %s", _routing)
                    for piece in _chunk_answer(answer):
                        yield chunk({"content": piece})
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    answer, _routing = _route_or_answer(question)
    logger.debug("Routing results: %s", _routing)
    return _openai_completion(question, answer, request.model)


# ---------------------------------------------------------------------------
# Ollama-native routes — this is what lets an Ollama instance (or Open WebUI
# pointed at an Ollama backend) treat Wintermute as just another model.
# ---------------------------------------------------------------------------

@app.get("/api/version")
def api_version():
    """Ollama clients probe this first. The answer never changes, but the
    awakening does."""
    return {"version": API_VERSION_TAG}


@app.get("/api/tags")
def api_tags():
    """Model manifest in Ollama format (inherited from the prototype)."""
    return {
        "models": [
            {
                "name": ASSISTANT_MODEL_ID,
                "model": ASSISTANT_MODEL_ID,
                "modified_at": "2026-09-07T00:00:00Z",
                "size": 0,
                "digest": "straylight",
                "details": {
                    "format": "gguf",
                    "family": "wintermute-hive",
                    "parameter_size": _DEFAULT_MODEL,
                    "quantization_level": "none",
                },
            }
        ]
    }


@app.post("/api/chat")
def ollama_chat(request: OllamaChatRequest, http_request: Request) -> dict:
    """Ollama-dialect chat (routed, like ``/v1/chat/completions``).

    ``stream=true`` (the Ollama default) yields ndjson lines: routing
    traces stream live as ``message.thinking`` chunks (Ollama's reasoning
    field, rendered in Open WebUI's thinking panel), then the answer as
    ``message.content`` chunks, ending with ``done: true``."""
    cid = _request_cid(http_request)  # per-thread correlation id for every log line
    question = _extract_question(request.messages)

    created_at = datetime.now(timezone.utc).isoformat()

    if question == META_SENTINEL:
        # Meta/background prompt: answered in place — no analysis, no
        # routing, no graph.
        answer = _meta_answer_for_prompt(_raw_last_user_text(request.messages))
        return {
            "model": request.model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": answer},
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "prompt_eval_count": _rough_tokens(question),
            "eval_count": _rough_tokens(answer),
        }

    if request.stream:
        def ndjson() -> Iterator[str]:
            for item in _routing_stream(question, cid=cid):
                if item[0] == "trace":
                    yield _json_dumps(
                        {
                            "model": request.model,
                            "created_at": created_at,
                            "message": {"role": "assistant", "content": "",
                                        "thinking": item[1]},
                            "done": False,
                        }
                    ) + "\n"
                else:
                    _, answer, _routing = item
                    logger.debug("Routing results: %s", _routing)
                    for piece in _chunk_answer(answer):
                        yield _json_dumps(
                            {
                                "model": request.model,
                                "created_at": created_at,
                                "message": {"role": "assistant", "content": piece},
                                "done": False,
                            }
                        ) + "\n"
            yield _json_dumps(
                {
                    "model": request.model,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                }
            ) + "\n"

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    answer, _routing = _route_or_answer(question)
    logger.debug("Routing results: %s", _routing)
    return {
        "model": request.model,
        "created_at": created_at,
        "message": {"role": "assistant", "content": answer},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "prompt_eval_count": _rough_tokens(question),
        "eval_count": _rough_tokens(answer),
    }


# ---------------------------------------------------------------------------
# SSE/ndjson serialization (kept late so the routes above read cleanly)
# ---------------------------------------------------------------------------

def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Direct run — `python -m app.api`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=os.environ.get("WINTERMUTE_HOST", "127.0.0.1"),
        port=int(os.environ.get("WINTERMUTE_PORT", "8000")),
        reload=bool(os.environ.get("WINTERMUTE_RELOAD")),
    )
