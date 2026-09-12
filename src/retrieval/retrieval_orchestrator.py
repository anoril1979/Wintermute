"""Retrieval orchestrator — the read-side entry point (deterministic).

    run_retrieval(requests)
        1. config gate   — retrieval.yaml validated at load; a malformed
           config ends gracefully with a ``config_error`` the calling LLM
           can relay to the user (fix the yaml);
        2. facts         — vector store state gathered once per prompt
           (read-only count probe);
        3. per request   — decision table (filters, top-k clamp, dormant
           gate, implemented gate) → the retrieval graph runs the lookup
           and returns the scored hits.

The requests come from the routing analyzer
(``src/routing/models.RetrievalRequest``): already classified, with
self-contained questions. There is NO retrieval-side LLM — the pipeline
is deterministic Python, the strict mirror of the ingestion tunnel.

The returned dict is a stable, plain-data contract for the task agent /
API layer. Statuses: ``ok`` (every request served), ``partial`` (at
least one served, others not), ``no_corpus``, ``not_implemented``,
``failed``, ``config_error``. ``requests`` carries one sub-result per
request (its own status, hits, explanation); ``hits`` is the flat list
for simple callers. ``traces`` carries the ordered events for the
thinking panel / the log file.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

from src.agents.contexts import RetrievalContext
from src.graphs.retrieval_graph import RetrievalGraph
from src.logging_setup import configure_logging
from src.retrieval.models import RetrievalBatch, RetrievalSpec
from src.retrieval.retrieval_router import (
    ROUTER_NO_CORPUS,
    RetrievalDecision,
    apply_decision_table,
    gather_facts,
)
from src.routing.models import RetrievalRequest
from src.tools.config_loader import ConfigError

logger = logging.getLogger(__name__)

# Top-level statuses (stable strings for callers/LLM).
STATUS_OK = "ok"
STATUS_PARTIAL = "partial"          # some requests served, others not
STATUS_NO_CORPUS = "no_corpus"
STATUS_NOT_IMPLEMENTED = "not_implemented"
STATUS_FAILED = "failed"
STATUS_CONFIG_ERROR = "config_error"


def run_retrieval(
    requests: List[RetrievalRequest],
    *,
    graph: Optional[RetrievalGraph] = None,
    on_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """Full retrieval flow for the lookup requests of one user prompt.

    Args:
        requests: the analyzer's retrieval requests (already classified,
            self-contained questions) — empty list is legitimate.
        graph: optional pre-built graph (tests); a default one is built
            otherwise.
        on_event: optional live observer of the trace events.

    Returns:
        A dict with a top-level ``status`` and per-request sub-results in
        ``requests`` (each with its own ``status``/``hits``). When at
        least one request was served, ``hits`` also carries the flat list
        of every served request's scored chunks. ``traces`` carries the
        ordered events. Never raises for expected failures: config
        problems, a dormant corpus and not-implemented request types are
        all reported as statuses.
    """
    # -- 1. config gate ---------------------------------------------------------
    from src.tools.config_loader import load_retrieval_config

    try:
        load_retrieval_config()
    except ConfigError as exc:
        logger.error("Retrieval configuration error: %s", exc)
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "requests": [],
            "hits": [],
            "traces": [],
        }

    traces: list = []

    def _record(event: Dict[str, Any]) -> None:
        traces.append(dict(event))
        if on_event is not None:
            try:
                on_event(event)
            except Exception as exc:  # noqa: BLE001 — tracing must not break
                logger.warning("Retrieval event observer failed: %s", exc)

    # -- 2. facts (once per prompt; gather_facts fails open internally) -----------
    facts = gather_facts()

    batch = RetrievalBatch.from_requests(list(requests or []))
    _record({
        "phase": "analysis",
        "kind": "understood",
        "message": (
            "%d lookup request(s): " % len(batch.specs)
            + "; ".join(f"[{s.kind.value}] {s.question}" for s in batch.specs)
        ),
        "data": {"requests": [s.summary() for s in batch.specs]},
    })

    # -- 3. graph, per classified request -----------------------------------------
    if graph is None:
        graph = RetrievalGraph()

    sub_results: List[Dict[str, Any]] = []
    flat_hits: List[Dict[str, Any]] = []

    for index, spec in enumerate(batch.specs, start=1):
        decision = apply_decision_table(facts, spec)
        sub = _solve_request(
            graph, decision, request_index=index, on_event=_record,
        )
        sub_results.append(sub)
        flat_hits.extend(sub.get("hits", []))

    served = [s for s in sub_results if s["status"] == STATUS_OK]
    if not sub_results:
        # Nothing asked, nothing served: an empty batch is a legitimate ok.
        status = STATUS_OK
    elif served and len(served) == len(sub_results):
        status = STATUS_OK
    elif served:
        status = STATUS_PARTIAL
    elif sub_results and all(
        s["status"] == STATUS_NO_CORPUS for s in sub_results
    ):
        status = STATUS_NO_CORPUS
    elif any(s["status"] == STATUS_NOT_IMPLEMENTED for s in sub_results):
        status = STATUS_NOT_IMPLEMENTED
    else:
        status = STATUS_FAILED

    return {
        "status": status,
        "message": (
            "%d/%d request(s) served" % (len(served), len(sub_results))
        ),
        "requests": sub_results,
        "hits": flat_hits,
        "traces": traces,
    }


def _solve_request(
    graph: RetrievalGraph,
    decision: RetrievalDecision,
    *,
    request_index: int,
    on_event: Any,
) -> Dict[str, Any]:
    """Solve ONE classified request through the graph.

    Same statuses as the batch contract (``ok``, ``no_corpus``,
    ``not_implemented``, ``failed``), scoped to the request; the caller
    aggregates them into the top-level status.
    """
    spec = decision.spec
    prefix = "[request %d] " % request_index

    def _emit(event_kind: str, message: str, **data: Any) -> None:
        on_event({
            "phase": "task",
            "kind": event_kind,
            "message": message,
            "data": {"request_index": request_index, **data},
        })

    _emit(
        "request_start",
        "%s[%s] %s" % (prefix, spec.kind.value, spec.question),
        lookup_kind=spec.kind.value,
    )

    if decision.status == ROUTER_NO_CORPUS:
        _emit(
            "memory_dormant",
            prefix + "no indexed document yet — retrieval stays dormant",
            chunk_count=0,
        )
        return {
            "status": STATUS_NO_CORPUS,
            "message": decision.explanation,
            "intent": spec.summary(),
            "hits": [],
        }

    if not decision.implemented:
        _emit(
            "retrieval_not_implemented",
            prefix + "request type '%s' is not served yet" % spec.kind.value,
            lookup_kind=spec.kind.value,
        )
        return {
            "status": STATUS_NOT_IMPLEMENTED,
            "message": decision.explanation,
            "intent": spec.summary(),
            "hits": [],
        }

    context = RetrievalContext(
        question=spec.question,
        on_event=on_event,
    )
    context.metadata.update(
        {
            "filters": decision.filters,
            "top_k": decision.top_k,
            "intent": spec.summary(),
        }
    )

    outcome = graph.run(context, kind=spec.kind.value)

    step_outcomes = outcome.as_list()
    # The LOOKUP step (semantic_search, ...) decides whether the request
    # was served; the answer step only phrases its results. A missing or
    # failed answerer never voids the search (skipped/failed answer →
    # the caller degrades to the raw hits).
    lookup = next(
        (s for s in step_outcomes if s.get("step") != "answer"),
        None,
    )
    served = bool(lookup) and lookup.get("status") == "ok"
    if not served:
        # The lookup failed or is not implemented (no registry entry):
        # report honestly with the step report for the caller.
        last = step_outcomes[-1] if step_outcomes else {}
        status = (
            STATUS_NOT_IMPLEMENTED
            if last.get("status") == "not_implemented"
            else STATUS_FAILED
        )
        message = str(last.get("detail", "retrieval step failed"))
        _emit("request_failed", prefix + message, status=status)
        return {
            "status": status,
            "message": message,
            "intent": spec.summary(),
            "steps": step_outcomes,
            "hits": [],
            "answer": "",
        }

    hits = context.outputs.get("hits", [])
    hit_dicts = [
        {
            "id": hit.id,
            "text": hit.text,
            "score": hit.score,
            "metadata": dict(hit.metadata),
        }
        for hit in hits
    ]
    # The answer step (when registered) phrases the hits into the final
    # user-facing reply; the search results stay carried alongside so a
    # phrasing failure never voids them.
    answer = context.outputs.get("answer") or ""
    _emit(
        "request_done",
        "%s%d chunk(s) retrieved" % (prefix, len(hit_dicts))
        + (" and answered" if answer else ""),
        hits=len(hit_dicts),
        answered=bool(answer),
    )
    return {
        "status": STATUS_OK,
        "intent": spec.summary(),
        "filters": decision.filters.summary() if decision.filters else {},
        "hits": hit_dicts,
        "answer": answer,
        "steps": step_outcomes,
    }


# ---------------------------------------------------------------------------
# CLI (direct invocation) — one request, semantic, from the raw question
# ---------------------------------------------------------------------------

def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Wintermute retrieval — ask the ingested corpus a question."
    )
    parser.add_argument("question", help="the question to answer from the corpus")
    args = parser.parse_args()

    configure_logging()
    # CLI mode is a trusted local invocation: one semantic request built
    # deterministically from the raw argument (no LLM analysis involved).
    request = RetrievalRequest(question=args.question)
    result = run_retrieval([request])
    print(f"status: {result['status']}")
    for sub in result.get("requests", []):
        intent = sub.get("intent", {})
        print(f"  [{sub['status']}] [{intent.get('kind', '?')}] {intent.get('question', '?')}")
        if sub.get("answer"):
            print(f"    answer: {sub['answer']}")
        for hit in sub.get("hits", []):
            meta = hit.get("metadata", {})
            print(
                f"    [{hit.get('score', 0):.3f}] {meta.get('doc_title', '?')} "
                f"p.{meta.get('page_number', '?')} :: {hit.get('text', '')[:120]}"
            )
    return 0 if result["status"] in (STATUS_OK, STATUS_PARTIAL, STATUS_NO_CORPUS) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
