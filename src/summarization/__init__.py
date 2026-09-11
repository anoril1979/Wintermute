"""Summarization layer: everything owned by the summarization process.

Currently:

* ``summarized_store`` — the LLM-summaries persistence layer
  (``save_summarized`` / ``load_summarized``): stores the fully-summarized
  ``DocumentExtract`` under ``summarization_output_dir`` with a content
  fingerprint, so a re-ingestion resumes from it with zero LLM calls and
  never serves stale summaries after the content changed.
"""

from src.summarization.summarized_store import (
    SummarizedJsonError,
    content_fingerprint,
    load_summarized,
    save_summarized,
    summarized_document_from_json_dict,
    summarized_document_to_json_dict,
    summarized_path_for,
    summarization_output_dir,
)

__all__ = [
    "SummarizedJsonError",
    "content_fingerprint",
    "load_summarized",
    "save_summarized",
    "summarized_document_from_json_dict",
    "summarized_document_to_json_dict",
    "summarized_path_for",
    "summarization_output_dir",
]
