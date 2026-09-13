"""Human-readable citation formatting for retrieved sources."""

from __future__ import annotations


def format_citation(source) -> str:
    """
    Render a retrieved chunk source as a citation label.

    Returns ``Source: <document>`` or ``Source: <document>, Page N`` when a
    page number is present on the source. Falls back gracefully when the
    document name or page is missing (e.g. chunks ingested before Team
    Lambda's ingestion fix ships), never emitting "Page None" or broken text.

    Accepts ``rag_service.SourceInfo``, ``schemas.SourceItem``, or any object
    exposing optional ``document``/``source``/``page`` attributes.
    """
    document_name = (
        getattr(source, "document", "") or getattr(source, "source", "") or ""
    ).strip()
    label = f"Source: {document_name}" if document_name else "Source: (unknown document)"
    page = getattr(source, "page", None)
    if page is not None and str(page).strip() != "":
        label = f"{label}, Page {page}"
    return label