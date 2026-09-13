"""Unit tests for source attribution / citation display (Phase 11)."""
from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from citations import format_citation  # noqa: E402
from rag_service import SourceInfo, _build_sources  # noqa: E402
from schemas import SourceItem  # noqa: E402


def test_citation_with_page_number():
    src = SourceInfo(
        id="pdf_chunk_1",
        distance=0.5,
        preview="stroma photosystem",
        source="photosynthesis_overview.txt",
        document="Photosynthesis Notes",
        page=4,
    )
    assert format_citation(src) == "Source: Photosynthesis Notes, Page 4"


def test_citation_without_page_number_gracefully_falls_back():
    src = SourceInfo(
        id="old_chunk_2",
        distance=0.5,
        preview="stroma photosystem",
        source="photosynthesis_overview.txt",
        document="Photosynthesis Notes",
    )
    label = format_citation(src)
    assert label == "Source: Photosynthesis Notes"
    assert "None" not in label
    assert "Page" not in label


def test_citation_with_no_document_or_page_has_no_broken_text():
    src = SourceInfo(id="legacy_chunk_3", distance=0.7, preview="chlorophyll")
    label = format_citation(src)
    assert "None" not in label
    assert "Page" not in label
    assert label.startswith("Source:")


def test_citation_accepts_source_item_from_schemas():
    item = SourceItem(
        id="pdf_chunk_1",
        distance=0.5,
        preview="stroma",
        source="photosynthesis_overview.txt",
        document="Photosynthesis Notes",
        page=4,
    )
    assert format_citation(item) == "Source: Photosynthesis Notes, Page 4"


def test_build_sources_reads_page_from_mocked_chunk_metadata():
    results = {
        "documents": [
            "Photosystem II sits in the thylakoid membrane.",
            "The Calvin cycle runs in the stroma.",
        ],
        "ids": ["pdf_chunk_0", "pdf_chunk_1"],
        "distances": [0.4, 0.5],
        "metadatas": [
            {"source": "notes.txt", "document": "Photosynthesis Notes", "page": 2},
            {"source": "notes.txt", "document": "Photosynthesis Notes"},
        ],
    }
    sources = _build_sources(results)
    assert len(sources) == 2
    assert sources[0].page == 2
    assert format_citation(sources[0]) == "Source: Photosynthesis Notes, Page 2"
    assert sources[1].page is None
    assert format_citation(sources[1]) == "Source: Photosynthesis Notes"


def test_cache_round_trip_preserves_page_from_source_info():
    import json

    from cache import CacheEntry, ask_result_to_cache_entry
    from rag_service import AskResult

    result = AskResult(
        answer="The Calvin cycle runs in the stroma.",
        refused=False,
        top_k=4,
        sources=[
            SourceInfo(
                id="pdf_chunk_0",
                distance=0.4,
                preview="stroma",
                source="notes.txt",
                document="Photosynthesis Notes",
                page=7,
            )
        ],
    )
    entry = ask_result_to_cache_entry(result)
    raw = entry.to_json()
    assert json.loads(raw)["sources"][0]["page"] == 7
    restored = CacheEntry.from_json(raw)
    assert restored.sources[0]["page"] == 7


def test_cache_round_trip_preserves_page_in_nondataclass_fallback():
    """Covers the non-dataclass fallback branch in ask_result_to_cache_entry."""
    from cache import CacheEntry, ask_result_to_cache_entry
    from rag_service import AskResult

    class LegacySource:
        id = "pdf_chunk_1"
        preview = "thylakoid"
        source = "notes.txt"
        document = "Photosynthesis Notes"
        page = 4

    result = AskResult(
        answer="Photosystem II sits in the thylakoid membrane.",
        refused=False,
        top_k=4,
        sources=[LegacySource()],
    )
    restored = CacheEntry.from_json(ask_result_to_cache_entry(result).to_json())
    assert restored.sources[0]["page"] == 4


def test_old_cache_entry_without_page_loads_and_defaults_to_none():
    """Backward compat: cached entries written before the page field was added."""
    import json

    from cache import CacheEntry
    from main import _cache_entry_to_result

    old_json = json.dumps(
        {
            "answer": "old answer",
            "refused": False,
            "top_k": 4,
            "grounded": True,
            "source_ids": ["pdf_chunk_0"],
            "sources": [
                {
                    "id": "pdf_chunk_0",
                    "distance": 0.5,
                    "preview": "stroma",
                    "source": "notes.txt",
                    "document": "Photosynthesis Notes",
                }
            ],
        }
    )
    entry = CacheEntry.from_json(old_json)
    assert "page" not in entry.sources[0]
    result = _cache_entry_to_result(entry, top_k=4)
    assert result.sources[0].page is None


def test_old_source_item_without_page_defaults_to_none():
    """Backward compat: API responses serialized before the page field existed."""
    item = SourceItem.model_validate(
        {
            "id": "pdf_chunk_0",
            "distance": 0.5,
            "preview": "stroma",
            "source": "notes.txt",
            "document": "Photosynthesis Notes",
        }
    )
    assert item.page is None
    assert "page" not in item.model_dump(exclude_none=True)