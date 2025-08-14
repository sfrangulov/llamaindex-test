import os
import asyncio

# Light import mode disables heavy index/DB initialization
os.environ["APP_LIGHT_IMPORT"] = "1"

import importlib
app = importlib.import_module("app")


def test_build_engine_with_filters_returns_engine_or_none():
    # In LIGHT_IMPORT, query_engine is None and build_retriever returns None
    engine = app._make_engine_with_filters([])
    assert engine is None or hasattr(engine, "aquery")


def test_fallback_engine_builds_without_error():
    # Should return a RetrieverQueryEngine or None in LIGHT_IMPORT
    try:
        eng = app._make_fallback_engine([])
        assert eng is None or hasattr(eng, "aquery")
    except Exception as e:
        # No heavy initialization should occur in light mode
        assert False, f"fallback engine raised: {e}"
