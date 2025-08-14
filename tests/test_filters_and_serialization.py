import os
import json
import asyncio
import types

# Light import mode disables heavy index/DB initialization
os.environ["APP_LIGHT_IMPORT"] = "1"

import importlib
app = importlib.import_module("app")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_make_filters_builds_exact_filters():
    filters = app._make_filters(file_name="a.pdf", section="s1", lang="ru", version="v1")
    keys = [f.key for f in filters]
    vals = [f.value for f in filters]
    assert keys == ["file_name", "section", "lang", "version"]
    assert vals == ["a.pdf", "s1", "ru", "v1"]


def test_build_sources_serializes_response_nodes():
    # Fake response with source_nodes list
    class Node:
        def __init__(self):
            self.metadata = {"file_name": "a.pdf", "section": "s1", "lang": "ru", "version": "v1", "window": None}
        def get_content(self, metadata_mode=None):
            return "hello"

    class Source:
        def __init__(self):
            self.node = Node()
            self.score = 0.42

    class Resp:
        def __init__(self):
            self.source_nodes = [Source()]
        def __str__(self):
            return "ok"

    sources = app._build_sources(Resp())
    assert isinstance(sources, list) and len(sources) == 1
    s = sources[0]
    assert s["file_name"] == "a.pdf"
    assert s["section"] == "s1"
    assert s["lang"] == "ru"
    assert s["version"] == "v1"
    assert s["text"] == "hello"
    assert isinstance(s["score"], float)


def test_compute_rerank_top_n_bounds():
    assert app._compute_rerank_top_n(15, True) == 12
    assert app._compute_rerank_top_n(5, True) == 5
    assert app._compute_rerank_top_n(0, True) == 1  # min guard
    assert app._compute_rerank_top_n(10, False) is None
