import os
import unittest


class TestSearchDocumentsIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Ensure env doesn't require real keys during test
        os.environ.setdefault("TOP_K", "2")
        os.environ.setdefault("USE_FUSION", "false")
        os.environ.setdefault("USE_HYDE", "false")
        os.environ.setdefault("USE_RERANK", "false")
        # Use built-in MockLLM by setting llm=None
        from llama_index.core import Settings

        Settings.llm = None  # triggers MockLLM in resolver

    async def test_search_documents_smoke(self):
        # Import after settings are in place
        import rag_engine
        # Monkeypatch: use BM25-only retriever to avoid embeddings/vector DB
        from llama_index.retrievers.bm25 import BM25Retriever
        nodes = rag_engine._load_nodes_for_bm25()
        rag_engine.build_retriever = lambda filters=None: BM25Retriever.from_defaults(  # type: ignore[attr-defined]
            nodes=nodes, similarity_top_k=2
        )

        search_documents = rag_engine.search_documents

        out = await search_documents("dummy query")
        self.assertIsInstance(out, str)
        self.assertIn("answer", out)


if __name__ == "__main__":
    unittest.main()
