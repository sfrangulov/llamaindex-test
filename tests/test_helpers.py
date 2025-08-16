import importlib
import os
import sys
import unittest


class TestHelpers(unittest.TestCase):
    def setUp(self):
        # Ensure we can import app.py from repo root
        self.app = None
        try:
            root = os.path.dirname(os.path.dirname(__file__))
            if root not in sys.path:
                sys.path.insert(0, root)
            self.app = importlib.import_module("app")
        except Exception as e:
            self.skipTest(f"app import failed (likely missing deps): {e}")

    def test_compute_rerank_top_n_bounds(self):
        app = self.app
        # When disabled, returns None
        self.assertIsNone(app._compute_rerank_top_n(10, False))
        # Lower bound
        self.assertEqual(app._compute_rerank_top_n(0, True), 1)
        # Upper cap at 12
        self.assertEqual(app._compute_rerank_top_n(100, True), 12)

    def test_make_filters_builds_exact_matches(self):
        app = self.app
        filters = app._make_filters(
            file_name="a.pdf", section="s1", lang="ru", version="v1"
        )
        keys = [f.key for f in filters]
        values = [f.value for f in filters]
        self.assertEqual(keys, ["file_name", "section", "lang", "version"])
        self.assertEqual(values, ["a.pdf", "s1", "ru", "v1"])

    def test_build_node_postprocessors_runs(self):
        app = self.app
        pps = app._build_node_postprocessors()
        self.assertIsInstance(pps, list)
        self.assertTrue(pps)  # non-empty


if __name__ == "__main__":
    unittest.main()
