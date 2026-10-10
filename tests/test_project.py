"""Offline regression checks. No real API keys or external API requests are needed."""

import asyncio
import importlib
import io
import json
import os
import pkgutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

import bank_rag
from bank_rag.config import INDEX_FILE, META_FILE, ROOT_DIR
from bank_rag.pipeline import merge_chunks


class ProjectChecks(unittest.TestCase):
    def test_modules_import_without_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_AGENTS_DISABLE_TRACING": "1"}):
            for module in pkgutil.walk_packages(bank_rag.__path__, bank_rag.__name__ + "."):
                importlib.import_module(module.name)

    def test_index_metadata_alignment_and_search(self):
        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.assertGreater(index.ntotal, 0)
        self.assertEqual(index.ntotal, len(metadata))
        self.assertEqual(index.d, 3072)
        self.assertTrue(all(item.get("text") for item in metadata))
        query = np.asarray(index.reconstruct(0), dtype="float32").reshape(1, -1)
        distances, indices = index.search(query, 1)
        self.assertAlmostEqual(float(distances[0][0]), 0.0, places=5)
        self.assertGreaterEqual(int(indices[0][0]), 0)

    def test_artifact_paths_survive_working_directory_change(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertTrue(INDEX_FILE.is_file())
                self.assertTrue(META_FILE.is_file())
            finally:
                os.chdir(original_cwd)

    def test_merge_does_not_reingest_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "2023_chunks.json").write_text(json.dumps([{"year": 2023, "text": "a"}]))
            (folder / "2024_chunks.json").write_text(json.dumps([{"year": 2024, "text": "b"}]))
            with patch.object(merge_chunks, "CHUNKS_DIR", folder), redirect_stdout(io.StringIO()):
                merge_chunks.main()
                first = (folder / "all_chunks.json").read_bytes()
                merge_chunks.main()
                second = (folder / "all_chunks.json").read_bytes()
            self.assertEqual(first, second)
            self.assertEqual(len(json.loads(second)), 2)

    def test_malformed_agent_plan_is_handled(self):
        from bank_rag.agents.orchestrator import safe_parse

        self.assertEqual(safe_parse("[]"), {})
        self.assertEqual(safe_parse("not json"), {})
        self.assertEqual(safe_parse('{"query_type": "numeric"}'), {"query_type": "numeric"})

    def test_missing_faiss_neighbors_are_not_used_as_negative_indices(self):
        from bank_rag.engine import rag_engine

        class SmallIndex:
            def search(self, query, count):
                return np.array([[0.0, 1e9]]), np.array([[0, -1]])

        metadata = [{"year": 2024, "text": "one actual result"}]
        with patch.object(rag_engine, "load_index", return_value=(SmallIndex(), metadata)), patch.object(
            rag_engine, "embed_text", return_value=np.zeros((1, 3072), dtype="float32")
        ):
            results = rag_engine.retrieve_chunks("question", top_k=2)
        self.assertEqual(len(results), 1)

    def test_unknown_numeric_pattern_returns_no_match(self):
        from bank_rag.engine.rag_engine import extract_verified_number

        self.assertEqual(extract_verified_number("unknown_metric", [{"text": "example"}]), (None, None))

    def test_ui_without_api_key_shows_setup_message(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_AGENTS_DISABLE_TRACING": "1"}):
            app = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("API key" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
