import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helper import documents, engine, models, notes
from helper.engine import ModelError

HERE = Path(__file__).resolve().parent
CODER = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def add_nomic(hub: Path, files=engine.EMBED_FILES) -> Path:
    repo = hub / ("models--" + engine.EMBED_REPO.replace("/", "--"))
    snapshot = repo / "snapshots" / "def456"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("def456")
    for name in files:
        (snapshot / name).write_text("{}" if name.endswith(".json") else "x")
    return snapshot


class EmbedWorkerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.hub = root / "hub"
        self.hub.mkdir()
        env = {"HF_HUB_CACHE": str(self.hub), "HIKARILM_HOME": str(root / "home"),
               "HIKARILM_MLX_PYTHON": sys.executable, "HIKARILM_MLX_WORKER": str(HERE / "fake_mlx_worker.py")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        engine.forget_scan()
        self.addCleanup(engine.unload_embed)
        self.addCleanup(engine.unload)
        self.addCleanup(engine.forget_scan)

    def test_not_downloaded_says_how_to_get_it(self):
        with self.assertRaises(ModelError) as caught:
            engine.embed(["search_query: hello"])
        self.assertIn("./start.sh", str(caught.exception))

    def test_a_partial_download_is_not_used(self):
        add_nomic(self.hub, files=("config.json", "vocab.txt"))
        self.assertIsNone(engine.embed_folder())

    def test_vectors_come_back_unit_length_from_one_worker(self):
        add_nomic(self.hub)
        first = engine.embed(["search_document: avl tree rotation", "search_document: pasta"])
        self.assertEqual(len(first), 2)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in first[0])), 1.0, places=5)
        pid = engine._embedder.process.pid
        engine.embed(["search_query: tree"])
        self.assertEqual(engine._embedder.process.pid, pid)

    def test_nomic_is_not_a_chat_model(self):
        add_nomic(self.hub)
        engine.forget_scan()
        self.assertNotIn(engine.EMBED_REPO, engine.names())

    def test_it_sits_beside_the_chat_model(self):
        add_nomic(self.hub)
        snapshot = self.hub / ("models--" + CODER.replace("/", "--")) / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot.parent.parent / "refs").mkdir()
        (snapshot.parent.parent / "refs" / "main").write_text("abc123")
        (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen2", "quantization": {"bits": 4}}))
        (snapshot / "model.safetensors").write_bytes(b"x" * 100)
        engine.forget_scan()
        engine.embed(["search_query: tree"])
        reply = engine.open_chat(CODER, [{"role": "user", "content": "hi"}])
        while reply.readline():
            pass
        reply.close()
        self.assertEqual(sorted(item["name"] for item in models.running()), sorted([CODER, models.EMBED_NAME]))
        engine.embed(["search_query: still here"])
        models.unload(models.EMBED_NAME)
        self.assertEqual([item["name"] for item in models.running()], [CODER])

    def test_a_crash_is_an_error_and_the_next_call_starts_again(self):
        add_nomic(self.hub)
        engine.embed(["search_query: first"])
        with self.assertRaises(ModelError):
            engine.embed(["crash"])
        self.assertEqual(len(engine.embed(["search_query: again"])), 1)

    def test_the_keepalive_sees_it(self):
        add_nomic(self.hub)
        models.embed(["search_query: tree"])
        self.assertIsNotNone(models.KEEPER.leaves_at(models.EMBED_NAME))
        models.KEEPER.forget(models.EMBED_NAME)

    def test_prepare_does_not_download_with_another_python(self):
        said = []
        self.assertFalse(engine.prepare_embed(said.append))
        self.assertEqual(said, [])


class StampTest(unittest.TestCase):
    """Vectors from Ollama's nomic are made again once with the MLX one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"HIKARILM_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def test_old_notes_are_embedded_again_and_skipped_until_then(self):
        notes.index_dir().mkdir(parents=True)
        old = {"source": "chat-a", "title": "Old", "chunks": [
            {"hash": notes._hash("avl tree"), "text": "avl tree", "vector": [1.0, 0.0]}]}
        (notes.index_dir() / "chat-a.json").write_text(json.dumps(old))
        self.assertEqual(notes.search("avl tree", self.embed, min_score=0.1), [])
        self.assertEqual(notes.sync("chat-a", "Old", ["avl tree"], self.embed), 1)
        self.assertEqual(notes.sync("chat-a", "Old", ["avl tree"], self.embed), 0)
        self.assertEqual([n["title"] for n in notes.search("avl tree", self.embed, min_score=0.1)], ["Old"])

    def test_old_document_vectors_are_made_again(self):
        doc = {"name": "paper.txt", "text": "one\n" * 10, "hash": "a" * 40, "message": 0}
        data = documents.load(doc)
        data["vectors"] = [[0.0, 1.0] for _ in data["chunks"]]
        documents.save(doc, data)
        documents.excerpts("question", [doc], 10_000, self.embed)
        self.assertEqual(self.calls[0], ["search_document: " + chunk["text"] for chunk in data["chunks"]])
        self.calls.clear()
        documents.excerpts("question", [doc], 10_000, self.embed)
        self.assertEqual(self.calls, [["search_query: question"]])


if __name__ == "__main__":
    unittest.main()
