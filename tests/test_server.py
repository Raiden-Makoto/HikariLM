import builtins
import json
import os
import symtable
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from helper import engine, server
from tests.test_engine import CODER, GEMMA, add_model, real_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


class NamesTest(unittest.TestCase):
    """A name used but never defined or imported only fails when that line runs, so a
    removed import (KEEP_ALIVE in 4.0.0) broke every chat while every test passed."""

    def test_every_global_name_is_defined(self):
        missing = []
        for path in sorted((ROOT / "helper").glob("*.py")) + [ROOT / "start.py"]:
            table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
            defined = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned()
                       or symbol.is_imported()} | set(dir(builtins)) | {"__file__", "__name__"}

            def walk(scope):
                for child in scope.get_children():
                    for symbol in child.get_symbols():
                        if symbol.is_referenced() and (symbol.is_global() or symbol.is_declared_global()) \
                                and symbol.get_name() not in defined:
                            missing.append(f"{path.name}: {child.get_name()} uses {symbol.get_name()}")
                    walk(child)

            walk(table)
            for symbol in table.get_symbols():
                if symbol.is_referenced() and not symbol.is_assigned() and not symbol.is_imported() \
                        and symbol.get_name() not in defined:
                    missing.append(f"{path.name}: uses {symbol.get_name()}")
        self.assertEqual(missing, [])


class ChatThroughServerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        hub = root / "hub"
        hub.mkdir()
        env = {"HF_HUB_CACHE": str(hub), "HIKARILM_HOME": str(root / "home"),
               "HIKARILM_MLX_PYTHON": sys.executable, "HIKARILM_MLX_WORKER": str(HERE / "fake_mlx_worker.py")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (CODER, GEMMA):
            add_model(hub, name, real_config(name))
        engine.forget_scan()
        self.addCleanup(engine.forget_scan)
        self.addCleanup(engine.unload)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def post(self, path: str, payload: dict) -> list[dict]:
        request = urllib.request.Request(self.base + path, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
        except urllib.error.HTTPError as error:
            self.fail(f"{path} answered {error.code}: {error.read().decode()}")

    def test_a_chat_reply_streams_to_the_end(self):
        lines = self.post("/api/chat", {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual([line for line in lines if line.get("error")], [])
        self.assertTrue(lines[-1].get("done"))
        self.assertTrue("".join((line.get("message") or {}).get("content", "") for line in lines))

    def test_the_welcome_list_says_what_is_downloaded(self):
        with urllib.request.urlopen(self.base + "/api/welcome", timeout=30) as response:
            body = json.loads(response.read())
        have = {model["repo"] for model in body["models"] if model["have"]}
        self.assertEqual(have, {CODER, GEMMA})
        self.assertEqual(body["chat"], 2)
        qwen = next(model for model in body["models"] if model["name"] == "Qwen3 8B")
        self.assertEqual(qwen["command"], "~/.hikarilm/venv/bin/hf download mlx-community/Qwen3-8B-4bit")

    def test_a_picture_is_described(self):
        lines = self.post("/api/describe", {"model": GEMMA, "image": PNG_B64, "name": "dot.png"})
        self.assertEqual([line for line in lines if line.get("error")], [])
        self.assertIsInstance(lines[-1].get("description"), str)


if __name__ == "__main__":
    unittest.main()
