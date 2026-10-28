import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from helper import imagegen, sdserver, server


class BugReachesThePageTest(unittest.TestCase):
    """A bug in the helper used to close the connection, and the page said only "Failed to fetch"."""

    def setUp(self):
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def test_bug_before_the_reply_is_a_json_error(self):
        with mock.patch.object(imagegen, "clean_request", side_effect=RuntimeError("boom")), \
                mock.patch("traceback.print_exc"):
            request = urllib.request.Request(self.base + "/api/generate", b'{"model": "x", "prompt": "a fox"}',
                                             {"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 500)
        body = json.loads(caught.exception.read())
        self.assertIn("The helper hit a bug: RuntimeError: boom", body["error"])


class KleinSetupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name) / "my home"
        patcher = mock.patch.dict(os.environ, {"LIGHTUSAGI_HOME": str(self.home)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_line_reads_back_and_files_are_ticked(self):
        setup = imagegen.klein_setup()
        self.assertTrue((self.home / "models").is_dir())
        self.assertEqual([f["found"] for f in setup["files"]], [False, False, False])
        (self.home / "models" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
        self.assertEqual([f["found"] for f in imagegen.klein_setup()["files"]], [False, False, True])
        models, problems = sdserver.parse_models(setup["line"])
        self.assertEqual(problems, [])
        self.assertEqual(models[0]["name"], "flux2-klein")
        args = models[0]["args"]
        self.assertEqual(args[args.index("--llm") + 1], str(self.home / "models" / "Qwen3-4B-Q4_K_M.gguf"))
        self.assertIn("--steps", args)

    def test_sd_server_found_where_the_steps_say(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertIsNone(sdserver.find_program(""))
            inside = self.home / "sd-server" / "sd-master-bin"
            inside.mkdir(parents=True)
            (inside / "sd-server").write_bytes(b"#!/bin/sh\n")
            found = sdserver.find_program("")
        self.assertEqual(found, str(inside / "sd-server"))
        self.assertTrue(os.access(found, os.X_OK))


if __name__ == "__main__":
    unittest.main()
