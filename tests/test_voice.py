import base64
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from helper import engine, server, voice, welcome
from helper.engine import ModelError

HERE = Path(__file__).resolve().parent
VOICES = ("af_heart", "af_bella", "am_adam", "bf_emma", "bm_george", "jf_alpha", "zf_xiaobei", "ef_dora")


def add_kokoro(hub: Path, voices=VOICES, weights=True) -> Path:
    repo = hub / ("models--" + voice.REPO.replace("/", "--"))
    snapshot = repo / "snapshots" / "abc123"
    (snapshot / "voices").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text(json.dumps({"istftnet": {}, "n_token": 178}))
    if weights:
        (snapshot / voice.WEIGHTS).write_bytes(b"x" * 10)
    for name in voices:
        (snapshot / "voices" / f"{name}.safetensors").write_bytes(b"x")
        (snapshot / "voices" / f"{name}.pt").write_bytes(b"x")
    return snapshot


class VoiceBase(unittest.TestCase):
    worker = True

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.hub = root / "hub"
        self.hub.mkdir()
        self.home = root / "home"
        env = {key: value for key, value in os.environ.items()
               if key not in ("HIKARILM_VOICE_PYTHON", "HIKARILM_VOICE_WORKER")}
        env.update({"HF_HUB_CACHE": str(self.hub), "HIKARILM_HOME": str(self.home)})
        if self.worker:
            env.update({"HIKARILM_VOICE_PYTHON": sys.executable,
                        "HIKARILM_VOICE_WORKER": str(HERE / "fake_voice_worker.py")})
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(voice.unload)
        self.said = []


class FolderTest(VoiceBase):
    def test_not_downloaded(self):
        self.assertIsNone(voice.folder())
        self.assertEqual(voice.voices(), [])
        self.assertFalse(voice.status()["downloaded"])

    def test_only_english_voices_are_offered(self):
        add_kokoro(self.hub)
        self.assertEqual(voice.voices(), ["af_bella", "af_heart", "am_adam", "bf_emma", "bm_george"])

    def test_a_download_without_the_weights_or_voices_does_not_count(self):
        add_kokoro(self.hub, weights=False)
        self.assertIsNone(voice.folder())
        shutil.rmtree(self.hub)
        self.hub.mkdir()
        add_kokoro(self.hub, voices=("jf_alpha",))
        self.assertIsNone(voice.folder())

    def test_kokoro_is_not_a_chat_model(self):
        add_kokoro(self.hub)
        engine.forget_scan()
        self.addCleanup(engine.forget_scan)
        self.assertNotIn(voice.REPO, engine.names())

    def test_the_welcome_list_has_kokoro(self):
        listed = {model["repo"]: model for model in welcome.listing()}
        self.assertIn("Reading aloud", welcome.GROUPS)
        self.assertEqual(listed[voice.REPO]["group"], "Reading aloud")
        self.assertFalse(listed[voice.REPO]["have"])
        add_kokoro(self.hub)
        self.assertTrue({model["repo"]: model for model in welcome.listing()}[voice.REPO]["have"])


class PrepareTest(VoiceBase):
    worker = False

    def test_nothing_to_do_without_kokoro(self):
        with mock.patch.object(engine, "_prepare_venv") as made:
            self.assertFalse(voice.prepare(self.said.append))
        made.assert_not_called()
        self.assertEqual(self.said, [])

    def test_packages_have_no_torch_and_a_loadable_scipy(self):
        self.assertIn("misaki", voice.PACKAGES)
        self.assertNotIn("misaki[en]", voice.PACKAGES)
        self.assertIn("scipy>=1.16.3", voice.PACKAGES)
        self.assertTrue(any("en_core_web_sm" in package for package in voice.PACKAGES))
        self.assertFalse(any(" " in package for package in voice.PACKAGES))

    def test_misaki_needs_python_3_11_or_3_12(self):
        add_kokoro(self.hub)
        with mock.patch.object(engine, "find_python", return_value=None) as found, \
                mock.patch.object(engine, "_prepare_venv") as made:
            self.assertFalse(voice.prepare(self.said.append))
        found.assert_called_with(11, 12)
        made.assert_not_called()
        self.assertIn("brew install python@3.12", self.said[0])

    def test_find_python_keeps_to_the_range(self):
        with mock.patch.object(sys, "version_info", (3, 13, 1)), \
                mock.patch.object(shutil, "which", lambda name: f"/x/{name}" if name in ("python3.13", "python3.11") else None), \
                mock.patch.object(os, "access", lambda path, mode: path.startswith("/x/")):
            self.assertEqual(engine.find_python(11, 12), "/x/python3.11")
            self.assertEqual(engine.find_python(11, 14), sys.executable)

    def test_a_venv_on_python_3_13_is_made_again(self):
        add_kokoro(self.hub)
        venv = voice.venv()
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.13.2\n")
        seen = {}

        def made(where, interpreter, packages, what, say, base_python):
            seen.update(exists=where.exists(), base=base_python)
            return False

        with mock.patch.object(engine, "_prepare_venv", made):
            self.assertFalse(voice.prepare(self.said.append, "/x/python3.12"))
        self.assertEqual(seen, {"exists": False, "base": "/x/python3.12"})

    def test_kokoro_that_does_not_load_is_set_up_again(self):
        add_kokoro(self.hub)
        venv = voice.venv()
        (venv / "bin").mkdir(parents=True)
        (venv / engine.READY_MARK).write_text(" ".join(voice.PACKAGES) + "\n")
        fake = venv / "bin" / "python"
        fake.write_text("#!/bin/sh\necho \"OSError: [E050] Can't find model 'en_core_web_sm'\" >&2\nexit 1\n")
        fake.chmod(0o755)
        self.assertFalse(voice.prepare(self.said.append, sys.executable))
        self.assertIn("does not load", self.said[-1])
        self.assertIn("en_core_web_sm", self.said[-1])
        self.assertFalse((venv / engine.READY_MARK).exists())


class RequestTest(VoiceBase):
    def test_lines_are_tidied_and_checked(self):
        add_kokoro(self.hub)
        lines, name, speed = voice.check_request({"lines": ["  Hello\n world. ", "", "  "], "voice": "bf_emma", "speed": 1.4})
        self.assertEqual((lines, name, speed), (["Hello world."], "bf_emma", 1.4))
        for bad in ({}, {"lines": "hi"}, {"lines": [1]}, {"lines": ["", " "]}, {"lines": ["x" * (voice.TEXT_CHARS + 1)]}):
            with self.assertRaises(ValueError):
                voice.check_request(bad)

    def test_an_unknown_voice_or_speed_falls_back(self):
        add_kokoro(self.hub)
        self.assertEqual(voice.check_request({"lines": ["a"], "voice": "jf_alpha", "speed": 9})[1:], ("af_heart", 2.0))
        self.assertEqual(voice.check_request({"lines": ["a"], "speed": "fast"})[1:], ("af_heart", 1.0))
        self.assertEqual(voice.check_request({"lines": ["a"], "speed": float("nan")})[2], 1.0)
        self.assertEqual(voice.check_request({"lines": ["a"], "speed": 0.1})[2], 0.5)


class SpeakTest(VoiceBase):
    def test_not_downloaded_says_how_to_get_it(self):
        with self.assertRaises(ModelError) as caught:
            list(voice.speak(["hi"], "af_heart", 1.0))
        self.assertIn(voice.GET, str(caught.exception))

    def test_each_line_comes_back_as_pcm(self):
        add_kokoro(self.hub)
        parts = list(voice.speak(["Hello there.", "Bye."], "bf_emma", 1.2))
        self.assertEqual([part["line"] for part in parts], [0, 1])
        self.assertEqual(len(base64.b64decode(parts[0]["audio"])), 2 * 100 * len("Hello there."))
        self.assertEqual((parts[0]["rate"], parts[0]["voice"], parts[0]["speed"]), (24000, "bf_emma", 1.2))
        pid = voice._worker.process.pid
        list(voice.speak(["Again."], "af_heart", 1.0))
        self.assertEqual(voice._worker.process.pid, pid)
        self.assertEqual(voice.running()[0]["name"], voice.NAME)

    def test_closing_a_reading_stops_it_and_the_next_one_starts_clean(self):
        add_kokoro(self.hub)
        reading = voice.speak(["slow", "slow", "slow", "slow"], "af_heart", 1.0)
        self.assertEqual(next(reading)["line"], 0)
        reading.close()
        parts = list(voice.speak(["Next."], "af_heart", 1.0))
        self.assertEqual([(part["line"], len(base64.b64decode(part["audio"]))) for part in parts], [(0, 1000)])

    def test_a_new_reading_ends_the_one_under_way(self):
        add_kokoro(self.hub)
        first = []

        def read_first():
            first.extend(voice.speak(["slow"] * 6, "af_heart", 1.0))

        thread = threading.Thread(target=read_first)
        thread.start()
        time.sleep(0.7)
        second = list(voice.speak(["Second."], "af_heart", 1.0))
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertLess(len(first), 6)
        self.assertEqual(len(second), 1)

    def test_an_error_is_said_and_the_worker_goes_on(self):
        add_kokoro(self.hub)
        with self.assertRaises(ModelError) as caught:
            list(voice.speak(["ok", "fail"], "af_heart", 1.0))
        self.assertIn("could not read", str(caught.exception))
        self.assertEqual(len(list(voice.speak(["fine"], "af_heart", 1.0))), 1)

    def test_a_worker_that_dies_is_started_again(self):
        add_kokoro(self.hub)
        with self.assertRaises(ModelError):
            list(voice.speak(["crash"], "af_heart", 1.0))
        self.assertEqual(voice.running(), [])
        self.assertEqual(len(list(voice.speak(["back"], "af_heart", 1.0))), 1)


class ServerTest(VoiceBase):
    def setUp(self):
        super().setUp()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def post(self, payload: dict):
        request = urllib.request.Request(self.base + "/api/speak", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return 200, [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_voice_status(self):
        with urllib.request.urlopen(self.base + "/api/voice", timeout=30) as response:
            body = json.loads(response.read())
        self.assertFalse(body["downloaded"])
        self.assertEqual(body["get"], voice.GET)
        add_kokoro(self.hub)
        with urllib.request.urlopen(self.base + "/api/voice", timeout=30) as response:
            body = json.loads(response.read())
        self.assertTrue(body["downloaded"] and body["ready"])
        self.assertIn("af_heart", body["voices"])

    def test_speak_streams_each_line_then_done(self):
        add_kokoro(self.hub)
        code, lines = self.post({"lines": ["One.", "Two."], "voice": "am_adam", "speed": 1})
        self.assertEqual(code, 200)
        self.assertEqual([line.get("line") for line in lines[:-1]], [0, 1])
        self.assertEqual(lines[-1], {"done": True})
        self.assertEqual(lines[0]["voice"], "am_adam")

    def test_speak_refuses_nothing_to_read_and_says_when_kokoro_is_missing(self):
        code, body = self.post({"lines": [" "]})
        self.assertEqual(code, 400)
        code, body = self.post({"lines": ["Hi."]})
        self.assertEqual(code, 503)
        self.assertIn(voice.GET, body["error"])

    def test_an_error_partway_comes_as_a_line(self):
        add_kokoro(self.hub)
        code, lines = self.post({"lines": ["ok", "fail"]})
        self.assertEqual(code, 200)
        self.assertIn("could not read", lines[-1]["error"])

    def test_kokoro_is_never_the_large_model(self):
        add_kokoro(self.hub)
        self.post({"lines": ["Hi."]})
        with urllib.request.urlopen(self.base + "/api/state", timeout=30) as response:
            self.assertIsNone(json.loads(response.read())["large_model"])
