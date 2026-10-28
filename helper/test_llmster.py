import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from helper import llmster, models, ollama

GIB = 1024 ** 3


class FakeLlmster:
    """Just enough of llmster's /api/v1 and /v1/chat/completions to test against."""

    def __init__(self, port=0):
        self.loaded: dict[str, int] = {}     # instance id -> context length
        self.calls: list[tuple[str, dict]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def _json(self, body, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path != "/api/v1/models":
                    return self._json({"error": "no"}, 404)
                self._json({"models": [
                    {"type": "llm", "key": "falcon3-10b-instruct-1.58bit", "architecture": "llama",
                     "size_bytes": 3 * GIB, "max_context_length": 32768, "capabilities": {"vision": False},
                     "loaded_instances": [{"id": key, "config": {"context_length": ctx}}
                                          for key, ctx in fake.loaded.items()]},
                    {"type": "embedding", "key": "text-embedding-nomic", "size_bytes": GIB // 4, "loaded_instances": []},
                ]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append((self.path, body))
                if self.path == "/api/v1/models/load":
                    fake.loaded[body["model"]] = body.get("context_length", 4096)
                    return self._json({"type": "llm", "instance_id": body["model"], "status": "loaded"})
                if self.path == "/api/v1/models/unload":
                    fake.loaded.pop(body["instance_id"], None)
                    return self._json({"instance_id": body["instance_id"]})
                if self.path == "/v1/chat/completions":
                    if body["model"] not in fake.loaded:
                        return self._json({"error": {"message": "model not loaded"}}, 400)
                    if not body.get("stream"):
                        return self._json({"choices": [{"message": {"role": "assistant", "content": "<think>x</think>notes"}}]})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for delta in ({"role": "assistant"}, {"reasoning_content": "Let me think. "},
                                  {"content": "Hello"}, {"content": " there"}):
                        self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode() + b"\n\n")
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n\n")
                    self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    return None
                return self._json({"error": "no"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class LlmsterTest(unittest.TestCase):
    MODEL = "lms:falcon3-10b-instruct-1.58bit"

    def setUp(self):
        self.fake = FakeLlmster()
        self.addCleanup(self.fake.stop)
        for name, value in (("HOST", self.fake.host), ("_absent_until", 0.0), ("_cached", None)):
            patcher = mock.patch.object(llmster, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_lists_chat_models_only(self):
        self.assertEqual(llmster.names(), [self.MODEL])
        self.assertEqual(llmster.sizes()[self.MODEL], 3 * GIB)
        self.assertEqual(llmster.running(), [])

    def test_show_has_ollamas_shape(self):
        shown = llmster.show(self.MODEL)
        self.assertEqual(shown["model_info"]["llama.context_length"], 32768)
        self.assertNotIn("thinking", shown["capabilities"])

    def test_loads_with_the_context_and_reloads_when_it_changes(self):
        self.assertEqual(llmster.ensure_loaded(self.MODEL, 8192), "falcon3-10b-instruct-1.58bit")
        self.assertEqual(self.fake.loaded, {"falcon3-10b-instruct-1.58bit": 8192})
        llmster.ensure_loaded(self.MODEL, 8192)
        self.assertEqual(sum(path.endswith("/load") for path, _ in self.fake.calls), 1)
        llmster.ensure_loaded(self.MODEL, 4096)
        self.assertEqual(self.fake.loaded, {"falcon3-10b-instruct-1.58bit": 4096})
        self.assertEqual([path.rsplit("/", 1)[1] for path, _ in self.fake.calls], ["load", "unload", "load"])
        self.assertEqual(llmster.running()[0]["name"], self.MODEL)

    def test_stream_reads_as_ollama_lines(self):
        stream = llmster.open_chat(self.MODEL, [{"role": "user", "content": "hi", "images": ["iVBORw0KGgo="]}], 4096)
        events = []
        while True:
            line = stream.readline()
            if not line:
                break
            events.append(json.loads(line))
        stream.close()
        self.assertEqual([e["message"]["thinking"] for e in events[:1]], ["Let me think. "])
        self.assertEqual("".join(e["message"]["content"] for e in events), "Hello there")
        self.assertTrue(events[-1]["done"])
        self.assertEqual((events[-1]["prompt_eval_count"], events[-1]["eval_count"]), (12, 5))
        sent = next(body for path, body in self.fake.calls if path == "/v1/chat/completions")
        self.assertEqual(sent["messages"][0]["content"][1]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgo=")

    def test_complete_and_unload(self):
        self.assertEqual(llmster.complete(self.MODEL, [{"role": "user", "content": "x"}], 4096, 100),
                         "<think>x</think>notes")
        sent = next(body for path, body in self.fake.calls if path == "/v1/chat/completions")
        self.assertEqual(sent["max_tokens"], 100)
        models.unload(self.MODEL)
        self.assertEqual(self.fake.loaded, {})

    def test_unknown_model(self):
        with self.assertRaises(ollama.OllamaError):
            llmster.show("lms:not-there")

    def test_both_servers_listed_and_either_can_be_missing(self):
        with mock.patch.object(ollama, "tags", return_value=["gemma3:4b"]):
            self.assertEqual(models.tags(), ["gemma3:4b", self.MODEL])
        with mock.patch.object(ollama, "tags", side_effect=ollama.OllamaError("Ollama is not running")):
            self.assertEqual(models.tags(), [self.MODEL])
        with mock.patch.object(ollama, "running", side_effect=ollama.OllamaError("down")):
            self.assertEqual(models.running(), [])

    def test_absent_llmster_is_skipped_quietly(self):
        self.fake.stop()
        with mock.patch.object(llmster, "_cached", None):
            self.assertEqual(llmster.names(), [])
            self.assertFalse(llmster.reachable())
            with mock.patch.object(ollama, "tags", side_effect=ollama.OllamaError("Ollama is not running")):
                with self.assertRaises(ollama.OllamaError):
                    models.tags()


if __name__ == "__main__":
    unittest.main()
