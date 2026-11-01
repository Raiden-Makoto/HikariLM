import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from helper import descriptions, engine, imagegen, rewrite, server, store, thinking
from tests.test_engine import CODER, GEMMA, R1, add_model, real_config
from tests.test_flux import KLEIN, FluxCase
from tests.test_server import PNG_B64


class CleanTest(unittest.TestCase):
    def test_the_prompt_comes_out_alone(self):
        for written in ("A small white rabbit-like creature.", "Prompt: A small white rabbit-like creature.",
                        "**Rewritten prompt:** A small white rabbit-like creature.",
                        "Here is the prompt: \"A small white rabbit-like creature.\"",
                        "<think>Usagi is from Chiikawa.</think>\nA small white rabbit-like creature.",
                        "```\nA small white rabbit-like creature.\n```",
                        "“A small   white\nrabbit-like creature.”"):
            self.assertEqual(rewrite.clean(written), "A small white rabbit-like creature.", written)

    def test_a_reference_goes_with_the_request(self):
        alone = rewrite.messages("Usagi from Chiikawa")[0]["content"]
        self.assertTrue(alone.endswith("Request: Usagi from Chiikawa"))
        self.assertNotIn("reference", alone)
        seen = rewrite.messages("her on a beach", "A cream-colored rabbit with long ears.")[0]["content"]
        self.assertIn("Request: her on a beach", seen)
        self.assertIn("A cream-colored rabbit with long ears.", seen)


class RewriteTest(FluxCase):
    def setUp(self):
        super().setUp()
        self.hub.mkdir(parents=True, exist_ok=True)
        for name in (CODER, GEMMA):
            add_model(self.hub, name, real_config(name))
        engine.forget_scan()
        self.addCleanup(engine.forget_scan)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def post(self, path: str, payload: dict):
        request = urllib.request.Request(self.base + path, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
        except urllib.error.HTTPError as error:
            return json.loads(error.read())

    def test_the_chat_model_rewrites_and_then_leaves_for_the_picture_model(self):
        lines = self.post("/api/rewrite", {"model": CODER, "request": "Usagi from Chiikawa"})
        prompt = lines[-1]["prompt"]
        self.assertTrue(prompt.startswith("echo: You write prompts for a picture model."))
        self.assertTrue(prompt.endswith("Request: Usagi from Chiikawa"))
        self.assertEqual(" ".join("".join(line.get("text", "") for line in lines[:-1]).split()), prompt)
        self.download(KLEIN)
        drawn = self.post("/api/generate", {"model": KLEIN, "prompt": "a cream rabbit", "request": "Usagi from Chiikawa",
                                            "reference": "A cream rabbit with long ears."})
        self.assertEqual(drawn[0]["unloading"], CODER)
        record = drawn[-1]["image"]
        self.assertEqual((record["prompt"], record["request"], record["reference"]),
                         ("a cream rabbit", "Usagi from Chiikawa", "A cream rabbit with long ears."))
        saved = json.loads((imagegen.images_dir() / (record["id"] + ".json")).read_text())
        self.assertEqual((saved["request"], saved["reference"]), ("Usagi from Chiikawa", "A cream rabbit with long ears."))
        kept = store.clean_picture(record)
        self.assertEqual((kept["request"], kept["reference"]), ("Usagi from Chiikawa", "A cream rabbit with long ears."))

    def test_a_prompt_drawn_as_written_keeps_no_request(self):
        self.assertNotIn("request", imagegen.clean_request({"model": KLEIN, "prompt": "a fox", "request": "a fox"}))
        self.assertNotIn("request", imagegen.clean_request({"model": KLEIN, "prompt": "a fox"}))
        with self.assertRaises(ValueError):
            imagegen.clean_request({"model": KLEIN, "prompt": "a fox", "request": 5})

    def test_a_model_that_always_thinks_never_rewrites(self):
        for name in ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", "mlx-community/QwQ-32B-4bit",
                     "lmstudio-community/gpt-oss-20b-MLX-8bit", "mlx-community/Qwen3-4B-Thinking-2507-4bit",
                     "mlx-community/Phi-4-reasoning-plus-4bit", "mlx-community/Magistral-Small-2506-4bit"):
            self.assertTrue(thinking.always_thinks(name), name)
        for name in (CODER, GEMMA, "mlx-community/Qwen3-8B-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit"):
            self.assertFalse(thinking.always_thinks(name), name)
        add_model(self.hub, R1, real_config(R1))
        engine.forget_scan()
        with urllib.request.urlopen(self.base + "/api/tags", timeout=10) as response:
            self.assertEqual(json.loads(response.read())["thinkers"], [R1])
        self.assertIn("thinks before every answer", self.post("/api/rewrite", {"model": R1, "request": "a fox"})["error"])

    def test_a_picture_model_cannot_rewrite(self):
        self.download(KLEIN)
        body = self.post("/api/rewrite", {"model": KLEIN, "request": "a fox"})
        self.assertIn("makes pictures", body["error"])
        self.assertIn("Write what", self.post("/api/rewrite", {"model": CODER, "request": " "})["error"])

    def test_each_step_says_which_model_it_loaded_and_which_left(self):
        self.post("/api/chat", {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
        look = self.post("/api/describe", {"model": GEMMA, "image": PNG_B64, "name": "usagi.png", "purpose": "look"})
        self.assertEqual(look[0], {"hikarilm": "loaded", "model": GEMMA, "unloaded": CODER})
        lines = self.post("/api/rewrite", {"model": CODER, "request": "her on a beach", "reference": look[-1]["description"]})
        self.assertEqual(lines[0], {"hikarilm": "loaded", "model": CODER, "unloaded": GEMMA})
        again = self.post("/api/rewrite", {"model": CODER, "request": "her in the rain"})
        self.assertEqual(again[0], {"hikarilm": "loaded", "model": CODER, "unloaded": None})

    def test_a_reference_is_described_for_drawing_apart_from_code(self):
        look = self.post("/api/describe", {"model": GEMMA, "image": PNG_B64, "name": "usagi.png", "purpose": "look"})
        self.assertIn("an artist who cannot see it", look[-1]["description"])
        code = self.post("/api/describe", {"model": GEMMA, "image": PNG_B64, "name": "usagi.png"})
        self.assertIn("write code from it", code[-1]["description"])
        self.assertFalse(code[-1]["cached"])
        again = self.post("/api/describe", {"model": GEMMA, "image": PNG_B64, "name": "usagi.png", "purpose": "look"})
        self.assertTrue(again[-1]["cached"])
        key = descriptions.image_key(PNG_B64)
        self.assertIsNotNone(descriptions.load(key + "-look"))
        self.assertIn("not one of", self.post("/api/describe", {"model": GEMMA, "image": PNG_B64, "purpose": "x"})["error"])


if __name__ == "__main__":
    unittest.main()
