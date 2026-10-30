import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helper import budget, engine, models, thinking
from helper.ollama import OllamaError

HERE = Path(__file__).resolve().parent
CODER = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
R1 = "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
GEMMA = "mlx-community/gemma-3-12b-it-4bit"


def add_model(hub: Path, name: str, config: dict, weights: int = 1000, incomplete: bool = False) -> Path:
    repo = hub / ("models--" + name.replace("/", "--"))
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "model.safetensors").write_bytes(b"x" * weights)
    if incomplete:
        (repo / "blobs").mkdir()
        (repo / "blobs" / "0123.incomplete").write_bytes(b"")
    return snapshot


def real_config(name: str) -> dict:
    return json.loads((HERE / "fixtures" / "mlx" / (name.split("/", 1)[1] + ".json")).read_text())


class EngineTest(unittest.TestCase):
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
        for name in (CODER, R1, GEMMA):
            add_model(self.hub, name, real_config(name), weights=2000 if name == GEMMA else 1000)
        engine.forget_scan()
        self.addCleanup(engine.unload)
        self.addCleanup(engine.forget_scan)

    def chat(self, model, text, **options):
        reply = engine.open_chat(model, [{"role": "user", "content": text}], options=options or None,
                                 think=options.pop("think", None))
        events = []
        try:
            while True:
                line = reply.readline()
                if not line:
                    return events
                events.append(json.loads(line))
        finally:
            reply.close()

    def test_finds_finished_mlx_downloads_only(self):
        add_model(self.hub, "mlx-community/still-coming-4bit", real_config(CODER), incomplete=True)
        add_model(self.hub, "Qwen/not-mlx", {k: v for k, v in real_config(CODER).items() if k != "quantization"})
        engine.forget_scan()
        self.assertEqual(sorted(engine.names()), sorted([CODER, R1, GEMMA]))
        self.assertEqual(engine.sizes()[GEMMA], 2000)
        own = Path(os.environ["HIKARILM_HOME"]) / "models" / "falcon3-10b-mlx"
        own.mkdir(parents=True)
        (own / "config.json").write_text(json.dumps(real_config(CODER)))
        (own / "weights.safetensors").write_bytes(b"x")
        engine.forget_scan()
        self.assertIn("falcon3-10b-mlx", engine.names())

    def test_shape_for_the_memory_budget(self):
        coder, r1, gemma = engine.show(CODER), engine.show(R1), engine.show(GEMMA)
        with mock.patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "4", "OLLAMA_KV_CACHE_TYPE": "q4_0"}):
            self.assertEqual(budget.per_token(coder), 28 * 4 * 256 * 2)
        self.assertEqual(budget.per_token(r1), 36 * 8 * 256 * 2)
        self.assertEqual(budget.per_token(gemma), 48 * 8 * 512 * 2)
        self.assertEqual(budget.trained_context(coder), 32768)
        self.assertEqual(budget.trained_context(gemma), 131072)
        self.assertEqual(budget.default_source(coder), (8192, "model"))

    def test_what_each_model_can_do(self):
        self.assertFalse(thinking.can_think(engine.show(CODER)))
        self.assertTrue(thinking.can_think(engine.show(R1)))
        self.assertIn("vision", engine.show(GEMMA)["capabilities"])
        self.assertNotIn("vision", engine.show(CODER)["capabilities"])

    def test_streams_like_ollama(self):
        events = self.chat(CODER, "hello there")
        self.assertEqual("".join(e["message"]["content"] for e in events), "echo: hello there")
        self.assertTrue(events[-1]["done"])
        self.assertEqual((events[-1]["prompt_eval_count"], events[-1]["eval_count"]), (11, 3))
        self.assertEqual(engine.running(), [{"name": CODER, "size": 1000, "expires": None}])

    def test_sends_options_and_think(self):
        sent = json.loads(self.chat(R1, "options", num_ctx=4096, num_predict=200, think=False)[0]["message"]["content"])
        self.assertEqual((sent["think"], sent["num_ctx"], sent["num_predict"]), (False, 4096, 200))
        self.assertFalse(sent["vision"])
        self.assertTrue(json.loads(self.chat(GEMMA, "options")[0]["message"]["content"])["vision"])

    def test_pictures_reach_the_worker(self):
        reply = engine.open_chat(GEMMA, [{"role": "user", "content": "options", "images": ["aGk="]}])
        try:
            sent = json.loads(json.loads(reply.readline())["message"]["content"])
        finally:
            reply.close()
        self.assertEqual(sent["images"], 1)

    def test_same_model_stays_loaded_and_another_replaces_it(self):
        first = json.loads(self.chat(CODER, "options")[0]["message"]["content"])["pid"]
        again = json.loads(self.chat(CODER, "options")[0]["message"]["content"])["pid"]
        self.assertEqual(first, again)
        other = json.loads(self.chat(R1, "options")[0]["message"]["content"])["pid"]
        self.assertNotEqual(first, other)
        self.assertEqual([item["name"] for item in engine.running()], [R1])

    def test_stop_part_way_then_answer_again(self):
        reply = engine.open_chat(CODER, [{"role": "user", "content": "slow"}])
        for _ in range(5):
            reply.readline()
        started = time.monotonic()
        reply.close()
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual("".join(e["message"]["content"] for e in self.chat(CODER, "after stop")), "echo: after stop")

    def test_unload_ends_the_worker(self):
        self.chat(CODER, "hi")
        models.unload(CODER)
        self.assertEqual(engine.running(), [])

    def test_crash_is_an_error_and_the_next_chat_starts_again(self):
        events = self.chat(CODER, "crash")
        self.assertIn("stopped", events[-1]["error"])
        self.assertEqual(engine.running(), [])
        self.assertEqual(self.chat(CODER, "back")[0]["message"]["content"], "echo:")

    def test_load_failure_is_worded(self):
        add_model(self.hub, "mlx-community/broken-4bit", real_config(CODER))
        engine.forget_scan()
        with self.assertRaises(OllamaError) as caught:
            engine.open_chat("mlx-community/broken-4bit", [{"role": "user", "content": "hi"}])
        self.assertIn("broken on purpose", str(caught.exception))
        self.assertEqual(self.chat(CODER, "still fine")[-1]["done"], True)

    def test_missing_venv_says_how_to_set_it_up(self):
        with mock.patch.dict(os.environ, {"HIKARILM_MLX_PYTHON": "/nowhere/python"}):
            with self.assertRaises(OllamaError) as caught:
                engine.open_chat(CODER, [{"role": "user", "content": "hi"}])
        self.assertIn("run ./start.sh again", str(caught.exception))

    def test_complete_joins_the_answer(self):
        self.assertEqual(engine.complete(CODER, [{"role": "user", "content": "a b"}], {"num_ctx": 2048}), "echo: a b")

    def test_models_route_by_name_and_work_without_ollama(self):
        with mock.patch("helper.ollama.tags", side_effect=OllamaError("down")):
            self.assertEqual(sorted(models.tags()), sorted([CODER, R1, GEMMA]))
        with mock.patch("helper.ollama.tags", return_value=["gemma3:4b"]):
            self.assertEqual(models.tags()[-1], "gemma3:4b")
        with mock.patch("helper.ollama.show", return_value={"from": "ollama"}) as ollama_show:
            self.assertEqual(models.show("gemma3:4b"), {"from": "ollama"})
            self.assertTrue(models.show(CODER)["mlx"])
            ollama_show.assert_called_once_with("gemma3:4b")
        with mock.patch("helper.ollama.running", side_effect=OllamaError("down")):
            self.assertEqual(models.running(), [])


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"HIKARILM_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("HIKARILM_MLX_PYTHON", None)
        self.said = []

    def run_prepare(self, fails=False):
        def run(command, check):
            if fails:
                raise engine.subprocess.CalledProcessError(1, command)
            if command[1:3] == ["-m", "venv"]:
                (Path(command[3]) / "bin").mkdir(parents=True)
                (Path(command[3]) / "bin" / "python").write_text("")
            return mock.Mock(returncode=0)

        with mock.patch.object(engine.subprocess, "run", side_effect=run) as ran:
            ready = engine.prepare(self.said.append, "/usr/bin/python3")
        return ready, [call.args[0] for call in ran.call_args_list]

    def test_first_start_makes_the_venv_and_installs_once(self):
        ready, commands = self.run_prepare()
        self.assertTrue(ready)
        venv = Path(self._tmp.name) / "venv"
        self.assertEqual(commands[0], ["/usr/bin/python3", "-m", "venv", str(venv)])
        self.assertEqual(commands[1], [str(venv / "bin" / "python"), "-m", "pip", "install", "--upgrade",
                                       "mlx-lm", "mlx-vlm"])
        self.assertEqual(self.run_prepare(), (True, []))

    def test_a_venv_made_by_hand_only_gets_the_packages(self):
        (Path(self._tmp.name) / "venv" / "bin").mkdir(parents=True)
        (Path(self._tmp.name) / "venv" / "bin" / "python").write_text("")
        _, commands = self.run_prepare()
        self.assertEqual([command[1:3] for command in commands], [["-m", "pip"]])

    def test_failure_is_said_and_tried_again_next_time(self):
        ready, _ = self.run_prepare(fails=True)
        self.assertFalse(ready)
        self.assertIn("Ollama's models still work", self.said[-1])
        self.assertTrue(self.run_prepare()[0])

    def test_another_python_is_left_alone(self):
        with mock.patch.dict(os.environ, {"HIKARILM_MLX_PYTHON": "/some/python"}):
            self.assertEqual(self.run_prepare(), (True, []))


class WorkerPromptTest(unittest.TestCase):
    """The real worker's prompt building, with a stand-in tokenizer (MLX is not needed for it)."""

    def setUp(self):
        sys.path.insert(0, str(HERE.parent / "helper"))
        self.addCleanup(sys.path.remove, str(HERE.parent / "helper"))
        import mlx_worker
        self.worker = mlx_worker

    def model(self, template, ending):
        class Tokenizer:
            chat_template = template

            def apply_chat_template(self, chat, add_generation_prompt, tokenize, **extra):
                self.extra = extra
                return "".join(m["content"] for m in chat) + ending

            def encode(self, text, add_special_tokens):
                return [ord(c) for c in text]

        runner = self.worker.TextModel.__new__(self.worker.TextModel)
        runner.tokenizer = Tokenizer()
        runner.template_switch = "enable_thinking" in template
        return runner

    def text(self, ids):
        return "".join(chr(i) for i in ids)

    def test_r1_told_not_to_think_gets_an_empty_reasoning_block(self):
        ids, lead = self.model("{{'<｜Assistant｜>'}}", "<A>")._prompt([{"role": "user", "content": "q"}], False)
        self.assertEqual(self.text(ids), "q<A>" + self.worker.EMPTY_THINK)
        self.assertEqual(lead, "")
        ids, _ = self.model("x", "<A><think>\n")._prompt([{"role": "user", "content": "q"}], False)
        self.assertEqual(self.text(ids), "q<A>" + self.worker.EMPTY_THINK)

    def test_template_that_opens_thinking_hands_the_tag_on(self):
        _, lead = self.model("x", "<A><think>\n")._prompt([{"role": "user", "content": "q"}], True)
        self.assertEqual(lead, "<think>\n")

    def test_qwen3_style_switch_is_used(self):
        runner = self.model("{% if enable_thinking %}", "<A>")
        ids, _ = runner._prompt([{"role": "user", "content": "q"}], False)
        self.assertEqual(runner.tokenizer.extra, {"enable_thinking": False})
        self.assertEqual(self.text(ids), "q<A>")

    def test_no_think_field_leaves_the_prompt_alone(self):
        ids, lead = self.model("x", "<A>")._prompt([{"role": "user", "content": "q"}], None)
        self.assertEqual((self.text(ids), lead), ("q<A>", ""))

    def test_reply_length(self):
        self.assertEqual(self.worker._room({"num_ctx": 4096}, 1000), 3096)
        self.assertEqual(self.worker._room({"num_ctx": 4096, "num_predict": 200}, 1000), 200)
        self.assertEqual(self.worker._room({"num_ctx": 1000}, 2000), 64)


if __name__ == "__main__":
    unittest.main()
