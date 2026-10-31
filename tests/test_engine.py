import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helper import budget, context, engine, models, thinking
from helper.engine import ModelError

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
        add_model(self.hub, "Qwen/Qwen2.5-Coder-7B-Instruct", {k: v for k, v in real_config(CODER).items() if k != "quantization"})
        engine.forget_scan()
        self.assertEqual(sorted(engine.names()), sorted([CODER, R1, GEMMA]))
        self.assertEqual(engine.sizes()[GEMMA], 2000)
        own = Path(os.environ["HIKARILM_HOME"]) / "models" / "falcon3-10b-mlx"
        own.mkdir(parents=True)
        (own / "config.json").write_text(json.dumps(real_config(CODER)))
        (own / "weights.safetensors").write_bytes(b"x")
        engine.forget_scan()
        self.assertIn("falcon3-10b-mlx", engine.names())

    def test_falcon3_bitnet_is_listed_from_tiis_own_repo(self):
        config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                  "quantization_config": {"modules_to_not_convert": ["lm_head"], "quant_method": "bitnet"}}
        add_model(self.hub, "tiiuae/Falcon3-10B-Instruct-1.58bit", config)
        add_model(self.hub, "tiiuae/Falcon3-10B-Instruct", {"model_type": "llama"})
        engine.forget_scan()
        self.assertIn("tiiuae/Falcon3-10B-Instruct-1.58bit", engine.names())
        self.assertNotIn("tiiuae/Falcon3-10B-Instruct", engine.names())

    def test_microsofts_bitnet_is_listed_packed_or_as_master_weights(self):
        for name, mode in (("microsoft/bitnet-b1.58-2B-4T", "offline"), ("microsoft/bitnet-b1.58-2B-4T-bf16", "online")):
            add_model(self.hub, name, {"model_type": "bitnet", "quantization_config": {
                "quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": mode}})
        engine.forget_scan()
        self.assertIn("microsoft/bitnet-b1.58-2B-4T", engine.names())
        self.assertIn("microsoft/bitnet-b1.58-2B-4T-bf16", engine.names())

    def test_any_mlx_model_is_listed_quantized_or_not(self):
        full = {k: v for k, v in real_config(CODER).items() if k != "quantization"}
        add_model(self.hub, "mlx-community/Llama-3.2-3B-Instruct-bf16", full)
        add_model(self.hub, "someone/phi-4-mini-mlx", full)
        add_model(self.hub, "someone/mistral-7b-instruct", full)
        engine.forget_scan()
        self.assertIn("mlx-community/Llama-3.2-3B-Instruct-bf16", engine.names())
        self.assertIn("someone/phi-4-mini-mlx", engine.names())
        self.assertNotIn("someone/mistral-7b-instruct", engine.names())
        own = Path(os.environ["HIKARILM_HOME"]) / "models" / "my-own-bf16"
        own.mkdir(parents=True)
        (own / "config.json").write_text(json.dumps(full))
        (own / "weights.safetensors").write_bytes(b"x")
        engine.forget_scan()
        self.assertIn("my-own-bf16", engine.names())

    def test_context_lengths(self):
        coder, gemma = engine.show(CODER), engine.show(GEMMA)
        self.assertEqual(budget.trained_context(coder), 32768)
        self.assertEqual(budget.trained_context(gemma), 131072)
        self.assertEqual(budget.default_source(coder), (8192, "model"))

    def test_what_each_model_can_do(self):
        self.assertFalse(thinking.can_think(engine.show(CODER)))
        self.assertTrue(thinking.can_think(engine.show(R1)))
        self.assertIn("vision", engine.show(GEMMA)["capabilities"])
        self.assertNotIn("vision", engine.show(CODER)["capabilities"])

    def test_streams_chat_lines(self):
        events = self.chat(CODER, "hello there")
        self.assertEqual("".join(e["message"]["content"] for e in events), "echo: hello there")
        self.assertTrue(events[-1]["done"])
        self.assertEqual((events[-1]["prompt_eval_count"], events[-1]["eval_count"]), (11, 3))
        self.assertEqual(engine.running(), [{"name": CODER, "size": 1000, "expires": None}])

    def test_sends_options_and_think(self):
        sent = json.loads(self.chat(R1, "options", num_ctx=4096, num_predict=200, think=False)[0]["message"]["content"])
        self.assertEqual((sent["think"], sent["num_ctx"], sent["num_predict"]), (False, 4096, 200))
        self.assertEqual(sent["kv_bits"], 8)
        self.assertFalse(sent["vision"])
        context.save_settings({**context.load_settings(), "kv_cache": {"bits": 4}})
        self.assertEqual(json.loads(self.chat(R1, "options")[0]["message"]["content"])["kv_bits"], 4)
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

    def test_unload_ends_the_worker_at_once(self):
        self.chat(CODER, "hi")
        process = engine._worker.process
        started = time.monotonic()
        models.unload(CODER)
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsNotNone(process.poll())
        self.assertEqual(engine.running(), [])

    def test_the_context_asked_for_is_used_as_is(self):
        shown = engine.show(GEMMA)
        self.assertEqual(budget.plan(65536, shown)["num_ctx"], 65536)
        self.assertEqual(budget.plan(None, shown)["num_ctx"], budget.default_source(shown)[0])
        self.assertEqual(budget.plan(10 ** 6, shown)["num_ctx"], 131072)

    def test_crash_is_an_error_and_the_next_chat_starts_again(self):
        events = self.chat(CODER, "crash")
        self.assertIn("stopped", events[-1]["error"])
        self.assertEqual(engine.running(), [])
        self.assertEqual(self.chat(CODER, "back")[0]["message"]["content"], "echo:")

    def test_load_failure_is_worded(self):
        add_model(self.hub, "mlx-community/broken-4bit", real_config(CODER))
        engine.forget_scan()
        with self.assertRaises(ModelError) as caught:
            engine.open_chat("mlx-community/broken-4bit", [{"role": "user", "content": "hi"}])
        self.assertIn("broken on purpose", str(caught.exception))
        self.assertEqual(self.chat(CODER, "still fine")[-1]["done"], True)

    def test_missing_venv_says_how_to_set_it_up(self):
        with mock.patch.dict(os.environ, {"HIKARILM_MLX_PYTHON": "/nowhere/python"}):
            with self.assertRaises(ModelError) as caught:
                engine.open_chat(CODER, [{"role": "user", "content": "hi"}])
        self.assertIn("run ./start.sh again", str(caught.exception))

    def test_complete_joins_the_answer(self):
        self.assertEqual(engine.complete(CODER, [{"role": "user", "content": "a b"}], {"num_ctx": 2048}), "echo: a b")

    def test_models_lists_the_mlx_models(self):
        self.assertEqual(sorted(models.tags()), sorted([CODER, R1, GEMMA]))
        self.assertTrue(models.show(CODER)["mlx"])
        self.assertEqual(models.running(), [])
        with self.assertRaises(ModelError):
            models.show("gemma3:4b")


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
        self.assertIn("Run ./start.sh again", self.said[-1])
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

    def test_every_bitnet_form_is_recognized(self):
        tii = {"modules_to_not_convert": ["lm_head"], "quant_method": "bitnet"}
        microsoft = {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "offline"}
        mlx_copy = {"group_size": 64, "bits": 4, "quant_method": "bitnet"}
        with tempfile.TemporaryDirectory() as folder:
            def settings(config):
                Path(folder, "config.json").write_text(json.dumps(config))
                return self.worker.bitnet_config(folder)

            self.assertEqual(settings({"quantization_config": tii}), tii)
            self.assertEqual(settings({"model_type": "bitnet", "quantization_config": microsoft}), microsoft)
            self.assertEqual(settings({"quantization": mlx_copy}), mlx_copy)
            self.assertIsNone(settings({"quantization": {"group_size": 64, "bits": 4}}))
            self.assertIsNone(settings({"quantization_config": {"quant_method": "fp8"}}))
            self.assertIsNone(settings({"model_type": "llama"}))

    def test_full_weights_are_packed_the_way_transformers_unpacks_them(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        weight = np.random.default_rng(1).standard_normal((16, 12)).astype(np.float32) * 0.02
        for divided in (True, False):
            packed, scale = self.worker.pack_ternary(weight, np, divided)
            self.assertEqual((packed.dtype, packed.shape), (np.uint8, (4, 12)))
            # transformers' unpack_weights: value i of each byte is row i * rows + r, minus 1.
            unpacked = np.concatenate([((packed >> (2 * i)) & 3).astype(np.int8) - 1 for i in range(4)])
            # transformers' WeightQuant: scale by 1 / mean size, round, clamp to -1..1, scale back.
            per = np.float32(1) / max(np.abs(weight).mean(), np.float32(1e-5))
            ternary = np.clip(np.round(weight * per), -1, 1)
            np.testing.assert_array_equal(unpacked, ternary)
            expected = per if divided else np.float32(1) / per
            self.assertAlmostEqual(float(scale), float(expected), places=7)
        with self.assertRaises(ValueError):
            self.worker.pack_ternary(np.ones((6, 4), np.float32), np, False)

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

    def test_r1_told_to_think_has_its_reasoning_opened(self):
        ids, lead = self.model("{{'<｜Assistant｜>'}}", "<A>")._prompt([{"role": "user", "content": "q"}], True)
        self.assertEqual(self.text(ids), "q<A><think>\n")
        self.assertEqual(lead, "<think>\n")
        split = thinking.InlineThink()
        thought, answer = split.feed(lead + "let me see")
        self.assertEqual((thought, answer), ("\nlet me see", ""))

    def test_turn_markers_end_the_reply(self):
        turn = self.worker.TurnEnd()
        self.assertEqual(turn.feed("def f():<|im"), "def f():")
        self.assertEqual(turn.feed("_end|>\n<|im_start|>user"), "")
        self.assertTrue(turn.ended)
        turn = self.worker.TurnEnd()
        self.assertEqual(turn.feed("if a <"), "if a ")
        self.assertEqual(turn.feed(" b:"), "< b:")
        self.assertFalse(turn.ended)
        self.assertEqual(turn.feed("done<end_of_turn>"), "done")
        turn = self.worker.TurnEnd()
        turn.feed("x <")
        self.assertEqual(turn.flush(), "<")

    def test_tokenizer_end_tokens_are_added(self):
        added = []

        class Tokenizer:
            eos_token = "<|im_end|>"

            def add_eos_token(self, token):
                added.append(token)

            def get_vocab(self):
                return {"<|im_end|>": 151645, "<|endoftext|>": 151643, "<|im_start|>": 151644, "hi": 1}

        self.worker._stop_at_turn_ends(Tokenizer())
        self.assertEqual(sorted(set(added)), ["<|endoftext|>", "<|im_end|>", "<|im_start|>"])

    def test_no_think_field_leaves_the_prompt_alone(self):
        ids, lead = self.model("x", "<A>")._prompt([{"role": "user", "content": "q"}], None)
        self.assertEqual((self.text(ids), lead), ("q<A>", ""))

    def test_reply_length(self):
        self.assertEqual(self.worker._room({"num_ctx": 4096}, 1000), 3096)
        self.assertEqual(self.worker._room({"num_ctx": 4096, "num_predict": 200}, 1000), 200)
        self.assertEqual(self.worker._room({"num_ctx": 1000}, 2000), 64)


if __name__ == "__main__":
    unittest.main()
