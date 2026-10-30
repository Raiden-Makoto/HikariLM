import base64
import os
import tempfile
import unittest
from unittest import mock

from helper import budget, imagegen, sdserver, store
from helper.router import Work

PNG = imagegen.PNG + b"rest of a picture"


class SupportTest(unittest.TestCase):
    def setUp(self):
        old = os.environ.pop("HIKARILM_IMAGES", None)
        if old is not None:
            self.addCleanup(os.environ.__setitem__, "HIKARILM_IMAGES", old)
        self.addCleanup(os.environ.pop, "HIKARILM_IMAGES", None)

    def test_only_apple_silicon_by_default(self):
        with mock.patch("platform.system", return_value="Linux"), mock.patch("platform.machine", return_value="x86_64"):
            got = imagegen.support()
        self.assertFalse(got["supported"])
        self.assertEqual(got["reason"], "Image generation needs a Mac with Apple Silicon.")
        with mock.patch("platform.system", return_value="Linux"), mock.patch("platform.machine", return_value="aarch64"):
            self.assertFalse(imagegen.support()["supported"])
        with mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"):
            self.assertTrue(imagegen.support()["supported"])

    def test_rosetta_asks_macos(self):
        answer = mock.Mock(stdout="1\n")
        with mock.patch("platform.system", return_value="Darwin"), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("subprocess.run", return_value=answer):
            self.assertTrue(imagegen.support()["supported"])
        answer.stdout = "0\n"
        with mock.patch("platform.system", return_value="Darwin"), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("subprocess.run", return_value=answer):
            self.assertFalse(imagegen.support()["supported"])

    def test_override(self):
        os.environ["HIKARILM_IMAGES"] = "on"
        self.assertTrue(imagegen.support()["supported"])
        os.environ["HIKARILM_IMAGES"] = "off"
        with mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"):
            self.assertFalse(imagegen.support()["supported"])


class ModelsTest(unittest.TestCase):
    def test_capabilities_win_and_names_are_the_fallback(self):
        self.assertTrue(imagegen.is_image_model("anything", {"capabilities": ["image"]}))
        self.assertFalse(imagegen.is_image_model("flux-named-chat", {"capabilities": ["completion"]}))
        self.assertFalse(imagegen.is_image_model("gemma3:4b", {"capabilities": ["completion", "vision"]}))
        self.assertTrue(imagegen.is_image_model("x/flux2-klein:4b", {}))
        self.assertTrue(imagegen.is_image_model("x/z-image-turbo", None))
        self.assertFalse(imagegen.is_image_model("qwen2.5-coder:7b", None))

    def test_split_puts_unreadable_models_with_chat(self):
        def show(name):
            if name == "broken":
                raise RuntimeError("no")
            return {"capabilities": ["image"] if name == "pic" else ["completion"]}
        self.assertEqual(imagegen.split(["qwen", "pic", "broken"], show), (["qwen", "broken"], ["pic"], []))

    def test_any_model_that_reports_vision_reads_pictures(self):
        def show(name):
            if name.startswith("named"):
                return {}
            return {"capabilities": ["completion", "vision"] if name in ("llava:7b", "my-own-model") else ["completion"]}
        names = ["llava:7b", "my-own-model", "gemma3-text-only", "qwen3:8b", "named-qwen2.5vl:7b", "named-llama"]
        chat, _, vision = imagegen.split(names, show)
        self.assertEqual(chat, names)
        self.assertEqual(vision, ["llava:7b", "my-own-model", "named-qwen2.5vl:7b"])


class RequestTest(unittest.TestCase):
    def test_clean_request(self):
        got = imagegen.clean_request({"model": " m ", "prompt": " a fox ", "size": "portrait", "seed": "42"})
        self.assertEqual((got["model"], got["prompt"], got["width"], got["height"], got["seed"], got["steps"]),
                         ("m", "a fox", 768, 1024, 42, None))
        self.assertIsInstance(imagegen.clean_request({"model": "m", "prompt": "x"})["seed"], int)
        for bad in ({"model": "m", "prompt": " "}, {"model": "m", "prompt": "x", "size": "huge"},
                    {"model": "m", "prompt": "x", "seed": -1}, {"model": "m", "prompt": "x", "steps": 500},
                    {"model": "", "prompt": "x"}, {"model": "m", "prompt": "x", "seed": True}):
            with self.assertRaises(ValueError):
                imagegen.clean_request(bad)

    def test_payload_sends_size_and_seed(self):
        request = imagegen.clean_request({"model": "m", "prompt": "x", "seed": 7, "steps": 4})
        payload = imagegen.ollama_payload(request, "5m")
        self.assertEqual((payload["width"], payload["height"], payload["steps"], payload["options"]),
                         (1024, 1024, 4, {"seed": 7}))


class FolderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"HIKARILM_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_read_delete(self):
        request = imagegen.clean_request({"model": "flux", "prompt": "a fox", "seed": 3})
        record = imagegen.save(imagegen.decode(base64.b64encode(PNG).decode()), request)
        self.assertEqual((record["prompt"], record["seed"]), ("a fox", 3))
        self.assertEqual(imagegen.read(record["id"]), PNG)
        self.assertTrue(imagegen.delete(record["id"]))
        self.assertFalse((imagegen.images_dir() / f"{record['id']}.json").exists())
        self.assertFalse(imagegen.delete(record["id"]))

    def test_a_chat_keeps_only_the_notes_of_its_picture(self):
        record = imagegen.save(PNG, imagegen.clean_request({"model": "sd:flux", "prompt": "a fox", "seed": 9}))
        store.save("c1", "sd:flux", [{"role": "user", "content": "a fox"},
                                     {"role": "assistant", "content": "", "picture": {**record, "image": "AAAA"}}])
        kept = store.load("c1")["messages"][1]["picture"]
        self.assertEqual((kept["id"], kept["seed"], kept["prompt"]), (record["id"], 9, "a fox"))
        self.assertNotIn("image", kept)
        self.assertEqual(store.picture_ids("c1"), [record["id"]])

    def test_bad_input_is_refused(self):
        for bad in ("", "not base64!", base64.b64encode(b"GIF89a").decode()):
            with self.assertRaises(ValueError):
                imagegen.decode(bad)
        for bad in ("../x", "20260101-000000-zzzzzz", ""):
            with self.assertRaises(ValueError):
                imagegen.read(bad)


class SdServerTest(unittest.TestCase):
    def test_model_lines(self):
        models, problems = sdserver.parse_models(
            "# my models\n"
            "flux2-klein: --diffusion-model ~/m/flux.gguf --vae ~/m/ae.sft --llm '/m/qwen 3.gguf'\n"
            "bad line without colon\n"
            "z: --listen-port 9\n"
            "flux2-klein: -m x\n")
        self.assertEqual([m["name"] for m in models], ["flux2-klein"])
        self.assertEqual(models[0]["args"][1], os.path.expanduser("~/m/flux.gguf"))
        self.assertEqual(models[0]["args"][-1], "/m/qwen 3.gguf")
        self.assertEqual(len(problems), 3)

    def test_size_counts_only_files_that_exist(self):
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"x" * 1000)
            handle.flush()
            self.assertEqual(sdserver.model_size(["--diffusion-model", handle.name, "--vae", "/nope"]), 1000)

    def test_keepalive(self):
        self.assertEqual(sdserver.keepalive_seconds("5m"), 300)
        self.assertEqual(sdserver.keepalive_seconds("30s"), 30)
        self.assertEqual(sdserver.keepalive_seconds("nonsense"), 300)

    def test_picker_lists_every_source(self):
        settings = {"pictures": {"program": "", "models": "flux: -m /x.gguf", "address": ""}}
        with mock.patch.object(sdserver, "find_program", return_value="/bin/sd-server"), \
                mock.patch.dict(os.environ, {"HIKARILM_IMAGES": "off"}):
            found, notes = imagegen.picture_models(settings, ["x/flux2-klein"])
        self.assertEqual([item["id"] for item in found], ["sd:flux"])
        self.assertTrue(any("hidden" in note for note in notes))
        with mock.patch.object(sdserver, "find_program", return_value=None):
            found, notes = imagegen.picture_models(settings, [])
        self.assertEqual(found, [])
        self.assertIn("sd-server was not found", notes[0])


class OllamaRefusesTest(unittest.TestCase):
    ERROR = "image generation models are not currently supported"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for patcher in (mock.patch.dict(os.environ, {"HIKARILM_HOME": self._tmp.name + "/home",
                                                     "HIKARILM_IMAGES": "on"}),
                        mock.patch("helper.ollama.version", side_effect=lambda: self.version)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.version = "0.13.0"
        self.settings = {"pictures": imagegen.PICTURE_DEFAULTS}

    def test_points_to_sd_server_and_hides_ollama_until_it_updates(self):
        self.assertEqual([item["id"] for item in imagegen.picture_models(self.settings, ["x/flux2-klein"])[0]],
                         ["x/flux2-klein"])
        message = imagegen.note_refusal(self.ERROR, self.version, "x/flux2-klein")
        self.assertIn("Ollama 0.13.0 cannot make pictures with x/flux2-klein", message)
        self.assertIn("sd-server", message)
        found, notes = imagegen.picture_models(self.settings, ["x/flux2-klein"])
        self.assertEqual(found, [])
        self.assertIn("sd-server", notes[0])
        self.version = "0.14.0"
        self.assertEqual(len(imagegen.picture_models(self.settings, ["x/flux2-klein"])[0]), 1)

    def test_other_errors_pass_through(self):
        self.assertIsNone(imagegen.note_refusal("model requires more system memory", self.version, "x/flux2-klein"))
        self.assertFalse(imagegen.ollama_refuses())


class WorkTest(unittest.TestCase):
    def test_a_picture_and_a_chat_never_overlap(self):
        work = Work()
        self.assertIsNone(work.start_chat())
        self.assertIn("chat reply", work.start_picture())
        work.end_chat()
        self.assertIsNone(work.start_picture())
        self.assertIn("already", work.start_picture())
        self.assertIn("picture is being made", work.start_chat())
        work.end_picture()
        self.assertIsNone(work.start_chat())


class MemoryTest(unittest.TestCase):
    def test_picture_model_must_fit(self):
        gib = budget.GIB
        budget.check_picture("flux", 8 * gib, [], (32 * gib, 24 * gib))
        with self.assertRaises(budget.NoRoom):
            budget.check_picture("flux", 8 * gib, [], (16 * gib, 4 * gib))
        budget.check_picture("flux", 8 * gib, [{"name": "qwen", "size": 6 * gib}], (16 * gib, 6 * gib))


if __name__ == "__main__":
    unittest.main()
