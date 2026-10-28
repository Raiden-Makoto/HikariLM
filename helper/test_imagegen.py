import base64
import os
import tempfile
import unittest
from unittest import mock

from helper import budget, imagegen
from helper.router import Work

PNG = imagegen.PNG + b"rest of a picture"


class SupportTest(unittest.TestCase):
    def setUp(self):
        old = os.environ.pop("LIGHTUSAGI_IMAGES", None)
        if old is not None:
            self.addCleanup(os.environ.__setitem__, "LIGHTUSAGI_IMAGES", old)
        self.addCleanup(os.environ.pop, "LIGHTUSAGI_IMAGES", None)

    def test_only_apple_silicon_by_default(self):
        with mock.patch("platform.system", return_value="Linux"), mock.patch("platform.machine", return_value="x86_64"):
            got = imagegen.support()
        self.assertFalse(got["supported"])
        self.assertIn("Image generation needs a Mac with Apple Silicon. This computer is Linux", got["reason"])
        with mock.patch("platform.system", return_value="Windows"), mock.patch("platform.machine", return_value="ARM64"):
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
            got = imagegen.support()
        self.assertFalse(got["supported"])
        self.assertIn("an Intel Mac", got["reason"])

    def test_override(self):
        os.environ["LIGHTUSAGI_IMAGES"] = "on"
        self.assertTrue(imagegen.support()["supported"])
        os.environ["LIGHTUSAGI_IMAGES"] = "off"
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
        self.assertEqual(imagegen.split(["qwen", "pic", "broken"], show), (["qwen", "broken"], ["pic"]))


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
        patcher = mock.patch.dict(os.environ, {"LIGHTUSAGI_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_list_read_delete(self):
        request = imagegen.clean_request({"model": "flux", "prompt": "a fox", "seed": 3})
        record = imagegen.save(imagegen.decode(base64.b64encode(PNG).decode()), request)
        self.assertEqual(imagegen.read(record["id"]), PNG)
        listed = imagegen.listing()
        self.assertEqual([(item["id"], item["prompt"], item["seed"]) for item in listed], [(record["id"], "a fox", 3)])
        self.assertTrue(imagegen.delete(record["id"]))
        self.assertEqual(imagegen.listing(), [])
        self.assertFalse(imagegen.delete(record["id"]))

    def test_picture_without_notes_still_shows(self):
        record = imagegen.save(PNG, imagegen.clean_request({"model": "m", "prompt": "x"}))
        (imagegen.images_dir() / f"{record['id']}.json").write_text("{damaged", encoding="utf-8")
        self.assertEqual(imagegen.listing()[0]["prompt"], "")

    def test_bad_input_is_refused(self):
        for bad in ("", "not base64!", base64.b64encode(b"GIF89a").decode()):
            with self.assertRaises(ValueError):
                imagegen.decode(bad)
        for bad in ("../x", "20260101-000000-zzzzzz", ""):
            with self.assertRaises(ValueError):
                imagegen.read(bad)


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
