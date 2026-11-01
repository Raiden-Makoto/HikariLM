import os
import tempfile
import unittest
from unittest import mock

from helper import imagegen, store
from helper.router import Work

PNG = imagegen.PNG + b"rest of a picture"


class AppleSiliconTest(unittest.TestCase):
    def test_only_apple_silicon(self):
        with mock.patch("platform.system", return_value="Linux"), mock.patch("platform.machine", return_value="aarch64"):
            self.assertFalse(imagegen.apple_silicon())
        with mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"):
            self.assertTrue(imagegen.apple_silicon())

    def test_rosetta_asks_macos(self):
        answer = mock.Mock(stdout="1\n")
        with mock.patch("platform.system", return_value="Darwin"), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("subprocess.run", return_value=answer):
            self.assertTrue(imagegen.apple_silicon())
        answer.stdout = "0\n"
        with mock.patch("platform.system", return_value="Darwin"), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("subprocess.run", return_value=answer):
            self.assertFalse(imagegen.apple_silicon())


class ModelsTest(unittest.TestCase):
    def test_vision_from_capabilities_and_unreadable_left_out(self):
        def show(name):
            if name == "broken":
                raise RuntimeError("no")
            return {"capabilities": ["completion", "vision"] if name == "gemma" else ["completion"]}
        self.assertEqual(imagegen.vision_models(["qwen", "gemma", "broken"], show), ["gemma"])


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


class FolderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"HIKARILM_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_read_delete(self):
        request = imagegen.clean_request({"model": "flux", "prompt": "a fox", "seed": 3})
        record = imagegen.save(PNG, request)
        self.assertEqual((record["prompt"], record["seed"]), ("a fox", 3))
        self.assertEqual(imagegen.read(record["id"]), PNG)
        self.assertTrue(imagegen.delete(record["id"]))
        self.assertFalse((imagegen.images_dir() / f"{record['id']}.json").exists())
        self.assertFalse(imagegen.delete(record["id"]))

    def test_a_chat_keeps_only_the_notes_of_its_picture(self):
        record = imagegen.save(PNG, imagegen.clean_request({"model": "flux", "prompt": "a fox", "seed": 9}))
        store.save("c1", "flux", [{"role": "user", "content": "a fox"},
                                     {"role": "assistant", "content": "", "picture": {**record, "image": "AAAA"}}])
        kept = store.load("c1")["messages"][1]["picture"]
        self.assertEqual((kept["id"], kept["seed"], kept["prompt"]), (record["id"], 9, "a fox"))
        self.assertNotIn("image", kept)
        self.assertEqual(store.picture_ids("c1"), [record["id"]])

    def test_bad_input_is_refused(self):
        request = imagegen.clean_request({"model": "flux", "prompt": "a fox"})
        for bad in (b"", b"GIF89a", "not bytes"):
            with self.assertRaises(ValueError):
                imagegen.save(bad, request)
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

if __name__ == "__main__":
    unittest.main()
