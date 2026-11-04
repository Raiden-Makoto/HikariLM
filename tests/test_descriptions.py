import base64
import os
import tempfile
import unittest

from helper import descriptions


class DescriptionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        old = os.environ.get("HIKARILM_HOME")
        os.environ["HIKARILM_HOME"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("HIKARILM_HOME") if old is None
                        else os.environ.__setitem__("HIKARILM_HOME", old))

    @staticmethod
    def pic(data: bytes) -> str:
        return base64.b64encode(data).decode()

    def test_same_picture_same_key_changed_picture_new_key(self):
        a = descriptions.image_key(self.pic(b"one"))
        self.assertEqual(a, descriptions.image_key(self.pic(b"one")))
        self.assertNotEqual(a, descriptions.image_key(self.pic(b"two")))

    def test_bad_pictures_are_refused(self):
        for bad in (None, "", 5, "not base64!!"):
            with self.assertRaises(ValueError):
                descriptions.image_key(bad)

    def test_save_then_load(self):
        key = descriptions.image_key(self.pic(b"tree"))
        self.assertIsNone(descriptions.load(key))
        descriptions.save(key, "root 5, left 3, right 8", "gemma3:4b", "tree.png")
        found = descriptions.load(key)
        self.assertEqual(found["text"], "root 5, left 3, right 8")
        self.assertEqual(found["model"], "gemma3:4b")
        self.assertEqual(found["image"], "tree.png")
        # No temp files are left behind.
        self.assertEqual([p.name for p in descriptions.descriptions_dir().iterdir()], [key + ".json"])

    def test_empty_or_damaged_description_is_not_used(self):
        key = descriptions.image_key(self.pic(b"x"))
        descriptions.save(key, "   ", "gemma3", "x.png")
        self.assertIsNone(descriptions.load(key))
        (descriptions.descriptions_dir() / f"{key}.json").write_text("{not json")
        self.assertIsNone(descriptions.load(key))

    def test_long_text_is_cut(self):
        key = descriptions.image_key(self.pic(b"long"))
        descriptions.save(key, "a" * (descriptions.MAX_TEXT + 500), "gemma3", "l.png")
        self.assertEqual(len(descriptions.load(key)["text"]), descriptions.MAX_TEXT)


if __name__ == "__main__":
    unittest.main()
