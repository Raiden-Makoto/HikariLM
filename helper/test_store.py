import os
import tempfile
import time
import unittest

from helper import store


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        previous = os.environ.get("LIGHTUSAGI_HOME")
        os.environ["LIGHTUSAGI_HOME"] = self.folder.name
        self.addCleanup(self._restore, previous)

    @staticmethod
    def _restore(previous) -> None:
        if previous is None:
            os.environ.pop("LIGHTUSAGI_HOME", None)
        else:
            os.environ["LIGHTUSAGI_HOME"] = previous

    def messages(self, text: str = "How does an AVL tree stay balanced?"):
        return [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "Rotations.", "thinking": "hmm", "model": "deepseekr1:8b"},
        ]

    def test_save_then_load_round_trips(self) -> None:
        store.save("abc-123", "deepseekr1:8b", self.messages())
        chat = store.load("abc-123")
        self.assertEqual(chat["model"], "deepseekr1:8b")
        self.assertEqual(chat["messages"], self.messages())

    def test_files_go_under_transcripts(self) -> None:
        store.save("abc-123", "gemma3", self.messages())
        self.assertTrue((store.transcripts_dir() / "abc-123.json").is_file())

    def test_title_comes_from_first_message_and_stays(self) -> None:
        store.save("one", "gemma3", self.messages("  Explain   rotations\nmore"))
        self.assertEqual(store.listing()[0]["title"], "Explain rotations")
        store.save("one", "gemma3", self.messages("something else"))
        self.assertEqual(store.listing()[0]["title"], "Explain rotations")

    def test_title_heuristic(self) -> None:
        cases = {
            "hey, can you please explain how AVL rotations work? thanks": "Explain how AVL rotations work",
            "Hi! I want you to find the bug in this code.\nmore": "Find the bug in this code",
            "```py\nx=1\n```\nwhy does this fail": "Why does this fail",
            "ok": "Ok",
            "hello": "Hello",
        }
        for text, expected in cases.items():
            self.assertEqual(store.make_title([{"role": "user", "content": text}]), expected)
        long = "explain " + "word " * 30
        title = store.make_title([{"role": "user", "content": long}])
        self.assertTrue(title.endswith("…") and len(title) <= store.TITLE_LENGTH)
        self.assertNotIn(" …", title)

    def test_files_are_kept(self) -> None:
        msgs = [{"role": "user", "content": "hi", "files": [{"name": "a.py", "text": "x=1"}, {"name": 3}]}]
        store.save("f1", "gemma3", msgs)
        self.assertEqual(store.load("f1")["messages"][0]["files"], [{"name": "a.py", "text": "x=1"}])

    def test_image_data_is_not_saved(self) -> None:
        msgs = [{"role": "user", "content": "hi", "files": [{"name": "t.png", "kind": "image", "image": "AAAA"}]}]
        store.save("img", "gemma3", msgs)
        self.assertEqual(store.load("img")["messages"][0]["files"], [{"name": "t.png", "kind": "image"}])

    def test_long_titles_are_cut(self) -> None:
        store.save("long", "gemma3", self.messages("x" * 200))
        title = store.listing()[0]["title"]
        self.assertEqual(len(title), store.TITLE_LENGTH)
        self.assertTrue(title.endswith("…"))

    def test_listing_is_newest_first(self) -> None:
        store.save("old", "gemma3", self.messages("old"))
        time.sleep(0.01)
        store.save("new", "gemma3", self.messages("new"))
        self.assertEqual([chat["id"] for chat in store.listing()], ["new", "old"])

    def test_listing_skips_damaged_files(self) -> None:
        store.save("good", "gemma3", self.messages())
        (store.transcripts_dir() / "bad.json").write_text("{nope", encoding="utf-8")
        self.assertEqual([chat["id"] for chat in store.listing()], ["good"])

    def test_rename_survives_later_saves(self) -> None:
        store.save("r1", "gemma3", self.messages())
        store.rename("r1", "  My   chat ")
        self.assertEqual(store.load("r1")["title"], "My chat")
        store.save("r1", "gemma3", self.messages())
        self.assertEqual(store.listing()[0]["title"], "My chat")

    def test_rename_rejects_bad_input(self) -> None:
        store.save("r2", "gemma3", self.messages())
        with self.assertRaises(ValueError):
            store.rename("r2", "   ")
        with self.assertRaises(ValueError):
            store.rename("../x", "a")
        with self.assertRaises(FileNotFoundError):
            store.rename("missing", "a")

    def test_draft_round_trip_and_keeps_files(self) -> None:
        files = [{"name": "a.py", "text": "x=1"}, {"name": "p.png", "image": "AAAA"}, {"name": 3}]
        self.assertTrue(store.save_draft("new", "half a thought", files))
        self.assertEqual(store.load_draft("new"), {"text": "half a thought", "files": files[:2]})
        store.save_draft("new", "more text")  # files not sent: keep them
        self.assertEqual(store.load_draft("new")["files"], files[:2])
        self.assertTrue((store.drafts_dir() / "new.json").is_file())

    def test_empty_draft_is_removed(self) -> None:
        store.save_draft("c1", "hi")
        self.assertFalse(store.save_draft("c1", "", []))
        with self.assertRaises(FileNotFoundError):
            store.load_draft("c1")

    def test_draft_ids_are_checked_and_deleting_chat_removes_draft(self) -> None:
        with self.assertRaises(ValueError):
            store.save_draft("../x", "a")
        store.save("c2", "gemma3", self.messages())
        store.save_draft("c2", "typing")
        store.delete("c2")
        with self.assertRaises(FileNotFoundError):
            store.load_draft("c2")

    def test_delete(self) -> None:
        store.save("gone", "gemma3", self.messages())
        self.assertTrue(store.delete("gone"))
        self.assertFalse(store.delete("gone"))
        self.assertEqual(store.listing(), [])

    def test_load_missing_chat(self) -> None:
        with self.assertRaises(FileNotFoundError):
            store.load("missing")

    def test_bad_ids_are_refused(self) -> None:
        for bad in ["../x", "a/b", "a.b", "", "x" * 65]:
            with self.assertRaises(ValueError):
                store.save(bad, "gemma3", self.messages())
            with self.assertRaises(ValueError):
                store.load(bad)

    def test_bad_messages_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            store.save("x", "gemma3", [])
        with self.assertRaises(ValueError):
            store.save("x", "gemma3", [{"role": "system", "content": "no"}])
        with self.assertRaises(ValueError):
            store.save("x", "", self.messages())

    def test_no_temp_files_are_left_behind(self) -> None:
        store.save("tidy", "gemma3", self.messages())
        self.assertEqual([p.name for p in store.transcripts_dir().iterdir()], ["tidy.json"])


if __name__ == "__main__":
    unittest.main()
