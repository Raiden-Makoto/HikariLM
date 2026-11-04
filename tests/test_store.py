import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helper import store


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        previous = os.environ.get("HIKARILM_HOME")
        os.environ["HIKARILM_HOME"] = self.folder.name
        self.addCleanup(self._restore, previous)

    @staticmethod
    def _restore(previous) -> None:
        if previous is None:
            os.environ.pop("HIKARILM_HOME", None)
        else:
            os.environ["HIKARILM_HOME"] = previous

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

    def test_pin_and_folder_survive_saves_and_show_in_listing(self) -> None:
        store.save("p1", "gemma3", self.messages())
        store.patch("p1", {"pinned": True, "folder": "  Trees   homework "})
        store.save("p1", "gemma3", self.messages("another"))
        item = store.listing()[0]
        self.assertTrue(item["pinned"])
        self.assertEqual(item["folder"], "Trees homework")
        store.patch("p1", {"pinned": False, "folder": None})
        item = store.listing()[0]
        self.assertFalse(item["pinned"])
        self.assertEqual(item["folder"], "")
        self.assertNotIn("folder", store.load("p1"))

    def test_patch_changes_only_what_is_given_and_checks_input(self) -> None:
        store.save("p2", "gemma3", self.messages("hello"))
        store.patch("p2", {"pinned": True})
        self.assertEqual(store.load("p2")["title"], "Hello")
        for bad in ({}, {"pinned": "yes"}, {"folder": 5}, {"title": " "}):
            with self.assertRaises(ValueError):
                store.patch("p2", bad)
        with self.assertRaises(FileNotFoundError):
            store.patch("nope", {"pinned": True})

    def _chats_in_folder(self) -> None:
        for name, folder, pinned in (("a", "Homework", False), ("b", "Homework", True), ("c", "Other", False), ("d", "", False)):
            store.save(name, "gemma3", self.messages(name))
            fields = {"pinned": pinned}
            if folder:
                fields["folder"] = folder
            store.patch(name, fields)

    def test_deleting_a_folder_keeps_every_chat(self) -> None:
        self._chats_in_folder()
        result = store.rename_folder("Homework", None)
        self.assertEqual((result["changed"], result["failed"]), (2, 0))
        by_id = {chat["id"]: chat for chat in store.listing()}
        self.assertEqual(len(by_id), 4)
        self.assertEqual(by_id["a"]["folder"], "")
        self.assertEqual(by_id["b"]["folder"], "")
        self.assertTrue(by_id["b"]["pinned"])  # pinning is untouched
        self.assertEqual(by_id["c"]["folder"], "Other")
        self.assertEqual(len(store.load("a")["messages"]), 2)
        # Deleting it again, or one that never existed, changes nothing and is not an error.
        self.assertEqual(store.rename_folder("Homework", None)["changed"], 0)
        self.assertEqual(store.rename_folder("Nope", "")["changed"], 0)

    def test_renaming_a_folder_and_merging(self) -> None:
        self._chats_in_folder()
        store.rename_folder("Homework", "School")
        self.assertEqual({c["folder"] for c in store.listing() if c["id"] in "ab"}, {"School"})
        store.rename_folder("School", "other")  # merges, keeping the existing spelling
        self.assertEqual({c["folder"] for c in store.listing() if c["id"] in "abc"}, {"Other"})
        store.rename_folder("Other", "OTHER")  # a change of case alone is allowed
        self.assertEqual({c["folder"] for c in store.listing() if c["id"] in "abc"}, {"OTHER"})
        with self.assertRaises(ValueError):
            store.rename_folder("  ", "x")

    def test_folder_names_ignore_case_when_moving_a_chat(self) -> None:
        store.save("m1", "gemma3", self.messages())
        store.save("m2", "gemma3", self.messages())
        store.patch("m1", {"folder": "Homework"})
        store.patch("m2", {"folder": "  homework "})
        self.assertEqual({c["folder"] for c in store.listing()}, {"Homework"})

    def test_a_damaged_chat_file_does_not_stop_a_folder_change(self) -> None:
        self._chats_in_folder()
        (store.transcripts_dir() / "broken.json").write_text("{not json", encoding="utf-8")
        result = store.rename_folder("Homework", None)
        self.assertEqual((result["changed"], result["failed"]), (2, 0))

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


class OldHomeTest(unittest.TestCase):
    """~/.lightusagi, from before the rename, moves to ~/.hikarilm once."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.home = Path(folder.name)
        patcher = mock.patch.object(Path, "home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = os.environ.pop("HIKARILM_HOME", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "HIKARILM_HOME", saved)

    def make_old(self) -> Path:
        old = self.home / ".lightusagi"
        (old / "transcripts").mkdir(parents=True)
        (old / "transcripts" / "a.json").write_text("{}")
        (old / "settings.json").write_text(json.dumps({"pictures": {"models": f"flux2-klein: --diffusion-model {old}/models/k.gguf"}}))
        return old

    def test_moves_the_folder_and_fixes_saved_paths(self) -> None:
        old = self.make_old()
        self.assertIn("Moved", store.move_old_home())
        new = self.home / ".hikarilm"
        self.assertFalse(old.exists())
        self.assertTrue((new / "transcripts" / "a.json").is_file())
        models = json.loads((new / "settings.json").read_text())["pictures"]["models"]
        self.assertEqual(models, f"flux2-klein: --diffusion-model {new}/models/k.gguf")
        self.assertEqual(store.move_old_home(), "")

    def test_leaves_both_alone_when_the_new_one_exists(self) -> None:
        old = self.make_old()
        (self.home / ".hikarilm").mkdir()
        self.assertEqual(store.move_old_home(), "")
        self.assertTrue(old.is_dir())

    def test_nothing_to_move(self) -> None:
        self.assertEqual(store.move_old_home(), "")
        self.assertFalse((self.home / ".hikarilm").exists())


if __name__ == "__main__":
    unittest.main()
