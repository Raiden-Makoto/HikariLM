import os
import re
import tempfile
import time
import unittest
import zlib

from helper import descriptions, notes, store

DIM = 64


def fake_embed(texts):
    """A word-count vector: texts sharing words are close, the way real embeddings are."""
    out = []
    for text in texts:
        vector = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in ("search", "document", "query", "question", "answer", "the", "a", "is", "of"):
                continue
            vector[zlib.crc32(word.encode()) % DIM] += 1.0
        out.append(vector)
    return out


class NotesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        old = os.environ.get("HIKARILM_HOME")
        os.environ["HIKARILM_HOME"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("HIKARILM_HOME") if old is None
                        else os.environ.__setitem__("HIKARILM_HOME", old))
        self.calls = []

        def embed(texts):
            self.calls.append(list(texts))
            return fake_embed(texts)

        self.embed = embed

    def chat(self, ident, question, answer, title="T"):
        data = {"title": title, "messages": [{"role": "user", "content": question},
                                             {"role": "assistant", "content": answer}]}
        t, texts = notes.chat_chunks(data)
        notes.sync(notes.chat_source(ident), t, texts, self.embed)

    def test_answer_is_kept_with_its_question(self):
        _, texts = notes.chat_chunks({"messages": [
            {"role": "user", "content": "How do I reverse a list?"},
            {"role": "assistant", "content": "Use reversed()."}]})
        self.assertEqual(texts, ["Question: How do I reverse a list?\nAnswer: Use reversed()."])

    def test_long_answers_are_split_at_line_breaks(self):
        answer = "\n".join(f"line {i} " + "x" * 100 for i in range(40))
        _, texts = notes.chat_chunks({"messages": [{"role": "user", "content": "q"},
                                                   {"role": "assistant", "content": answer}]})
        self.assertGreater(len(texts), 1)
        self.assertTrue(all(len(t) <= notes.CHUNK + 400 for t in texts))

    def test_search_finds_the_related_chat_only(self):
        self.chat("a", "binary tree rotation", "An AVL tree rotates left and right after an insert.", "AVL")
        self.chat("b", "pasta recipe", "Boil water, add salt and the pasta.", "Pasta")
        found = notes.search("how does avl tree rotation work", self.embed, min_score=0.3)
        self.assertEqual([n["title"] for n in found], ["AVL"])

    def test_current_chat_is_left_out(self):
        self.chat("a", "binary tree rotation", "An AVL tree rotates left and right.", "AVL")
        found = notes.search("avl tree rotation", self.embed, exclude=notes.chat_source("a"), min_score=0.3)
        self.assertEqual(found, [])

    def test_only_new_text_is_embedded(self):
        self.chat("a", "q one", "answer one", "T")
        self.calls.clear()
        self.chat("a", "q one", "answer one", "New title")
        self.assertEqual(self.calls, [])
        data = {"title": "T", "messages": [{"role": "user", "content": "q one"}, {"role": "assistant", "content": "answer one"},
                                           {"role": "user", "content": "q two"}, {"role": "assistant", "content": "answer two"}]}
        title, texts = notes.chat_chunks(data)
        notes.sync("chat-a", title, texts, self.embed)
        self.assertEqual(sum(len(c) for c in self.calls), 1)

    def test_no_notes_means_no_embedding_call(self):
        self.assertEqual(notes.search("anything at all", self.embed), [])
        self.assertEqual(self.calls, [])

    def test_one_note_per_source_and_top_limit(self):
        for i in range(5):
            self.chat(f"c{i}", "avl tree rotation", f"avl tree rotation notes {i}", f"C{i}")
        found = notes.search("avl tree rotation", self.embed, min_score=0.3)
        self.assertEqual(len(found), notes.TOP)
        self.assertEqual(len({n["source"] for n in found}), notes.TOP)

    def test_removing_and_emptying_a_source(self):
        self.chat("a", "q", "a real answer here", "T")
        self.assertEqual(notes.count()["sources"], 1)
        notes.sync("chat-a", "T", [], self.embed)
        self.assertEqual(notes.count()["sources"], 0)
        notes.remove("chat-missing")

    def test_bad_source_names_are_refused(self):
        with self.assertRaises(ValueError):
            notes.sync("../evil", "T", ["x"], self.embed)

    def test_indexer_indexes_chats_and_pictures_in_the_background(self):
        store.save("abc", "m", [{"role": "user", "content": "binary tree rotation"},
                                {"role": "assistant", "content": "An AVL tree rotates left and right."}])
        key = descriptions.image_key("aGVsbG8=")
        descriptions.save(key, "root 5, left child 3, right child 8", "gemma3", "tree.png")
        indexer = notes.Indexer(self.embed)
        indexer.everything()
        deadline = time.time() + 5
        while time.time() < deadline and notes.count()["sources"] < 2:
            time.sleep(0.05)
        self.assertEqual(notes.count()["sources"], 2)
        found = notes.search("tree with root and left child", self.embed, min_score=0.2)
        self.assertIn("Picture tree.png", [n["title"] for n in found])
        indexer.forget_chat("abc")
        self.assertEqual(notes.count()["sources"], 1)


if __name__ == "__main__":
    unittest.main()
