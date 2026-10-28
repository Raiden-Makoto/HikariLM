import os
import tempfile
import unittest

from helper import documents
from tests.test_notes import fake_embed


def paper(pages=6, per_page=4):
    """A paper with page marks, like the page's PDF reader writes, and one topic per page."""
    topics = ["introduction motivation", "related work transformers", "method attention kernel",
              "experiments datasets imagenet", "results accuracy table", "conclusion future"]
    out = []
    for page in range(1, pages + 1):
        topic = topics[(page - 1) % len(topics)]
        out.append(f"\n--- page {page} ---\n")
        for paragraph in range(per_page):
            out.append(f"Page {page} paragraph {paragraph} about {topic}. " + "filler words here " * 20 + "\n\n")
    return "".join(out)


class DocumentsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        old = os.environ.get("LIGHTUSAGI_HOME")
        os.environ["LIGHTUSAGI_HOME"] = self._tmp.name
        self.addCleanup(lambda: os.environ.__setitem__("LIGHTUSAGI_HOME", old) if old is not None
                        else os.environ.pop("LIGHTUSAGI_HOME", None))
        text = paper()
        self.doc = documents.gather([{"role": "user", "content": "hi", "files": [{"name": "paper.pdf", "text": text}]}])[0]

    def test_split_keeps_pages_and_size(self):
        chunks = documents.split(paper())
        self.assertTrue(all(len(chunk["text"]) <= documents.CHUNK for chunk in chunks))
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[-1]["page"], 6)
        self.assertNotIn("--- page", "".join(chunk["text"] for chunk in chunks))
        self.assertTrue(all("Page 3 " not in chunk["text"] or chunk["page"] in (2, 3) for chunk in chunks))

    def test_gather_once_each_in_order(self):
        messages = [{"role": "user", "content": "a", "files": [{"name": "a.txt", "text": "one"}]},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "b", "files": [{"name": "a copy.txt", "text": "one"},
                                                              {"name": "b.txt", "text": "two"}]}]
        docs = documents.gather(messages)
        self.assertEqual([(doc["name"], doc["message"]) for doc in docs], [("a.txt", 0), ("b.txt", 2)])

    def test_whole_document_requests(self):
        for question in ("Give me a detailed summary", "summarize this", "What is this paper about?",
                         "key findings?", "tl;dr", "go through it section by section", ""):
            self.assertTrue(documents.wants_whole(question), question)
        for question in ("What dataset did they use?", "explain equation 3", "who are the authors"):
            self.assertFalse(documents.wants_whole(question), question)

    def test_excerpts_fit_and_follow_the_question(self):
        for embed in (fake_embed, None):
            text, info = documents.excerpts("Which datasets, imagenet?", [self.doc], 900, embed)
            self.assertLessEqual(documents.tokens(text), 900 + 150)
            self.assertIn("Page 1 paragraph 0", text)        # the start always comes along
            self.assertIn("imagenet", text)
            self.assertIn("[page 4]", text)
            self.assertIn("[…]", text)
            self.assertEqual(info["search"], "meaning" if embed else "words")
            self.assertLess(info["files"][0]["used"], info["files"][0]["total"])

    def test_vectors_are_kept(self):
        calls = []

        def counting(texts):
            calls.append(len(texts))
            return fake_embed(texts)

        documents.excerpts("imagenet", [self.doc], 900, counting)
        first = sum(calls)
        documents.excerpts("attention kernel", [self.doc], 900, counting)
        self.assertEqual(sum(calls) - first, 1)   # only the question

    def test_embedding_failure_falls_back_to_words(self):
        def broken(texts):
            raise OSError("nomic-embed-text is not installed")

        text, info = documents.excerpts("imagenet", [self.doc], 900, broken)
        self.assertEqual(info["search"], "words")
        self.assertIn("imagenet", text)

    def test_notes_read_everything_and_are_kept(self):
        asked = []

        def ask(prompt, limit):
            asked.append((prompt, limit))
            return f"- note {len(asked)}"

        seen = []
        text, info = documents.notes([self.doc], "qwen3:8b", 2048, 1500, ask,
                                     lambda name, part, parts: seen.append((part, parts)) or True)
        parts = info["files"][0]["parts"]
        self.assertGreater(parts, 1)
        self.assertEqual(len(asked), parts)
        self.assertEqual(seen, [(n, parts) for n in range(1, parts + 1)])
        whole = "".join(prompt for prompt, _ in asked)
        for page in range(1, 7):
            self.assertIn(f"Page {page} paragraph 3", whole)   # nothing was cut off
        self.assertTrue(all(documents.tokens(prompt) <= 2048 - documents.NOTE_MAX for prompt, _ in asked))
        self.assertIn("- note 1", text)
        self.assertIn(f"Part {parts} of {parts}", text)
        again, _ = documents.notes([self.doc], "qwen3:8b", 2048, 1500, ask, lambda *a: True)
        self.assertEqual(len(asked), parts)
        self.assertEqual(again, text)

    def test_notes_condensed_to_fit(self):
        def ask(prompt, limit):
            return "word " * min(limit, 400)

        text, _ = documents.notes([self.doc], "m", 2048, 400, ask, lambda *a: True)
        self.assertLessEqual(documents.tokens(text), 400 + 60)

    def test_stopped_while_reading(self):
        calls = []
        got = documents.notes([self.doc], "m", 2048, 1500, lambda p, n: calls.append(p) or "x",
                              lambda name, part, parts: part < 2)
        self.assertIsNone(got)
        self.assertEqual(len(calls), 1)

    def test_reasoning_is_taken_out_of_notes(self):
        # An R1 distill writes its reasoning into the answer; one call runs out of tokens while still thinking.
        answers = iter(["<think>Okay, the user wants notes on part 1.</think>\n\n- Part one covers the introduction and motivation.",
                        "<think>Okay, so I need to write notes but the user didn't give me the document"])

        def ask(prompt, limit):
            return next(answers, "- Later parts: method, experiments and results in detail.")

        text, _ = documents.notes([self.doc], "deepseek-r1:8b", 2048, 1500, ask, lambda *a: True)
        self.assertNotIn("<think>", text)
        self.assertNotIn("the user", text)
        self.assertIn("- Part one covers the introduction", text)
        self.assertIn("Page 2 paragraph", text)    # the part's own text stands in for the empty note

    def test_old_notes_are_not_reused(self):
        data = documents.load(self.doc)
        data["notes"] = {"deepseek-r1:8b|1348|224": ["<think>no document"] * 9}
        documents.save(self.doc, data)
        calls = []
        documents.notes([self.doc], "deepseek-r1:8b", 2048, 1500,
                        lambda p, n: calls.append(p) or "- a proper note about this part of the paper", lambda *a: True)
        self.assertTrue(calls)

    def test_the_model_is_told_the_document_is_there(self):
        notes_text, _ = documents.notes([self.doc], "m", 2048, 1500, lambda p, n: "- a proper note about the paper here",
                                        lambda *a: True)
        passages, _ = documents.excerpts("imagenet", [self.doc], 900, None)
        for text in (notes_text, passages):
            self.assertIn("The user attached the file paper.pdf", text)
            self.assertIn("do not say the document was not provided", text)
            self.assertIn('<document name="paper.pdf"', text)
            self.assertTrue(text.rstrip().endswith("</document>"))

    def test_rooms(self):
        self.assertEqual(documents.reply_room(4096), 819)
        self.assertEqual(documents.reply_room(2048), 768)
        self.assertEqual(documents.thinking_room(4096, 2048), 512)
        self.assertEqual(documents.thinking_room(32768, 2048), 2048)
        self.assertEqual(documents.thinking_room(32768, None), 4096)


if __name__ == "__main__":
    unittest.main()
