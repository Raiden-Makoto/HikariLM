import unittest

from helper import context, store, thinking


def split_all(pieces):
    split = thinking.InlineThink()
    thought = answer = ""
    for piece in pieces:
        t, a = split.feed(piece)
        thought, answer = thought + t, answer + a
    t, a = split.flush()
    return thought + t, answer + a


class InlineThinkTest(unittest.TestCase):
    def test_tags_split_across_pieces(self):
        self.assertEqual(split_all(["\n<th", "ink>step one", " step two</th", "ink>\n\nThe answer"]),
                         ("step one step two", "The answer"))

    def test_an_answer_without_tags_is_untouched(self):
        self.assertEqual(split_all(["<table>", " x"]), ("", "<table> x"))
        self.assertEqual(split_all(["Hello <think> later"]), ("", "Hello <think> later"))

    def test_unfinished_thinking_stays_thinking(self):
        self.assertEqual(split_all(["<think>still going", " </thi"]), ("still going </thi", ""))

    def test_nothing_is_held_once_answering(self):
        split = thinking.InlineThink()
        split.feed("<think>a</think>")
        self.assertEqual(split.feed("</"), ("", "</"))


class SettingTest(unittest.TestCase):
    def test_defaults_and_limits(self):
        self.assertEqual(thinking.clean(None), {"enabled": True, "budget": 2048})
        self.assertEqual(thinking.clean({"budget": None})["budget"], None)
        self.assertEqual(thinking.clean({"budget": 5})["budget"], thinking.LOWEST)
        self.assertEqual(thinking.clean({"budget": "x"})["budget"], 2048)
        self.assertFalse(thinking.clean({"enabled": False})["enabled"])
        self.assertEqual(context.clean_settings({})["thinking"], thinking.DEFAULTS)

    def test_think_value(self):
        can = {"capabilities": ["completion", "thinking"]}
        on, off = {"enabled": True}, {"enabled": False}
        self.assertIsNone(thinking.think_value("qwen2.5", {"capabilities": ["completion"]}, on))
        self.assertIs(thinking.think_value("qwen3", can, on), True)
        self.assertIs(thinking.think_value("qwen3", can, off), False)
        self.assertEqual(thinking.think_value("gpt-oss:20b", can, off), "low")

    def test_answer_now_keeps_the_chat_and_adds_the_reasoning(self):
        sent = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"}]
        again = thinking.answer_now(sent, " because ")
        self.assertEqual(again[:2], sent[:2])
        self.assertTrue(again[2]["content"].startswith("q2\n\n[Your reasoning so far"))
        self.assertIn("because", again[2]["content"])
        self.assertEqual(sent[2]["content"], "q2")

    def test_saved_chat_keeps_how_long_it_thought(self):
        kept = store.clean_messages([{"role": "user", "content": "q"},
                                     {"role": "assistant", "content": "a", "thinking": "hm",
                                      "thoughtSeconds": 12.4, "thinkingCut": True}])[1]
        self.assertEqual((kept["thoughtSeconds"], kept["thinkingCut"]), (12, True))


if __name__ == "__main__":
    unittest.main()
