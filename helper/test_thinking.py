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

    def test_only_a_possible_close_tag_is_held(self):
        split = thinking.InlineThink()
        split.feed("<think>a</think>")
        self.assertEqual(split.feed("x </"), ("", "x "))
        self.assertEqual(split.feed("b> y"), ("", "</b> y"))

    ISSUE_3 = ("| Which one is better? | It depends. |\n\nLet me know if you need more details! </think> Okay, "
               "let's break down 2NF, 3NF and 4NF.\n\nThe choice depends on the system. </think>\n\n"
               "**2NF** removes partial dependencies.")

    def test_reasoning_without_an_opening_tag(self):
        for size in (1, 3, 40):
            pieces = [self.ISSUE_3[i:i + size] for i in range(0, len(self.ISSUE_3), size)]
            split = thinking.InlineThink()
            thought = answer = taken = ""
            for piece in pieces:
                t, a = split.feed(piece)
                back = split.take_back()
                self.assertTrue(answer.endswith(back))
                answer = answer[:len(answer) - len(back)] + a
                thought, taken = thought + back + t, taken + back
            t, a = split.flush()
            thought, answer = thought + t, answer + a
            self.assertEqual(answer.strip(), "**2NF** removes partial dependencies.")
            self.assertNotIn("think>", thought)
            self.assertIn("Which one is better?", thought)
            self.assertIn("The choice depends on the system.", thought)
            self.assertEqual(thinking.without_thinking(self.ISSUE_3), "**2NF** removes partial dependencies.")


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
        self.assertTrue(again[2]["content"].startswith("q2\n\n[What follows is your own reasoning"))
        self.assertIn("because", again[2]["content"])
        self.assertEqual(sent[2]["content"], "q2")

    def test_saved_chat_keeps_how_long_it_thought(self):
        kept = store.clean_messages([{"role": "user", "content": "q"},
                                     {"role": "assistant", "content": "a", "thinking": "hm",
                                      "thoughtSeconds": 12.4, "thinkingCut": True}])[1]
        self.assertEqual((kept["thoughtSeconds"], kept["thinkingCut"]), (12, True))


class WithoutThinkingTest(unittest.TestCase):
    def test_strips_reasoning(self):
        self.assertEqual(thinking.without_thinking("<think>hmm</think>\n\nThe answer."), "The answer.")
        self.assertEqual(thinking.without_thinking("The answer."), "The answer.")
        self.assertEqual(thinking.without_thinking("<think>still thinking when the tokens ran out"), "")

    def test_answer_now_says_whose_reasoning_it_is(self):
        sent = thinking.answer_now([{"role": "user", "content": "Summarize"}], "I think...")
        self.assertIn("your own reasoning", sent[-1]["content"])
        self.assertIn("not an earlier answer", sent[-1]["content"])


if __name__ == "__main__":
    unittest.main()
