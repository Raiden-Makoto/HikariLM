import os
import tempfile
import unittest

from helper import budget, context, store

# What /api/show reports for a qwen2.5 7B, trimmed to what matters.
QWEN = {
    "parameters": "stop \"<|im_end|>\"",
    "model_info": {
        "general.architecture": "qwen2",
        "qwen2.block_count": 28,
        "qwen2.attention.head_count": 28,
        "qwen2.attention.head_count_kv": 4,
        "qwen2.embedding_length": 3584,
        "qwen2.context_length": 32768,
    },
}


def chat(turns, chars=400):
    out = []
    for index in range(turns):
        out.append({"role": "user", "content": f"question {index} " + "q" * chars})
        out.append({"role": "assistant", "content": f"answer {index} " + "a" * chars})
    return out


class BudgetTest(unittest.TestCase):
    def test_default_context(self):
        self.assertEqual(budget.default_source(QWEN), (4096, "default"))
        self.assertEqual(budget.default_source({"parameters": "num_ctx 8192\nstop x"}), (8192, "model"))

    def test_the_length_asked_for_is_sent(self):
        plan = budget.plan(12288, QWEN)
        self.assertEqual((plan["num_ctx"], plan["default"], plan["default_from"]), (12288, 4096, "default"))
        self.assertEqual(budget.plan(None, QWEN)["num_ctx"], 4096)

    def test_never_past_what_the_model_was_trained_for(self):
        self.assertEqual(budget.plan(131072, QWEN)["num_ctx"], 32768)
        self.assertEqual(budget.plan(100, QWEN)["num_ctx"], budget.MIN_CONTEXT)


class CompactionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        old = os.environ.get("HIKARILM_HOME")
        os.environ["HIKARILM_HOME"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("HIKARILM_HOME") if old is None
                        else os.environ.__setitem__("HIKARILM_HOME", old))
        self.settings = context.load_settings()

    def test_retention_is_clamped_like_open_webui(self):
        self.assertEqual(context.clamp_retention(5), 10)
        self.assertEqual(context.clamp_retention(90), 50)
        self.assertEqual(context.clamp_retention("x"), 40)

    def test_boundary_matches_open_webui(self):
        messages = chat(5) + [{"role": "user", "content": "now"}]   # 11 messages
        # keep max(2, 11*40//100=4) -> target 7, last user turn at or before 7 is 6
        self.assertEqual(context.find_boundary(messages, 40), 6)
        self.assertEqual(messages[6]["role"], "user")
        self.assertEqual(context.find_boundary(chat(1), 40), 0)

    def test_small_chat_is_left_alone(self):
        plan = context.compaction_needed(chat(2), self.settings, 4096)
        self.assertIsNone(plan["cut"])
        self.assertEqual(plan["start"], 0)

    def test_long_chat_is_cut_at_a_user_turn(self):
        messages = chat(20) + [{"role": "user", "content": "next"}]
        plan = context.compaction_needed(messages, self.settings, 4096)
        self.assertIsNotNone(plan["cut"])
        self.assertEqual(messages[plan["cut"]]["role"], "user")
        self.assertGreater(plan["tokens"], plan["threshold"])
        self.assertEqual(plan["threshold"], int(4096 * context.AUTO_SHARE))

    def test_checkpoint_is_resumed_and_moves_forward(self):
        messages = chat(20, chars=800) + [{"role": "user", "content": "next"}]
        messages[30] = {**messages[30], "contextSummary": "earlier things"}
        plan = context.compaction_needed(messages, self.settings, 4096)
        self.assertEqual(plan["start"], 30)
        self.assertEqual(plan["summary"], "earlier things")
        self.assertIsNone(plan["cut"])  # 11 messages after the checkpoint fit
        plan = context.compaction_needed(messages, self.settings, 2048)
        self.assertGreater(plan["cut"], 30)

    def test_reported_usage_wins_over_the_estimate(self):
        messages = chat(2)
        messages[-1] = {**messages[-1], "usage": {"prompt_eval_count": 3500, "eval_count": 200}}
        messages.append({"role": "user", "content": "short"})
        plan = context.compaction_needed(messages, self.settings, 4096)
        self.assertGreaterEqual(plan["tokens"], 3700)
        self.assertIsNotNone(plan["cut"])

    def test_disabled_never_cuts(self):
        self.settings["compaction"]["enabled"] = False
        plan = context.compaction_needed(chat(30) + [{"role": "user", "content": "x"}], self.settings, 2048)
        self.assertIsNone(plan["cut"])

    def test_threshold_never_leaves_no_room_for_the_reply(self):
        self.settings["compaction"]["threshold"] = 100000
        self.assertEqual(context.threshold_for(self.settings, 8192), int(8192 * context.CAP_SHARE))
        self.settings["compaction"]["threshold"] = 3000
        self.assertEqual(context.threshold_for(self.settings, 8192), 3000)

    def test_summary_goes_into_the_system_message_and_bookkeeping_is_dropped(self):
        out = context.with_summary([{"role": "user", "content": "hi", "contextSummary": "s", "usage": {"eval_count": 1}}],
                                   "old talk")
        self.assertEqual(out[0], {"role": "system", "content": "[CONVERSATION SUMMARY]\nold talk"})
        self.assertEqual(out[1], {"role": "user", "content": "hi"})
        out = context.with_summary([{"role": "system", "content": "be brief"}, {"role": "user", "content": "x"}], "s")
        self.assertEqual(out[0]["content"], "be brief\n[CONVERSATION SUMMARY]\ns")

    def test_prompt_is_open_webuis_and_fits_the_context(self):
        compacted = chat(50, chars=2000)
        prompt = context.summary_prompt("", compacted, chat(1), "before", 3000)
        self.assertIn("### Messages Being Compacted:", prompt)
        self.assertIn("USER: question 0", prompt)
        self.assertIn("before", prompt)
        self.assertLessEqual(len(prompt), 3000 * 4 + 500)

    def test_fallback_summary(self):
        text = context.fallback_summary("prev", [{"role": "user", "content": "x" * 900}])
        self.assertTrue(text.startswith("prev\n- user: "))
        self.assertLessEqual(len(text), 4000)

    def test_settings_round_trip_and_validation(self):
        saved = context.save_settings({"context": {"all": 8192, "models": {"qwen": 16384}},
                                       "compaction": {"retention": 70, "threshold": "", "enabled": False}})
        self.assertEqual(saved["compaction"]["retention"], 50)
        self.assertIsNone(saved["compaction"]["threshold"])
        self.assertEqual(context.load_settings(), saved)
        self.assertEqual(context.requested_context(saved, "qwen"), 16384)
        self.assertEqual(context.requested_context(saved, "other"), 8192)
        with self.assertRaises(ValueError):
            context.save_settings({"context": {"all": 10}})

    def test_store_keeps_checkpoints_and_usage(self):
        store.save("c1", "m", [{"role": "user", "content": "q", "contextSummary": " sum "},
                               {"role": "assistant", "content": "a", "usage": {"prompt_eval_count": 5, "eval_count": True}}])
        loaded = store.load("c1")["messages"]
        self.assertEqual(loaded[0]["contextSummary"], "sum")
        self.assertEqual(loaded[1]["usage"], {"prompt_eval_count": 5})


if __name__ == "__main__":
    unittest.main()
