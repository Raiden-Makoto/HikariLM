import os
import tempfile
import unittest

from helper import budget, context, store

GIB = 1024 ** 3

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
    def setUp(self):
        for name in ("OLLAMA_KV_CACHE_TYPE", "OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH", "HIKARILM_MEMORY_GB",
                     "OLLAMA_HOST"):
            old = os.environ.pop(name, None)
            if old is not None:
                self.addCleanup(os.environ.__setitem__, name, old)

    def test_cache_size_comes_from_the_model_shape(self):
        # 28 layers x 4 kv heads x (128 + 128) x 2 bytes
        self.assertEqual(budget.per_token(QWEN), 28 * 4 * 256 * 2)
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        self.assertEqual(budget.per_token(QWEN), 28 * 4 * 256)

    def test_unknown_shape_uses_a_cautious_guess(self):
        self.assertEqual(budget.per_token({}), budget.FALLBACK_PER_TOKEN)

    def test_default_context(self):
        self.assertEqual(budget.default_source(QWEN)[0], 4096)
        self.assertEqual(budget.default_source({"parameters": "num_ctx 8192\nstop x"})[0], 8192)
        os.environ["OLLAMA_CONTEXT_LENGTH"] = "16384"
        self.assertEqual(budget.default_source(QWEN), (16384, "OLLAMA_CONTEXT_LENGTH"))
        self.assertEqual(budget.default_source({"parameters": "num_ctx 131072"}), (131072, "model"))
        del os.environ["OLLAMA_CONTEXT_LENGTH"]
        self.assertEqual(budget.default_source(QWEN), (4096, "ollama"))

    def test_plan_reports_the_default_and_takes_any_length(self):
        plan = budget.plan("qwen", 12288, QWEN, 5 * GIB, [], (32 * GIB, 24 * GIB))
        self.assertEqual((plan["num_ctx"], plan["default"], plan["default_from"]), (12288, 4096, "ollama"))
        big = {**QWEN, "parameters": "num_ctx 131072"}
        plan = budget.plan("qwen", None, big, int(4.5 * GIB), [], (8 * GIB, 6 * GIB))
        self.assertEqual(plan["default"], 131072)
        self.assertTrue(plan["clamped"])

    def test_asked_for_context_fits_when_there_is_room(self):
        plan = budget.plan("qwen", 16384, QWEN, 5 * GIB, [], (32 * GIB, 24 * GIB))
        self.assertEqual(plan["num_ctx"], 16384)
        self.assertFalse(plan["clamped"])

    def test_context_is_cut_to_what_fits(self):
        # 8 GB machine, 4 GB free, a 5 GB model: about 1 GB left for the cache.
        plan = budget.plan("qwen", 131072, QWEN, int(4.5 * GIB), [], (8 * GIB, 6 * GIB))
        self.assertTrue(plan["clamped"])
        self.assertLess(plan["num_ctx"], 131072)
        self.assertLessEqual(plan["num_ctx"], 32768)  # never past what the model was trained for
        self.assertEqual(plan["num_ctx"] % budget.STEP, 0)
        used = int(4.5 * GIB) + plan["num_ctx"] * budget.per_token(QWEN) + budget.RESERVE
        self.assertLessEqual(used, 8 * GIB * budget.SHARE)

    def test_a_model_that_does_not_fit_is_refused(self):
        with self.assertRaises(budget.NoRoom) as caught:
            budget.plan("big", None, QWEN, 20 * GIB, [], (16 * GIB, 12 * GIB))
        self.assertIn("Not enough memory", str(caught.exception))

    def test_the_model_being_replaced_counts_as_free(self):
        loaded = [{"name": "gemma3:12b", "size": 9 * GIB}, {"name": "nomic-embed-text:latest", "size": GIB}]
        with self.assertRaises(budget.NoRoom):
            budget.plan("qwen", 8192, QWEN, 5 * GIB, [], (16 * GIB, 2 * GIB))
        plan = budget.plan("qwen", 8192, QWEN, 5 * GIB, loaded, (16 * GIB, 2 * GIB))
        self.assertEqual(plan["num_ctx"], 8192)

    def test_memory_is_only_read_when_ollama_is_here(self):
        for host in ("", "127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "0.0.0.0"):
            self.assertTrue(budget.ollama_is_local(host), host)
        self.assertFalse(budget.ollama_is_local("http://192.168.1.20:11434"))
        self.assertFalse(budget.ollama_is_local("gpu-box:11434"))

    def test_this_computers_memory_can_be_read(self):
        mem = budget.memory()
        self.assertIsNotNone(mem)
        self.assertGreater(mem[0], 0)
        os.environ["HIKARILM_MEMORY_GB"] = "8"
        self.assertEqual(budget.memory(), (8 * GIB, None))

    def test_parallel_slots_multiply_the_cache(self):
        os.environ["OLLAMA_NUM_PARALLEL"] = "4"
        self.assertEqual(budget.per_token(QWEN), 28 * 4 * 256 * 2 * 4)


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               20000.
Pages active:                            300000.
Pages inactive:                          290000.
Pages speculative:                         5000.
Pages throttled:                              0.
Pages wired down:                        150000.
Pages purgeable:                          10000.
"Translation faults":                 123456789.
Pages copy-on-write:                    1234567.
Pages zero filled:                     98765432.
Pages reactivated:                        12345.
Pages purged:                              6789.
File-backed pages:                       500000.
Anonymous pages:                         350000.
Pages stored in compressor:              120000.
Pages occupied by compressor:             50000.
Decompressions:                           11111.
Compressions:                             22222.
Pageins:                                  33333.
Pageouts:                                     0.
Swapins:                                      0.
Swapouts:                                     0.
"""


class MacMemoryTest(unittest.TestCase):
    TOTAL = 1048576 * 16384   # 16 GB

    def test_cached_files_count_as_free(self):
        held = 150000 + (350000 - 10000) + 50000
        self.assertEqual(budget.mac_available(VM_STAT, self.TOTAL), self.TOTAL - held * 16384)
        old = (20000 + 290000 + 5000 + 10000) * 16384
        self.assertGreater(budget.mac_available(VM_STAT, self.TOTAL), old + 2 * budget.GIB)

    def test_cached_part_is_reported(self):
        self.assertEqual(budget.mac_cached(VM_STAT, self.TOTAL), (500000 + 10000) * 16384)

    def test_older_macos(self):
        stats = "\n".join(line for line in VM_STAT.splitlines()
                          if not line.startswith(("Anonymous", "File-backed")))
        self.assertEqual(budget.mac_available(stats, self.TOTAL), (20000 + 290000 + 5000 + 10000) * 16384)

    def test_never_past_total(self):
        stats = VM_STAT.replace("150000.", "0.").replace("350000.", "0.").replace("Pages occupied by compressor:             50000.", "Pages occupied by compressor: 0.")
        self.assertEqual(budget.mac_available(stats, self.TOTAL), self.TOTAL)

    def test_fits_once_the_cache_counts(self):
        # A 6 GB model with under 5 GB free or inactive, but plenty held only as cached files.
        mem = (self.TOTAL, budget.mac_available(VM_STAT, self.TOTAL))
        shown = {"model_info": {"general.architecture": "qwen2", "qwen2.block_count": 28,
                                "qwen2.attention.head_count": 28, "qwen2.attention.head_count_kv": 4,
                                "qwen2.embedding_length": 3584, "qwen2.context_length": 32768}}
        result = budget.plan("big", 4096, shown, 6 * budget.GIB, [], mem)
        self.assertEqual(result["num_ctx"], 4096)
        with self.assertRaises(budget.NoRoom):
            budget.plan("big", 4096, shown, 6 * budget.GIB, [], (self.TOTAL, (20000 + 290000 + 5000 + 10000) * 16384))


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
