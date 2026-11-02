import json
import unittest
import urllib.error
import urllib.request

from helper import context, mlx_worker, rules, store
from tests.test_engine import CODER
from tests.test_server import ChatThroughServerTest

RULE = "You are a senior engineer. Answer with the code first."


class RulesCase(unittest.TestCase):
    setUp = ChatThroughServerTest.setUp
    post = ChatThroughServerTest.post

    def call(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def first_message(self, messages: list[dict], **extra) -> dict:
        """What the model was sent first, as the fake worker tells it."""
        lines = self.post("/api/chat", {"model": CODER, "messages": messages + [{"role": "user", "content": "system"}],
                                        **extra})
        self.assertEqual([line for line in lines if line.get("error")], [])
        return json.loads("".join((line.get("message") or {}).get("content", "") for line in lines))


class SavedRulesTest(RulesCase):
    def test_kept_in_rules_md_with_a_switch(self):
        self.assertEqual(rules.load(), {"text": "", "on": True, "cut": 0, "limit": rules.MAX_CHARS})
        code, body = self.call("PUT", "/api/rules", {"text": RULE + "\r\n"})
        self.assertEqual((code, body["text"], body["on"]), (200, RULE + "\n", True))
        self.assertEqual((store.home() / "rules.md").read_text(), RULE + "\n")
        code, body = self.call("PUT", "/api/rules", {"on": False})
        self.assertEqual((body["text"], body["on"]), (RULE + "\n", False))
        code, body = self.call("GET", "/api/rules")
        self.assertEqual((body["on"], body["folder_files"]), (False, list(rules.FOLDER_FILES)))

    def test_limits(self):
        code, body = self.call("PUT", "/api/rules", {"text": "x" * (rules.MAX_CHARS + 1)})
        self.assertEqual(code, 400)
        self.assertEqual(self.call("PUT", "/api/rules", {"on": "yes"})[0], 400)
        # Edited outside HikariLM past the limit: only the first part is used, and it says so.
        store.home().mkdir(parents=True, exist_ok=True)
        (store.home() / "rules.md").write_text("y" * (rules.MAX_CHARS + 10))
        self.assertEqual((len(rules.load()["text"]), rules.load()["cut"]), (rules.MAX_CHARS, 10))


class ChatTest(RulesCase):
    def setUp(self):
        super().setUp()
        rules.save({"text": RULE})

    def test_your_rules_go_first(self):
        first = self.first_message([], rules={"on": True})
        self.assertEqual(first["role"], "system")
        self.assertIn(RULE, first["content"])
        self.assertEqual(first["count"], 2)

    def test_before_the_code_editor_instructions(self):
        editor = {"role": "system", "content": "Write code for the open file."}
        folder = {"name": "HIKARILM.md", "text": "Use tabs."}
        first = self.first_message([editor], rules={"on": True, "folder": folder})
        mine, theirs, last = (first["content"].index(RULE), first["content"].index("Use tabs."),
                              first["content"].index("Write code for the open file."))
        self.assertLess(mine, theirs)
        self.assertLess(theirs, last)
        self.assertIn("from its HIKARILM.md", first["content"])
        self.assertEqual(first["count"], 2)

    def test_only_when_asked_for(self):
        # Apply, rewriting and the rest send no "rules", so they get none.
        first = self.first_message([])
        self.assertEqual((first["role"], first["content"]), ("user", "system"))
        self.assertEqual(self.first_message([], rules={"on": False})["role"], "user")

    def test_the_switch_and_a_folder_file_alone(self):
        rules.save({"on": False})
        self.assertEqual(self.first_message([], rules={"on": True})["role"], "user")
        first = self.first_message([], rules={"on": True, "folder": {"name": "AGENTS.md", "text": "Use tabs."}})
        self.assertNotIn(RULE, first["content"])
        self.assertIn("from its AGENTS.md", first["content"])

    def test_a_folder_file_must_be_one_of_the_known_names(self):
        first = self.first_message([], rules={"on": True, "folder": {"name": "notes.md", "text": "Use tabs."}})
        self.assertNotIn("Use tabs.", first["content"])

    def test_the_page_is_told_and_they_count_in_the_context(self):
        lines = self.post("/api/chat", {"model": CODER, "messages": [{"role": "user", "content": "hi"}],
                                        "rules": {"on": True}})
        told = next(line for line in lines if line.get("hikarilm") == "rules")
        self.assertEqual(told["from"], ["yours"])
        self.assertGreater(told["tokens"], 5)
        tokens = next(line for line in lines if line.get("hikarilm") == "context")["tokens"]
        plain = self.post("/api/chat", {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
        self.assertGreater(tokens, next(line for line in plain if line.get("hikarilm") == "context")["tokens"])

    def test_compaction_checkpoints_still_point_at_the_page_messages(self):
        context.save_settings({**context.load_settings(), "compaction": {"enabled": True, "threshold": 256,
                                                                          "retention": 10, "max_tokens": 100}})
        chat = []
        for n in range(12):
            chat += [{"role": "user", "content": f"question {n} " + "word " * 60},
                     {"role": "assistant", "content": f"answer {n} " + "word " * 60}]
        chat.append({"role": "user", "content": "last"})
        found = []
        for extra in ({}, {"rules": {"on": True}}):
            lines = self.post("/api/chat", {"model": CODER, "messages": chat, **extra})
            done = [line for line in lines if line.get("hikarilm") == "compaction" and line.get("done")]
            self.assertEqual(len(done), 1, lines[:4])
            found.append(done[0]["checkpoint"])
        self.assertEqual(found[0], found[1])


class PerChatTest(RulesCase):
    def test_a_chat_can_turn_them_off(self):
        store.save("abc", CODER, [{"role": "user", "content": "hi"}])
        self.assertTrue(store.listing()[0]["rules"])
        code, body = self.call("PATCH", "/api/chats/abc", {"rules": False})
        self.assertEqual((code, body["rules"]), (200, False))
        self.assertFalse(store.listing()[0]["rules"])
        # Saving a reply keeps it off.
        store.save("abc", CODER, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        self.assertFalse(store.listing()[0]["rules"])
        self.call("PATCH", "/api/chats/abc", {"rules": True})
        self.assertNotIn("rules", store.load("abc"))
        self.assertEqual(self.call("PATCH", "/api/chats/abc", {"rules": "no"})[0], 400)


class TemplateTest(unittest.TestCase):
    def test_a_template_without_a_system_role_gets_them_in_the_first_message(self):
        def gemma2(chat):
            if chat[0]["role"] == "system":
                raise ValueError("System role not supported")
            return chat

        chat = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "hi"}]
        self.assertEqual(mlx_worker.with_system(gemma2, chat), [{"role": "user", "content": "Rules.\n\nhi"}])
        self.assertEqual(mlx_worker.with_system(lambda turns: turns, chat), chat)

        def broken(chat):
            raise ValueError("something else")

        with self.assertRaises(ValueError):
            mlx_worker.with_system(broken, chat)


if __name__ == "__main__":
    unittest.main()
