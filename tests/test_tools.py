import json
import time
import unittest

from helper import context, mlx_worker, store, tools
from helper.calculator import MAX_CHARS, CalcError, calculate
from tests.test_engine import CODER
from tests.test_server import ChatThroughServerTest


class CalculatorTest(unittest.TestCase):
    def test_right_answers(self):
        cases = {
            "sqrt(4761)": "69", "2^3 + 1": "9", "-2**2": "-4", "5!": "120", "(2+3)!": "120",
            "7/2": "3.5", "10/5": "2", "7//2": "3", "7 % 3": "1", "sin(pi/6)": "0.5", "sind(30)": "0.5",
            "cos(0)": "1", "degrees(atan(1))": "45", "log(8, 2)": "3", "ln(e)": "1", "log10(1000)": "3",
            "exp(0)": "1", "1/3": "0.333333333333333", "comb(52, 5)": "2598960", "perm(5, 2)": "20",
            "√2": "1.4142135623731", "√(16)": "4", "3×4÷2": "6", "root(-27, 3)": "-3", "cbrt(-8)": "-2",
            "hypot(3, 4)": "5", "gcd(48, 18)": "6", "lcm(4, 6)": "12", "round(2.567, 2)": "2.57",
            "floor(-2.5)": "-3", "ceil(2.1)": "3", "abs(-3)": "3", "max(1, 5, 3)": "5", "0.1 + 0.2": "0.3",
            "gamma(5)": "24", "1e20": "1e+20", "tau / pi": "2",
        }
        for expression, expected in cases.items():
            with self.subTest(expression):
                self.assertEqual(calculate(expression), expected)

    def test_double_precision_like_a_calculator(self):
        self.assertEqual(calculate("1234567 * 7654321"), "9449772114007")
        self.assertEqual(calculate("999999999999999"), "999999999999999")
        self.assertEqual(calculate("2^64 - 1"), "1.84467440737096e+19")
        self.assertEqual(calculate("2**100"), "1.26765060022823e+30")
        self.assertEqual(calculate("170!"), "7.257415615308e+306")
        self.assertEqual(calculate("exp(709)"), "8.21840746155497e+307")
        self.assertEqual(calculate("1e-300 * 1e-7"), "1e-307")
        self.assertEqual(calculate("0 * 1e-300"), "0")

    def test_outside_the_range_is_overflow_or_underflow(self):
        for expression in ("1e308 * 10", "1e400", "10^400", "9**9**9", "171!", "exp(710)", "gamma(172)",
                           "comb(1030, 515)", "comb(10**6, 5*10**5)", "-1e308 * 10", "1" + "0" * 400):
            with self.subTest(expression), self.assertRaises(CalcError) as caught:
                calculate(expression)
            self.assertTrue(str(caught.exception).startswith("overflow"), str(caught.exception))
        for expression in ("1e-200 * 1e-200", "1e-310", "2^-2000", "0.5^2000", "exp(-1000)", "1e-300 / 1e10",
                           "1e-308 - 0.99e-308"):
            with self.subTest(expression), self.assertRaises(CalcError) as caught:
                calculate(expression)
            self.assertTrue(str(caught.exception).startswith("underflow"), str(caught.exception))

    def test_mistakes_come_back_in_words(self):
        cases = {"1/0": "division by zero", "5 % 0": "division by zero", "0^-1": "division by zero",
                 "sqrt(-1)": "domain", "(-8)**(1/3)": "not a real number",
                 "x + 1": "unknown name 'x'", "sqrt": "is a function", "factorial(2.5)": "whole number",
                 "factorial(-1)": "negative", "2 +": "could not be read", "(1": "could not be read",
                 "": "give an expression"}
        for expression, words in cases.items():
            with self.subTest(expression), self.assertRaises(CalcError) as caught:
                calculate(expression)
            self.assertIn(words, str(caught.exception))

    def test_nothing_but_math(self):
        for expression in ("__import__('os').system('ls')", "open('/etc/passwd')", "(1).real", "'a' * 3",
                           "[1, 2]", "2, 3", "lambda: 1", "True + 1", "x := 3", "[x for x in range(9)]",
                           "sqrt.__class__", "print(1)", "eval('1')", "{1: 2}", "1 if 1 else 2", "1 < 2",
                           "sqrt(x=4)", "a.b(1)"):
            with self.subTest(expression), self.assertRaises(CalcError):
                calculate(expression)

    def test_limits_answer_quickly(self):
        started = time.monotonic()
        for expression in ("9**9**9", "factorial(10**9)", "comb(10**15, 10**14)", "perm(10**12, 10**6)",
                           "gcd(1e300, 1)", "1" + "+1" * 400):
            with self.subTest(expression):
                try:
                    calculate(expression)
                except CalcError:
                    pass
        with self.assertRaises(CalcError):
            calculate("1" * (MAX_CHARS + 1))
        self.assertLess(time.monotonic() - started, 1.0)


def watch(pieces) -> tuple[str, dict | None, tools.CallWatch]:
    seen = tools.CallWatch()
    shown = "".join(seen.feed(piece) for piece in pieces)
    return shown + seen.flush(), seen.call, seen


def chunks(text: str, size: int = 3) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


class CallWatchTest(unittest.TestCase):
    def test_qwen_hermes_call_in_pieces(self):
        text = 'Let me check. <tool_call>\n{"name": "calculate", "arguments": {"expression": "sqrt(4761)"}}\n</tool_call> 42!'
        shown, call, seen = watch(chunks(text))
        self.assertEqual(shown, "Let me check. ")
        self.assertEqual(call, {"name": "calculate", "expression": "sqrt(4761)"})
        self.assertEqual(seen.written(), "Let me check. ")

    def test_other_makers_formats(self):
        formats = {
            "mistral": '[TOOL_CALLS] [{"name": "calculate", "arguments": {"expression": "2^10"}}]',
            "deepseek": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>calculate\n```json\n"
                        '{"expression": "2^10"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            "gemma": '```tool_code\nprint(calculate(expression="2^10"))\n```',
            "arguments as a string": '<tool_call>{"name": "calculate", "arguments": "{\\"expression\\": \\"2^10\\"}"}</tool_call>',
            "parameters": '<tool_call>{"name": "calculate", "parameters": {"expression": "2^10"}}</tool_call>',
        }
        for name, text in formats.items():
            with self.subTest(name):
                shown, call, _ = watch(chunks("Sure. " + text, 4))
                self.assertEqual(shown, "Sure. ")
                self.assertEqual(call["expression"], "2^10")
                self.assertNotIn("error", call)

    def test_an_unknown_tool_is_answered_with_an_error(self):
        _, call, _ = watch(['<tool_call>{"name": "python", "arguments": {"code": "1"}}</tool_call>'])
        self.assertIn("no tool named python", tools.run(call)["error"])

    def test_text_that_only_looks_like_a_start_is_shown(self):
        self.assertEqual(watch(chunks("a < b and [TOOL is fine", 2))[:2], ("a < b and [TOOL is fine", None))
        self.assertEqual(watch(["<tool_call> never finished"])[:2], ("<tool_call> never finished", None))

    def test_run_and_the_messages_it_adds(self):
        answer = tools.run({"name": "calculate", "expression": "1/0"})
        self.assertEqual(answer, {"name": "calculate", "expression": "1/0", "error": "division by zero"})
        call_message, result = tools.as_messages("Let me see.", {}, tools.run({"name": "calculate", "expression": "6*7"}), 3)
        self.assertEqual(call_message["tool_calls"][0]["function"], {"name": "calculate", "arguments": {"expression": "6*7"}})
        self.assertEqual(call_message["tool_calls"][0]["id"], "calc00003")
        self.assertEqual((result["role"], result["content"], result["tool_call_id"]), ("tool", "42", "calc00003"))


class WorkerChatTest(unittest.TestCase):
    CHAT = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "sqrt 4761?"},
            *tools.as_messages("", {}, tools.run({"name": "calculate", "expression": "sqrt(4761)"}), 1)]

    def test_a_template_with_tools_gets_them_as_they_are(self):
        chat = mlx_worker.tool_chat(self.CHAT, tools.TOOLS, tools.AS_TEXT, True)
        self.assertEqual(chat[0]["content"], "Rules.")
        self.assertEqual(chat[2]["tool_calls"][0]["function"]["name"], "calculate")
        self.assertEqual((chat[3]["role"], chat[3]["content"]), ("tool", "69"))

    def test_one_without_gets_text(self):
        chat = mlx_worker.tool_chat(self.CHAT, tools.TOOLS, tools.AS_TEXT, False)
        self.assertTrue(chat[0]["content"].startswith("Rules.\n\nYou have a calculator tool."))
        self.assertIn('<tool_call>\n{"name": "calculate", "arguments": {"expression": "sqrt(4761)"}}', chat[2]["content"])
        self.assertNotIn("tool_calls", chat[2])
        self.assertEqual(chat[3], {"role": "user", "content": "<tool_response>\n69\n</tool_response>"})
        # With the tool off (or used up), its description is left out.
        self.assertEqual(mlx_worker.tool_chat(self.CHAT[:2], None, tools.AS_TEXT, False)[0]["content"], "Rules.")


class LoopTest(unittest.TestCase):
    setUp = ChatThroughServerTest.setUp
    post = ChatThroughServerTest.post

    def chat(self, question: str) -> tuple[str, list[dict], list[dict]]:
        lines = self.post("/api/chat", {"model": CODER, "messages": [{"role": "user", "content": question}]})
        self.assertEqual([line for line in lines if line.get("error")], [])
        self.assertTrue(lines[-1].get("done"))
        shown = "".join((line.get("message") or {}).get("content", "") for line in lines)
        return shown, [line for line in lines if line.get("hikarilm") == "tool"], lines

    def test_the_model_goes_on_from_the_result(self):
        shown, calls, _ = self.chat("calc: sqrt(4761)")
        self.assertEqual(calls, [{"hikarilm": "tool", "name": "calculate", "expression": "sqrt(4761)", "result": "69"}])
        self.assertTrue(shown.startswith("Let me work it out. The answer is 69. "))
        self.assertNotIn("tool_call", shown)
        self.assertNotIn("made it up", shown)
        sent = json.loads(shown.split(". ", 2)[2])
        self.assertEqual(sent["roles"], ["user", "assistant", "tool"])
        self.assertEqual(sent["calls"], [{"expression": "sqrt(4761)"}])
        self.assertEqual(sent["written"], ["Let me work it out. "])

    def test_an_error_goes_back_to_the_model(self):
        shown, calls, _ = self.chat("calc: 1/0")
        self.assertEqual(calls[0]["tool_error"], "division by zero")
        self.assertTrue(shown.startswith("Let me work it out. The answer is Error: division by zero. "))

    def test_a_call_from_its_thinking(self):
        _, calls, lines = self.chat("calc-think: 2^10")
        self.assertEqual(calls[0]["result"], "1024")
        thought = "".join((line.get("message") or {}).get("thinking", "") for line in lines)
        self.assertIn("I should use the calculator.", thought)
        self.assertNotIn("tool_call", thought)

    def test_twenty_calls_an_answer(self):
        shown, calls, _ = self.chat("calc-many: 1+1")
        self.assertEqual(len(calls), tools.MAX_CALLS + 1)
        self.assertTrue(all(call.get("result") == "2" for call in calls[:-1]))
        self.assertIn("20 times", calls[-1]["tool_error"])
        self.assertIn('"tools": false', shown)

    def test_the_switch(self):
        context.save_settings({**context.load_settings(), "tools": {"calculator": False}})
        shown, calls, lines = self.chat("calc: sqrt(4761)")
        self.assertEqual((shown, calls), ("no tools", []))
        plain = next(line for line in lines if line.get("hikarilm") == "context")["tokens"]
        context.save_settings({**context.load_settings(), "tools": {"calculator": True}})
        _, _, lines = self.chat("calc: sqrt(4761)")
        offered = next(line for line in lines if line.get("hikarilm") == "context")["tokens"]
        self.assertEqual(offered - plain, tools.text_tokens())

    def test_saved_with_the_chat(self):
        store.save("abc", CODER, [{"role": "user", "content": "sqrt 4761?"},
                                  {"role": "assistant", "content": "69", "tools": [
                                      {"expression": "sqrt(4761)", "result": "69"},
                                      {"expression": "1/0", "error": "division by zero"},
                                      {"expression": "no result"}, "junk"]}])
        self.assertEqual(store.load("abc")["messages"][1]["tools"],
                         [{"expression": "sqrt(4761)", "result": "69"}, {"expression": "1/0", "error": "division by zero"}])


class SettingsTest(unittest.TestCase):
    def test_on_to_start(self):
        self.assertEqual(context.clean_settings({})["tools"], {"calculator": True})
        self.assertEqual(context.clean_settings({"tools": {"calculator": False}})["tools"], {"calculator": False})


if __name__ == "__main__":
    unittest.main()
