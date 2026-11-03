import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helper import context, engine, mlx_worker, store, tools
from helper.calculator import MAX_CHARS, CalcError, calculate
from tests.test_engine import CODER, GEMMA, add_model, real_config
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
            "gpt-oss": 'to=functions.calculate<|channel|>commentary json<|message|>'
                       '{"expression": "2^10"}<|call|>',
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

    def test_json_calls_with_no_marker(self):
        # Issue 8: Qwen2.5 Coder wrote its calls as ```json blocks, Falcon3 as bare JSON.
        coder = '```json\n{\n  "name": "calculate",\n  "arguments": {\n    "expression": "sqrt(4761)"\n  }\n}\n```'
        falcon = '{\n  "name": "function",\n  "arguments": {\n    "expression": "2 ** 40"\n  }\n}'
        for text, expression in ((coder, "sqrt(4761)"), ("Sure.\n" + coder, "sqrt(4761)"), (falcon, "2 ** 40")):
            with self.subTest(text[:12]):
                shown, call, _ = watch(chunks(text, 3))
                self.assertEqual(shown, text.split("```")[0] if text.startswith("Sure") else "")
                self.assertEqual(call, {"name": "calculate", "expression": expression})

    def test_gemma_tool_code_as_plain_python(self):
        written = {"print(math.sqrt(4761))": "sqrt(4761)", "sqrt(4761)": "sqrt(4761)",
                   "[calculate(expression=2**40)]": "2 ** 40", "print(calculate(expression='2^10'))": "2^10",
                   "tools.calculate(expression=\"1234567 * 7654321\")": "1234567 * 7654321",
                   "print(np.sin(np.radians(30)) * 12.5)": "sin(radians(30)) * 12.5"}
        for code, expression in written.items():
            with self.subTest(code):
                shown, call, _ = watch(chunks("Let me see.\n```tool_code\n" + code + "\n```\nmade up", 3))
                self.assertEqual(shown, "Let me see.\n")
                self.assertEqual(call, {"name": "calculate", "expression": expression})
        self.assertEqual(tools.run(watch(["```tool_code\nprint(math.sqrt(4761))\n```"])[1])["result"], "69")
        self.assertIn("error", tools.run(watch(["```tool_code\nx = 1\ny = 2\n```"])[1]))

    def test_bare_json_after_the_reasoning(self):
        # Issue 9: DeepSeek R1 thought first, then wrote the call as bare JSON.
        text = '<think>\nI should use the calculator.\n</think>\n\n{"name":"calculate","arguments":{"expression":"sqrt(4761)"}}'
        shown, call, seen = watch(chunks(text, 3))
        self.assertEqual(shown.rstrip(), "<think>\nI should use the calculator.\n</think>")
        self.assertEqual(call, {"name": "calculate", "expression": "sqrt(4761)"})
        self.assertEqual(watch(chunks("<think>hm</think>\nIt is {1, 2}.", 2))[:2], ("<think>hm</think>\nIt is {1, 2}.", None))

    def test_wrappers_qwen_coder_made_up(self):
        # Issue 9: each of these was Qwen2.5 Coder's whole answer.
        made_up = {
            "```xml\n<function-call>\n  <name>calculate</name>\n  <arguments>\n    <expression>sqrt(4761)</expression>\n"
            "  </arguments>\n</function-call>\n```": "sqrt(4761)",
            '```xml\n<tools>\n    {\n        "name": "calculate",\n        "arguments": {\n            "expression": "2**40"\n'
            "        }\n    }\n</tools>\n```": "2**40",
            "```xml\n<function name=\"calculate\" arguments='{\"expression\": \"1234567 * 7654321\"}'/>\n```": "1234567 * 7654321",
            '<tools>\n  {\n    "name": "calculate",\n    "arguments": {\n      "expression": "sin(30 * pi / 180) * 12.5"\n'
            "    }\n  }\n</tools>": "sin(30 * pi / 180) * 12.5",
            "<function-call><name>calculate</name><arguments><expression>2 &lt; 3</expression></arguments></function-call>": "2 < 3",
        }
        for text, expression in made_up.items():
            with self.subTest(text[:30]):
                shown, call, _ = watch(chunks(text, 3))
                self.assertEqual(shown, "")
                self.assertEqual(call, {"name": "calculate", "expression": expression})

    def test_xml_that_is_no_call_is_shown(self):
        for text in ('```xml\n<note name="x"><to>Tove</to></note>\n```', "<tools>a list of tools</tools> here",
                     '<function name="python" arguments=\'{"code": "1"}\'/>', "```xml\n<a/>\n```"):
            with self.subTest(text[:20]):
                self.assertEqual(watch(chunks(text, 3))[:2], (text, None))

    def test_json_that_is_no_call_is_shown(self):
        for text in ('Here:\n```json\n{"a": 1}\n```\nDone.', '{"a": [1, 2]} is JSON.', "```json\nnot json\n```",
                     'It is {"name": "calculate", "arguments": {"expression": "1"}} in the middle.', "  \n[1] a note"):
            with self.subTest(text[:12]):
                self.assertEqual(watch(chunks(text, 3))[:2], (text, None))

    def test_a_call_named_after_the_function_it_wants(self):
        calls = {'{"name": "sqrt", "arguments": {"x": 4761}}': "sqrt(4761)",
                 '{"name": "sqrt", "arguments": {"expression": "sqrt(4761)"}}': "sqrt(4761)",
                 '{"name": "sqrt", "arguments": {"expression": "4761"}}': "sqrt(4761)",
                 '{"name": "hypot", "arguments": {"a": 3, "b": 4}}': "hypot(3, 4)",
                 '{"name": "calculate", "arguments": {"expr": "2^10"}}': "2^10",
                 '{"name": "functions.calculate", "arguments": {"expression": "2^10"}}': "2^10"}
        for body, expression in calls.items():
            with self.subTest(body):
                _, call, _ = watch(["<tool_call>" + body + "</tool_call>"])
                self.assertEqual(call, {"name": "calculate", "expression": expression})
        self.assertEqual(tools.run(watch(['<tool_call>{"name": "hypot", "arguments": {"a": 3, "b": 4}}</tool_call>'])[1])
                         ["result"], "5")
        _, call, _ = watch(['<tool_call>{"name": "calculate", "arguments": {}}</tool_call>'])
        self.assertEqual(call["error"], 'calculate takes {"expression": "..."}')

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


class OfferedTest(unittest.TestCase):
    ON = {"tools": {"calculator": True}}

    def test_who_gets_the_calculator(self):
        self.assertTrue(tools.offered(self.ON, "mlx-community/Qwen3-8B-4bit"))
        self.assertTrue(tools.offered(self.ON, GEMMA))
        self.assertFalse(tools.offered({"tools": {"calculator": False}}, "mlx-community/Qwen3-8B-4bit"))
        # Issue 9: R1 never called it; Falcon3 (BitNet) called it wrong over and over.
        self.assertFalse(tools.offered(self.ON, "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"))
        self.assertFalse(tools.offered(self.ON, "tiiuae/Falcon3-10B-Instruct-1.58bit", bitnet=True))

    def test_bitnet_models_are_told_apart_by_their_config(self):
        hub = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, hub, True)
        add_model(hub, "tiiuae/Falcon3-10B-Instruct-1.58bit",
                  {"model_type": "llama", "quantization_config": {"quant_method": "bitnet"}})
        add_model(hub, GEMMA, real_config(GEMMA))
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            engine.forget_scan()
            self.addCleanup(engine.forget_scan)
            self.assertTrue(engine.bitnet_model("tiiuae/Falcon3-10B-Instruct-1.58bit"))
            self.assertFalse(engine.bitnet_model(GEMMA))
            self.assertFalse(engine.bitnet_model("not/downloaded"))


class TemplateKindTest(unittest.TestCase):
    def test_only_templates_that_read_calls_the_standard_way_get_them_as_they_are(self):
        qwen = "{% if tools %}...{% for tool_call in message.tool_calls %}{% set tool_call = tool_call.function %}"
        deepseek = "{{ tool['function']['name'] }}"  # no tools of its own
        falcon = "{% if tools %}<tools>{{ tools|tojson }}</tools>{{ message.tool_calls|tojson(indent=2) }}"
        self.assertTrue(mlx_worker.native_tools(qwen))
        self.assertFalse(mlx_worker.native_tools(deepseek))
        self.assertFalse(mlx_worker.native_tools(falcon))

    def test_a_template_that_wants_roles_to_alternate_gets_the_system_message_in_the_first(self):
        def mistral(chat):
            roles = [turn["role"] for turn in chat]
            if roles != ["user", "assistant", "user", "assistant", "user"][:len(roles)]:
                raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
            return chat

        chat = mlx_worker.tool_chat([{"role": "user", "content": "sqrt 4489?"}], tools.TOOLS, tools.AS_TEXT, False)
        self.assertEqual(chat[0]["role"], "system")
        rendered = mlx_worker.with_system(mistral, chat)
        self.assertEqual([turn["role"] for turn in rendered], ["user"])
        self.assertTrue(rendered[0]["content"].startswith("You have a calculator tool."))
        self.assertTrue(rendered[0]["content"].endswith("\n\nsqrt 4489?"))


class TemplateFillTest(unittest.TestCase):
    """Issue 7: Qwen3 2507 keeps its template in chat_template.jinja alone, which older
    transformers do not read, so the tokenizer came without one."""

    class Tokenizer:
        chat_template = None

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        (self.folder / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))

    def test_a_template_in_chat_template_jinja_alone_is_used(self):
        (self.folder / "chat_template.jinja").write_text("{{ messages }}")
        tokenizer = self.Tokenizer()
        mlx_worker.fill_template(tokenizer, str(self.folder))
        self.assertEqual(tokenizer.chat_template, "{{ messages }}")

    def test_the_other_places_a_template_is_kept(self):
        (self.folder / "chat_template.json").write_text(json.dumps({"chat_template": "json"}))
        self.assertEqual(mlx_worker.template_in(str(self.folder)), "json")
        (self.folder / "tokenizer_config.json").write_text(json.dumps(
            {"chat_template": [{"name": "tool_use", "template": "tools"}, {"name": "default", "template": "plain"}]}))
        self.assertEqual(mlx_worker.template_in(str(self.folder)), "plain")

    def test_a_template_the_tokenizer_has_is_kept(self):
        (self.folder / "chat_template.jinja").write_text("from the file")
        tokenizer = self.Tokenizer()
        tokenizer.chat_template = "its own"
        mlx_worker.fill_template(tokenizer, str(self.folder))
        self.assertEqual(tokenizer.chat_template, "its own")

    def test_no_template_anywhere_says_so_for_chat_but_not_for_pictures(self):
        with self.assertRaises(RuntimeError) as caught:
            mlx_worker.fill_template(self.Tokenizer(), str(self.folder))
        self.assertIn("no chat template", str(caught.exception))
        tokenizer = self.Tokenizer()
        mlx_worker.fill_template(tokenizer, str(self.folder), required=False)
        self.assertIsNone(tokenizer.chat_template)


class LoopTest(unittest.TestCase):
    setUp = ChatThroughServerTest.setUp
    post = ChatThroughServerTest.post

    def chat(self, question: str, model: str = GEMMA) -> tuple[str, list[dict], list[dict]]:
        lines = self.post("/api/chat", {"model": model, "messages": [{"role": "user", "content": question}]})
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

    def test_the_same_call_again_gets_the_results_as_text(self):
        # Falcon3 asked for sqrt(4489) over and over.
        shown, calls, _ = self.chat("calc-repeat: sqrt(4489)")
        self.assertEqual([call["result"] for call in calls], ["67"])
        self.assertTrue(shown.endswith("no tools | [Calculator results: sqrt(4489) = 67. Answer with these; "
                                       "the calculator cannot be used again in this answer.]"), shown)
        self.assertNotIn("tool_call", shown)

    def test_ten_different_calls_an_answer(self):
        shown, calls, _ = self.chat("calc-many: 1")
        self.assertEqual([call["result"] for call in calls], [str(1 + n) for n in range(tools.MAX_CALLS)])
        self.assertIn("[Calculator results: 1+0 = 1; 1+1 = 2;", shown)

    def test_coders_never_get_the_calculator(self):
        # Qwen2.5 Coder is there to write code; with the calculator it wrote made-up calls.
        shown, calls, _ = self.chat("calc: 2+2", CODER)
        self.assertEqual((shown, calls), ("no tools", []))
        self.assertTrue(tools.offered({"tools": {"calculator": True}}, "mlx-community/Qwen3-8B-4bit"))
        for name in (CODER, "mlx-community/Devstral-Small-2505-4bit", "mlx-community/codegemma-7b-it-4bit"):
            self.assertFalse(tools.offered({"tools": {"calculator": True}}, name), name)

    def test_three_failed_calls_get_the_results_as_text(self):
        # Falcon3 kept calling a tool by the wrong name until the ten calls ran out.
        shown, calls, _ = self.chat("calc-many: 1/0")
        self.assertEqual([call["tool_error"] for call in calls], ["division by zero"] * tools.MAX_ERRORS)
        self.assertIn("no tools | [Calculator results: 1/0+0 = error: division by zero;", shown)

    def test_a_call_after_that_ends_the_answer(self):
        shown, calls, lines = self.chat("calc-stubborn: 2+2")
        self.assertEqual([call["result"] for call in calls], ["4"])
        self.assertNotIn("tool_call", shown)
        self.assertEqual(sum(1 for line in lines if line.get("done")), 1)

    def test_answer_from_keeps_the_chat_and_adds_to_the_question(self):
        chat = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "sqrt 4489?", "images": ["x"]}]
        out = tools.answer_from(chat, [{"expression": "sqrt(4489)", "result": "67"},
                                       {"expression": "1/0", "error": "division by zero"}])
        self.assertEqual(chat[1]["content"], "sqrt 4489?")
        self.assertEqual(out[1]["images"], ["x"])
        self.assertTrue(out[1]["content"].startswith(
            "sqrt 4489?\n\n[Calculator results: sqrt(4489) = 67; 1/0 = error: division by zero."))
        self.assertEqual(tools.same_call({"name": "calculate", "expression": "sqrt( 4489 )"}),
                         tools.same_call({"name": "calculate", "expression": "sqrt(4489)"}))

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
