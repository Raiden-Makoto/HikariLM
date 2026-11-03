import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helper import context, engine, mlx_worker, python_tool, store, tools
from tests.test_engine import CODER, GEMMA, add_model, real_config
from tests.test_server import ChatThroughServerTest

# Scripts run with this Python. Off a Mac they run with no sandbox, which the tests alone may
# ask for; on a Mac they run in the sandbox (tests/check_python.py tries to get out of it).
TOOL_PYTHON = {"HIKARILM_TOOL_PYTHON": sys.executable, "HIKARILM_TEST_UNSANDBOXED": "1"}
MAC = sys.platform == "darwin"


def with_tool_python(test: unittest.TestCase) -> None:
    patcher = mock.patch.dict(os.environ, TOOL_PYTHON)
    patcher.start()
    test.addCleanup(patcher.stop)


class RunTest(unittest.TestCase):
    def setUp(self):
        with_tool_python(self)

    def test_what_it_prints_and_the_last_value(self):
        answer = python_tool.run("import math\nprint(math.sqrt(4761))\n2 ** 10")
        self.assertEqual((answer["name"], answer["output"], answer["pictures"]), ("python", "69.0\n1024\n", []))
        self.assertNotIn("error", answer)
        self.assertEqual(python_tool.run("x = None\nx")["output"], "")

    def test_errors_come_back_with_their_traceback(self):
        answer = python_tool.run("x = 1\n1 / 0")
        self.assertEqual(answer["error"], "ZeroDivisionError: division by zero")
        self.assertIn('File "<script>", line 2', answer["output"])
        self.assertNotIn("run.py", answer["output"])
        self.assertTrue(python_tool.run("def (")["error"].startswith("SyntaxError"))
        self.assertEqual(python_tool.run("raise SystemExit(3)")["error"], "the script stopped with an error")
        self.assertNotIn("error", python_tool.run("raise SystemExit"))

    def test_what_it_cannot_be_given(self):
        self.assertIn("give the code", python_tool.run("  ")["error"])
        self.assertIn("longer than", python_tool.run("1" * (python_tool.CODE_CHARS + 1))["error"])

    def test_it_stops_after_its_time(self):
        with mock.patch.object(python_tool, "SECONDS", 2):
            started = time.monotonic()
            answer = python_tool.run("print('begun', flush=True)\nwhile True: pass")
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(answer["error"], "stopped after 2 seconds")
        self.assertEqual(answer["output"], "begun\n")

    def test_it_stops_past_its_memory(self):
        with mock.patch.object(python_tool, "MEMORY", 100 * 1024 * 1024):
            answer = python_tool.run("held = b'x' * (300 * 1024 * 1024)\nimport time\ntime.sleep(10)")
        self.assertTrue(answer["error"].startswith("stopped: it used more than"), answer)
        self.assertLess(answer["seconds"], 8)

    def test_long_output_is_cut_and_a_flood_stopped(self):
        answer = python_tool.run("print('a' * 30000)")
        self.assertTrue(answer["output"].startswith("a" * python_tool.OUTPUT))
        self.assertTrue(answer["output"].endswith(f"[… output cut at {python_tool.OUTPUT:,} characters]"))
        self.assertNotIn("error", answer)
        with mock.patch.object(python_tool, "FLOOD", 200_000):
            answer = python_tool.run("while True: print('a' * 1000)")
        self.assertEqual(answer["error"], "stopped: it printed more than 200,000 bytes")

    def test_a_file_past_the_limit_is_an_error(self):
        with mock.patch.object(python_tool, "FILE_BYTES", 1_000_000):
            answer = python_tool.run("open('big', 'wb').write(b'x' * 2_000_000)")
        self.assertIn("File too large", answer["error"])

    def test_its_files_go_with_its_folder(self):
        answer = python_tool.run("import os\nopen('note.txt', 'w').write('hi')\nprint(os.getcwd())")
        folder = Path(answer["output"].strip())
        self.assertEqual(folder.name, "work")
        self.assertFalse(folder.exists())

    @unittest.skipIf(MAC, "on a Mac it always runs in the sandbox")
    def test_never_without_the_sandbox_but_in_the_tests(self):
        with mock.patch.dict(os.environ, {"HIKARILM_TEST_UNSANDBOXED": ""}):
            self.assertFalse(python_tool.available())
            self.assertIn("only on a Mac", python_tool.run("print(1)")["error"])
        self.assertTrue(python_tool.available())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed here")
    def test_plots_come_back_as_pictures(self):
        answer = python_tool.run("import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.figure()\n"
                                 "plt.bar(['a', 'b'], [3, 5])")
        self.assertEqual(len(answer["pictures"]), 2)
        self.assertTrue(all(p.startswith("data:image/png;base64,iVBORw0KGgo") for p in answer["pictures"]))
        self.assertEqual(answer["output"], "")
        self.assertIn("[2 plots made; the user sees them with your answer]", python_tool.as_text(answer))

    def test_what_the_model_is_given_back(self):
        self.assertEqual(python_tool.as_text({"output": "69\n"}), "69")
        self.assertEqual(python_tool.as_text({"output": ""}), "(no output; print what you want to see)")
        self.assertEqual(python_tool.as_text({"output": "", "error": "stopped after 30 seconds"}),
                         "Error: stopped after 30 seconds")
        self.assertEqual(python_tool.as_text({"output": "half\n", "error": "stopped after 30 seconds"}),
                         "half\nError: stopped after 30 seconds")
        self.assertEqual(python_tool.as_text({"output": "Traceback…\nNameError: x", "error": "NameError: x"}),
                         "Traceback…\nNameError: x")


class SandboxProfileTest(unittest.TestCase):
    def test_what_the_profile_refuses_and_lets_in(self):
        text = python_tool.profile("/s", ["/v", "/base"], ["/v/bin/python"])
        lines = text.splitlines()
        for rule in ("(deny network*)", "(deny process-fork)", "(deny process-exec)", "(deny file-write*)",
                     "(deny mach-lookup)", "(deny appleevent-send)", "(deny signal)", "(allow signal (target self))"):
            self.assertIn(rule, lines)
        self.assertIn('(allow process-exec (literal "/v/bin/python"))', lines)
        self.assertTrue(any(line.startswith('(allow file-write* (subpath "/s")') for line in lines))
        denied = next(i for i, line in enumerate(lines) if line.startswith("(deny file-read*"))
        allowed = next(i for i, line in enumerate(lines) if line.startswith("(allow file-read*"))
        self.assertLess(denied, allowed)
        self.assertIn('(subpath "/Users")', lines[denied])
        self.assertEqual(lines[allowed], '(allow file-read* (subpath "/s") (subpath "/v") (subpath "/base"))')
        self.assertLess(lines.index("(deny process-exec)"), lines.index('(allow process-exec (literal "/v/bin/python"))'))


class PrepareTest(unittest.TestCase):
    """The venv is made with Python 3.11 or newer, where pip gets a scipy macOS 27 can load."""

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, True)
        env = {key: value for key, value in os.environ.items() if key != "HIKARILM_TOOL_PYTHON"}
        patcher = mock.patch.dict(os.environ, {**env, "HIKARILM_HOME": self.home}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.said = []

    def test_scipy_is_one_macos_can_load(self):
        self.assertIn("scipy>=1.16.3", python_tool.PACKAGES)

    def test_minor_version_from_pyvenv_cfg(self):
        cfg = Path(self.home) / "pyvenv.cfg"
        cfg.write_text("home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.10.12\n")
        self.assertEqual(python_tool._minor(cfg), 10)
        cfg.write_text("version_info = 3.13.1.final.0\n")
        self.assertEqual(python_tool._minor(cfg), 13)
        self.assertIsNone(python_tool._minor(Path(self.home) / "none.cfg"))

    def test_base_is_this_python_when_new_enough(self):
        with mock.patch.object(sys, "version_info", (3, 12, 0)):
            self.assertEqual(python_tool.base(), sys.executable)

    def test_base_looks_for_a_newer_python(self):
        with mock.patch.object(sys, "version_info", (3, 10, 4)), \
                mock.patch.object(shutil, "which", lambda name: "/x/python3.12" if name == "python3.12" else None), \
                mock.patch.object(os, "access", lambda path, mode: path == "/x/python3.12"):
            self.assertEqual(python_tool.base(), "/x/python3.12")
        with mock.patch.object(sys, "version_info", (3, 9, 6)), mock.patch.object(shutil, "which", lambda name: None), \
                mock.patch.object(os, "access", lambda path, mode: False):
            self.assertIsNone(python_tool.base())

    def test_no_new_enough_python_says_what_to_install(self):
        with mock.patch.object(python_tool, "base", lambda: None), \
                mock.patch.object(engine, "_prepare_venv") as made:
            self.assertFalse(python_tool.prepare(self.said.append))
        made.assert_not_called()
        self.assertIn("brew install python@3.13", self.said[0])

    def test_a_venv_made_with_an_old_python_is_made_again(self):
        folder = python_tool.venv()
        folder.mkdir(parents=True)
        (folder / "pyvenv.cfg").write_text("version = 3.10.12\n")
        (folder / engine.READY_MARK).write_text("numpy sympy pandas scipy matplotlib\n")
        seen = {}

        def made(where, interpreter, packages, what, say, base_python):
            seen.update(exists=where.exists(), base=base_python, packages=packages)
            return False

        with mock.patch.object(engine, "_prepare_venv", made):
            self.assertFalse(python_tool.prepare(self.said.append, "/x/python3.13"))
        self.assertEqual(seen, {"exists": False, "base": "/x/python3.13", "packages": python_tool.PACKAGES})
        self.assertIn("Python 3.10", self.said[0])

    def test_a_venv_on_a_new_python_is_kept(self):
        folder = python_tool.venv()
        folder.mkdir(parents=True)
        (folder / "pyvenv.cfg").write_text("version = 3.12.7\n")
        with mock.patch.object(engine, "_prepare_venv", lambda *a: False):
            python_tool.prepare(self.said.append, "/x/python3.13")
        self.assertTrue(folder.exists())

    def test_packages_that_do_not_load_are_reported_and_set_up_again(self):
        folder = python_tool.venv()
        (folder / "bin").mkdir(parents=True)
        (folder / engine.READY_MARK).write_text(" ".join(python_tool.PACKAGES) + "\n")
        fake = folder / "bin" / "python"
        fake.write_text("#!/bin/sh\ncase \"$2\" in *scipy.sparse*) echo \"ImportError: dlopen(_spropack.so): "
                        "section '__DATA/__thread_bss' has a zero-fill section type\" >&2; exit 1;; esac\n"
                        "echo '[]'\n")
        fake.chmod(0o755)
        self.assertFalse(python_tool.prepare(self.said.append, sys.executable))
        self.assertIn("do not load", self.said[-1])
        self.assertIn("__thread_bss", self.said[-1])
        self.assertFalse((folder / engine.READY_MARK).exists())
        self.assertFalse((folder / python_tool.PATHS).exists())


def watch(pieces) -> tuple[str, dict | None, tools.CallWatch]:
    seen = tools.CallWatch()
    shown = "".join(seen.feed(piece) for piece in pieces)
    return shown + seen.flush(), seen.call, seen


def chunks(text: str, size: int = 3) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


PRINT = "print(2 ** 10)"
OLD_SUM = "from math import *\nsqrt(4761)"  # a 4.8 calculator call, as code


class CallWatchTest(unittest.TestCase):
    def test_qwen_hermes_call_in_pieces(self):
        text = ('Let me check. <tool_call>\n{"name": "python", "arguments": {"code": "import math\\nprint(math.sqrt(4761))"}}'
                "\n</tool_call> 42!")
        shown, call, seen = watch(chunks(text))
        self.assertEqual(shown, "Let me check. ")
        self.assertEqual(call, {"name": "python", "code": "import math\nprint(math.sqrt(4761))"})
        self.assertEqual(seen.written(), "Let me check. ")

    def test_other_makers_formats(self):
        code = json.dumps({"code": PRINT})
        formats = {
            "mistral": '[TOOL_CALLS] [{"name": "python", "arguments": ' + code + "}]",
            "deepseek": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>python\n```json\n"
                        + code + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "gemma": "```tool_code\n" + PRINT + "\n```",
            "gemma calling it": '```tool_code\npython(code="print(2 ** 10)")\n```',
            "gpt-oss": "to=functions.python<|channel|>commentary json<|message|>" + code + "<|call|>",
            "arguments as a string": '<tool_call>{"name": "python", "arguments": ' + json.dumps(code) + "}</tool_call>",
            "parameters": '<tool_call>{"name": "python", "parameters": ' + code + "}</tool_call>",
            "another name": '<tool_call>{"name": "code_interpreter", "arguments": {"source": "print(2 ** 10)"}}</tool_call>',
        }
        for name, text in formats.items():
            with self.subTest(name):
                shown, call, _ = watch(chunks("Sure. " + text, 4))
                self.assertEqual(shown, "Sure. ")
                self.assertEqual(call, {"name": "python", "code": PRINT})

    def test_calculator_calls_from_4_8_still_run(self):
        for text in ('<tool_call>{"name": "calculate", "arguments": {"expression": "sqrt(4761)"}}</tool_call>',
                     '```tool_code\ncalculate(expression="sqrt(4761)")\n```'):
            with self.subTest(text[:12]):
                self.assertEqual(watch([text])[1], {"name": "python", "code": OLD_SUM})
        self.assertEqual(watch(['<tool_call>{"name": "calculate", "arguments": {"expression": "2^10"}}</tool_call>'])[1]
                         ["code"], "from math import *\n2**10")

    def test_an_unknown_tool_is_answered_with_an_error(self):
        _, call, _ = watch(['<tool_call>{"name": "weather", "arguments": {"city": "Oslo"}}</tool_call>'])
        self.assertIn("no tool named weather", tools.run(call)["error"])
        _, call, _ = watch(['<tool_call>{"name": "python", "arguments": {}}</tool_call>'])
        self.assertEqual(tools.run(call)["error"], 'python takes {"code": "..."}')

    def test_json_calls_with_no_marker(self):
        # Issue 8: Qwen2.5 Coder wrote its calls as ```json blocks, Falcon3 as bare JSON.
        coder = '```json\n{\n  "name": "python",\n  "arguments": {\n    "code": "print(2 ** 10)"\n  }\n}\n```'
        falcon = '{\n  "name": "function",\n  "arguments": {\n    "code": "print(2 ** 10)"\n  }\n}'
        for text in (coder, "Sure.\n" + coder, falcon):
            with self.subTest(text[:12]):
                shown, call, _ = watch(chunks(text, 3))
                self.assertEqual(shown, text.split("```")[0] if text.startswith("Sure") else "")
                self.assertEqual(call, {"name": "python", "code": PRINT})

    def test_python_blocks_are_the_model_showing_code(self):
        text = "Like this:\n```python\nprint(2 ** 10)\n```\nIt prints 1024."
        self.assertEqual(watch(chunks(text, 3))[:2], (text, None))

    def test_bare_json_after_the_reasoning(self):
        # Issue 9: DeepSeek R1 thought first, then wrote the call as bare JSON.
        text = '<think>\nI should run it.\n</think>\n\n{"name":"python","arguments":{"code":"print(2 ** 10)"}}'
        shown, call, seen = watch(chunks(text, 3))
        self.assertEqual(shown.rstrip(), "<think>\nI should run it.\n</think>")
        self.assertEqual(call, {"name": "python", "code": PRINT})
        self.assertEqual(watch(chunks("<think>hm</think>\nIt is {1, 2}.", 2))[:2], ("<think>hm</think>\nIt is {1, 2}.", None))

    def test_wrappers_qwen_coder_made_up(self):
        # Issue 9: Qwen2.5 Coder made these up for the calculator; they are read for Python too.
        made_up = {
            "```xml\n<function-call>\n  <name>python</name>\n  <arguments>\n    <code>print(2 ** 10)</code>\n"
            "  </arguments>\n</function-call>\n```": PRINT,
            '```xml\n<tools>\n    {\n        "name": "python",\n        "arguments": {\n            "code": "print(2 ** 10)"\n'
            "        }\n    }\n</tools>\n```": PRINT,
            "```xml\n<function name=\"python\" arguments='{\"code\": \"print(2 ** 10)\"}'/>\n```": PRINT,
            "<function-call><name>python</name><arguments><code>print(2 &lt; 3)</code></arguments></function-call>":
                "print(2 < 3)",
            "<function-call><name>calculate</name><arguments><expression>sqrt(4761)</expression></arguments>"
            "</function-call>": OLD_SUM,
        }
        for text, code in made_up.items():
            with self.subTest(text[:30]):
                shown, call, _ = watch(chunks(text, 3))
                self.assertEqual(shown, "")
                self.assertEqual(call, {"name": "python", "code": code})

    def test_xml_that_is_no_call_is_shown(self):
        for text in ('```xml\n<note name="x"><to>Tove</to></note>\n```', "<tools>a list of tools</tools> here",
                     '<function name="weather" arguments=\'{"city": "Oslo"}\'/>', "```xml\n<a/>\n```"):
            with self.subTest(text[:20]):
                self.assertEqual(watch(chunks(text, 3))[:2], (text, None))

    def test_json_that_is_no_call_is_shown(self):
        for text in ('Here:\n```json\n{"a": 1}\n```\nDone.', '{"a": [1, 2]} is JSON.', "```json\nnot json\n```",
                     'It is {"name": "python", "arguments": {"code": "1"}} in the middle.', "  \n[1] a note"):
            with self.subTest(text[:12]):
                self.assertEqual(watch(chunks(text, 3))[:2], (text, None))

    def test_text_that_only_looks_like_a_start_is_shown(self):
        self.assertEqual(watch(chunks("a < b and [TOOL is fine", 2))[:2], ("a < b and [TOOL is fine", None))
        self.assertEqual(watch(["<tool_call> never finished"])[:2], ("<tool_call> never finished", None))

    def test_the_messages_a_run_adds(self):
        answer = {"name": "python", "code": "print(6 * 7)", "output": "42\n", "pictures": []}
        call_message, result = tools.as_messages("Let me see.", {}, answer, 3)
        self.assertEqual(call_message["tool_calls"][0]["function"], {"name": "python", "arguments": {"code": "print(6 * 7)"}})
        self.assertEqual(call_message["tool_calls"][0]["id"], "call00003")
        self.assertEqual((result["role"], result["content"], result["tool_call_id"]), ("tool", "42", "call00003"))


class WorkerChatTest(unittest.TestCase):
    CHAT = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "sqrt 4761?"},
            *tools.as_messages("", {}, {"name": "python", "code": "print(69)", "output": "69\n"}, 1)]

    def test_a_template_with_tools_gets_them_as_they_are(self):
        chat = mlx_worker.tool_chat(self.CHAT, tools.TOOLS, tools.AS_TEXT, True)
        self.assertEqual(chat[0]["content"], "Rules.")
        self.assertEqual(chat[2]["tool_calls"][0]["function"]["name"], "python")
        self.assertEqual((chat[3]["role"], chat[3]["content"]), ("tool", "69"))

    def test_one_without_gets_text(self):
        chat = mlx_worker.tool_chat(self.CHAT, tools.TOOLS, tools.AS_TEXT, False)
        self.assertTrue(chat[0]["content"].startswith("Rules.\n\nYou have a Python tool."))
        self.assertIn('<tool_call>\n{"name": "python", "arguments": {"code": "print(69)"}}', chat[2]["content"])
        self.assertNotIn("tool_calls", chat[2])
        # Issue 9: Gemma 3 took a bare result for the user's answer to a quiz and asked a new question.
        self.assertEqual(chat[3], {"role": "user", "content": "<tool_response>\n69\n</tool_response>\nThat is what "
                                   "your Python code printed, not a message from me. Now answer my question with it."})
        # With the tool off (or used up), its description is left out.
        self.assertEqual(mlx_worker.tool_chat(self.CHAT[:2], None, tools.AS_TEXT, False)[0]["content"], "Rules.")

    def test_the_example_in_the_text_is_a_call(self):
        example = tools.AS_TEXT.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        self.assertEqual(watch(["<tool_call>" + example + "</tool_call>"])[1],
                         {"name": "python", "code": "import math\nprint(math.sqrt(4761))"})


class OfferedTest(unittest.TestCase):
    ON = {"tools": {"python": True}}

    def setUp(self):
        with_tool_python(self)

    def test_who_gets_it(self):
        self.assertTrue(tools.offered(self.ON, "mlx-community/Qwen3-8B-4bit"))
        self.assertTrue(tools.offered(self.ON, GEMMA))
        self.assertFalse(tools.offered({"tools": {"python": False}}, "mlx-community/Qwen3-8B-4bit"))
        # Issue 9: R1 never called the calculator; Falcon3 (BitNet) called it wrong over and over.
        self.assertFalse(tools.offered(self.ON, "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"))
        self.assertFalse(tools.offered(self.ON, "tiiuae/Falcon3-10B-Instruct-1.58bit", bitnet=True))
        # Not before Python for the models is set up.
        with mock.patch.dict(os.environ, {"HIKARILM_TOOL_PYTHON": "/nowhere/python"}):
            self.assertFalse(tools.offered(self.ON, GEMMA))

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
        self.assertTrue(rendered[0]["content"].startswith("You have a Python tool."))
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
    post = ChatThroughServerTest.post

    def setUp(self):
        ChatThroughServerTest.setUp(self)
        with_tool_python(self)

    def chat(self, question: str, model: str = GEMMA) -> tuple[str, list[dict], list[dict]]:
        lines = self.post("/api/chat", {"model": model, "messages": [{"role": "user", "content": question}]})
        self.assertEqual([line for line in lines if line.get("error")], [])
        self.assertTrue(lines[-1].get("done"))
        shown = "".join((line.get("message") or {}).get("content", "") for line in lines)
        return shown, [line for line in lines if line.get("hikarilm") == "tool"], lines

    def test_the_model_goes_on_from_the_output(self):
        shown, calls, lines = self.chat("calc: print(69)")
        self.assertEqual([(c["name"], c["code"], c["output"], c["pictures"]) for c in calls],
                         [("python", "print(69)", "69\n", [])])
        started = [line for line in lines if line.get("hikarilm") == "tool_start"]
        self.assertEqual(started, [{"hikarilm": "tool_start", "name": "python", "code": "print(69)"}])
        self.assertLess(lines.index(started[0]), lines.index(calls[0]))
        self.assertTrue(shown.startswith("Let me work it out. The answer is 69. "))
        self.assertNotIn("tool_call", shown)
        self.assertNotIn("made it up", shown)
        sent = json.loads(shown.split(". ", 2)[2])
        self.assertEqual(sent["roles"], ["user", "assistant", "tool"])
        self.assertEqual(sent["calls"], [{"code": "print(69)"}])
        self.assertEqual(sent["written"], ["Let me work it out. "])

    def test_an_error_goes_back_to_the_model(self):
        shown, calls, _ = self.chat("calc: 1/0")
        self.assertEqual(calls[0]["tool_error"], "ZeroDivisionError: division by zero")
        self.assertIn("ZeroDivisionError: division by zero. ", shown)

    def test_a_call_from_its_thinking(self):
        _, calls, lines = self.chat("calc-think: print(2 ** 10)")
        self.assertEqual(calls[0]["output"], "1024\n")
        thought = "".join((line.get("message") or {}).get("thinking", "") for line in lines)
        self.assertIn("I should run some Python.", thought)
        self.assertNotIn("tool_call", thought)

    def test_the_same_call_again_gets_the_results_as_text(self):
        # Falcon3 asked for sqrt(4489) over and over.
        shown, calls, _ = self.chat("calc-repeat: print(67)")
        self.assertEqual([call["output"] for call in calls], ["67\n"])
        self.assertTrue(shown.endswith("no tools | [Python results so far:\n\nCode:\nprint(67)\nOutput:\n67\n\n"
                                       "Answer with these; Python cannot be used again in this answer.]"), shown)
        self.assertNotIn("tool_call", shown)

    def test_ten_different_calls_an_answer(self):
        shown, calls, _ = self.chat("calc-many: 1")
        self.assertEqual([call["output"] for call in calls], [f"{1 + n}\n" for n in range(tools.MAX_CALLS)])
        self.assertIn("[Python results so far:\n\nCode:\nprint(1 + 0)\nOutput:\n1\n\nCode:\nprint(1 + 1)", shown)

    def test_coders_never_get_the_tool(self):
        # Qwen2.5 Coder is there to write code; with the calculator it wrote made-up calls.
        shown, calls, _ = self.chat("calc: print(4)", CODER)
        self.assertEqual((shown, calls), ("no tools", []))
        self.assertTrue(tools.offered({"tools": {"python": True}}, "mlx-community/Qwen3-8B-4bit"))
        for name in (CODER, "mlx-community/Devstral-Small-2505-4bit", "mlx-community/codegemma-7b-it-4bit"):
            self.assertFalse(tools.offered({"tools": {"python": True}}, name), name)

    def test_three_failed_calls_get_the_results_as_text(self):
        # Falcon3 kept calling a tool by the wrong name until the ten calls ran out.
        shown, calls, _ = self.chat("calc-many: 1/0")
        self.assertEqual([call["tool_error"] for call in calls], ["ZeroDivisionError: division by zero"] * tools.MAX_ERRORS)
        self.assertIn("no tools | [Python results so far:\n\nCode:\nprint(1/0 + 0)\nOutput:\nTraceback", shown)

    def test_a_call_after_that_ends_the_answer(self):
        shown, calls, lines = self.chat("calc-stubborn: print(2 + 2)")
        self.assertEqual([call["output"] for call in calls], ["4\n"])
        self.assertNotIn("tool_call", shown)
        self.assertEqual(sum(1 for line in lines if line.get("done")), 1)

    def test_answer_from_keeps_the_chat_and_adds_to_the_question(self):
        chat = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "sqrt 4489?", "images": ["x"]}]
        out = tools.answer_from(chat, [{"code": "print(67)", "output": "67\n"},
                                       {"code": "1/0", "output": "", "error": "ZeroDivisionError: division by zero"}])
        self.assertEqual(chat[1]["content"], "sqrt 4489?")
        self.assertEqual(out[1]["images"], ["x"])
        self.assertTrue(out[1]["content"].startswith(
            "sqrt 4489?\n\n[Python results so far:\n\nCode:\nprint(67)\nOutput:\n67\n\nCode:\n1/0\nOutput:\n"
            "Error: ZeroDivisionError: division by zero\n\n"))
        self.assertEqual(tools.same_call({"name": "python", "code": "print(67)  \n\n"}),
                         tools.same_call({"name": "python", "code": "print(67)"}))

    def test_the_switch(self):
        context.save_settings({**context.load_settings(), "tools": {"python": False}})
        shown, calls, lines = self.chat("calc: print(69)")
        self.assertEqual((shown, calls), ("no tools", []))
        plain = next(line for line in lines if line.get("hikarilm") == "context")["tokens"]
        context.save_settings({**context.load_settings(), "tools": {"python": True}})
        _, _, lines = self.chat("calc: print(69)")
        offered = next(line for line in lines if line.get("hikarilm") == "context")["tokens"]
        self.assertEqual(offered - plain, tools.text_tokens())

    def test_saved_with_the_chat(self):
        plot = "data:image/png;base64,iVBORw0KGgo="
        store.save("abc", CODER, [{"role": "user", "content": "sqrt 4761?"},
                                  {"role": "assistant", "content": "69", "tools": [
                                      {"code": "print(69)", "output": "69\n", "pictures": [plot, "javascript:x", 3]},
                                      {"code": "1/0", "output": "Traceback…", "error": "ZeroDivisionError"},
                                      {"expression": "sqrt(4761)", "result": "69"},
                                      {"expression": "no result"}, {"output": "no code"}, "junk"]}])
        self.assertEqual(store.load("abc")["messages"][1]["tools"],
                         [{"code": "print(69)", "output": "69\n", "pictures": [plot]},
                          {"code": "1/0", "output": "Traceback…", "error": "ZeroDivisionError"},
                          {"expression": "sqrt(4761)", "result": "69"}])


class SettingsTest(unittest.TestCase):
    def test_on_to_start_and_the_calculator_switch_carries_over(self):
        self.assertEqual(context.clean_settings({})["tools"], {"python": True})
        self.assertEqual(context.clean_settings({"tools": {"python": False}})["tools"], {"python": False})
        self.assertEqual(context.clean_settings({"tools": {"calculator": False}})["tools"], {"python": False})


if __name__ == "__main__":
    unittest.main()
