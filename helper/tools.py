"""The tool the chat models can call (Python, run in a sandbox), and spotting a call in a reply.

The tool goes to the model through its chat template's own `tools` (worker), or, for a template
without them, as text in the system message. Models write calls in their makers' formats; the
ones below are read. A call is taken out of what the page sees, the stream is stopped there, and
the helper answers it (server._chat) with python_tool.
"""

from __future__ import annotations

import ast
import html
import json
import re

from helper import python_tool

MAX_CALLS = 10   # different calls an answer; then it answers from the results it has
MAX_ERRORS = 3   # calls that failed; then the same

DESCRIPTION = (
    "Runs Python 3 code and gives back what it prints. Use it for arithmetic, math, statistics, "
    "data and plots instead of working them out by hand, then answer from the output. "
    "numpy, sympy, pandas, scipy and matplotlib are installed, with the standard library. "
    "Print what you want to see; the value of the last line is shown too. Plots left open in "
    "matplotlib are shown to the user as pictures. Each call starts fresh: nothing is kept between "
    f"calls. There is no internet, no files but its own folder, and no other programs; it stops after "
    f"{python_tool.SECONDS} seconds or {python_tool.MEMORY / 1024 ** 3:.2f} GB of memory, and output past "
    f"{python_tool.OUTPUT:,} characters is cut.")

PYTHON = {"type": "function", "function": {
    "name": "python",
    "description": DESCRIPTION,
    "parameters": {"type": "object", "required": ["code"], "properties": {
        "code": {"type": "string", "description": "The Python code, such as: import math\nprint(math.sqrt(4761))"}}}}}

TOOLS = [PYTHON]

# For a template without tools: the same, as text, in the format Qwen and Hermes models use.
AS_TEXT = (
    "You have a Python tool. " + DESCRIPTION + "\n\n"
    "To use it, write exactly this and nothing after it, with the code as a JSON string:\n"
    '<tool_call>\n{"name": "python", "arguments": {"code": "import math\\nprint(math.sqrt(4761))"}}\n</tool_call>\n'
    "The output comes back in the next message, then you go on.")

# Names models give the tool. "calculate" is the calculator's (4.8), which a model may still
# write from an older chat; its sum is run as Python with math's names.
NAMES = {"python", "run_python", "python_exec", "execute_python", "exec_python", "execute", "run",
         "run_code", "execute_code", "code_interpreter", "python_interpreter", "ipython", "calculate"}

# Coding models are there to write code; given a tool, Qwen2.5 Coder wrote made-up calls instead
# of answers. The same names the page takes for coders (codeRank in models.js).
CODERS = re.compile(r"code|devstral", re.I)
# Models that were given the calculator and did not use it well (issue 9): DeepSeek R1 8B thought
# about it and never called it, from the system message or the question. BitNet models (Falcon3
# 1.58-bit) are left out too: Falcon3 called it by the wrong name over and over.
POOR_CALLERS = re.compile(r"deepseek-r1", re.I)


def offered(settings: dict, model: str, bitnet: bool = False) -> bool:
    """Whether this model gets the tool: it is on in Settings, Python for the models is set up,
    and the model is no coder, no BitNet model and none of POOR_CALLERS."""
    name = model or ""
    return (bool(settings["tools"]["python"]) and not bitnet and python_tool.available()
            and not CODERS.search(name) and not POOR_CALLERS.search(name))


def text_tokens() -> int:
    """About how much of the context the tool's description takes."""
    return len(json.dumps(TOOLS)) // 4


def run(call: dict) -> dict:
    """Answer a call: python_tool.run's result, or the call's own error."""
    if call.get("error"):
        return {"name": "python", "code": call.get("code") or "", "output": "", "error": call["error"]}
    return python_tool.run(call["code"])


def _tidy(code) -> str:
    return "\n".join(line.rstrip() for line in str(code or "").strip().splitlines())


def same_call(call: dict) -> str:
    """Calls that ask for the same thing give the same key."""
    return f"{_tidy(call.get('code'))}\0{call.get('error', '')}"


def _clip(text: str, size: int = 3000) -> str:
    return text if len(text) <= size else text[:size] + " […]"


def answer_from(messages: list[dict], answers: list[dict]) -> list[dict]:
    """The chat as it was asked, with what the scripts printed so far added to the question as
    text, for a last answer with no tool. Plain text, since some templates (Falcon3's) leave
    tool results out when no tools are given."""
    runs = [f"Code:\n{_clip(a.get('code') or '')}\nOutput:\n{_clip(python_tool.as_text(a))}" for a in answers]
    note = ("[Python results so far:\n\n" + "\n\n".join(runs) + "\n\nAnswer with these; Python cannot be "
            "used again in this answer.]") if runs else "[Answer now, without Python.]"
    out = list(messages)
    for at in range(len(out) - 1, -1, -1):
        if out[at]["role"] == "user":
            out[at] = {**out[at], "content": str(out[at].get("content") or "") + "\n\n" + note}
            return out
    return out + [{"role": "user", "content": note}]


def as_messages(written: str, call: dict, answer: dict, number: int) -> list[dict]:
    """The call and its result, to add to the chat the way chat templates take them."""
    call_id = f"call{number:05d}"
    return [{"role": "assistant", "content": written,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "python", "arguments": {"code": answer.get("code") or ""}}}]},
            {"role": "tool", "name": "python", "tool_call_id": call_id, "content": python_tool.as_text(answer)}]


# --- spotting a call ----------------------------------------------------------
# Where each format starts. Qwen2.5 and Qwen3 (Hermes): <tool_call>{json}</tool_call>. Mistral:
# [TOOL_CALLS][{json}]. DeepSeek: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>
# python ```json {args} ```. Gemma: ```tool_code (Python) ```. gpt-oss (Harmony):
# to=functions.python<|channel|>commentary json<|message|>{args}<|call|>.
# Qwen2.5 Coder often writes its call as a ```json block instead, and Falcon3 and DeepSeek R1 as
# bare JSON at the start of the answer; those count only when the JSON has a name and arguments
# (LOOSE_STARTS). Qwen2.5 Coder also makes up wrappers: <tools>{json}</tools>, ```xml blocks,
# <function-call><name>…</name><arguments><code>…</code></arguments></function-call>, and
# <function name="python" arguments='{json}'/>. Those count only when they name the tool and give
# code (WRAPPED, with where each ends). Gemma 3, shown the <tool_call> format, writes it as a
# ```tool_call block of JSON. A ```python block is the model showing code, not a call.
WRAPPED = {"```tool_call": ("```",), "```xml": ("```",), "<tools>": ("</tools>",), "<function-call>": ("</function-call>",),
           "<function_call>": ("</function_call>",), "<function ": ("/>", "</function>")}
STARTS = ("<tool_call>", "[TOOL_CALLS]", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "```tool_code",
          "to=functions.", "```json", *WRAPPED)
LOOSE_STARTS = ("```json", "")
_XML_NAME = re.compile(r"""<name>\s*([\w.]+)\s*</name>|\bname\s*=\s*["']([\w.]+)["']""")
_XML_ARGUMENTS = re.compile(r"""\barguments\s*=\s*(?:'([^']*)'|"((?:[^"\\]|\\.)*)")""")
_XML_CODE = re.compile(r"<(code|expression)>(.*?)</\1>", re.S)


def _json_after(text: str, opener: str = "{["):
    """The first JSON value in text that starts with one of opener, if it is complete yet."""
    at = min((i for i in (text.find(c) for c in opener) if i >= 0), default=-1)
    if at < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[at:])[0]
    except ValueError:
        return None


def _code(name: str, arguments) -> str | None:
    """The code a call gives: under "code", or the one string it has when it names the tool.
    The calculator's {"expression": "sqrt(4761)"} becomes code that prints it."""
    if not isinstance(arguments, dict):
        return arguments if isinstance(arguments, str) and name in NAMES else None
    if isinstance(arguments.get("code"), str):
        return arguments["code"]
    if isinstance(arguments.get("expression"), str):
        return "from math import *\n" + arguments["expression"].replace("^", "**")
    values = [value for key, value in arguments.items() if key != "name"]
    if name in NAMES and len(values) == 1 and isinstance(values[0], str):
        return values[0]
    return None


def _call_from(value) -> dict | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None
    if "function" in value and isinstance(value["function"], dict):
        value = value["function"]
    arguments = value.get("arguments", value.get("parameters", value))
    name = str(value.get("name", "python")).strip()
    name = name[len("functions."):] if name.startswith("functions.") else name
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    code = _code(name.lower(), arguments)
    if isinstance(code, str) and code.strip():
        # There is one tool, so a call by another name with code in it is taken as that.
        return {"name": "python", "code": code}
    if name.lower() in NAMES:
        return {"name": "python", "code": None, "error": 'python takes {"code": "..."}'}
    return {"name": name, "code": None, "error": f"there is no tool named {name}; the one tool is python"}


NOT_A_CALL = object()
THINK_END = "</think>"


def _looks_like_call(value) -> bool:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict) and isinstance(value.get("function"), dict):
        value = value["function"]
    return (isinstance(value, dict) and isinstance(value.get("name"), str)
            and any(key in value for key in ("arguments", "parameters")))


def _tool_code(block: str) -> str:
    """The code in Gemma's ```tool_code block. It is Python; when it is one call of the tool,
    python(code="…"), the code is that string."""
    block = block.strip()
    try:
        node = ast.parse(block, mode="eval").body
    except SyntaxError:
        return block
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() in NAMES
            and len(node.args) + len(node.keywords) == 1):
        value = node.args[0] if node.args else node.keywords[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if node.func.id == "calculate":
                return "from math import *\n" + value.value.replace("^", "**")
            return value.value
    return block


def _parse_wrapped(start: str, rest: str):
    ends = [at for at in (rest.find(end) for end in WRAPPED[start]) if at >= 0]
    if not ends:
        return None
    body = rest[:min(ends)]
    value = _json_after(body)
    if _looks_like_call(value):
        call = _call_from(value)
    else:
        name = _XML_NAME.search(start + body)
        if not name:
            return NOT_A_CALL
        arguments = None
        attribute = _XML_ARGUMENTS.search(body)
        code = _XML_CODE.search(body)
        if attribute:
            try:
                arguments = json.loads(attribute.group(1) if attribute.group(1) is not None
                                       else attribute.group(2).replace('\\"', '"'))
            except ValueError:
                arguments = None
        elif code:
            arguments = {code.group(1): html.unescape(code.group(2).strip("\n"))}
        call = _call_from({"name": name.group(1) or name.group(2), "arguments": arguments})
    # Made-up wrappers are taken only for the tool; anything else is the model's text.
    return NOT_A_CALL if call is None or "error" in call else call


def _parse(start: str, rest: str) -> dict | None:
    """The call written after `start`, or None while it is not complete."""
    if start in WRAPPED:
        return _parse_wrapped(start, rest)
    if start == "```tool_code":
        if "```" not in rest:
            return None
        code = _tool_code(rest.split("```", 1)[0])
        return {"name": "python", "code": code} if code else {"name": "python", "code": None,
                                                                "error": "the tool_code block is empty"}
    if start.startswith("<｜tool"):
        head = rest.split("{", 1)[0]
        value = _json_after(rest, "{")
        if value is None:
            return None
        name = (re.findall(r"(\w+)\s*(?:```|$)", head) or ["python"])[-1]
        return _call_from({"name": name, "arguments": value})
    if start == "to=functions.":
        name = re.match(r"\w*", rest).group(0) or "?"
        value = _json_after(rest.split("<|message|>", 1)[-1], "{")
        return None if value is None else _call_from({"name": name, "arguments": value})
    if start in LOOSE_STARTS:
        body = rest.lstrip()
        if not body:
            return None
        if body[0] not in ("{[" if start else "{"):
            return NOT_A_CALL
        value = _json_after(body)
        if value is None:
            return None
        return _call_from(value) if _looks_like_call(value) else NOT_A_CALL
    value = _json_after(rest)
    return None if value is None else _call_from(value)


class CallWatch:
    """Fed the reply as it streams. Gives back what the page may see, and the call once one is
    complete. Text that could be the start of a call is held back until it is clear."""

    def __init__(self) -> None:
        self.raw = ""      # everything the model wrote this round
        self.shown = 0     # how much of raw went to the page
        self.start = -1    # where a call starts in raw
        self.marker = ""
        self.call: dict | None = None
        self.opened = False  # whether the answer has anything but spaces yet
        self.answer_at = 0   # where the answer starts: after </think>, if it thought
        self.searched = 0    # where to look for a start from, past JSON that was no call

    def feed(self, text: str) -> str:
        self.raw += text
        if self.call:
            return ""
        while True:
            if self.start < 0 and self.opened:
                ended = self.raw.find(THINK_END, self.answer_at)
                if ended >= 0:
                    # The answer starts after the reasoning, so bare JSON may come again.
                    self.opened, self.answer_at = False, ended + len(THINK_END)
            if self.start < 0 and not self.opened:
                # Bare JSON counts only as the first thing in the answer.
                answer = self.raw[self.answer_at:]
                lead = self.answer_at + len(answer) - len(answer.lstrip())
                if lead == len(self.raw):
                    out, self.shown = self.raw[self.shown:self.answer_at], max(self.shown, self.answer_at)
                    return out
                self.opened = True
                if self.raw[lead] == "{":
                    self.start, self.marker = lead, ""
            if self.start < 0:
                found = [(self.raw.find(s, max(self.shown, self.searched)), s) for s in STARTS]
                found = [(at, s) for at, s in found if at >= 0]
                if found:
                    self.start, self.marker = min(found)
            if self.start < 0:
                break
            call = _parse(self.marker, self.raw[self.start + len(self.marker):])
            if call is NOT_A_CALL:
                self.searched = self.start + max(1, len(self.marker))
                self.start, self.marker = -1, ""
                continue
            self.call = call
            out, self.shown = self.raw[self.shown:self.start], max(self.shown, self.start)
            return out
        keep = max((n for s in STARTS for n in range(1, min(len(s), len(self.raw) - self.shown) + 1)
                    if self.raw.endswith(s[:n])), default=0)
        end = len(self.raw) - keep
        out, self.shown = self.raw[self.shown:end], max(self.shown, end)
        return out

    def flush(self) -> str:
        """At the end of the reply: what was held back, if it never became a call."""
        if self.call:
            return ""
        out, self.shown = self.raw[self.shown:], len(self.raw)
        return out

    def written(self) -> str:
        """What the model wrote before its call, for the chat it goes on from."""
        return self.raw[:self.start] if self.start >= 0 else self.raw
