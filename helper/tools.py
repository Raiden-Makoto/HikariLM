"""The tools the chat models can call (the calculator for now), and spotting a call in a reply.

The tool goes to the model through its chat template's own `tools` (worker), or, for a template
without them, as text in the system message. Models write calls in their makers' formats; the
ones below are read. A call is taken out of what the page sees, the stream is stopped there, and
the helper answers it (server._chat).
"""

from __future__ import annotations

import ast
import html
import json
import re

from helper import calculator

MAX_CALLS = 10   # different calls an answer; then it answers from the results it has
MAX_ERRORS = 3   # calls that could not be worked out; then the same

DESCRIPTION = (
    "A scientific calculator. Use it for any arithmetic or math instead of working it out by hand, "
    "then answer from the result. The expression is written like Python or a calculator: "
    "+ - * / // % and ** or ^ for powers, brackets, 5! for factorial; "
    "sqrt, cbrt, root(x, n), exp, ln, log(x) (natural), log(x, base), log10, log2; "
    "sin cos tan and asin acos atan atan2 in radians, sind cosd tand in degrees, sinh cosh tanh; "
    "degrees, radians, hypot, factorial, gamma, comb, perm, gcd, lcm, floor, ceil, abs, round(x, digits), "
    "min, max; constants pi, e, tau. Numbers are double-precision floats (about 15 significant digits, "
    "up to about 1.8e308); past that range the result is an overflow or underflow error. "
    "One expression per call.")

CALCULATOR = {"type": "function", "function": {
    "name": "calculate",
    "description": DESCRIPTION,
    "parameters": {"type": "object", "required": ["expression"], "properties": {
        "expression": {"type": "string", "description": "The expression, such as sqrt(4761) or 2^40 * 3.5"}}}}}

TOOLS = [CALCULATOR]

# For a template without tools: the same, as text, in the format Qwen and Hermes models use.
AS_TEXT = (
    "You have a calculator tool. " + DESCRIPTION + "\n\n"
    "To use it, write exactly this and nothing after it:\n"
    '<tool_call>\n{"name": "calculate", "arguments": {"expression": "sqrt(4761)"}}\n</tool_call>\n'
    "The result comes back in the next message, then you go on.")


# Coding models are there to write code; given the calculator, Qwen2.5 Coder wrote made-up calls
# instead of answers. The same names the page takes for coders (codeRank in models.js).
CODERS = re.compile(r"code|devstral", re.I)


def offered(settings: dict, model: str) -> bool:
    """Whether this model gets the calculator: it is on in Settings and the model is no coder."""
    return bool(settings["tools"]["calculator"]) and not CODERS.search(model or "")


def text_tokens() -> int:
    """About how much of the context the tool's description takes."""
    return len(json.dumps(TOOLS)) // 4


def run(call: dict) -> dict:
    """Answer a call: {"expression", "result"} or {"expression", "error"}."""
    expression = call.get("expression")
    out = {"name": call.get("name") or "calculate", "expression": expression if isinstance(expression, str) else ""}
    if call.get("error"):
        return {**out, "error": call["error"]}
    try:
        return {**out, "result": calculator.calculate(expression)}
    except calculator.CalcError as error:
        return {**out, "error": str(error)}


def same_call(call: dict) -> str:
    """Calls that ask for the same thing give the same key."""
    expression = call.get("expression") if isinstance(call.get("expression"), str) else ""
    return f"{str(call.get('name')).strip().lower()}:{''.join(expression.split())}:{call.get('error', '')}"


def answer_from(messages: list[dict], answers: list[dict]) -> list[dict]:
    """The chat as it was asked, with the results worked out so far added to the question as
    text, for a last answer with no tool. Plain text, since some templates (Falcon3's) leave
    tool results out when no tools are given."""
    lines = [f"{a['expression']} = " + (a["result"] if "result" in a else "error: " + a.get("error", ""))
             for a in answers]
    note = ("[Calculator results: " + "; ".join(lines) + ". Answer with these; the calculator cannot be used "
            "again in this answer.]") if lines else "[Answer now, without the calculator.]"
    out = list(messages)
    for at in range(len(out) - 1, -1, -1):
        if out[at]["role"] == "user":
            out[at] = {**out[at], "content": str(out[at].get("content") or "") + "\n\n" + note}
            return out
    return out + [{"role": "user", "content": note}]


def as_messages(written: str, call: dict, answer: dict, number: int) -> list[dict]:
    """The call and its result, to add to the chat the way chat templates take them."""
    call_id = f"calc{number:05d}"
    arguments = {"expression": answer["expression"]}
    result = answer.get("result") or "Error: " + answer.get("error", "")
    return [{"role": "assistant", "content": written,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": answer["name"], "arguments": arguments}}]},
            {"role": "tool", "name": answer["name"], "tool_call_id": call_id, "content": result}]


# --- spotting a call ----------------------------------------------------------
# Where each format starts. Qwen2.5 and Qwen3 (Hermes): <tool_call>{json}</tool_call>. Mistral:
# [TOOL_CALLS][{json}]. DeepSeek: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>
# calculate ```json {args} ```. Gemma: ```tool_code calculate("…") ```. gpt-oss (Harmony):
# to=functions.calculate<|channel|>commentary json<|message|>{args}<|call|>.
# Qwen2.5 Coder often writes its call as a ```json block instead, and Falcon3 and DeepSeek R1 as
# bare JSON at the start of the answer; those count only when the JSON has a name and arguments
# (LOOSE_STARTS). Qwen2.5 Coder also makes up wrappers: <tools>{json}</tools>, ```xml blocks,
# <function-call><name>…</name><arguments><expression>…</expression></arguments></function-call>,
# and <function name="calculate" arguments='{json}'/>. Those count only when they name the
# calculator and give a sum (WRAPPED, with where each ends).
WRAPPED = {"```xml": ("```",), "<tools>": ("</tools>",), "<function-call>": ("</function-call>",),
           "<function_call>": ("</function_call>",), "<function ": ("/>", "</function>")}
STARTS = ("<tool_call>", "[TOOL_CALLS]", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "```tool_code",
          "to=functions.", "```json", *WRAPPED)
LOOSE_STARTS = ("```json", "")
_XML_NAME = re.compile(r"""<name>\s*([\w.]+)\s*</name>|\bname\s*=\s*["']([\w.]+)["']""")
_XML_ARGUMENTS = re.compile(r"""\barguments\s*=\s*(?:'([^']*)'|"((?:[^"\\]|\\.)*)")""")
_XML_EXPRESSION = re.compile(r"<expression>(.*?)</expression>", re.S)
_CALL_IN_CODE = re.compile(r"""(\w+)\(\s*(?:expression\s*=\s*)?("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*\)""")


def _json_after(text: str, opener: str = "{["):
    """The first JSON value in text that starts with one of opener, if it is complete yet."""
    at = min((i for i in (text.find(c) for c in opener) if i >= 0), default=-1)
    if at < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[at:])[0]
    except ValueError:
        return None


def _expression(name: str, arguments) -> str | None:
    """The sum a call asks for. Small models (Falcon3) name the tool after the function they
    want, {"name": "sqrt", "arguments": {"x": 4761}}, or give the expression under another key."""
    if isinstance(arguments, dict) and isinstance(arguments.get("expression"), str):
        expression = arguments["expression"]
    elif name != "calculate" and name not in calculator.FUNCTIONS:
        return None
    elif isinstance(arguments, dict):
        values = [value for key, value in arguments.items() if key != "name"]
        if len(values) == 1 and isinstance(values[0], str):
            expression = values[0]
        elif values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            expression = ", ".join(str(v) for v in values)
        else:
            return None
    elif isinstance(arguments, (int, float)) and not isinstance(arguments, bool):
        expression = str(arguments)
    else:
        return None
    if name in calculator.FUNCTIONS and _NUMBERS.fullmatch(expression):
        expression = f"{name}({expression})"
    return expression


_NUMBERS = re.compile(r"\s*-?[\d.]+(?:[eE][-+]?\d+)?(?:\s*,\s*-?[\d.]+(?:[eE][-+]?\d+)?)*\s*")


def _call_from(value) -> dict | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None
    if "function" in value and isinstance(value["function"], dict):
        value = value["function"]
    arguments = value.get("arguments", value.get("parameters", value))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"expression": arguments}
    name = str(value.get("name", "calculate")).strip()
    name = name[len("functions."):] if name.startswith("functions.") else name
    expression = _expression(name, arguments)
    if isinstance(expression, str):
        # There is one tool, so a call by another name with a sum in it is taken as that.
        return {"name": "calculate", "expression": expression}
    call = {"name": name, "expression": None}
    if name == "calculate" or name in calculator.FUNCTIONS:
        call["error"] = 'calculate takes {"expression": "..."}'
    else:
        call["error"] = f"there is no tool named {name}; the one tool is calculate"
    return call


NOT_A_CALL = object()
THINK_END = "</think>"


def _looks_like_call(value) -> bool:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict) and isinstance(value.get("function"), dict):
        value = value["function"]
    return (isinstance(value, dict) and isinstance(value.get("name"), str)
            and any(key in value for key in ("arguments", "parameters")))


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
        expression = _XML_EXPRESSION.search(body)
        if attribute:
            try:
                arguments = json.loads(attribute.group(1) if attribute.group(1) is not None
                                       else attribute.group(2).replace('\\"', '"'))
            except ValueError:
                arguments = None
        elif expression:
            arguments = {"expression": html.unescape(expression.group(1).strip())}
        call = _call_from({"name": name.group(1) or name.group(2), "arguments": arguments})
    # Made-up wrappers are taken only for the calculator; anything else is the model's text.
    return NOT_A_CALL if call is None or "error" in call else call


def _parse(start: str, rest: str) -> dict | None:
    """The call written after `start`, or None while it is not complete."""
    if start in WRAPPED:
        return _parse_wrapped(start, rest)
    if start == "```tool_code":
        match = _CALL_IN_CODE.search(rest)
        if not match:
            return None
        name, expression = match.group(1), ast.literal_eval(match.group(2))
        return _call_from({"name": name, "arguments": {"expression": expression}})
    if start.startswith("<｜tool"):
        head = rest.split("{", 1)[0]
        value = _json_after(rest, "{")
        if value is None:
            return None
        name = "calculate" if "calculate" in head else (re.findall(r"(\w+)\s*(?:```|$)", head) or ["?"])[-1]
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
