"""The tools the chat models can call (the calculator for now), and spotting a call in a reply.

The tool goes to the model through its chat template's own `tools` (worker), or, for a template
without them, as text in the system message. Models write calls in their makers' formats; the
ones below are read. A call is taken out of what the page sees, the stream is stopped there, and
the helper answers it (server._chat).
"""

from __future__ import annotations

import ast
import json
import re

from helper import calculator

MAX_CALLS = 20   # an answer; then the model is told to answer with what it has

DESCRIPTION = (
    "A scientific calculator. Use it for any arithmetic or math instead of working it out by hand, "
    "then answer from the result. The expression is written like Python or a calculator: "
    "+ - * / // % and ** or ^ for powers, brackets, 5! for factorial, whole numbers of any size; "
    "sqrt, cbrt, root(x, n), exp, ln, log(x) (natural), log(x, base), log10, log2; "
    "sin cos tan and asin acos atan atan2 in radians, sind cosd tand in degrees, sinh cosh tanh; "
    "degrees, radians, hypot, factorial, gamma, comb, perm, gcd, lcm, floor, ceil, abs, round(x, digits), "
    "min, max; constants pi, e, tau. One expression per call.")

CALCULATOR = {"type": "function", "function": {
    "name": "calculate",
    "description": DESCRIPTION,
    "parameters": {"type": "object", "required": ["expression"], "properties": {
        "expression": {"type": "string", "description": "The expression, such as sqrt(4761) or 2^64 - 1"}}}}}

TOOLS = [CALCULATOR]

# For a template without tools: the same, as text, in the format Qwen and Hermes models use.
AS_TEXT = (
    "You have a calculator tool. " + DESCRIPTION + "\n\n"
    "To use it, write exactly this and nothing after it:\n"
    '<tool_call>\n{"name": "calculate", "arguments": {"expression": "sqrt(4761)"}}\n</tool_call>\n'
    "The result comes back in the next message, then you go on.")


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
# calculate ```json {args} ```. Gemma: ```tool_code calculate("…") ```.
STARTS = ("<tool_call>", "[TOOL_CALLS]", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "```tool_code")
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
    name = value.get("name", "calculate")
    expression = arguments.get("expression") if isinstance(arguments, dict) else None
    call = {"name": name, "expression": expression}
    if name != "calculate":
        call["error"] = f"there is no tool named {name}; the one tool is calculate"
    elif not isinstance(expression, str):
        call["error"] = 'calculate takes {"expression": "..."}'
    return call


def _parse(start: str, rest: str) -> dict | None:
    """The call written after `start`, or None while it is not complete."""
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

    def feed(self, text: str) -> str:
        self.raw += text
        if self.call:
            return ""
        if self.start < 0:
            found = [(self.raw.find(s, self.shown), s) for s in STARTS]
            found = [(at, s) for at, s in found if at >= 0]
            if found:
                self.start, self.marker = min(found)
        if self.start >= 0:
            self.call = _parse(self.marker, self.raw[self.start + len(self.marker):])
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
