"""Thinking, kept short.

A thinking model can reason for thousands of tokens before it answers, and a
small local model can loop there until the context runs out. So the helper
counts the thinking as it streams. At the limit it stops Ollama, then asks the
same model again with thinking turned off and the reasoning so far handed to
it, so the model answers at once instead of starting to think from the top.

Models that Ollama says can think get `think` set explicitly, so their
reasoning arrives apart from the answer. Older templates (an R1 distill, say)
write it into the answer between <think> and </think>; that part is split off
here, so the page sees one shape either way.
"""

from __future__ import annotations

DEFAULTS = {"enabled": True, "budget": 2048}
LOWEST, HIGHEST = 128, 32768
OPEN, CLOSE = "<think>", "</think>"


def clean(raw) -> dict:
    """Blank budget means no limit; a missing one means the default."""
    raw = raw if isinstance(raw, dict) else {}
    budget = raw.get("budget", DEFAULTS["budget"])
    if budget is not None:
        try:
            budget = min(max(int(budget), LOWEST), HIGHEST)
        except (TypeError, ValueError):
            budget = DEFAULTS["budget"]
    return {"enabled": raw.get("enabled", True) is not False, "budget": budget}


def can_think(shown: dict | None) -> bool:
    return "thinking" in ((shown or {}).get("capabilities") or [])


def _levels_only(model: str) -> bool:
    # gpt-oss always thinks; it takes a level instead of on and off.
    return "gpt-oss" in model.lower()


def off(model: str):
    return "low" if _levels_only(model) else False


def think_value(model: str, shown: dict | None, settings: dict):
    """What to send as `think`, or None to leave it out (a model that cannot think
    refuses the field)."""
    if not can_think(shown):
        return None
    return True if settings["enabled"] else off(model)


def answer_now(messages: list[dict], thought: str) -> list[dict]:
    """The same chat, with the cut-off reasoning given back on the last message."""
    last = dict(messages[-1])
    last["content"] = (last.get("content") or "") + (
        "\n\n[What follows is your own reasoning about this request, not a message from the user and not an "
        "earlier answer. It was cut off at the thinking limit:]\n" + thought.strip() +
        "\n[Stop reasoning now. Using the reasoning above and everything given above it, write the final answer.]")
    return messages[:-1] + [last]


def without_thinking(text: str) -> str:
    """An answer with any leading <think>...</think> taken out. A reply that never got
    past its thinking (it ran out of tokens first) has no answer at all."""
    split = InlineThink()
    _, answer = split.feed(text)
    _, rest = split.flush()
    return (answer + rest).strip()


class InlineThink:
    """Splits a leading <think>...</think> out of streamed answer text. Tags may arrive
    cut across pieces, so a possible half tag is held back until the next piece."""

    def __init__(self) -> None:
        self.state = "start"
        self.buffer = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Returns (thinking, answer) for this piece."""
        if self.state == "answer":
            return "", text
        self.buffer += text
        if self.state == "start":
            lead = self.buffer.lstrip()
            if not lead or (OPEN.startswith(lead) and lead != OPEN):
                return "", ""
            if not lead.startswith(OPEN):
                self.state = "answer"
                out, self.buffer = self.buffer, ""
                return "", out
            self.state = "thinking"
            self.buffer = lead[len(OPEN):]
        end = self.buffer.find(CLOSE)
        if end >= 0:
            thought, rest = self.buffer[:end], self.buffer[end + len(CLOSE):].lstrip("\n")
            self.state, self.buffer = "answer", ""
            return thought, rest
        keep = next((n for n in range(min(len(CLOSE) - 1, len(self.buffer)), 0, -1)
                     if CLOSE.startswith(self.buffer[-n:])), 0)
        thought = self.buffer[:len(self.buffer) - keep]
        self.buffer = self.buffer[len(self.buffer) - keep:]
        return thought, ""

    def flush(self) -> tuple[str, str]:
        rest, self.buffer = self.buffer, ""
        return (rest, "") if self.state == "thinking" else ("", rest)
