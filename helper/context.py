"""Context length and context compaction, the way Open WebUI does it.

When a chat's estimated size passes a token threshold, the older part is
summarized by the same model and the newest messages are kept word for word.
The summary is saved on the first kept message (`contextSummary`) as a
checkpoint. Later turns send that summary plus the messages after it, and when
those grow past the threshold again the old summary is folded into a new one
and the checkpoint moves forward. The chat on screen always keeps every message.

This follows open_webui/utils/context_compaction.py: the same prompt, the same
token estimate (reported usage, else characters / 4), the same split (the oldest
~60% at a user turn, at least two messages kept, retention clamped to 10-50%),
and the summary placed in the system message as "[CONVERSATION SUMMARY]".
Two things differ for a small local machine: compaction is on by default, and
the threshold defaults to a share of the context length that is really in use,
instead of a fixed 80,000 tokens that no local context reaches.
"""

from __future__ import annotations

import json
import os
import tempfile

from helper import imagegen, store, thinking

DEFAULT_PROMPT = """### Task:
Summarize the conversation history that will be compacted out of the active chat context.

### Instructions:
- Preserve key decisions, user preferences, and constraints.
- Preserve files, artifacts, tool results, and code changes that matter going forward.
- Preserve the current task state, unresolved questions, and next steps.
- Be factual and specific. Do not invent details.
- Keep the summary concise, but complete enough for the assistant to continue without the removed messages.

### Previous Summary:
{{PREVIOUS_SUMMARY}}

### Messages Being Compacted:
{{COMPACTED_MESSAGES}}

### Recent Messages Kept In Context:
{{RECENT_MESSAGES}}"""

AUTO_SHARE = 0.75      # compact at this share of the context length, unless a threshold is set
CAP_SHARE = 0.9        # a threshold above this share would leave no room for the reply
MAX_CONTEXT = 262144

DEFAULTS = {
    "pictures": dict(imagegen.PICTURE_DEFAULTS),
    "thinking": dict(thinking.DEFAULTS),
    "context": {"all": None, "models": {}},
    "compaction": {"enabled": True, "threshold": None, "retention": 40, "max_tokens": 1000, "prompt": ""},
    "kv_cache": {"bits": 8},
}
KV_BITS = (0, 8, 4)    # off, or the bits each number of the MLX cache is kept in


# --- settings -----------------------------------------------------------------

def _settings_path():
    return store.home() / "settings.json"


def _positive(value, low: int, high: int) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("use a whole number") from None
    if not low <= number <= high:
        raise ValueError(f"use a number from {low} to {high}")
    return number


def clamp_retention(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 40
    return min(50, max(10, number))


def clean_settings(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("settings must be an object")
    context = raw.get("context") if isinstance(raw.get("context"), dict) else {}
    compaction = raw.get("compaction") if isinstance(raw.get("compaction"), dict) else {}
    models = {}
    for name, value in (context.get("models") or {}).items():
        if isinstance(name, str) and name.strip() and len(name) < 200:
            number = _positive(value, 512, MAX_CONTEXT)
            if number:
                models[name.strip()] = number
    prompt = compaction.get("prompt") if isinstance(compaction.get("prompt"), str) else ""
    kv = raw.get("kv_cache") if isinstance(raw.get("kv_cache"), dict) else {}
    bits = kv.get("bits", 8)
    return {
        "kv_cache": {"bits": bits if bits in KV_BITS and not isinstance(bits, bool) else 8},
        "pictures": imagegen.clean_picture_settings(raw.get("pictures")),
        "thinking": thinking.clean(raw.get("thinking")),
        "context": {"all": _positive(context.get("all"), 512, MAX_CONTEXT), "models": models},
        "compaction": {
            "enabled": compaction.get("enabled", True) is not False,
            "threshold": _positive(compaction.get("threshold"), 256, MAX_CONTEXT),
            "retention": clamp_retention(compaction.get("retention", 40)),
            "max_tokens": _positive(compaction.get("max_tokens"), 64, 8192) or 1000,
            "prompt": prompt.strip()[:8000],
        },
    }


def load_settings() -> dict:
    try:
        return clean_settings(json.loads(_settings_path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return json.loads(json.dumps(DEFAULTS))


def save_settings(raw) -> dict:
    settings = clean_settings(raw)
    folder = store.home()
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=folder, prefix=".settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
        os.replace(temp, _settings_path())
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return settings


def requested_context(settings: dict, model: str) -> int | None:
    context = settings["context"]
    return context["models"].get(model) or context["all"]


def threshold_for(settings: dict, num_ctx: int) -> int:
    """Tokens above which a chat is compacted, never so high that the reply has no room."""
    cap = int(num_ctx * CAP_SHARE)
    chosen = settings["compaction"]["threshold"] or int(num_ctx * AUTO_SHARE)
    return min(chosen, cap)


# --- estimating ---------------------------------------------------------------

def estimate_tokens(value) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return max(1, len(value) // 4) if value else 0


def estimate_messages(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += 4 + estimate_tokens(message.get("content"))
        total += 1000 * len(message.get("images") or [])
    return total


def usage_tokens(usage) -> int:
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("prompt_eval_count") or 0) + int(usage.get("eval_count") or 0)


def apply_checkpoint(messages: list[dict]) -> tuple[int, str | None]:
    """Where the latest summary starts, and the summary. (0, None) if there is none."""
    for index in range(len(messages) - 1, -1, -1):
        value = messages[index].get("contextSummary")
        if isinstance(value, str) and value.strip():
            return index, value
    return 0, None


def context_tokens(messages: list[dict], system_prompt: str, summary: str | None) -> int:
    """Reported usage of the latest reply plus an estimate of what came after it, else an estimate of everything."""
    for index in range(len(messages) - 1, -1, -1):
        tokens = usage_tokens(messages[index].get("usage"))
        if tokens:
            return tokens + estimate_messages(messages[index + 1:])
    return estimate_tokens(system_prompt) + estimate_tokens(summary or "") + estimate_messages(messages)


def find_boundary(messages: list[dict], retention: int = 40) -> int:
    retention = clamp_retention(retention)
    keep = max(2, len(messages) * retention // 100)
    target = max(1, len(messages) - keep)
    boundaries = [index for index, message in enumerate(messages) if message.get("role") == "user"][1:]
    target = max(target, min(boundaries, default=0))
    return next((index for index in reversed(boundaries) if index <= target), 0)


# --- the summary --------------------------------------------------------------

def format_messages(messages: list[dict]) -> str:
    return "\n".join(f"{str(message.get('role', '')).upper()}: {message.get('content', '')}" for message in messages)


def _middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...\n" + text[-(limit - half):]


def summary_prompt(template: str, compacted: list[dict], recent: list[dict], previous: str | None,
                   room_tokens: int) -> str:
    """The prompt, cut in the middle of each part if needed so it fits the context."""
    template = template.strip() or DEFAULT_PROMPT
    room = max(2000, room_tokens * 4)
    parts = {
        "{{PREVIOUS_SUMMARY}}": previous or "",
        "{{COMPACTED_MESSAGES}}": format_messages(compacted),
        "{{RECENT_MESSAGES}}": format_messages(recent),
    }
    fixed = len(template) - sum(len(name) for name in parts)
    share = {"{{PREVIOUS_SUMMARY}}": 0.2, "{{COMPACTED_MESSAGES}}": 0.6, "{{RECENT_MESSAGES}}": 0.2}
    budget = max(1000, room - fixed)
    if sum(len(text) for text in parts.values()) > budget:
        parts = {name: _middle(text, int(budget * share[name])) for name, text in parts.items()}
    prompt = template
    for name, text in parts.items():
        prompt = prompt.replace(name, text)
    last_user = next((m.get("content", "") for m in reversed([*compacted, *recent]) if m.get("role") == "user"), "")
    return prompt.replace("{{prompt}}", last_user)


def fallback_summary(previous: str | None, compacted: list[dict]) -> str:
    parts = [previous] if previous else []
    for message in compacted:
        content = message.get("content")
        if content:
            parts.append(f"- {message.get('role', 'unknown')}: {str(content)[:500]}")
    return "\n".join(parts)[:4000]


def with_summary(messages: list[dict], summary: str | None) -> list[dict]:
    """Messages for the model: the summary appended to the system message, and no bookkeeping fields."""
    out = []
    for message in messages:
        entry = {"role": message["role"], "content": message["content"]}
        if message.get("images"):
            entry["images"] = message["images"]
        out.append(entry)
    if not summary:
        return out
    text = "[CONVERSATION SUMMARY]\n" + summary
    if out and out[0]["role"] == "system":
        out[0] = {**out[0], "content": out[0]["content"] + "\n" + text}
        return out
    return [{"role": "system", "content": text}, *out]


def compaction_needed(messages: list[dict], settings: dict, num_ctx: int) -> dict:
    """Decide what to send. Returns the start of the active window, its summary, and, if
    the window is over the threshold, where to cut it."""
    system = [m for m in messages[:1] if m.get("role") == "system"]
    body_start = len(system)
    start, previous = apply_checkpoint(messages[body_start:])
    start += body_start
    window = messages[start:]
    threshold = threshold_for(settings, num_ctx)
    tokens = context_tokens(window, system[0]["content"] if system else "", previous)
    plan = {"start": start, "summary": previous, "tokens": tokens, "threshold": threshold, "cut": None}
    if not settings["compaction"]["enabled"] or tokens <= threshold:
        return plan
    boundary = find_boundary(window, settings["compaction"]["retention"])
    if boundary <= 0 or boundary >= len(window):
        return plan
    plan["cut"] = start + boundary
    return plan
