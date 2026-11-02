"""Your rules for the models, the way Claude Code reads CLAUDE.md.

Yours are kept in ~/.hikarilm/rules.md, plain Markdown that can be edited outside HikariLM
too, with the switch in rules.json beside it. In the code editor the open folder's own file
(HIKARILM.md, or else AGENTS.md, CLAUDE.md or .cursorrules) comes after yours; the page
reads it and sends it with each message. Both go at the start of the system message, before
the code editor's instructions, and only for chat answers: rewriting, Apply, compaction,
notes and pictures have instructions of their own that a rule could break.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from helper import store

MAX_CHARS = 8000
FOLDER_FILES = ("HIKARILM.md", "AGENTS.md", "CLAUDE.md", ".cursorrules")


def _path() -> Path:
    return store.home() / "rules.md"


def _switch_path() -> Path:
    return store.home() / "rules.json"


def load() -> dict:
    try:
        text = _path().read_text(encoding="utf-8")
    except OSError:
        text = ""
    try:
        on = json.loads(_switch_path().read_text(encoding="utf-8")).get("on") is not False
    except (OSError, ValueError, AttributeError):
        on = True
    return {"text": text[:MAX_CHARS], "on": on, "cut": max(0, len(text) - MAX_CHARS), "limit": MAX_CHARS}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def save(payload: dict) -> dict:
    """Change the text, the switch, or both."""
    if "text" in payload:
        text = payload["text"]
        if not isinstance(text, str):
            raise ValueError("rules must be text")
        if len(text) > MAX_CHARS:
            raise ValueError(f"rules are limited to {MAX_CHARS:,} characters")
        _write(_path(), text.replace("\r\n", "\n"))
    if "on" in payload:
        if not isinstance(payload["on"], bool):
            raise ValueError("on must be true or false")
        _write(_switch_path(), json.dumps({"on": payload["on"]}))
    return load()


def _folder(value) -> dict | None:
    if not isinstance(value, dict) or value.get("name") not in FOLDER_FILES or not isinstance(value.get("text"), str):
        return None
    text = value["text"].strip()
    return {"name": value["name"], "text": text[:MAX_CHARS]} if text else None


def text_for(request) -> tuple[str, list[str]]:
    """The rules to put first in the system message, and where they came from. A request
    that does not ask for them (`{"on": true}`) gets none: only chat answers do."""
    if not isinstance(request, dict) or request.get("on") is not True:
        return "", []
    parts, names = [], []
    mine = load()
    if mine["on"] and mine["text"].strip():
        parts.append("The user's rules. Follow them in every answer:\n\n" + mine["text"].strip())
        names.append("yours")
    folder = _folder(request.get("folder"))
    if folder:
        parts.append(f"Rules for this project, from its {folder['name']}. Follow them too:\n\n{folder['text']}")
        names.append(folder["name"])
    return "\n\n".join(parts), names


def apply(messages: list[dict], request) -> tuple[list[dict], list[str], int]:
    """The messages with the rules at the start of the system message, where they came from,
    and how many messages were added in front (1 when there was no system message)."""
    text, names = text_for(request)
    if not text:
        return messages, [], 0
    if messages and messages[0]["role"] == "system":
        first = {**messages[0], "content": text + "\n\n" + messages[0]["content"]}
        return [first, *messages[1:]], names, 0
    return [{"role": "system", "content": text}, *messages], names, 1
