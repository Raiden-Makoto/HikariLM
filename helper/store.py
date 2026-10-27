"""Chat transcripts on disk, one JSON file per conversation.

Files live in ~/.lightusagi/transcripts. Nothing here keeps a chat in memory:
each call reads or writes one file and lets go.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,64}$")
TITLE_LENGTH = 60


def home() -> Path:
    return Path(os.environ.get("LIGHTUSAGI_HOME") or Path.home() / ".lightusagi")


def transcripts_dir() -> Path:
    return home() / "transcripts"


def _path(chat_id: str) -> Path:
    if not isinstance(chat_id, str) or not ID_PATTERN.match(chat_id):
        raise ValueError("chat id is not valid")
    return transcripts_dir() / f"{chat_id}.json"


def clean_files(files) -> list[dict]:
    if not isinstance(files, list):
        return []
    kept = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        if isinstance(item.get("text"), str):
            kept.append({"name": item["name"], "text": item["text"]})
        elif item.get("kind") == "image":
            # Only the name is kept. The picture itself is not saved.
            kept.append({"name": item["name"], "kind": "image"})
    return kept


def clean_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages are required")
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message needs a role and content")
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("each message needs a role and content")
        entry = {"role": role, "content": content}
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            entry["thinking"] = thinking
        model = message.get("model")
        if isinstance(model, str) and model:
            entry["model"] = model
        files = clean_files(message.get("files"))
        if files:
            entry["files"] = files
        cleaned.append(entry)
    return cleaned


FILLER = re.compile(
    r"^(?:hey|hi|hello|yo|ok|okay|so|um|uh|well|please|pls|"
    r"(?:can|could|would|will) you(?: please)?|"
    r"i(?: would|'d) like (?:you )?to|i (?:want|need) you to|i (?:want|need)|"
    r"help me(?: to)?|tell me|let's|lets)\b[\s,.!:;-]*",
    re.IGNORECASE,
)


def _first_text(content: str) -> str:
    """First useful prose line, skipping fenced code and blank lines."""
    in_fence = False
    fallback = ""
    for line in content.splitlines():
        text = " ".join(line.split())
        if text.startswith("```"):
            in_fence = not in_fence
            continue
        if not text:
            continue
        if in_fence:
            fallback = fallback or text
            continue
        return text
    return fallback


def _shorten(text: str) -> str:
    if len(text) <= TITLE_LENGTH:
        return text
    cut = text[: TITLE_LENGTH - 1]
    space = cut.rfind(" ")
    if space >= TITLE_LENGTH // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def make_title(messages: list[dict]) -> str:
    """Name a chat from its first message: drop greetings and polite filler,
    keep the first sentence, capitalise, and cut at a word boundary."""
    for message in messages:
        if message["role"] != "user":
            continue
        original = _first_text(message["content"])
        if not original:
            continue
        text = original
        while True:
            stripped = FILLER.sub("", text, count=1)
            if stripped == text:
                break
            text = stripped
        text = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        text = text.strip().rstrip(".!?,;: ").strip()
        if len(text) < 3:
            text = original.rstrip(".!?,;: ") or original
        return _shorten(text[:1].upper() + text[1:])
    return "New chat"


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("transcript is not an object")
    return data


def save(chat_id: str, model, messages) -> dict:
    path = _path(chat_id)
    cleaned = clean_messages(messages)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    now = time.time()
    created = now
    title = make_title(cleaned)
    if path.exists():
        try:
            existing = _read(path)
            created = float(existing.get("created", now))
            title = str(existing.get("title") or title)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    data = {
        "id": chat_id,
        "title": title,
        "created": created,
        "updated": now,
        "model": model.strip(),
        "messages": cleaned,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return {"id": chat_id, "title": title, "updated": now}


def load(chat_id: str) -> dict:
    path = _path(chat_id)
    try:
        return _read(path)
    except json.JSONDecodeError as error:
        raise ValueError("transcript is damaged") from error


def rename(chat_id: str, title) -> dict:
    path = _path(chat_id)
    if not isinstance(title, str) or not " ".join(title.split()):
        raise ValueError("title is required")
    title = " ".join(title.split())[:TITLE_LENGTH * 2]
    data = _read(path)
    data["title"] = title
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return {"id": chat_id, "title": title}


def delete(chat_id: str) -> bool:
    path = _path(chat_id)
    try:
        path.unlink()
    except FileNotFoundError:
        delete_draft(chat_id)
        return False
    delete_draft(chat_id)
    return True


# Drafts: what is typed and attached in the message box but not sent yet.
# One file per chat in ~/.lightusagi/drafts, named by chat id ("new" is the
# draft for a chat that has not been started). Deleted when the message is sent.
def drafts_dir() -> Path:
    return home() / "drafts"


def _draft_path(key: str) -> Path:
    if not isinstance(key, str) or not ID_PATTERN.match(key):
        raise ValueError("draft id is not valid")
    return drafts_dir() / f"{key}.json"


def _clean_draft_files(files) -> list[dict]:
    kept = []
    for item in files if isinstance(files, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        if isinstance(item.get("text"), str):
            kept.append({"name": item["name"], "text": item["text"]})
        elif isinstance(item.get("image"), str):
            kept.append({"name": item["name"], "image": item["image"]})
    return kept


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def save_draft(key: str, text, files=None) -> bool:
    """Save a draft. files=None keeps the files already saved; a list replaces them.
    A draft with no text and no files is removed. Returns True if one is kept."""
    path = _draft_path(key)
    if not isinstance(text, str):
        raise ValueError("text is required")
    if files is None:
        try:
            kept = _clean_draft_files(_read(path).get("files"))
        except (OSError, ValueError, json.JSONDecodeError):
            kept = []
    else:
        kept = _clean_draft_files(files)
    if not text and not kept:
        delete_draft(key)
        return False
    _write(path, {"text": text, "files": kept, "updated": time.time()})
    return True


def load_draft(key: str) -> dict:
    path = _draft_path(key)
    try:
        data = _read(path)
    except json.JSONDecodeError as error:
        raise ValueError("draft is damaged") from error
    return {"text": str(data.get("text", "")), "files": _clean_draft_files(data.get("files"))}


def delete_draft(key: str) -> None:
    try:
        _draft_path(key).unlink()
    except (FileNotFoundError, ValueError):
        pass


def listing() -> list[dict]:
    folder = transcripts_dir()
    if not folder.is_dir():
        return []
    chats = []
    for path in folder.glob("*.json"):
        if not ID_PATTERN.match(path.stem):
            continue
        try:
            data = _read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        chats.append(
            {
                "id": path.stem,
                "title": str(data.get("title") or "New chat"),
                "updated": float(data.get("updated", 0)),
            }
        )
    chats.sort(key=lambda chat: chat["updated"], reverse=True)
    return chats
