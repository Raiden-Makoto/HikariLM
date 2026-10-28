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
        summary = clean_summary(message.get("contextSummary"))
        if summary:
            entry["contextSummary"] = summary
        usage = clean_usage(message.get("usage"))
        if usage:
            entry["usage"] = usage
        picture = clean_picture(message.get("picture"))
        if picture and role == "assistant":
            entry["picture"] = picture
        cleaned.append(entry)
    return cleaned


PICTURE_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def clean_picture(value) -> dict:
    """A reply that is a picture: which file in images/, and how it was made. Never the picture itself."""
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not PICTURE_ID.match(value["id"]):
        return {}
    kept = {"id": value["id"]}
    for key, limit in (("prompt", 4000), ("model", 200), ("size", 20)):
        if isinstance(value.get(key), str):
            kept[key] = value[key][:limit]
    for key in ("seed", "width", "height", "steps"):
        number = value.get(key)
        if isinstance(number, int) and not isinstance(number, bool) and 0 <= number < 2 ** 32:
            kept[key] = number
    return kept


def picture_ids(chat_id: str) -> list[str]:
    """The pictures a chat made, so deleting the chat can delete them too."""
    try:
        messages = load(chat_id).get("messages") or []
    except (FileNotFoundError, ValueError, OSError):
        return []
    return [m["picture"]["id"] for m in messages if isinstance(m.get("picture"), dict) and "id" in m["picture"]]


MAX_SUMMARY = 20_000


def clean_summary(value) -> str:
    """A compaction checkpoint: the summary of everything before this message."""
    return value.strip()[:MAX_SUMMARY] if isinstance(value, str) and value.strip() else ""


def clean_usage(value) -> dict:
    """The token counts Ollama reported for a reply."""
    if not isinstance(value, dict):
        return {}
    kept = {}
    for key in ("prompt_eval_count", "eval_count"):
        number = value.get(key)
        if isinstance(number, int) and not isinstance(number, bool) and 0 <= number < 10_000_000:
            kept[key] = number
    return kept


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
    kept = {}
    if path.exists():
        try:
            existing = _read(path)
            created = float(existing.get("created", now))
            title = str(existing.get("title") or title)
            # Pin and folder belong to the sidebar, not to a reply being saved.
            if existing.get("pinned") is True:
                kept["pinned"] = True
            if existing.get("folder"):
                kept["folder"] = str(existing["folder"])
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    data = {
        "id": chat_id,
        "title": title,
        "created": created,
        "updated": now,
        "model": model.strip(),
        "messages": cleaned,
        **kept,
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


FOLDER_LENGTH = 40


def clean_folder(name) -> str:
    """A folder name, or "" for no folder."""
    if name is None:
        return ""
    if not isinstance(name, str):
        raise ValueError("folder must be text")
    return " ".join(name.split())[:FOLDER_LENGTH]


def _folder_names() -> set[str]:
    return {chat["folder"] for chat in listing() if chat["folder"]}


def _same_folder_spelling(name: str, leaving: str = "") -> str:
    """Folders are matched without regard to case, so "homework" joins an
    existing "Homework" instead of making a second folder that looks the same."""
    for existing in sorted(_folder_names()):
        if existing != leaving and existing.casefold() == name.casefold():
            return existing
    return name


def rename_folder(old, new) -> dict:
    """Rename a folder on every chat in it, or remove it when new is empty.
    Removing a folder never deletes a chat: its chats just lose the folder.
    If new is the name of another folder, the two merge. One unreadable chat
    file does not stop the rest; it is counted in "failed"."""
    old = clean_folder(old)
    if not old:
        raise ValueError("folder is required")
    new = clean_folder(new)
    if new:
        new = _same_folder_spelling(new, leaving=old)
    changed = failed = 0
    folder = transcripts_dir()
    paths = sorted(folder.glob("*.json")) if folder.is_dir() else []
    for path in paths:
        if not ID_PATTERN.match(path.stem):
            continue
        try:
            data = _read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue  # a damaged file is not shown in the sidebar either
        if data.get("folder") != old:
            continue
        if new:
            data["folder"] = new
        else:
            data.pop("folder", None)
        try:
            _write(path, data)
            changed += 1
        except OSError:
            failed += 1
    return {"from": old, "to": new, "changed": changed, "failed": failed}


def patch(chat_id: str, fields: dict) -> dict:
    """Change a chat's title, pinned flag, or folder without touching its messages.
    Only the keys present in fields change. A folder of "" or null means no folder."""
    path = _path(chat_id)
    data = _read(path)
    if "title" in fields:
        title = fields["title"]
        if not isinstance(title, str) or not " ".join(title.split()):
            raise ValueError("title is required")
        data["title"] = " ".join(title.split())[: TITLE_LENGTH * 2]
    if "pinned" in fields:
        if not isinstance(fields["pinned"], bool):
            raise ValueError("pinned must be true or false")
        data["pinned"] = fields["pinned"]
    if "folder" in fields:
        folder = clean_folder(fields["folder"])
        if folder:
            data["folder"] = _same_folder_spelling(folder, leaving=str(data.get("folder") or ""))
        else:
            data.pop("folder", None)
    if not any(key in fields for key in ("title", "pinned", "folder")):
        raise ValueError("nothing to change")
    _write(path, data)
    return {
        "id": chat_id,
        "title": data["title"],
        "pinned": bool(data.get("pinned")),
        "folder": data.get("folder", ""),
    }


def rename(chat_id: str, title) -> dict:
    return patch(chat_id, {"title": title})


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
                "pinned": data.get("pinned") is True,
                "folder": str(data.get("folder") or ""),
            }
        )
    chats.sort(key=lambda chat: chat["updated"], reverse=True)
    return chats
