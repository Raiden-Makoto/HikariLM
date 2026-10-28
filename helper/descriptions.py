"""Text that a vision model wrote for a picture, kept so it is only written once.

Qwen and R1 cannot see pictures. Gemma writes the picture down as text, and the
text is saved in ~/.lightusagi/descriptions/. It is named after a hash of the
picture's bytes, so the same picture finds its description again, and a changed
picture does not.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from helper import store

MAX_TEXT = 20_000

PROMPT = (
    "Describe this picture as plain text for someone who cannot see it and has to write code from it. "
    "If it is a diagram (a tree, graph, flowchart, table, or similar), write down its structure exactly: "
    "every node or box with its value or label, and for each one what it connects to "
    "(for a tree: its left and right child). "
    "If it shows text or code, copy that text exactly. "
    "Say what is there and nothing else. No introduction and no guessing about what it is for."
)


def descriptions_dir() -> Path:
    return store.home() / "descriptions"


def image_key(image: object) -> str:
    """A name for the picture, from its bytes. Raises ValueError if it is not base64."""
    if not isinstance(image, str) or not image:
        raise ValueError("the picture is missing")
    try:
        raw = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the picture is not valid base64") from None
    return hashlib.sha256(raw).hexdigest()[:32]


def _path(key: str) -> Path:
    if not store.ID_PATTERN.match(key):
        raise ValueError("description id is not valid")
    return descriptions_dir() / f"{key}.json"


def load(key: str) -> dict | None:
    try:
        data = json.loads(_path(key).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str) or not data["text"].strip():
        return None
    return data


def save(key: str, text: str, model: str, name: str) -> None:
    folder = descriptions_dir()
    folder.mkdir(parents=True, exist_ok=True)
    body = {
        "image": name[:200],
        "model": model,
        "created": time.time(),
        "text": text[:MAX_TEXT],
    }
    fd, temp = tempfile.mkstemp(dir=folder, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle)
        os.replace(temp, _path(key))
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def delete(key: str) -> None:
    try:
        _path(key).unlink()
    except FileNotFoundError:
        pass
