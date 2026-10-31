"""Remember what was answered and what was read, so a later question can find it.

What is indexed is text HikariLM already saved: the questions and answers in
each chat, and the text Gemma wrote for each picture. Files in a workspace are
not indexed; a small folder is sent whole. Vectors come from nomic-embed-text
(passed in as `embed`), one small file per chat or picture in
~/.hikarilm/index/, so deleting a chat removes its notes and nothing else.

Not used in 3.0: nomic ran in Ollama, which is gone. 3.1 runs it in MLX and
turns this back on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from helper import descriptions, store

CHUNK = 1200
QUESTION = 300
BATCH = 16
TOP = 3
MIN_SCORE = 0.6
MAX_CHUNKS = 20_000

Embed = Callable[[list[str]], list[list[float]]]


def index_dir() -> Path:
    return store.home() / "index"


def _path(source: str) -> Path:
    if not store.ID_PATTERN.match(source):
        raise ValueError("note source is not valid")
    return index_dir() / f"{source}.json"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _unit(vector: list[float]) -> list[float]:
    size = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [round(x / size, 5) for x in vector]


def _split(text: str) -> list[str]:
    """Cut at line breaks where it can, so a chunk is about CHUNK characters."""
    text = text.strip()
    if len(text) <= CHUNK:
        return [text] if text else []
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > CHUNK:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:CHUNK])
            line = line[CHUNK:]
        if len(current) + len(line) > CHUNK and current:
            parts.append(current)
            current = ""
        current += line
    if current.strip():
        parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def chat_source(chat_id: str) -> str:
    return f"chat-{chat_id}"


def description_source(key: str) -> str:
    return f"pic-{key}"


def chat_chunks(chat: dict) -> tuple[str, list[str]]:
    """Each answer, with the question that led to it."""
    title = str(chat.get("title") or "Chat")
    texts: list[str] = []
    question = ""
    for message in chat.get("messages", []):
        if message.get("role") == "user":
            question = str(message.get("content", "")).strip()[:QUESTION]
        elif message.get("role") == "assistant":
            answer = str(message.get("content", "")).strip()
            for number, part in enumerate(_split(answer)):
                head = f"Question: {question}\nAnswer: " if question and number == 0 else ""
                texts.append(head + part)
    return title, texts


def description_chunks(data: dict) -> tuple[str, list[str]]:
    name = str(data.get("image") or "picture")
    return f"Picture {name}", _split(str(data.get("text", "")))


def _load(source: str) -> dict | None:
    try:
        data = json.loads(_path(source).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def remove(source: str) -> None:
    try:
        _path(source).unlink()
    except FileNotFoundError:
        pass


def sync(source: str, title: str, texts: list[str], embed: Embed) -> int:
    """Make the notes for one source match `texts`. Only new text is embedded.

    Returns how many chunks were embedded. A source with no text has no file.
    """
    if not texts:
        remove(source)
        return 0
    old = _load(source) or {}
    known = {chunk["hash"]: chunk["vector"] for chunk in old.get("chunks", []) if "hash" in chunk and "vector" in chunk}
    fresh = [text for text in texts if _hash(text) not in known]
    for start in range(0, len(fresh), BATCH):
        batch = fresh[start:start + BATCH]
        vectors = embed(["search_document: " + text for text in batch])
        for text, vector in zip(batch, vectors):
            known[_hash(text)] = _unit(vector)
    body = {
        "source": source,
        "title": title,
        "chunks": [{"hash": _hash(text), "text": text, "vector": known[_hash(text)]} for text in texts],
    }
    folder = index_dir()
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=folder, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle)
        os.replace(temp, _path(source))
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return len(fresh)


def _files() -> list[Path]:
    folder = index_dir()
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def count() -> dict:
    chunks = 0
    files = _files()
    for path in files:
        data = _load(path.stem)
        if data:
            chunks += len(data.get("chunks", []))
    return {"sources": len(files), "notes": chunks}


def search(query: str, embed: Embed, exclude: str = "", top: int = TOP, min_score: float = MIN_SCORE) -> list[dict]:
    """The notes closest to a question. Nothing is embedded if there are no notes."""
    query = query.strip()
    files = [path for path in _files() if path.stem != exclude]
    if not query or not files:
        return []
    vector = _unit(embed(["search_query: " + query])[0])
    scored: list[dict] = []
    seen = 0
    for path in files:
        data = _load(path.stem)
        if not data:
            continue
        for chunk in data.get("chunks", []):
            seen += 1
            if seen > MAX_CHUNKS:
                break
            other = chunk.get("vector")
            if not isinstance(other, list) or len(other) != len(vector):
                continue
            score = sum(a * b for a, b in zip(vector, other))
            if score >= min_score:
                scored.append({"title": data.get("title", ""), "text": chunk.get("text", ""),
                               "source": data.get("source", path.stem), "score": round(score, 3)})
    scored.sort(key=lambda note: note["score"], reverse=True)
    # One note per source, so one long chat cannot fill all the places.
    picked, used = [], set()
    for note in scored:
        if note["source"] in used:
            continue
        used.add(note["source"])
        picked.append(note)
        if len(picked) >= top:
            break
    return picked


class Indexer:
    """Does the embedding in the background, one job at a time, so saving a chat is never slowed by it."""

    def __init__(self, embed: Embed) -> None:
        self._embed = embed
        self._jobs: queue.Queue = queue.Queue()
        self._waiting: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self.last_error = ""
        threading.Thread(target=self._run, daemon=True).start()

    def chat(self, chat_id: str) -> None:
        self._add(("chat", chat_id))

    def picture(self, key: str) -> None:
        self._add(("pic", key))

    def forget_chat(self, chat_id: str) -> None:
        try:
            remove(chat_source(chat_id))
        except (ValueError, OSError):
            pass

    def everything(self) -> None:
        """Catch up on chats and pictures saved before now. Only new text is embedded."""
        try:
            for item in store.listing():
                self.chat(item["id"])
            if descriptions.descriptions_dir().is_dir():
                for path in descriptions.descriptions_dir().glob("*.json"):
                    self.picture(path.stem)
        except OSError:
            pass

    def _add(self, job: tuple[str, str]) -> None:
        with self._lock:
            if job in self._waiting:
                return
            self._waiting.add(job)
        self._jobs.put(job)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            with self._lock:
                self._waiting.discard(job)
            try:
                self.work(job)
                self.last_error = ""
            except Exception as error:  # a failed note must never stop the helper
                self.last_error = str(error)

    def work(self, job: tuple[str, str]) -> None:
        kind, ident = job
        if kind == "chat":
            try:
                chat = store.load(ident)
            except (ValueError, OSError):
                remove(chat_source(ident))
                return
            title, texts = chat_chunks(chat)
            sync(chat_source(ident), title, texts, self._embed)
        else:
            data = descriptions.load(ident)
            if data is None:
                remove(description_source(ident))
                return
            title, texts = description_chunks(data)
            sync(description_source(ident), title, texts, self._embed)
