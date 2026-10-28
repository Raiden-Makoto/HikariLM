"""Fit attached documents into the context, however long they are.

A model reads only what fits in its context, and that context also has to hold
the conversation, the model's thinking, and its reply. A paper pasted whole
into a 4,096-token context is silently cut by Ollama. So every send, the helper
works out the room left for documents and then:

- sends them whole when they fit;
- for a request about the whole document (a summary, an overview, the key
  points), reads it all first: the model writes notes on each part in turn,
  each part sized to fit, and answers from those notes. The notes are kept, so
  a follow-up does not read the document again;
- otherwise sends the passages closest to the question, found with
  nomic-embed-text (or by shared words when nomic is not installed), as many as
  fit, in the order they appear.

Pieces and their vectors are kept in ~/.lightusagi/documents/, named after a
hash of the text, so the same file in another chat is not split or embedded
again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

from helper import store

CHUNK = 1200            # characters in one passage
BATCH = 16
KEEP_FILES = 100        # documents/ keeps this many, newest first
NOTE_MIN, NOTE_MAX = 120, 700
PART_OVERHEAD = 400     # tokens for the note prompt around a part
PAGE_MARK = re.compile(r"^--- page (\d+) ---$")
WHOLE = re.compile(
    r"\b(summar\w*|overview|tl;?dr|outline|gist|recap|abstract|critique|review|"
    r"key (points?|findings?|ideas?|takeaways?|results?|contributions?)|"
    r"main (points?|ideas?|findings?|arguments?|results?|contributions?)|"
    r"what('s| is) (this|it|the)( \w+)? about|(whole|entire|full) (\w+ )?(paper|document|article|pdf|file|report|book|text)|"
    r"(every|each|all) (sections?|pages?|chapters?|parts?)|section by section|page by page|chapter by chapter|go through)\b",
    re.I)
WORD = re.compile(r"[a-z0-9]{3,}")
STOP = set("the and for are but not you all any can had her was one our out has have this that with from they "
           "will what when where which who why how does did its into than then them these those there their about "
           "would could should been being were also just more most some such only other very your".split())

NOTE_PROMPT = """Below is part {part} of {parts} of the document "{name}"{pages}.

<document_part>
{text}
</document_part>

Write detailed notes on this part, so that someone who never sees it can answer questions about it and write a full summary of the document. Keep the section titles and, in order, every claim, method, definition, equation, number, result and example. Use short bullet points. Add nothing that is not in the text and do not comment on the task. At most about {words} words."""

CONDENSE_PROMPT = """Below are notes on consecutive parts of the document "{name}".

<notes>
{text}
</notes>

Merge them into one set of notes in the same order. Keep every section title, claim, method, number and result; drop only repetition. Use short bullet points and add nothing new. At most about {words} words."""

Embed = Callable[[list[str]], list[list[float]]]
Ask = Callable[[str, int], str]


def tokens(text: str) -> int:
    """A cautious estimate: papers with numbers and symbols run short of 4 characters a token."""
    return len(text) * 10 // 32 + 1


def wants_whole(question: str) -> bool:
    """Whether a question is about the whole document rather than one part of it."""
    return not question.strip() or bool(WHOLE.search(question))


def reply_room(num_ctx: int) -> int:
    return max(768, min(4096, num_ctx // 5))


def thinking_room(num_ctx: int, budget: int | None) -> int:
    """The least the thinking keeps when documents are fitted in."""
    return min(budget or num_ctx, max(512, num_ctx // 8))


# --- documents in a chat --------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def gather(messages: list[dict]) -> list[dict]:
    """Every text file in the chat, once each, in the order they were attached."""
    seen, docs = set(), []
    for index, message in enumerate(messages):
        for file in message.get("files") or []:
            text = file["text"]
            key = _hash(text)
            if not text.strip() or key in seen:
                continue
            seen.add(key)
            docs.append({"name": file["name"], "text": text, "hash": key, "message": index})
    return docs


def block(name: str, text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"File: {name}\n{fence}\n{text}\n{fence}"


def mention(names: list[str]) -> str:
    return "[Attached: " + ", ".join(names) + "]"


# --- pieces ---------------------------------------------------------------

def split(text: str) -> list[dict]:
    """Passages of about CHUNK characters, cut at line breaks, each with the page it starts on."""
    pieces: list[dict] = []
    current, start_page, page = "", None, None

    def close():
        nonlocal current
        if current.strip():
            pieces.append({"text": current.strip(), "page": start_page})
        current = ""

    for line in text.splitlines(keepends=True):
        mark = PAGE_MARK.match(line.strip())
        if mark:
            page = int(mark.group(1))
            if not current.strip():
                start_page = page
            continue
        while len(line) > CHUNK:
            close()
            start_page = page
            current = line[:CHUNK]
            close()
            line = line[CHUNK:]
        if len(current) + len(line) > CHUNK and current.strip():
            close()
        if not current.strip():
            start_page = page
        current += line
    close()
    return pieces


def _folder() -> Path:
    return store.home() / "documents"


def _path(key: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", key):
        raise ValueError("document key is not valid")
    return _folder() / f"{key}.json"


def load(doc: dict) -> dict:
    """The saved pieces for a document, split now if it is new."""
    try:
        data = json.loads(_path(doc["hash"]).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            return data
    except (FileNotFoundError, ValueError, OSError):
        pass
    return {"name": doc["name"], "chunks": split(doc["text"]), "vectors": None, "notes": {}}


def save(doc: dict, data: dict) -> None:
    folder = _folder()
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=folder, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(temp, _path(doc["hash"]))
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    files = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old in files[KEEP_FILES:]:
        try:
            old.unlink()
        except OSError:
            pass


def _pages(chunks: list[dict]) -> str:
    pages = [chunk["page"] for chunk in chunks if chunk.get("page")]
    if not pages:
        return ""
    return f", page {pages[0]}" if pages[0] == pages[-1] else f", pages {pages[0]}–{pages[-1]}"


# --- passages closest to the question -----------------------------------

def _unit(vector: list[float]) -> list[float]:
    size = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / size for x in vector]


def _by_meaning(question: str, entries: list[tuple], embed: Embed) -> list[float]:
    """Cosine scores from nomic. Vectors are made once per document and kept."""
    for doc, data in {id(e[0]): (e[0], e[1]) for e in entries}.values():
        if data.get("vectors") and len(data["vectors"]) == len(data["chunks"]):
            continue
        vectors = []
        texts = [chunk["text"] for chunk in data["chunks"]]
        for start in range(0, len(texts), BATCH):
            got = embed(["search_document: " + text for text in texts[start:start + BATCH]])
            vectors += [[round(x, 5) for x in _unit(vector)] for vector in got]
        data["vectors"] = vectors
        save(doc, data)
    query = _unit(embed(["search_query: " + question])[0])
    return [sum(a * b for a, b in zip(query, data["vectors"][index])) for _, data, index in entries]


def _by_words(question: str, entries: list[tuple]) -> list[float]:
    """Shared rare words, for when nomic-embed-text is not there."""
    wanted = {word for word in WORD.findall(question.lower()) if word not in STOP}
    texts = [set(WORD.findall(data["chunks"][index]["text"].lower())) for _, data, index in entries]
    count = len(texts) or 1
    rarity = {word: math.log(1 + count / (1 + sum(word in words for words in texts))) for word in wanted}
    return [sum(rarity[word] for word in wanted if word in words) for words in texts]


def excerpts(question: str, docs: list[dict], room: int, embed: Embed | None) -> tuple[str, dict]:
    """As many of the closest passages as fit in `room` tokens, in document order."""
    loaded = [(doc, load(doc)) for doc in docs]
    entries = [(doc, data, index) for doc, data in loaded for index in range(len(data["chunks"]))]
    how = "meaning"
    try:
        if embed is None:
            raise LookupError
        scores = _by_meaning(question, entries, embed)
    except Exception:
        how = "words"
        scores = _by_words(question, entries)
    order = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)
    # The start of each document (title, abstract) first, then the closest passages.
    firsts = [i for i, (_, _, index) in enumerate(entries) if index == 0]
    picked, used = set(), 0
    for i in firsts + order:
        if i in picked:
            continue
        cost = tokens(entries[i][1]["chunks"][entries[i][2]]["text"]) + 12
        if used + cost > room:
            continue
        picked.add(i)
        used += cost
    parts, files = [], []
    for doc, data in loaded:
        chosen = sorted(index for i, (d, _, index) in enumerate(entries) if d is doc and i in picked)
        files.append({"name": doc["name"], "used": len(chosen), "total": len(data["chunks"])})
        if not chosen:
            continue
        lines, last = [], None
        for index in chosen:
            chunk = data["chunks"][index]
            if last is not None and index != last + 1:
                lines.append("[…]")
            where = f"[page {chunk['page']}] " if chunk.get("page") else ""
            lines.append(where + chunk["text"])
            last = index
        parts.append(f"Passages from {doc['name']}. It is too long to send whole, so these are the parts "
                     f"closest to the question, in the order they appear ([…] marks a gap):\n\n" + "\n\n".join(lines))
    return "\n\n".join(parts), {"mode": "excerpts", "search": how, "files": files}


# --- reading the whole document -----------------------------------------

def _groups(chunks: list[dict], limit: int) -> list[list[dict]]:
    groups, current, size = [], [], 0
    for chunk in chunks:
        cost = tokens(chunk["text"]) + 2
        if current and size + cost > limit:
            groups.append(current)
            current, size = [], 0
        current.append(chunk)
        size += cost
    if current:
        groups.append(current)
    return groups


def notes(docs: list[dict], model: str, num_ctx: int, room: int, ask: Ask,
          progress: Callable[[str, int, int], bool]) -> tuple[str, dict] | None:
    """Read every document in parts that fit the context, write notes on each part, and
    return notes that fit in `room` tokens. None if stopped (progress returned False)."""
    part_size = max(600, num_ctx - NOTE_MAX - PART_OVERHEAD)
    loaded = [(doc, load(doc)) for doc in docs]
    split_up = [(doc, data, _groups(data["chunks"], part_size)) for doc, data in loaded]
    # Fixed for a context length, not the room left, so the notes are kept and reused.
    cap = max(NOTE_MIN, min(NOTE_MAX, part_size // 6))
    sections, files = [], []
    for doc, data, groups in split_up:
        key = f"{model}|{part_size}|{cap}"
        kept = data.setdefault("notes", {}).get(key)
        written = kept if isinstance(kept, list) and len(kept) == len(groups) else []
        for number, group in enumerate(groups, 1):
            if number <= len(written):
                continue
            if not progress(doc["name"], number, len(groups)):
                return None
            prompt = NOTE_PROMPT.format(part=number, parts=len(groups), name=doc["name"], pages=_pages(group),
                                        text="\n\n".join(chunk["text"] for chunk in group), words=cap * 3 // 4)
            written.append(ask(prompt, cap + 64).strip() or "(nothing to note)")
            data["notes"][key] = written
            save(doc, data)
        labelled = [f"Part {number} of {len(groups)}{_pages(group)}:\n{text}"
                    for number, (group, text) in enumerate(zip(groups, written), 1)]
        labelled = _condense(doc["name"], labelled, room * tokens(doc["text"]) // max(1, sum(tokens(d["text"]) for d in docs)),
                             num_ctx, ask)
        files.append({"name": doc["name"], "parts": len(groups)})
        sections.append(f"Notes on {doc['name']}, written by reading all of it in {len(groups)} "
                        f"part{'s' if len(groups) != 1 else ''}, in order:\n\n" + "\n\n".join(labelled))
    text = "\n\n".join(sections)
    limit = room * 32 // 10
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n[the notes were cut here to fit]"
    return text, {"mode": "notes", "files": files}


def _condense(name: str, labelled: list[str], room: int, num_ctx: int, ask: Ask) -> list[str]:
    """Merge neighbouring notes until they fit in `room` tokens."""
    while sum(tokens(text) for text in labelled) > room and len(labelled) > 1:
        groups = _groups([{"text": text} for text in labelled], max(600, num_ctx - NOTE_MAX - PART_OVERHEAD))
        if len(groups) == len(labelled):
            groups = [labelled[i:i + 2] for i in range(0, len(labelled), 2)]
            groups = [[{"text": text} for text in group] for group in groups]
        cap = max(NOTE_MIN, min(NOTE_MAX, room // len(groups) - 20))
        labelled = [ask(CONDENSE_PROMPT.format(name=name, text="\n\n".join(item["text"] for item in group),
                                               words=cap * 3 // 4), cap + 64).strip() for group in groups]
    return labelled
