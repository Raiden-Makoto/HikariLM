"""Talk to a local Ollama server. The helper does not load models itself."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
KEEP_ALIVE = os.environ.get("LIGHTUSAGI_KEEP_ALIVE", "5m")


class OllamaError(Exception):
    pass


def _request(path: str, payload: dict | None = None, timeout: float | None = 600):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        try:
            message = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            message = detail or error.reason
        raise OllamaError(str(message)) from error
    except urllib.error.URLError as error:
        raise OllamaError(f"Ollama is not reachable at {OLLAMA_HOST}: {error.reason}") from error


def unload(model: str) -> None:
    """Ask Ollama to drop a model now. Used only when switching large models."""
    with _request("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""}) as response:
        response.read()


def tags() -> list[str]:
    with _request("/api/tags", timeout=10) as response:
        body = json.loads(response.read().decode())
    names = []
    for item in body.get("models", []):
        name = item.get("name") or item.get("model")
        if name:
            names.append(name)
    return names


EMBED_MODEL = "nomic-embed-text"


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into vectors with nomic-embed-text.

    nomic is small and sits beside the large model, so this never unloads it.
    """
    if not any(name.split(":", 1)[0] == EMBED_MODEL for name in tags()):
        raise OllamaError(f"{EMBED_MODEL} is not installed in Ollama, so earlier chats cannot be searched.")
    with _request("/api/embed", {"model": EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE}, timeout=120) as response:
        body = json.loads(response.read().decode())
    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise OllamaError("Ollama sent back an unexpected answer for the embedding.")
    return vectors


def sizes() -> dict[str, int]:
    """Bytes on disk for each installed model, which is about what its weights take in memory."""
    with _request("/api/tags", timeout=10) as response:
        body = json.loads(response.read().decode())
    return {item.get("name") or item.get("model"): int(item.get("size") or 0) for item in body.get("models", [])}


def running() -> list[dict]:
    """What Ollama has in memory now, with how much each one takes."""
    with _request("/api/ps", timeout=10) as response:
        body = json.loads(response.read().decode())
    return [{"name": item.get("name") or item.get("model"), "size": int(item.get("size") or 0)}
            for item in body.get("models", [])]


_shown: dict[str, dict] = {}


def show(model: str) -> dict:
    """A model's details (layers, heads, trained context). They do not change, so they are kept."""
    if model not in _shown:
        with _request("/api/show", {"model": model}, timeout=30) as response:
            _shown[model] = json.loads(response.read().decode())
    return _shown[model]


def complete(model: str, messages: list[dict], options: dict, keep_alive: str = KEEP_ALIVE) -> str:
    """One answer, not streamed. Used to write a compaction summary."""
    with _request("/api/chat", {"model": model, "messages": messages, "stream": False,
                                "keep_alive": keep_alive, "options": options}) as response:
        body = json.loads(response.read().decode())
    message = body.get("message") or {}
    return str(message.get("content") or "")


def open_generate(payload: dict):
    """Ask an image model for a picture, streamed: progress lines, then one with the picture.
    The caller closes the response; closing it early stops the picture."""
    return _request("/api/generate", payload)


def open_chat(model: str, messages: list[dict], keep_alive: str = KEEP_ALIVE, options: dict | None = None):
    """Open a streaming chat. The caller closes the response."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": keep_alive,
    }
    if options:
        payload["options"] = options
    return _request("/api/chat", payload)
