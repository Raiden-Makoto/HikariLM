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


def open_chat(model: str, messages: list[dict], keep_alive: str = KEEP_ALIVE):
    """Open a streaming chat. The caller closes the response."""
    return _request(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive,
        },
    )
