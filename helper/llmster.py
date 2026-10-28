"""Talk to LM Studio's llmster server (or the LM Studio app's server), for models Ollama does not have.

llmster lists its models with their size, context length and what they can do,
loads and unloads them on request, and answers through the OpenAI-style
/v1/chat/completions, which takes a whole chat history. The helper loads a
model itself, with the context length it worked out, and unloads it when it
switches models or the keepalive runs out, so llmster's models keep to the
same one-large-model rule as Ollama's.

The address is LMSTUDIO_HOST (http://127.0.0.1:1234 by default) and the token,
when llmster asks for one, LM_API_TOKEN. When nothing answers there, llmster
is simply left out.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

from helper.ollama import OllamaError

HOST = os.environ.get("LMSTUDIO_HOST", "http://127.0.0.1:1234").rstrip("/")
if "://" not in HOST:
    HOST = "http://" + HOST
TOKEN = os.environ.get("LM_API_TOKEN", "")
PREFIX = "lms:"
ABSENT_FOR = 30          # seconds to skip llmster after it did not answer
FRESH_FOR = 2            # seconds one model list is reused, so a page load asks once
_absent_until = 0.0
_cached: tuple[float, list[dict]] | None = None
_lock = threading.Lock()


class LlmsterError(OllamaError):
    pass


def is_lms(model: str) -> bool:
    return model.startswith(PREFIX)


def key(model: str) -> str:
    return model[len(PREFIX):] if is_lms(model) else model


def _request(path: str, payload: dict | None = None, timeout: float | None = 600):
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f"{HOST}{path}", data=data, headers=headers,
                                     method="POST" if payload is not None else "GET")
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        try:
            body = json.loads(detail)
            message = body.get("error", detail)
            if isinstance(message, dict):
                message = message.get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            message = detail or error.reason
        raise LlmsterError(f"LM Studio: {message}") from error
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise LlmsterError(f"LM Studio's llmster is not reachable at {HOST}: {reason}") from error


def catalog(quiet: bool = False) -> list[dict]:
    """Every model llmster has downloaded, as /api/v1/models lists them. With quiet,
    an llmster that does not answer gives an empty list, and is not asked again for a while."""
    global _absent_until, _cached
    now = time.monotonic()
    if _cached and now - _cached[0] < FRESH_FOR:
        return _cached[1]
    if quiet and now < _absent_until:
        return []
    try:
        with _request("/api/v1/models", timeout=3 if quiet else 10) as response:
            body = json.loads(response.read().decode())
    except LlmsterError:
        if quiet:
            _absent_until = now + ABSENT_FOR
            return []
        raise
    _absent_until = 0.0
    models = body.get("models")
    models = models if isinstance(models, list) else []
    _cached = (now, models)
    return models


def reachable() -> bool:
    catalog(quiet=True)
    return time.monotonic() >= _absent_until


def _changed() -> None:
    """A model was loaded or unloaded, so the kept list is out of date."""
    global _cached
    _cached = None


def _entry(model: str) -> dict:
    for item in catalog():
        if item.get("key") == key(model):
            return item
    raise LlmsterError(f"LM Studio does not have {key(model)}. Download it with `lms get`, or pick another model.")


def names() -> list[str]:
    """The models you can chat with. Embedding and decision models are left out."""
    return [PREFIX + item["key"] for item in catalog(quiet=True) if item.get("type") == "llm" and item.get("key")]


def sizes() -> dict[str, int]:
    return {PREFIX + item["key"]: int(item.get("size_bytes") or 0) for item in catalog(quiet=True) if item.get("key")}


def running() -> list[dict]:
    """The models llmster has in memory. It sets no expiry; the helper's keepalive covers them."""
    out = []
    for item in catalog(quiet=True):
        if item.get("loaded_instances") and item.get("type") == "llm":
            out.append({"name": PREFIX + item["key"], "size": int(item.get("size_bytes") or 0), "expires": None})
    return out


def show(model: str) -> dict:
    """The model's details in the shape Ollama's /api/show has, so the context and memory
    checks treat both alike. llmster does not say a model's layers and heads, so the
    cache estimate falls back to its cautious guess."""
    item = _entry(model)
    arch = str(item.get("architecture") or "llmster")
    info = {"general.architecture": arch}
    if item.get("max_context_length"):
        info[f"{arch}.context_length"] = int(item["max_context_length"])
    capabilities = ["completion"]
    if (item.get("capabilities") or {}).get("vision"):
        capabilities.append("vision")
    return {"model_info": info, "capabilities": capabilities, "llmster": True}


def unload(model: str) -> None:
    """Unload every instance of this model."""
    for item in catalog(quiet=True):
        if item.get("key") != key(model):
            continue
        for instance in item.get("loaded_instances") or []:
            with _request("/api/v1/models/unload", {"instance_id": instance["id"]}, timeout=60) as response:
                response.read()
            _changed()


def ensure_loaded(model: str, num_ctx: int | None) -> str:
    """The instance to chat with, loaded with this context length. An instance with a
    different context length is unloaded first, so only one copy is ever in memory."""
    with _lock:
        item = _entry(model)
        for instance in item.get("loaded_instances") or []:
            if not num_ctx or (instance.get("config") or {}).get("context_length") == num_ctx:
                return instance["id"]
        unload(model)
        payload = {"model": item["key"]}
        if num_ctx:
            payload["context_length"] = num_ctx
        with _request("/api/v1/models/load", payload, timeout=900) as response:
            body = json.loads(response.read().decode())
        _changed()
        return str(body.get("instance_id") or item["key"])


def _image_type(data: str) -> str:
    for start, kind in (("iVBOR", "png"), ("/9j/", "jpeg"), ("UklGR", "webp"), ("R0lG", "gif")):
        if data.startswith(start):
            return kind
    return "png"


def _messages(messages: list[dict]) -> list[dict]:
    """Ollama's message shape (images as a list of base64) in OpenAI's (content parts)."""
    out = []
    for message in messages:
        images = message.get("images") or []
        if not images:
            out.append({"role": message["role"], "content": message["content"]})
            continue
        parts = [{"type": "text", "text": message["content"]}]
        parts += [{"type": "image_url", "image_url": {"url": f"data:image/{_image_type(image)};base64,{image}"}}
                  for image in images]
        out.append({"role": message["role"], "content": parts})
    return out


class Stream:
    """llmster's server-sent events, read as Ollama's lines: one JSON object per line with
    message.content and message.thinking, and a last one with done and the token counts."""

    def __init__(self, response) -> None:
        self._response = response
        self._usage: dict = {}
        self._finished = False

    def readline(self) -> bytes:
        while not self._finished:
            raw = self._response.readline()
            if not raw:
                return self._end()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return self._end()
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("error"):
                error = chunk["error"]
                return self._line({"error": "LM Studio: " + str(error.get("message") if isinstance(error, dict) else error)})
            if chunk.get("usage"):
                self._usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = str(delta.get("content") or "")
                reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
                if content or reasoning:
                    return self._line({"message": {"role": "assistant", "content": content, "thinking": reasoning},
                                       "done": False})
        return b""

    def _end(self) -> bytes:
        if self._finished:
            return b""
        self._finished = True
        return self._line({"message": {"role": "assistant", "content": ""}, "done": True,
                           "prompt_eval_count": int(self._usage.get("prompt_tokens") or 0),
                           "eval_count": int(self._usage.get("completion_tokens") or 0)})

    @staticmethod
    def _line(event: dict) -> bytes:
        return json.dumps(event).encode() + b"\n"

    def close(self) -> None:
        self._response.close()


def open_chat(model: str, messages: list[dict], num_ctx: int | None) -> Stream:
    instance = ensure_loaded(model, num_ctx)
    payload = {"model": instance, "messages": _messages(messages), "stream": True,
               "stream_options": {"include_usage": True}}
    return Stream(_request("/v1/chat/completions", payload))


def complete(model: str, messages: list[dict], num_ctx: int | None, max_tokens: int | None) -> str:
    instance = ensure_loaded(model, num_ctx)
    payload = {"model": instance, "messages": _messages(messages), "stream": False}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    with _request("/v1/chat/completions", payload) as response:
        body = json.loads(response.read().decode())
    choices = body.get("choices") or [{}]
    return str((choices[0].get("message") or {}).get("content") or "")
