"""One way to reach a model, whether Ollama or LM Studio's llmster has it.

llmster's models carry the `lms:` prefix; every other name is Ollama's. The rest
of the helper calls these and never asks which server it is talking to.
"""

from __future__ import annotations

from helper import llmster, ollama
from helper.ollama import KEEP_ALIVE, OllamaError


def tags() -> list[str]:
    """Every model either server has. Fails only when neither answers."""
    try:
        names = ollama.tags()
    except OllamaError:
        names = llmster.names()
        if not names:
            raise
        return names
    return names + llmster.names()


def sizes() -> dict[str, int]:
    out = {}
    try:
        out.update(ollama.sizes())
    except OllamaError:
        pass
    out.update(llmster.sizes())
    return out


def running() -> list[dict]:
    """What is in memory on both. Fails only when neither answers."""
    try:
        loaded = ollama.running()
    except OllamaError:
        if not llmster.reachable():
            raise
        return llmster.running()
    return loaded + llmster.running()


def show(model: str) -> dict:
    return llmster.show(model) if llmster.is_lms(model) else ollama.show(model)


def unload(model: str) -> None:
    if llmster.is_lms(model):
        llmster.unload(model)
    else:
        ollama.unload(model)


def open_chat(model: str, messages: list[dict], keep_alive: str = KEEP_ALIVE, options: dict | None = None,
              think=None):
    if llmster.is_lms(model):
        return llmster.open_chat(model, messages, (options or {}).get("num_ctx"))
    return ollama.open_chat(model, messages, keep_alive, options, think)


def complete(model: str, messages: list[dict], options: dict, keep_alive: str = KEEP_ALIVE, think=None) -> str:
    if llmster.is_lms(model):
        return llmster.complete(model, messages, options.get("num_ctx"), options.get("num_predict"))
    return ollama.complete(model, messages, options, keep_alive, think)
