"""One way to reach a model, whether MLX (engine.py) or Ollama runs it.

A name the engine finds in the Hugging Face cache is an MLX model; every other
name is Ollama's. The rest of the helper calls these and never asks which one it
is talking to.
"""

from __future__ import annotations

from helper import engine, ollama
from helper.ollama import KEEP_ALIVE, OllamaError


def tags() -> list[str]:
    """MLX models first, then Ollama's. Fails only when there are none and Ollama does not answer."""
    mlx = engine.names()
    try:
        return mlx + ollama.tags()
    except OllamaError:
        if not mlx:
            raise
        return mlx


def sizes() -> dict[str, int]:
    out = {}
    try:
        out.update(ollama.sizes())
    except OllamaError:
        pass
    out.update(engine.sizes())
    return out


def running() -> list[dict]:
    """What is in memory in both. Fails only when Ollama does not answer and MLX has nothing loaded."""
    mlx = engine.running()
    try:
        return mlx + ollama.running()
    except OllamaError:
        if not mlx and not engine.names():
            raise
        return mlx


def show(model: str) -> dict:
    return engine.show(model) if engine.owns(model) else ollama.show(model)


def unload(model: str) -> None:
    if engine.owns(model):
        engine.unload(model)
    else:
        ollama.unload(model)


def open_chat(model: str, messages: list[dict], keep_alive: str = KEEP_ALIVE, options: dict | None = None,
              think=None):
    if engine.owns(model):
        return engine.open_chat(model, messages, keep_alive, options, think)
    return ollama.open_chat(model, messages, keep_alive, options, think)


def complete(model: str, messages: list[dict], options: dict, keep_alive: str = KEEP_ALIVE, think=None) -> str:
    if engine.owns(model):
        return engine.complete(model, messages, options, keep_alive, think)
    return ollama.complete(model, messages, options, keep_alive, think)
