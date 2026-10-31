"""The one way the rest of the helper reaches a model. Every model runs in MLX
(engine.py); this keeps the keepalive next to it. nomic-embed-text has a worker
of its own and is held to the same keepalive."""

from __future__ import annotations

import os

from helper import engine, keepalive
from helper.engine import ModelError

KEEP_ALIVE = os.environ.get("HIKARILM_KEEP_ALIVE", "5m")
# Every request HikariLM makes with a model runs inside KEEPER.using(model).
KEEPER = keepalive.Keeper(keepalive.seconds(KEEP_ALIVE))

EMBED_NAME = engine.EMBED_NAME

__all__ = ["EMBED_NAME", "KEEP_ALIVE", "KEEPER", "ModelError", "complete", "embed", "open_chat", "running", "show",
           "sizes", "tags", "unload"]


def tags() -> list[str]:
    return engine.names()


def sizes() -> dict[str, int]:
    return engine.sizes()


def running() -> list[dict]:
    """What is in memory: the chat model, and nomic beside it when it has been used."""
    return engine.running() + engine.embed_running()


def show(model: str) -> dict:
    return engine.show(model)


def unload(model: str) -> None:
    if model == EMBED_NAME:
        engine.unload_embed()
    else:
        engine.unload(model)


def embed(texts: list[str]) -> list[list[float]]:
    with KEEPER.using(EMBED_NAME):
        return engine.embed(texts)


def open_chat(model: str, messages: list[dict], keep_alive: str = KEEP_ALIVE, options: dict | None = None,
              think=None):
    return engine.open_chat(model, messages, keep_alive, options, think)


def complete(model: str, messages: list[dict], options: dict, keep_alive: str = KEEP_ALIVE, think=None) -> str:
    return engine.complete(model, messages, options, keep_alive, think)
