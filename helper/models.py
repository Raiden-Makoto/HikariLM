"""The one way the rest of the helper reaches a model. Every model runs in MLX
(engine.py); this keeps the keepalive next to it."""

from __future__ import annotations

import os

from helper import engine, keepalive
from helper.engine import ModelError

KEEP_ALIVE = os.environ.get("HIKARILM_KEEP_ALIVE", "5m")
# Every request HikariLM makes with a model runs inside KEEPER.using(model).
KEEPER = keepalive.Keeper(keepalive.seconds(KEEP_ALIVE))

__all__ = ["KEEP_ALIVE", "KEEPER", "ModelError", "complete", "open_chat", "running", "show", "sizes", "tags",
           "unload"]


def tags() -> list[str]:
    return engine.names()


def sizes() -> dict[str, int]:
    return engine.sizes()


def running() -> list[dict]:
    return engine.running()


def show(model: str) -> dict:
    return engine.show(model)


def unload(model: str) -> None:
    engine.unload(model)


def open_chat(model: str, messages: list[dict], keep_alive: str = KEEP_ALIVE, options: dict | None = None,
              think=None):
    return engine.open_chat(model, messages, keep_alive, options, think)


def complete(model: str, messages: list[dict], options: dict, keep_alive: str = KEEP_ALIVE, think=None) -> str:
    return engine.complete(model, messages, options, keep_alive, think)
