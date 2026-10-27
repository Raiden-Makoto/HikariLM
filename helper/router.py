"""Decide when a large model has to leave memory.

Follow-ups reuse the loaded model. nomic-embed-text is small enough to sit
beside it, so loading nomic never forces the large model out. A different
large model does.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

NOMIC = "nomic-embed-text"


def is_nomic(model: str) -> bool:
    return model.split(":", 1)[0] == NOMIC


def chat_models(names: list[str]) -> list[str]:
    """Models you can chat with. nomic-embed-text only indexes text."""
    return [name for name in names if not is_nomic(name)]


class ModelRouter:
    def __init__(self, unload: Callable[[str], None]) -> None:
        self._unload = unload
        self._lock = threading.Lock()
        self.large: str | None = None

    def begin(self, model: str) -> None:
        """Unload the previous large model when this request needs a different one."""
        if is_nomic(model):
            return
        with self._lock:
            previous = self.large
            if previous is not None and previous != model:
                self._unload(previous)
                self.large = None

    def commit(self, model: str) -> None:
        """Remember the large model only after Ollama has accepted the request."""
        if is_nomic(model):
            return
        with self._lock:
            self.large = model
