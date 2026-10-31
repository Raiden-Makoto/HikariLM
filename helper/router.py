"""Decide when a large model has to leave memory.

Follow-ups reuse the loaded model. A different large model makes it leave first.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class Work:
    """What runs now. A picture and a chat reply each need their own large model,
    so one never starts while the other runs; that would put two in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chats = 0
        self.picture = False

    def start_chat(self) -> str | None:
        """None if a chat may start, else why not."""
        with self._lock:
            if self.picture:
                return "A picture is being made. Wait for it to finish, or stop it, then send again."
            self.chats += 1
            return None

    def end_chat(self) -> None:
        with self._lock:
            self.chats = max(0, self.chats - 1)

    def start_picture(self) -> str | None:
        with self._lock:
            if self.picture:
                return "A picture is already being made. Wait for it, or stop it."
            if self.chats:
                return "A chat reply is still being written. Wait for it to finish, or stop it, then generate."
            self.picture = True
            return None

    def end_picture(self) -> None:
        with self._lock:
            self.picture = False


class ModelRouter:
    def __init__(self, unload: Callable[[str], None]) -> None:
        self._unload = unload
        self._lock = threading.Lock()
        self.large: str | None = None

    def begin(self, model: str) -> None:
        """Unload the previous large model when this request needs a different one."""
        with self._lock:
            previous = self.large
            if previous is not None and previous != model:
                self._unload(previous)
                self.large = None

    def forget(self, model: str) -> None:
        """The model has left memory on its own (keepalive), so there is nothing to unload."""
        with self._lock:
            if self.large == model:
                self.large = None

    def commit(self, model: str) -> None:
        """Remember the large model only once it has accepted the request."""
        with self._lock:
            self.large = model
