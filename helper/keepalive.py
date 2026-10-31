"""Hold models to the keepalive.

Every request sets a keepalive: how long its model stays in memory once the
request ends. The helper checks what is loaded every half minute and unloads a
model whose keepalive has passed since its last request ended, as long as
nothing of HikariLM's is using it.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager

GRACE = 30           # seconds past the deadline before stepping in
EVERY = 30           # seconds between checks


def seconds(value, default: float = 300.0) -> float:
    """A duration ("5m", "1h30m", "90s", "300", "-1") in seconds. Negative means never
    unload, which is infinity here."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    try:
        number = float(text)
        return float("inf") if number < 0 else number
    except ValueError:
        pass
    if text.startswith("-"):
        return float("inf")
    parts = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?", text)
    if not parts or not any(parts.groups()):
        return default
    hours, minutes, secs, millis = (float(group or 0) for group in parts.groups())
    return hours * 3600 + minutes * 60 + secs + millis / 1000


class Keeper:
    def __init__(self, keep_alive: float, clock: Callable[[], float] = time.time) -> None:
        self.keep_alive = keep_alive
        self._clock = clock
        self._lock = threading.Lock()
        self._busy: dict[str, int] = {}
        self._ended: dict[str, tuple[float, float]] = {}  # model -> (when the last request ended, its keepalive)

    @contextmanager
    def using(self, model: str, keep_alive=None):
        """Around every request HikariLM makes with a model, with the keepalive it sent."""
        keep = self.keep_alive if keep_alive is None else seconds(keep_alive, self.keep_alive)
        with self._lock:
            self._busy[model] = self._busy.get(model, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._busy[model] -= 1
                if not self._busy[model]:
                    del self._busy[model]
                self._ended[model] = (self._clock(), keep)

    def busy(self, model: str) -> bool:
        with self._lock:
            return model in self._busy

    def leaves_at(self, model: str):
        """When this watchdog will unload the model, or None while it is busy or unknown."""
        with self._lock:
            ended = self._ended.get(model)
            if ended is None or model in self._busy:
                return None
            return ended[0] + ended[1]

    def overdue(self, loaded: list[dict]) -> list[str]:
        """The loaded models that should have left memory by now."""
        now = self._clock()
        out = []
        with self._lock:
            for item in loaded:
                name = item.get("name")
                if not name or name in self._busy:
                    continue
                ended = self._ended.get(name)
                if ended is not None and now > ended[0] + ended[1] + GRACE:
                    out.append(name)
        return out

    def forget(self, model: str) -> None:
        with self._lock:
            self._ended.pop(model, None)

    def check(self, running: Callable[[], list[dict]], unload: Callable[[str], None]) -> list[str]:
        """Unload what has outstayed. Returns the names unloaded."""
        done = []
        for name in self.overdue(running()):
            if self.busy(name):
                continue
            unload(name)
            self.forget(name)
            done.append(name)
        return done

    def watch(self, running, unload, stopping: threading.Event, every: float = EVERY,
              on_unload: Callable[[str], None] | None = None) -> threading.Thread:
        def loop():
            while not stopping.wait(every):
                try:
                    for name in self.check(running, unload):
                        if on_unload:
                            on_unload(name)
                except Exception:
                    # A model busy loading: try again at the next check.
                    pass

        thread = threading.Thread(target=loop, name="keepalive", daemon=True)
        thread.start()
        return thread
