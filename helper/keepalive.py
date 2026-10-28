"""Hold models to the keepalive, even when Ollama does not.

Every request sets a keepalive, and Ollama is meant to unload the model when it
runs out. It does not always: a model can stay after a reply was cut off midway
(Stop, the thinking limit), or after a request that never finished. A model
left in memory takes the room the next one needs, so the helper checks Ollama's
list of loaded models every half minute and unloads one that has outstayed:

- a model LightUsagi used, once its own keepalive has passed since the last
  request ended and nothing of LightUsagi's is using it;
- any model Ollama itself says should have gone (its expiry is past), since
  Ollama has evidently missed it.

A model some other program pinned on purpose (an expiry more than a day away,
as `keep_alive: -1` gives) is left alone.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime

GRACE = 30           # seconds past the deadline before stepping in
EVERY = 30           # seconds between checks
PINNED = 86400       # an expiry further away than this was set on purpose


def seconds(value, default: float = 300.0) -> float:
    """An Ollama duration ("5m", "1h30m", "90s", "300", "-1") in seconds. Negative means
    never unload, which is infinity here."""
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


def parse_expiry(value) -> float | None:
    """Ollama's expires_at (RFC 3339, nanoseconds, any offset) as seconds since 1970."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    text = re.sub(r"\.(\d{1,5})(?=[+-]|$)", lambda m: "." + m.group(1).ljust(6, "0"), text)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None or moment.year < 2000:
        return None
    return moment.timestamp()


def key(name: str) -> str:
    """Ollama lists `gemma3` as `gemma3:latest`."""
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


class Keeper:
    def __init__(self, keep_alive: float, clock: Callable[[], float] = time.time) -> None:
        self.keep_alive = keep_alive
        self._clock = clock
        self._lock = threading.Lock()
        self._busy: dict[str, int] = {}
        self._ended: dict[str, tuple[float, float]] = {}  # model -> (when the last request ended, its keepalive)

    @contextmanager
    def using(self, model: str, keep_alive=None):
        """Around every request LightUsagi makes with a model, with the keepalive it sent."""
        model = key(model)
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
            return key(model) in self._busy

    def leaves_at(self, model: str):
        """When this watchdog will unload the model, or None while it is busy or unknown."""
        with self._lock:
            ended = self._ended.get(key(model))
            if ended is None or key(model) in self._busy:
                return None
            return ended[0] + ended[1]

    def overdue(self, loaded: list[dict]) -> list[str]:
        """The loaded models that should have left memory by now."""
        now = self._clock()
        out = []
        with self._lock:
            for item in loaded:
                name, expires = item.get("name"), item.get("expires")
                if not name or key(name) in self._busy:
                    continue
                if expires is not None and expires - now > PINNED:
                    continue
                ended = self._ended.get(key(name))
                ours_passed = ended is not None and now > ended[0] + ended[1] + GRACE
                ollama_passed = expires is not None and now > expires + GRACE
                if ours_passed or ollama_passed:
                    out.append(name)
        return out

    def forget(self, model: str) -> None:
        with self._lock:
            self._ended.pop(key(model), None)

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
                    # Ollama not running, or busy: try again at the next check.
                    pass

        thread = threading.Thread(target=loop, name="keepalive", daemon=True)
        thread.start()
        return thread
