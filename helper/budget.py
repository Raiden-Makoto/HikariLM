"""Keep a model and its context inside the memory this computer really has.

Ollama keeps a cache for every token of context (the KV cache), and its size
grows with the context length. A context length that is too big for the
machine makes Ollama push the system into swap, or fail. So before every
request the helper works out how much context fits: the model's weights plus
its cache must stay under a share of the memory, and leave room for everything
else. A context length the user asked for is cut down to what fits, and a
model that does not fit at all is refused before it is loaded.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from urllib.parse import urlparse

GIB = 1024 ** 3
SHARE = 0.75            # at most this much of all memory for Ollama's model and cache
RESERVE = 1 * GIB       # always left free for the system, the browser and Ollama itself
MIN_CONTEXT = 2048
STEP = 256
DEFAULT_CONTEXT = 4096
FALLBACK_PER_TOKEN = 160 * 1024   # a cautious guess when a model does not say its shape
CACHE_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}


class NoRoom(ValueError):
    """The model cannot run safely on this computer right now. Worded for the user."""


def _gb(value: float) -> str:
    return f"{value / GIB:.1f} GB"


LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}


def ollama_is_local(host: str | None = None) -> bool:
    """Whether Ollama runs on this computer, so this computer's memory is the one that counts."""
    host = host if host is not None else os.environ.get("OLLAMA_HOST", "")
    if host and "://" not in host:
        host = "http://" + host
    name = (urlparse(host).hostname or "").lower()
    return name in LOCAL_HOSTS or name.startswith("127.") or name == socket.gethostname().lower()


def _proc() -> tuple[int, int | None]:
    with open("/proc/meminfo", encoding="ascii") as handle:
        values = {}
        for line in handle:
            name, _, rest = line.partition(":")
            values[name] = int(rest.split()[0]) * 1024
    return values["MemTotal"], values.get("MemAvailable")


def _mac() -> tuple[int, int | None]:
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                               timeout=5).stdout.strip())
    stats = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    page = int(re.search(r"page size of (\d+)", stats).group(1))
    free = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        found = re.search(label + r":\s+(\d+)", stats)
        if found:
            free += int(found.group(1)) * page
    return total, free


def _windows() -> tuple[int, int | None]:
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                    ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong)]

    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.total, status.available


def _sysconf() -> tuple[int, int | None]:
    page = os.sysconf("SC_PAGE_SIZE")
    total = os.sysconf("SC_PHYS_PAGES") * page
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * page
    except (ValueError, OSError):
        available = None
    return total, available


def memory() -> tuple[int, int | None] | None:
    """(total, available) in bytes for the computer Ollama runs on. Available is None if it
    cannot be told. None if the memory cannot be known, so nothing is checked."""
    forced = os.environ.get("LIGHTUSAGI_MEMORY_GB")
    if forced:
        try:
            return int(float(forced) * GIB), None
        except ValueError:
            pass
    if not ollama_is_local():
        return None
    if sys.platform == "win32":
        readers = (_windows,)
    elif sys.platform == "darwin":
        readers = (_mac, _sysconf)
    else:
        readers = (_proc, _sysconf)
    for reader in readers:
        try:
            total, available = reader()
            if total > 0:
                return total, available
        except (OSError, KeyError, ValueError, IndexError, AttributeError, subprocess.SubprocessError):
            continue
    return None


def _info(shown: dict) -> tuple[dict, str]:
    info = shown.get("model_info") or {}
    return info, str(info.get("general.architecture") or "")


def _number(value) -> int:
    if isinstance(value, list):
        value = max((v for v in value if isinstance(v, (int, float))), default=0)
    return int(value) if isinstance(value, (int, float)) else 0


def per_token(shown: dict) -> int:
    """Bytes of cache for one token of context, from the model's own shape."""
    info, arch = _info(shown)
    layers = _number(info.get(f"{arch}.block_count"))
    heads = _number(info.get(f"{arch}.attention.head_count"))
    kv_heads = _number(info.get(f"{arch}.attention.head_count_kv")) or heads
    width = _number(info.get(f"{arch}.embedding_length"))
    key = _number(info.get(f"{arch}.attention.key_length")) or (width // heads if heads else 0)
    value = _number(info.get(f"{arch}.attention.value_length")) or key
    if not (layers and kv_heads and key):
        return FALLBACK_PER_TOKEN
    cache = CACHE_BYTES.get(os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16"), 2.0)
    parallel = max(1, int(os.environ.get("OLLAMA_NUM_PARALLEL", "1") or 1))
    return int(layers * kv_heads * (key + value) * cache * parallel)


def trained_context(shown: dict) -> int:
    info, arch = _info(shown)
    return _number(info.get(f"{arch}.context_length"))


def default_context(shown: dict) -> int:
    """What Ollama would use without being told: the model's own setting, Ollama's, or 4096."""
    found = re.search(r"^\s*num_ctx\s+(\d+)", str(shown.get("parameters") or ""), re.M)
    if found:
        return int(found.group(1))
    try:
        return int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "")) or DEFAULT_CONTEXT
    except ValueError:
        return DEFAULT_CONTEXT


def plan(model: str, requested: int | None, shown: dict, size: int, loaded: list[dict],
         mem: tuple[int, int | None] | None) -> dict:
    """The context length to send, and the most that fits right now.

    `loaded` is what Ollama has in memory. A large model other than this one is
    unloaded before this one loads, so its memory counts as free; nomic stays.
    """
    want = requested or default_context(shown)
    trained = trained_context(shown)
    if trained:
        want = min(want, trained)
    want = max(MIN_CONTEXT, want)
    result = {"num_ctx": want, "requested": requested, "max_fit": None, "clamped": False, "trained": trained or None}
    if mem is None:
        return result
    total, available = mem
    leaving = sum(item["size"] for item in loaded if not item["name"].startswith("nomic-embed-text"))
    room = total * SHARE
    if available is not None:
        room = min(room, available + leaving)
    room -= RESERVE + size
    cost = per_token(shown)
    fit = int(room // cost) // STEP * STEP if room > 0 else 0
    if trained:
        fit = min(fit, trained)
    result["max_fit"] = fit
    if fit < MIN_CONTEXT:
        free = (available + leaving) if available is not None else total * SHARE
        raise NoRoom(
            f"Not enough memory to run {model} safely. It needs about {_gb(size + RESERVE + cost * MIN_CONTEXT)}, "
            f"and about {_gb(max(0, free))} can be used now. Close some programs or choose a smaller model."
        )
    if want > fit:
        result["num_ctx"] = fit
        result["clamped"] = True
    return result
