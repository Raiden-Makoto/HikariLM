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


def mac_available(stats: str, total: int) -> int:
    """What macOS can hand to a new program, from `vm_stat`.

    Cached files (file-backed pages) count as free: macOS drops them the moment
    memory is needed, and after a while they fill most of the RAM. So this is all
    of the memory less what is really held, the way Activity Monitor's "Memory
    Used" counts it: wired, app memory (anonymous pages less purgeable ones), and
    what the compressor holds."""
    page = int(re.search(r"page size of (\d+)", stats).group(1))

    def pages(label: str) -> int | None:
        found = re.search(re.escape(label) + r":\s+(\d+)", stats)
        return int(found.group(1)) if found else None

    wired, anonymous = pages("Pages wired down"), pages("Anonymous pages")
    if wired is not None and anonymous is not None:
        held = wired + max(0, anonymous - (pages("Pages purgeable") or 0)) + (pages("Pages occupied by compressor") or 0)
        return max(0, min(total, total - held * page))
    # An older macOS without those lines: count what it lists as free or droppable.
    labels = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    return min(total, sum((pages(label) or 0) for label in labels) * page)


def _mac() -> tuple[int, int | None]:
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                               timeout=5).stdout.strip())
    stats = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    return total, mac_available(stats, total)


def _sysconf() -> tuple[int, int | None]:
    page = os.sysconf("SC_PAGE_SIZE")
    total = os.sysconf("SC_PHYS_PAGES") * page
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * page
    except (ValueError, OSError):
        available = None
    return total, available


def memory(host: str | None = None) -> tuple[int, int | None] | None:
    """(total, available) in bytes for the computer Ollama (or the server at `host`) runs on.
    Available is None if it cannot be told. None if the memory cannot be known, so nothing
    is checked."""
    forced = os.environ.get("HIKARILM_MEMORY_GB")
    if forced:
        try:
            return int(float(forced) * GIB), None
        except ValueError:
            pass
    if not ollama_is_local(host):
        return None
    for reader in (_mac, _sysconf) if sys.platform == "darwin" else (_sysconf,):
        try:
            total, available = reader()
            if total > 0:
                return total, available
        except (OSError, KeyError, ValueError, IndexError, subprocess.SubprocessError):
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
    if shown.get("mlx"):
        # MLX keeps one 16-bit cache; Ollama's settings do not reach it.
        return int(layers * kv_heads * (key + value) * 2)
    cache = CACHE_BYTES.get(os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16"), 2.0)
    parallel = max(1, int(os.environ.get("OLLAMA_NUM_PARALLEL", "1") or 1))
    return int(layers * kv_heads * (key + value) * cache * parallel)


def trained_context(shown: dict) -> int:
    info, arch = _info(shown)
    return _number(info.get(f"{arch}.context_length"))


def default_source(shown: dict) -> tuple[int, str]:
    """What Ollama would use without being told, and where that comes from: the model's
    own setting, OLLAMA_CONTEXT_LENGTH, or 4096."""
    found = re.search(r"^\s*num_ctx\s+(\d+)", str(shown.get("parameters") or ""), re.M)
    if found:
        return int(found.group(1)), "model"
    try:
        value = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", ""))
        if value > 0:
            return value, "OLLAMA_CONTEXT_LENGTH"
    except ValueError:
        pass
    return DEFAULT_CONTEXT, "ollama"


PICTURE_WORK = 2 * GIB   # what a picture model needs beyond its weights while it draws


def check_picture(model: str, size: int, loaded: list[dict], mem: tuple[int, int | None] | None) -> None:
    """Raise NoRoom if a picture model cannot run safely. It has no context to cut, so it fits or not."""
    if mem is None:
        return
    total, available = mem
    leaving = sum(item["size"] for item in loaded if not item["name"].startswith("nomic-embed-text"))
    room = total * SHARE
    if available is not None:
        room = min(room, available + leaving)
    needed = size + PICTURE_WORK + RESERVE
    if needed > room:
        raise NoRoom(f"Not enough memory to run {model} safely. It needs about {_gb(needed)}, and about "
                     f"{_gb(max(0, room))} can be used now. Close some programs or choose a smaller model.")


def plan(model: str, requested: int | None, shown: dict, size: int, loaded: list[dict],
         mem: tuple[int, int | None] | None) -> dict:
    """The context length to send, and the most that fits right now.

    `loaded` is what Ollama has in memory. A large model other than this one is
    unloaded before this one loads, so its memory counts as free; nomic stays.
    """
    default, source = default_source(shown)
    want = requested or default
    trained = trained_context(shown)
    if trained:
        want = min(want, trained)
    want = max(MIN_CONTEXT, want)
    result = {"num_ctx": want, "requested": requested, "max_fit": None, "clamped": False, "trained": trained or None,
              "default": default, "default_from": source}
    if mem is None:
        return result
    total, available = mem
    leaving = sum(item["size"] for item in loaded if not item["name"].startswith("nomic-embed-text"))
    usable = total * SHARE
    if available is not None:
        usable = min(usable, available + leaving)
    fixed = int(shown.get("window_bytes") or 0)
    room = usable - RESERVE - size - fixed
    cost = per_token(shown)
    fit = int(room // cost) // STEP * STEP if room > 0 else 0
    if trained:
        fit = min(fit, trained)
    result["max_fit"] = fit
    if fit < MIN_CONTEXT:
        cache = fixed + cost * MIN_CONTEXT
        free = "" if available is None else f", {_gb(available)} free now" + (
            f" plus {_gb(leaving)} from the model it replaces" if leaving else "")
        raise NoRoom(
            f"Not enough memory to run {model} safely. It needs about {_gb(size + cache + RESERVE)} "
            f"({_gb(size)} model, {_gb(cache)} for {MIN_CONTEXT} tokens of context, {_gb(RESERVE)} kept spare), "
            f"and about {_gb(max(0, usable))} can be used (at most {SHARE:.0%} of {_gb(total)}{free}). "
            "Close some programs or choose a smaller model."
        )
    if want > fit:
        result["num_ctx"] = fit
        result["clamped"] = True
    return result
