"""The context length each request gets.

It is the length the user asked for in Settings, or else the model's own
default, never past what the model was trained for. The helper does not try to
work out memory: macOS manages that itself.
"""

from __future__ import annotations

import re

MIN_CONTEXT = 2048
DEFAULT_CONTEXT = 4096


def _info(shown: dict) -> tuple[dict, str]:
    info = shown.get("model_info") or {}
    return info, str(info.get("general.architecture") or "")


def _number(value) -> int:
    if isinstance(value, list):
        value = max((v for v in value if isinstance(v, (int, float))), default=0)
    return int(value) if isinstance(value, (int, float)) else 0


def trained_context(shown: dict) -> int:
    info, arch = _info(shown)
    return _number(info.get(f"{arch}.context_length"))


def default_source(shown: dict) -> tuple[int, str]:
    """The length used when Settings leaves it blank, and where it comes from: the
    model's own default (engine.show), or 4096."""
    found = re.search(r"^\s*num_ctx\s+(\d+)", str(shown.get("parameters") or ""), re.M)
    if found:
        return int(found.group(1)), "model"
    return DEFAULT_CONTEXT, "default"


def plan(requested: int | None, shown: dict) -> dict:
    """The context length to send."""
    default, source = default_source(shown)
    want = requested or default
    trained = trained_context(shown)
    if trained:
        want = min(want, trained)
    return {"num_ctx": max(MIN_CONTEXT, want), "requested": requested, "trained": trained or None,
            "default": default, "default_from": source}
