"""FLUX.2 picture models: which downloaded folders HikariLM draws with, and which model each is.

mflux runs every FLUX.2 klein model with one pipeline and is told which one it has by
its own name for it ("flux2-klein-9b"). Its saved folders carry no config, so the name
comes from the repo's name, checked against the transformer's blocks in the weights.

Only the models themselves are listed: Black Forest Labs' repos and mlx-community's
copies of them, or a folder in ~/.hikarilm/models named the same way. Fine-tunes and
LoRA adapters are left out, with a note saying why.
"""

from __future__ import annotations

import json
import re
import struct
import threading
import time
from pathlib import Path

from helper import store

# mflux's name: (label, double-stream blocks, single-stream blocks, distilled)
VARIANTS = {
    "flux2-klein-4b": ("FLUX.2 klein 4B", 5, 20, True),
    "flux2-klein-9b": ("FLUX.2 klein 9B", 8, 24, True),
    "flux2-klein-9b-kv": ("FLUX.2 klein 9B KV", 8, 24, True),
    "flux2-klein-base-4b": ("FLUX.2 klein base 4B", 5, 20, False),
    "flux2-klein-base-9b": ("FLUX.2 klein base 9B", 8, 24, False),
}
# Distilled models draw in 4 steps without guidance; the base models take 50 with it.
STEPS = {True: 4, False: 50}
GUIDANCE = {True: 1.0, False: 4.0}
MAKERS = ("black-forest-labs", "mlx-community")
PARTS = ("transformer", "text_encoder", "vae")
NAME = re.compile(r"flux2-klein(?P<base>-base)?-(?P<size>4b|9b)(?P<kv>-kv)?(?:-(?P<bits>\d)-?bit|-bf16|-fp16|-mlx)*")
BLOCK = re.compile(r"(?:^|\.)(single_)?transformer_blocks\.(\d+)\.")
MAX_HEADER = 100_000_000
SCAN_SECONDS = 5


def _plain(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "")


def variant_of(name: str) -> tuple[str, int | None] | None:
    """mflux's name for the model a repo or folder is named for, and the bits its name gives."""
    found = NAME.fullmatch(_plain(name.split("/")[-1]))
    if not found:
        return None
    key = "flux2-klein-" + ("base-" if found["base"] else "") + found["size"] + ("-kv" if found["kv"] else "")
    if key not in VARIANTS:
        return None
    return key, int(found["bits"]) if found["bits"] else None


def _header_keys(path: Path) -> list[str]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > MAX_HEADER:
            raise ValueError("header too large")
        header = json.loads(handle.read(size).decode("utf-8"))
    return [key for key in header if key != "__metadata__"]


def _weight_names(folder: Path) -> tuple[list[str], dict]:
    """The weight names in a component folder, and the index's metadata if it has one."""
    for index in sorted(folder.glob("*.safetensors.index.json")):
        data = json.loads(index.read_text(encoding="utf-8"))
        return list((data.get("weight_map") or {}).keys()), data.get("metadata") or {}
    names: list[str] = []
    for path in sorted(folder.glob("*.safetensors")):
        names += _header_keys(path)
    return names, {}


def blocks(names: list[str]) -> tuple[int, int]:
    """How many double-stream and single-stream blocks the weights hold."""
    double, single = set(), set()
    for name in names:
        found = BLOCK.search(name)
        if found:
            (single if found[1] else double).add(int(found[2]))
    return len(double), len(single)


def identify(name: str, folder: Path) -> tuple[dict | None, str]:
    """The model in this folder, or None and a note for the user (empty for a folder that
    is not a FLUX.2 model at all)."""
    plain = _plain(name)
    if "flux2" not in plain and "flux-2" not in plain:
        return None, ""
    named = variant_of(name)
    maker = name.split("/")[0] if "/" in name else ""
    if named is None or (maker and maker not in MAKERS):
        return None, (f"{name} is left out: HikariLM runs only the FLUX.2 klein models themselves, from "
                      "Black Forest Labs or mlx-community, not fine-tunes or LoRA adapters.")
    key, bits = named
    missing = [part for part in PARTS if not any((folder / part).glob("*.safetensors"))]
    if missing or not (folder / "tokenizer").is_dir():
        return None, f"{name} is left out: it is not a whole FLUX.2 model (no {', '.join(missing) or 'tokenizer'} folder)."
    label, double, single, distilled = VARIANTS[key]
    try:
        names, metadata = _weight_names(folder / "transformer")
    except (OSError, ValueError, struct.error):
        return None, f"{name} is left out: its transformer weights could not be read."
    if blocks(names) != (double, single):
        return None, f"{name} is left out: its name says {label}, but its weights are a different model."
    try:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    if isinstance(index, dict) and isinstance(index.get("is_distilled"), bool) and index["is_distilled"] != distilled:
        return None, f"{name} is left out: its name says {label}, but its model_index.json says otherwise."
    level = str(metadata.get("quantization_level") or "")
    bits = int(level) if level.isdigit() else bits
    return {"id": name, "label": label + (f", {bits}-bit" if bits else ""), "variant": key, "folder": folder,
            "steps": STEPS[distilled], "guidance": GUIDANCE[distilled], "size": _size(folder)}, ""


def _size(folder: Path) -> int:
    total = 0
    for part in PARTS:
        for path in (folder / part).glob("*.safetensors"):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


_scan: tuple[float, dict[str, dict], list[str]] = (0.0, {}, [])
_scan_lock = threading.Lock()


def scan() -> tuple[dict[str, dict], list[str]]:
    """Every FLUX.2 model downloaded in full, by name, and a note for each one left out."""
    from helper import engine

    global _scan
    with _scan_lock:
        if time.monotonic() - _scan[0] < SCAN_SECONDS:
            return _scan[1], _scan[2]
        out: dict[str, dict] = {}
        notes: list[str] = []
        hub = engine.hub()
        places = []
        for repo in sorted(hub.glob("models--*")) if hub.is_dir() else []:
            if any(repo.glob("blobs/*.incomplete")):
                continue
            snapshot = engine.snapshot(repo)
            if snapshot:
                places.append((repo.name[len("models--"):].replace("--", "/", 1), snapshot))
        own = store.home() / "models"
        places += [(folder.name, folder) for folder in sorted(own.iterdir()) if folder.is_dir()] if own.is_dir() else []
        for name, folder in places:
            model, note = identify(name, folder)
            if model:
                out[name] = model
            elif note:
                notes.append(note)
        _scan = (time.monotonic(), out, notes)
        return out, notes


def forget_scan() -> None:
    global _scan
    with _scan_lock:
        _scan = (0.0, {}, [])


def find(model: str) -> dict | None:
    return scan()[0].get(model)
