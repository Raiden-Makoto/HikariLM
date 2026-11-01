"""Making pictures: which models make them, the requests, and the images/ folder.

Pictures are drawn by FLUX.2 models through mflux (flux.py finds them, engine.py
runs them). They are listed apart and never answer messages.
"""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from helper import flux, store

SIZES = {
    "square": (1024, 1024),
    "portrait": (768, 1024),
    "landscape": (1024, 768),
    "wide": (1280, 720),
    "small": (512, 512),
}
MAX_PROMPT = 4000
MAX_REFERENCE = 20_000
MAX_SEED = 2 ** 31 - 1
MAX_STEPS = 100
ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
PNG = b"\x89PNG\r\n\x1a\n"


def apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine() == "arm64":
        return True
    # A Python running under Rosetta says x86_64 on an M-series Mac, so ask macOS.
    try:
        answer = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=5)
        return answer.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


SUGGESTED = "mlx-community/FLUX.2-Klein-4B-4bit"
GET_ONE = f"~/.hikarilm/venv/bin/hf download {SUGGESTED}"


def vision_models(names: list[str], show) -> list[str]:
    """The chat models that can read pictures. One whose details cannot be read is left out."""
    vision = []
    for name in names:
        try:
            if "vision" in (show(name).get("capabilities") or []):
                vision.append(name)
        except Exception:
            pass
    return vision


# --- which models make pictures -------------------------------------------------------

def is_picture_model(model: str) -> bool:
    return flux.find(model) is not None


def picture_models() -> tuple[list[dict], list[str]]:
    """Every downloaded FLUX.2 model, for the picker, and a plain note for each one left out.
    Two copies of the same model keep their repo names, so both can be told apart."""
    found, notes = flux.scan()
    items = [{"id": name, "label": model["label"]} for name, model in found.items()]
    labels = [item["label"] for item in items]
    for item in items:
        if labels.count(item["label"]) > 1:
            item["label"] = item["id"].split("/")[-1]
    return items, notes


# --- a request ------------------------------------------------------------------

def clean_request(payload: dict) -> dict:
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Write what the picture should show.")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"The prompt is longer than {MAX_PROMPT} characters.")
    size = payload.get("size", "square")
    if size not in SIZES:
        raise ValueError("size is not one of the choices")
    seed = payload.get("seed")
    if seed is None or seed == "":
        seed = secrets.randbelow(MAX_SEED)
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise ValueError("seed must be a whole number")
    try:
        seed = int(seed)
    except ValueError:
        raise ValueError("seed must be a whole number") from None
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be from 0 to {MAX_SEED}")
    steps = payload.get("steps")
    if steps in (None, ""):
        steps = None
    else:
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            raise ValueError("steps must be a whole number") from None
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be from 1 to {MAX_STEPS}")
    # What was asked for, when a chat model rewrote it into the prompt.
    request = payload.get("request")
    if request is not None and (not isinstance(request, str) or len(request) > MAX_PROMPT):
        raise ValueError(f"request must be text of at most {MAX_PROMPT} characters")
    request = (request or "").strip()
    # What a vision model saw in a reference picture, which the rewrite was made from.
    reference = payload.get("reference")
    if reference is not None and (not isinstance(reference, str) or len(reference) > MAX_REFERENCE):
        raise ValueError(f"reference must be text of at most {MAX_REFERENCE} characters")
    reference = (reference or "").strip()
    width, height = SIZES[size]
    cleaned = {"model": model.strip(), "prompt": prompt.strip(), "size": size, "width": width, "height": height,
               "seed": seed, "steps": steps}
    if request and request != cleaned["prompt"]:
        cleaned["request"] = request
    if reference:
        cleaned["reference"] = reference
    return cleaned


# --- images/ -----------------------------------------------------------------------

def images_dir() -> Path:
    return store.home() / "images"


def _path(image_id: str, suffix: str) -> Path:
    if not isinstance(image_id, str) or not ID_PATTERN.match(image_id):
        raise ValueError("picture id is not valid")
    return images_dir() / f"{image_id}{suffix}"


def _write(path: Path, data: bytes) -> None:
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".image.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def save(data: bytes, request: dict) -> dict:
    """Write the picture and its notes. The picture goes last, so a crash never shows half of one."""
    if not isinstance(data, bytes) or not data.startswith(PNG):
        raise ValueError("The picture model finished without a PNG picture.")
    folder = images_dir()
    folder.mkdir(parents=True, exist_ok=True)
    image_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    record = {"id": image_id, "prompt": request["prompt"], "model": request["model"], "seed": request["seed"],
              "size": request["size"], "width": request["width"], "height": request["height"],
              "steps": request["steps"], "created": time.time()}
    for key in ("request", "reference"):
        if request.get(key):
            record[key] = request[key]
    _write(_path(image_id, ".json"), json.dumps(record, indent=2).encode())
    _write(_path(image_id, ".png"), data)
    return record


def read(image_id: str) -> bytes:
    return _path(image_id, ".png").read_bytes()


def delete(image_id: str) -> bool:
    removed = False
    for suffix in (".png", ".json"):
        try:
            _path(image_id, suffix).unlink()
            removed = True
        except FileNotFoundError:
            pass
    return removed
