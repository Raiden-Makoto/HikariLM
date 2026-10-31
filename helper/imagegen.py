"""Making pictures: which models make them, the requests, and the images/ folder.

Pictures are made by sd-server, which the helper starts, or by a picture server
the user runs. Their models are listed apart and never answer messages.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import platform
import re
import secrets
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from helper import store

SIZES = {
    "square": (1024, 1024),
    "portrait": (768, 1024),
    "landscape": (1024, 768),
    "wide": (1280, 720),
    "small": (512, 512),
}
MAX_PROMPT = 4000
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


# FLUX.2 klein 4B for sd-server, as stable-diffusion.cpp's docs/flux2.md sets it up. None
# of these needs a Hugging Face account (the small-decoder VAE stands in for FLUX.2-dev's,
# which is behind a sign-in).
KLEIN_FILES = (
    ("--diffusion-model", "flux-2-klein-4b-Q4_0.gguf",
     "https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_0.gguf"),
    ("--vae", "full_encoder_small_decoder.safetensors",
     "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors"),
    ("--llm", "Qwen3-4B-Q4_K_M.gguf",
     "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"),
)
KLEIN_OPTIONS = "--cfg-scale 1.0 --steps 4"


def models_dir() -> Path:
    return store.home() / "models"


def klein_setup() -> dict:
    """The files FLUX.2 klein needs, whether each is in the models folder yet, and the line
    for Settings that uses them."""
    folder = models_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    files = [{"name": name, "url": url, "found": (folder / name).is_file()} for _, name, url in KLEIN_FILES]
    line = "flux2-klein: " + " ".join(f"{flag} {shlex.quote(str(folder / name))}" for flag, name, _ in KLEIN_FILES)
    return {"folder": str(folder), "files": files, "line": f"{line} {KLEIN_OPTIONS}"}


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


# --- where pictures come from -------------------------------------------------------

SD_PREFIX = "sd:"          # an sd-server model the helper starts and stops
REMOTE_PREFIX = "remote:"  # a picture server the user runs, reached by its address


def is_picture_model(model: str) -> bool:
    return model.startswith((SD_PREFIX, REMOTE_PREFIX))
PICTURE_DEFAULTS = {"program": "", "models": "", "address": ""}


def clean_picture_settings(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key, limit in (("program", 1000), ("models", 20000), ("address", 500)):
        value = raw.get(key)
        out[key] = value.strip()[:limit] if isinstance(value, str) else ""
    if out["address"] and not out["address"].startswith(("http://", "https://")):
        out["address"] = "http://" + out["address"]
    return out


def picture_models(settings: dict) -> tuple[list[dict], list[str]]:
    """Every model that can make a picture now, for the picker, and a plain note for each
    source that is set up but not usable."""
    from helper import sdserver

    found, notes = [], []
    pictures = settings.get("pictures") or PICTURE_DEFAULTS
    models, problems = sdserver.parse_models(pictures["models"])
    notes += problems
    if models:
        program = sdserver.find_program(pictures["program"])
        if program:
            found += [{"id": SD_PREFIX + model["name"], "label": model["name"], "kind": "sd"} for model in models]
        else:
            notes.append(f"sd-server was not found{' at ' + pictures['program'] if pictures['program'] else ''}, "
                         "so its models are hidden. Use \"Set up picture making…\".")
    if pictures["address"]:
        try:
            name = sdserver.current_model(pictures["address"])
            found.append({"id": REMOTE_PREFIX + name, "label": name + " (picture server)", "kind": "remote"})
        except sdserver.PictureError as error:
            notes.append(str(error))
    return found, notes


def find_sd_model(settings: dict, model_id: str) -> dict | None:
    from helper import sdserver

    name = model_id[len(SD_PREFIX):]
    models, _ = sdserver.parse_models((settings.get("pictures") or PICTURE_DEFAULTS)["models"])
    return next((model for model in models if model["name"] == name), None)


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
    width, height = SIZES[size]
    return {"model": model.strip(), "prompt": prompt.strip(), "size": size, "width": width, "height": height,
            "seed": seed, "steps": steps}


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


def decode(image_b64) -> bytes:
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("The picture server sent no picture.")
    try:
        data = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("The picture server sent a picture that could not be read.") from None
    if not data.startswith(PNG):
        raise ValueError("The picture server sent something that is not a PNG picture.")
    return data


def save(data: bytes, request: dict) -> dict:
    """Write the picture and its notes. The picture goes last, so a crash never shows half of one."""
    folder = images_dir()
    folder.mkdir(parents=True, exist_ok=True)
    image_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    record = {"id": image_id, "prompt": request["prompt"], "model": request["model"], "seed": request["seed"],
              "size": request["size"], "width": request["width"], "height": request["height"],
              "steps": request["steps"], "created": time.time()}
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
