"""LoRAs from Hugging Face, drawn on FLUX.2 klein base 4B.

The user names a repo; it is looked up on the Hugging Face API, the LoRA file in it is
picked and checked, and it is kept in ~/.hikarilm/loras.json with its trigger word. It is
downloaded by hand (the command is shown), and is ready once the file is in the Hugging
Face cache. mflux bakes it into the model when the model loads.

Hugging Face is asked with curl, which every Mac has: Python from python.org cannot open
HTTPS until its certificates are installed.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

from helper import flux, store
from helper.names import label

VARIANT = "flux2-klein-base-4b"
API = "https://huggingface.co/api/models/"
FILES = "https://huggingface.co/{repo}/resolve/main/{file}"
DOWNLOAD = "~/.hikarilm/venv/bin/hf download "
REPO = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}$")
MAX_TRIGGER = 100
MAX_STRENGTH = 2.0
MAX_HEADER = 50_000_000
# ai-toolkit saves a checkpoint every so many steps beside the LoRA: name_000002750.safetensors.
STEP_FILE = re.compile(r"_\d{6,}\.safetensors$")
DOUBLE = re.compile(r"(?<!single_)(?:double_blocks|transformer_blocks)[._](\d+)[._]")
SINGLE = re.compile(r"(?:single_blocks|single_transformer_blocks)[._](\d+)[._]")
WIDTH_4B, WIDTH_9B = 3072, 4096


class LoraError(ValueError):
    pass


# --- Hugging Face ------------------------------------------------------------------

def _token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    home = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface"
    try:
        return (home / "token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _curl(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    """The status and the body. Raises LoraError when Hugging Face cannot be reached."""
    command = ["curl", "-sS", "-L", "--max-time", "20", "-w", "\n%{http_code}"]
    token = _token()
    for key, value in {**({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})}.items():
        command += ["-H", f"{key}: {value}"]
    try:
        done = subprocess.run(command + [url], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise LoraError(f"Could not reach Hugging Face: {error}") from None
    body, _, code = done.stdout.rpartition(b"\n")
    if done.returncode != 0 or not code.strip().isdigit():
        reason = done.stderr.decode("utf-8", "replace").strip().removeprefix("curl: ")
        raise LoraError(f"Could not reach Hugging Face: {reason or 'no answer'}")
    return int(code), body


def _remote_header(repo: str, file: str) -> dict:
    url = FILES.format(repo=repo, file=file)
    code, start = _curl(url, {"Range": "bytes=0-7"})
    if code not in (200, 206) or len(start) < 8:
        raise LoraError(f"Could not read {file} from Hugging Face (HTTP {code}).")
    size = struct.unpack("<Q", start[:8])[0]
    if size > MAX_HEADER:
        raise LoraError(f"{file} is not a safetensors file HikariLM can read.")
    code, body = _curl(url, {"Range": f"bytes=8-{7 + size}"})
    try:
        return json.loads(body[:size])
    except ValueError:
        raise LoraError(f"{file} is not a safetensors file HikariLM can read.") from None


def parse(text: str) -> tuple[str, str]:
    """The repo and the file someone typed: "user/name", "user/name:file.safetensors" or the
    repo's address on huggingface.co."""
    text = (text or "").strip()
    text = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", text)
    text = re.sub(r"/(?:tree|blob|resolve)/[^/]+/?", ":", text, count=1).rstrip("/:")
    repo, _, file = text.partition(":")
    if not REPO.match(repo) or ".." in repo:
        raise LoraError("Write the repo the way Hugging Face names it: user/name.")
    if file and (not file.endswith(".safetensors") or ".." in file.split("/")):
        raise LoraError("The file must be a .safetensors file in the repo.")
    return repo, file


def _bases(card: dict) -> list[str]:
    base = card.get("base_model")
    return [base] if isinstance(base, str) else [item for item in base or [] if isinstance(item, str)]


def _name(variant: str) -> str:
    return f"{label(variant)} {'9B' if '9b' in variant else '4B'}"


def made_for(bases: list[str]) -> tuple[str, str]:
    """What the card says the LoRA was made for: a refusal, or a note ("" for the base 4B)."""
    if not bases:
        return "", "Its page does not say which model it was made for, so only its file can tell."
    variants = [flux.variant_of(base) for base in bases]
    if any(found and found[0] == VARIANT for found in variants):
        return "", ""
    if any(found and found[0] == "flux2-klein-4b" for found in variants):
        return "", ("It was made for the quick FLUX.2 klein 4B. It draws on the base 4B, though it may look a "
                    "little different from its examples.")
    other = next((found[0] for found in variants if found), "")
    if other:
        return f"It was made for {_name(other)}, and LoRAs work on FLUX.2 klein base 4B.", ""
    if any(re.search(r"flux\.?-?1", base, re.I) for base in bases):
        return "It was made for FLUX.1, not FLUX.2 klein.", ""
    return f"It was made for {bases[0]}, not FLUX.2 klein.", ""


def check_header(header: dict) -> str:
    """Why the file cannot be drawn on the base 4B, or "" when it can."""
    meta = header.get("__metadata__") or {}
    keys = [key for key in header if key != "__metadata__"]
    if any("lora_magnitude_vector" in key for key in keys) or (
            any(".dora_scale" in key for key in keys) and any(re.search(r"lora_(?:A|down)", key) for key in keys)):
        return "It is a DoRA, which mflux does not load."
    if not any(re.search(r"lora_(?:A|B|down|up)|lokr_", key) for key in keys):
        return "It is not a LoRA."
    trained_on = str(meta.get("model") or "")
    if trained_on and not trained_on.startswith(("flux2-klein-4b", "flux2-klein-base-4b")):
        named = _name(trained_on) if trained_on.startswith("flux2-klein") else trained_on
        return f"It was trained on {named}, and LoRAs work on FLUX.2 klein base 4B."
    widths = {entry["shape"][1] for key, entry in header.items()
              if key != "__metadata__" and re.search(r"lora_(?:A|down)", key) and (DOUBLE.search(key) or SINGLE.search(key))
              and isinstance(entry, dict) and len(entry.get("shape") or []) == 2}
    if WIDTH_9B in widths and WIDTH_4B not in widths:
        return "It was made for FLUX.2 klein 9B, and LoRAs work on FLUX.2 klein base 4B."
    double = {int(found[1]) for key in keys for found in [DOUBLE.search(key)] if found}
    single = {int(found[1]) for key in keys for found in [SINGLE.search(key)] if found}
    if (double and max(double) >= flux.VARIANTS[VARIANT][0]) or (single and max(single) >= flux.VARIANTS[VARIANT][1]):
        return "It was made for a larger model than FLUX.2 klein 4B (FLUX.1, or the 9B)."
    if not double and not single:
        return "It has no layers HikariLM knows on FLUX.2 klein."
    return ""


def trigger_of(card: dict, meta: dict) -> str:
    """The trigger word its page or its trainer gives, or ""."""
    prompt = card.get("instance_prompt")
    if isinstance(prompt, list):
        prompt = next((item for item in prompt if isinstance(item, str)), "")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()[:MAX_TRIGGER]
    # kohya and ai-toolkit name the picture folder "10_trigger".
    try:
        folders = json.loads(meta.get("ss_tag_frequency") or "{}")
    except ValueError:
        folders = {}
    for folder in folders if isinstance(folders, dict) else []:
        found = re.match(r"^\d+_(.+)$", str(folder))
        if found and found[1].strip():
            return found[1].strip()[:MAX_TRIGGER]
    return ""


def _pick(files: list[dict], wanted: str) -> str:
    names = [item["name"] for item in files]
    if wanted:
        if wanted not in names:
            raise LoraError(f"There is no {wanted} in that repo.")
        return wanted
    plain = [name for name in names if "/" not in name and not STEP_FILE.search(name)] or names
    plain.sort(key=lambda name: ("comfy" in name.lower(), len(name), name))
    return plain[0]


def look_up(text: str) -> dict:
    """What a repo holds, for the user to add: the file, its size, what it was made for, the
    trigger word and the download command, or "error" saying why it cannot be drawn."""
    repo, wanted = parse(text)
    code, body = _curl(API + repo + "?blobs=true")
    if code in (401, 403, 404):
        raise LoraError(f"There is no public repo called {repo} on Hugging Face.")
    if code != 200:
        raise LoraError(f"Hugging Face answered HTTP {code} for {repo}.")
    try:
        info = json.loads(body)
    except ValueError:
        raise LoraError("Hugging Face sent an answer HikariLM could not read.") from None
    files = [{"name": item["rfilename"], "size": int(item.get("size") or 0)}
             for item in info.get("siblings") or [] if str(item.get("rfilename", "")).endswith(".safetensors")]
    if not files:
        raise LoraError(f"{repo} has no .safetensors file, so it is not a LoRA mflux can load.")
    file = _pick(files, wanted)
    card = info.get("cardData") or {}
    refusal, note = made_for(_bases(card))
    out = {"repo": repo, "file": file, "id": f"{repo}:{file}", "files": files,
           "size": next(item["size"] for item in files if item["name"] == file),
           "made_for": _bases(card)[:1], "note": note, "gated": bool(info.get("gated")),
           "command": f"{DOWNLOAD}{repo} {file}", "trigger": trigger_of(card, {})}
    if not refusal:
        header = _remote_header(repo, file)
        refusal = check_header(header)
        out["trigger"] = out["trigger"] or trigger_of({}, header.get("__metadata__") or {})
    if refusal:
        out["error"] = refusal
    return out


# --- the saved list ---------------------------------------------------------------

_lock = threading.Lock()


def _path() -> Path:
    return store.home() / "loras.json"


def saved() -> list[dict]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for item in data.get("loras") or [] if isinstance(data, dict) else []:
        try:
            repo, file = parse(f"{item['repo']}:{item['file']}")
        except (KeyError, TypeError, LoraError):
            continue
        out.append({"repo": repo, "file": file, "id": f"{repo}:{file}",
                    "trigger": clean_trigger(item.get("trigger")), "tags": item.get("tags") is True,
                    "note": str(item.get("note") or "")[:400],
                    "size": int(item.get("size") or 0) if str(item.get("size") or "0").isdigit() else 0})
    return out


def _write(items: list[dict]) -> None:
    folder = store.home()
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=folder, prefix=".loras.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"loras": [{key: item[key] for key in ("repo", "file", "trigger", "tags", "note", "size")}
                                 for item in items]}, handle, indent=2)
        os.replace(temp, _path())
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def clean_trigger(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_TRIGGER]


def add(found: dict) -> dict:
    """Keep a looked-up LoRA. One that cannot be drawn is never kept."""
    if found.get("error"):
        raise LoraError(found["error"])
    item = {"repo": found["repo"], "file": found["file"], "id": found["id"], "trigger": clean_trigger(found.get("trigger")),
            "tags": False, "note": found.get("note") or "", "size": int(found.get("size") or 0)}
    with _lock:
        items = [entry for entry in saved() if entry["id"] != item["id"]] + [item]
        _write(items)
    return item


def _change(lora_id: str, key: str, value) -> dict:
    with _lock:
        items = saved()
        for item in items:
            if item["id"] == lora_id:
                item[key] = value
                _write(items)
                return item
    raise LoraError("That LoRA is not in the list.")


def set_trigger(lora_id: str, trigger) -> dict:
    """Many LoRAs need no trigger word: an empty one puts nothing in front of the prompt."""
    return _change(lora_id, "trigger", clean_trigger(trigger))


def set_tags(lora_id: str, tags) -> dict:
    """Whether the LoRA was trained on booru tags, so Rewrite turns the prompt into tags for it."""
    return _change(lora_id, "tags", tags is True)


def remove(lora_id: str) -> bool:
    with _lock:
        items = saved()
        kept = [item for item in items if item["id"] != lora_id]
        if len(kept) != len(items):
            _write(kept)
        return len(kept) != len(items)


# --- downloaded ---------------------------------------------------------------------

def local_file(item: dict) -> Path | None:
    """The LoRA's file in the Hugging Face cache, once it is downloaded in full."""
    from helper import engine

    folder = engine.hub() / ("models--" + item["repo"].replace("/", "--"))
    if engine.downloading(folder):
        return None
    snapshot = engine.snapshot(folder)
    path = snapshot / item["file"] if snapshot else None
    return path if path and path.is_file() else None


def _local_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > MAX_HEADER:
            raise ValueError("header too large")
        return json.loads(handle.read(size).decode("utf-8"))


def listing() -> list[dict]:
    """Every kept LoRA, with whether it is downloaded and can be drawn."""
    out = []
    for item in saved():
        path = local_file(item)
        entry = {**item, "label": item["repo"].split("/")[-1], "ready": False, "error": "",
                 "command": f"{DOWNLOAD}{item['repo']} {item['file']}"}
        if path:
            try:
                entry["error"] = check_header(_local_header(path))
            except (OSError, ValueError, struct.error):
                entry["error"] = "Its file could not be read."
            entry["ready"] = not entry["error"]
        out.append(entry)
    return out


def find(lora_id: str) -> dict | None:
    """A kept, downloaded LoRA that can be drawn, with its file; None otherwise."""
    for item in listing():
        if item["id"] == lora_id and item["ready"]:
            return {**item, "path": str(local_file(item))}
    return None


def with_trigger(prompt: str, trigger: str) -> str:
    """The prompt with the trigger word in front, unless it is already in it."""
    if not trigger or trigger.lower() in prompt.lower():
        return prompt
    return f"{trigger}, {prompt}"
