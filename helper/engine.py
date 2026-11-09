"""MLX models, run by helper/mlx_worker.py in a process of their own.

The models are the ones `hf download` put in the Hugging Face cache (and any MLX
model folder in ~/.hikarilm/models). The worker runs with the Python in
~/.hikarilm/venv, which has mlx-lm and mlx-vlm; the helper never imports MLX.
One model is loaded at a time. Unloading ends the worker, which gives all of
its memory back at once.

Answers are handed on as streamed chat lines ({"message": ..., "done": ...}), and
a model's details as a small dict of its shape (`show`), which budget.py and
thinking.py read.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from helper import store
from helper.names import label

WORKER = Path(__file__).with_name("mlx_worker.py")
LOAD_SECONDS = 600       # a large model read from a slow disk
WAIT_SECONDS = 600       # a long prompt is read before the first piece comes
STOP_SECONDS = 20
DEFAULT_CONTEXT = 8192
SCAN_SECONDS = 5
DOWNLOAD_QUIET = 600
# Settings a config leaves out because they are the library's defaults.
DEFAULTS = {"gemma3_text": {"head_dim": 256, "max_position_embeddings": 131072, "sliding_window_pattern": 6}}
# R1 distills write <think> themselves, so their templates do not show it.
THINKERS = re.compile(r"deepseek-r1|qwq", re.IGNORECASE)
SETUP = "MLX is not set up yet. Run  hikarilm restart  in Terminal; it installs mlx-lm and mlx-vlm once."
PICTURE_SETUP = ("mflux, which draws the pictures, is not set up yet. Run  hikarilm restart  in Terminal; "
                 "it installs mflux once a FLUX.2 model is downloaded.")


class ModelError(Exception):
    """Worded for the user."""


# scipy is not ours but comes with mlx-vlm (through mlx-audio). macOS 27 cannot load scipy 1.15
# and older, and pip install --upgrade leaves an old one that still meets mlx-audio's
# scipy>=1.10, so the venv asks for one it can load. That needs Python 3.11 or newer.
LOADABLE_SCIPY = "scipy>=1.16.3"
PACKAGES = ("mlx-lm", "mlx-vlm", LOADABLE_SCIPY)
OLDEST = 11
SCIPY_LOADS = "import scipy.sparse.linalg, scipy.integrate, scipy.special"
READY_MARK = ".hikarilm-ready"


def venv() -> Path:
    return store.home() / "venv"


def python() -> Path:
    return Path(os.environ.get("HIKARILM_MLX_PYTHON") or venv() / "bin" / "python")


def _prepare_venv(folder: Path, interpreter: Path, packages: tuple[str, ...], what: str, say,
                  base_python: str) -> bool:
    mark = folder / READY_MARK
    try:
        if mark.read_text(encoding="utf-8").split() == list(packages):
            return True
    except OSError:
        pass
    say(f"Setting up {what} in {folder} (once)…")
    try:
        if not interpreter.exists():
            subprocess.run([base_python, "-m", "venv", str(folder)], check=True)
        subprocess.run([str(interpreter), "-m", "pip", "install", "--upgrade", *packages], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        say(f"  {what} could not be set up ({error}). Run  hikarilm restart  to retry.")
        return False
    mark.write_text(" ".join(packages) + "\n", encoding="utf-8")
    say(f"  {what} is ready.")
    return True


def find_python(oldest: int, newest: int) -> str | None:
    """A Python 3.oldest to 3.newest to make a venv with: the one running HikariLM, or another
    one from Homebrew, python.org or the PATH, newest first."""
    if (3, oldest) <= tuple(sys.version_info[:2]) <= (3, newest):
        return sys.executable
    for minor in range(newest, oldest - 1, -1):
        name = f"python3.{minor}"
        for found in (shutil.which(name), f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}",
                      f"/Library/Frameworks/Python.framework/Versions/3.{minor}/bin/{name}"):
            if found and os.access(found, os.X_OK):
                return found
    return None


def venv_minor(folder: Path) -> int | None:
    """The 3.x a venv was made with, from its pyvenv.cfg."""
    try:
        for line in (folder / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in ("version", "version_info"):
                parts = value.strip().split(".")
                return int(parts[1]) if parts[0] == "3" and len(parts) > 1 else None
    except (OSError, ValueError):
        pass
    return None


def _new_enough(folder: Path, what: str, say, base_python: str | None) -> str | None:
    """A Python 3.11 or newer to make the venv with, after deleting one made with an older
    Python; None, said, when there is none."""
    base_python = base_python if base_python and _minor_of(base_python) >= OLDEST else find_python(OLDEST, 14)
    if not base_python:
        say(f"  {what} needs Python 3.{OLDEST} or newer (this is {sys.version.split()[0]}), because older ones get "
            "a scipy macOS cannot load. Install one with  brew install python@3.12  or from python.org, then run  "
            "hikarilm restart.")
        return None
    made = venv_minor(folder)
    if made is not None and made < OLDEST:
        say(f"{what} was set up with Python 3.{made}; setting it up again with a newer one…")
        shutil.rmtree(folder, ignore_errors=True)
    return base_python


def _minor_of(executable: str) -> int:
    if executable == sys.executable:
        return sys.version_info[1]
    found = re.search(r"python3\.(\d+)", os.path.basename(executable))
    return int(found.group(1)) if found else 0


def _scipy_loads(folder: Path, interpreter: Path, what: str, say) -> bool:
    """macOS 27 refuses an old scipy only when it is imported, so import it once here."""
    try:
        done = subprocess.run([str(interpreter), "-c", SCIPY_LOADS], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        done = subprocess.CompletedProcess([], 1, "", str(error))
    if done.returncode:
        lines = (done.stderr or done.stdout).strip().splitlines()
        say(f"  {what} is installed but its scipy does not load: {lines[-1] if lines else done.returncode}")
        (folder / READY_MARK).unlink(missing_ok=True)
        return False
    return True


def prepare(say, base_python: str = sys.executable) -> bool:
    """Make ~/.hikarilm/venv and install the MLX packages into it, once. True when it is ready.
    Nothing is checked when HIKARILM_MLX_PYTHON names another Python."""
    if os.environ.get("HIKARILM_MLX_PYTHON"):
        return True
    if not (base_python := _new_enough(venv(), "MLX", say, base_python)):
        return False
    fresh = not _ready_mark(venv(), PACKAGES)
    return (_prepare_venv(venv(), python(), PACKAGES, "MLX", say, base_python)
            and (not fresh or _scipy_loads(venv(), python(), "MLX", say)))


def _ready_mark(folder: Path, packages: tuple[str, ...]) -> bool:
    try:
        return (folder / READY_MARK).read_text(encoding="utf-8").split() == list(packages)
    except OSError:
        return False


# mflux pins its own MLX and wants torch and transformers 5, so it has a venv of its
# own and never changes the one the chat models run in.
PICTURE_PACKAGES = ("mflux", LOADABLE_SCIPY)
PICTURE_WORKER = Path(__file__).with_name("flux_worker.py")


def picture_venv() -> Path:
    return store.home() / "pictures-venv"


def picture_python() -> Path:
    return Path(os.environ.get("HIKARILM_PICTURE_PYTHON") or picture_venv() / "bin" / "python")


def picture_worker() -> Path:
    return Path(os.environ.get("HIKARILM_PICTURE_WORKER") or PICTURE_WORKER)


def prepare_pictures(say, base_python: str = sys.executable) -> bool:
    """Make ~/.hikarilm/pictures-venv with mflux, once. True when it is ready."""
    if os.environ.get("HIKARILM_PICTURE_PYTHON"):
        return True
    what = "mflux (for pictures)"
    if not (base_python := _new_enough(picture_venv(), what, say, base_python)):
        return False
    fresh = not _ready_mark(picture_venv(), PICTURE_PACKAGES)
    return (_prepare_venv(picture_venv(), picture_python(), PICTURE_PACKAGES, what, say, base_python)
            and (not fresh or _scipy_loads(picture_venv(), picture_python(), what, say)))


def pictures_ready() -> bool:
    if os.environ.get("HIKARILM_PICTURE_PYTHON"):
        return picture_python().exists()
    try:
        return (picture_venv() / READY_MARK).read_text(encoding="utf-8").split() == list(PICTURE_PACKAGES)
    except OSError:
        return False


def worker() -> Path:
    return Path(os.environ.get("HIKARILM_MLX_WORKER") or WORKER)


def hub() -> Path:
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            return Path(os.environ[variable]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _config(folder: Path) -> dict | None:
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _is_mlx(folder: Path, trusted: bool = False) -> bool:
    """Quantized weights are MLX's own. Full-precision ones look like any Hugging Face model,
    so those count only from mlx-community, a repo named for MLX, or ~/.hikarilm/models.
    BitNet weights (Falcon3 1.58-bit) are packed the way mlx-lm reads them, so they count too."""
    config = _config(folder)
    return (config is not None and ("quantization" in config or trusted or is_bitnet(config))
            and any(folder.glob("*.safetensors")))


# Models with a config.json and weights that do not chat. Whisper (speech to text) says
# "model_type": "whisper"; Kokoro (text to speech) has no model_type at all, which every
# language model's config has. Encoder-only and audio models are listed for the same reason.
NOT_CHAT = {"whisper", "wav2vec2", "hubert", "parakeet", "kokoro", "bert", "nomic_bert", "roberta",
            "xlm-roberta", "modernbert", "clip", "siglip", "t5", "encodec", "mimi", "dac"}


def is_chat(config: dict | None) -> bool:
    """Whether a model's config is a chat (language) model's, not speech, audio or an encoder."""
    model_type = str((config or {}).get("model_type") or "").lower()
    return bool(model_type) and model_type not in NOT_CHAT


def is_bitnet(config: dict | None) -> bool:
    quantization = (config or {}).get("quantization_config")
    return isinstance(quantization, dict) and quantization.get("quant_method") == "bitnet"


def bitnet_model(model: str) -> bool:
    """Whether a downloaded model is a BitNet one (Falcon3 1.58-bit)."""
    folder = found().get(model)
    return folder is not None and is_bitnet(_config(folder))


def snapshot(repo: Path) -> Path | None:
    try:
        return repo / "snapshots" / (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        found = sorted((repo / "snapshots").glob("*"))
        return found[-1] if found else None


def downloading(repo: Path) -> bool:
    """Whether a download into this cache folder is running. Hugging Face writes each file to
    blobs/*.incomplete as it comes; one left by a download that was stopped is never written
    again, and stops counting after DOWNLOAD_QUIET."""
    now = time.time()
    for path in repo.glob("blobs/*.incomplete"):
        try:
            if now - path.stat().st_mtime < DOWNLOAD_QUIET:
                return True
        except OSError:
            pass
    return False


def missing_files(folder: Path) -> list[str]:
    """The weight files a model's indexes name that are not in its folder (a download that
    stopped part of the way), and links to files that are gone."""
    missing = []
    for index in folder.rglob("*.safetensors.index.json"):
        try:
            shards = set((json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}).values())
        except (OSError, ValueError, AttributeError):
            continue
        absent = [shard for shard in sorted(shards) if not (index.parent / shard).is_file()]
        # Some repos keep an index from before the weights were saved again in other files
        # (mlx-community/gemma-3-12b-it-4bit: it names five, the repo has two). MLX loads the
        # .safetensors files that are there, so an index none of whose files exist is ignored.
        if len(absent) < len(shards):
            missing += [str((index.parent / shard).relative_to(folder)) for shard in absent]
    weights = list(folder.rglob("*.safetensors"))
    missing += [str(path.relative_to(folder)) for path in weights if not path.exists()]
    # Shards named model-00001-of-00003.safetensors must all be there.
    for path in weights:
        match = SHARD.fullmatch(path.name)
        if match:
            total, width = int(match.group(3)), len(match.group(2))
            missing += [str(path.parent.relative_to(folder) / f"{match.group(1)}-{n:0{width}d}-of-{match.group(3)}.safetensors")
                        for n in range(1, total + 1)
                        if not (path.parent / f"{match.group(1)}-{n:0{width}d}-of-{match.group(3)}.safetensors").is_file()]
    return sorted(set(missing))


SHARD = re.compile(r"(.+)-(\d+)-of-(\d+)\.safetensors")


def _repo_name(repo: Path) -> str:
    return repo.name[len("models--"):].replace("--", "/", 1)


def _trusted(name: str) -> bool:
    return name.startswith("mlx-community/") or "mlx" in re.split(r"[-_.]", name.split("/")[-1].lower())


def _snapshots(repo: Path) -> list[Path]:
    """The cache's copies of a repo: the one refs/main names first, then the newest. A later
    fetch of one file (a README, a template) moves refs/main to a new copy that has only that
    file, while the whole model is still in the copy before."""
    main = snapshot(repo)
    try:
        others = sorted((p for p in (repo / "snapshots").iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        others = []
    return ([main] if main and main.is_dir() else []) + [p for p in others if p != main]


def _complete_snapshot(repo: Path, trusted: bool) -> Path | None:
    for folder in _snapshots(repo):
        if _is_mlx(folder, trusted) and is_chat(_config(folder)) and not missing_files(folder):
            return folder
    return None


def why_missing(model: str) -> str:
    """Why a model in the Hugging Face cache is not in found(), for check scripts."""
    repo = hub() / ("models--" + model.replace("/", "--"))
    if not repo.is_dir():
        return f"there is no {repo}"
    if downloading(repo):
        return "a download into it is still running (blobs/*.incomplete written in the last few minutes)"
    copies = _snapshots(repo)
    if not copies:
        return f"{repo / 'snapshots'} is empty"
    reasons = []
    for folder in copies:
        config = _config(folder)
        if config is None:
            reasons.append(f"{folder.name[:8]}: no readable config.json")
        elif not is_chat(config):
            kind = config.get("model_type") or "no model_type"
            reasons.append(f"{folder.name[:8]}: not a chat model ({kind}; speech, audio or an encoder)")
        elif not any(folder.glob("*.safetensors")):
            reasons.append(f"{folder.name[:8]}: no .safetensors files")
        elif not ("quantization" in config or _trusted(model) or is_bitnet(config)):
            reasons.append(f"{folder.name[:8]}: full-precision weights from a repo not named for MLX")
        elif missing_files(folder):
            reasons.append(f"{folder.name[:8]}: missing {', '.join(missing_files(folder)[:3])}")
        else:
            return "it is complete; run again (the model list is cached for a few seconds)"
    return "; ".join(reasons)


_scan: tuple[float, dict[str, Path]] = (0.0, {})
_scan_lock = threading.Lock()


def found() -> dict[str, Path]:
    """Name -> folder for every MLX model downloaded in full. A download still running is left out."""
    global _scan
    with _scan_lock:
        if time.monotonic() - _scan[0] < SCAN_SECONDS:
            return _scan[1]
        out: dict[str, Path] = {}
        cache = hub()
        for repo in sorted(cache.glob("models--*")) if cache.is_dir() else []:
            if downloading(repo):
                continue
            name = _repo_name(repo)
            folder = _complete_snapshot(repo, _trusted(name))
            if folder:
                out[name] = folder
        own = store.home() / "models"
        for folder in sorted(own.iterdir()) if own.is_dir() else []:
            if folder.is_dir() and _is_mlx(folder, trusted=True) and is_chat(_config(folder)):
                out[folder.name] = folder
        _scan = (time.monotonic(), out)
        return out


def forget_scan() -> None:
    global _scan
    with _scan_lock:
        _scan = (0.0, {})


def owns(model: str) -> bool:
    return model in found()


def _folder(model: str) -> Path:
    folder = found().get(model)
    if folder is None:
        raise ModelError(f"{label(model)} is not downloaded. Get it with:  ~/.hikarilm/venv/bin/hf download {model}")
    return folder


def names() -> list[str]:
    return list(found())


def _size(folder: Path) -> int:
    total = 0
    for path in folder.glob("*.safetensors"):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def sizes() -> dict[str, int]:
    """Each model's size on disk, for the model list."""
    return {name: _size(folder) for name, folder in found().items()}


def _template(folder: Path) -> str:
    text = ""
    try:
        text += str(json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template") or "")
    except (OSError, ValueError, AttributeError):
        pass
    for name in ("chat_template.jinja", "chat_template.json"):
        try:
            text += (folder / name).read_text(encoding="utf-8")
        except OSError:
            pass
    return text


def show(model: str) -> dict:
    """The model's shape from its config.json: what budget.py and thinking.py read."""
    folder = _folder(model)
    config = _config(folder) or {}
    text = {**config, **(config.get("text_config") or {})}
    arch = str(text.get("model_type") or "mlx")
    text = {**DEFAULTS.get(arch, {}), **{key: value for key, value in text.items() if value is not None}}
    trained = int(text.get("max_position_embeddings") or 0)
    info = {"general.architecture": arch}
    for key, field in (("block_count", "num_hidden_layers"), ("attention.head_count", "num_attention_heads"),
                       ("attention.head_count_kv", "num_key_value_heads"), ("embedding_length", "hidden_size"),
                       ("attention.key_length", "head_dim"), ("context_length", "max_position_embeddings")):
        if isinstance(text.get(field), int):
            info[f"{arch}.{key}"] = text[field]
    capabilities = ["completion"]
    if "vision_config" in config:
        capabilities.append("vision")
    template = _template(folder)
    if THINKERS.search(model) or "<think>" in template or "enable_thinking" in template:
        capabilities.append("thinking")
    default = min(trained, DEFAULT_CONTEXT) if trained else DEFAULT_CONTEXT
    return {"model_info": info, "capabilities": capabilities, "parameters": f"num_ctx {default}", "mlx": True}


class _Worker:
    """The worker process and the lines it writes, read by a thread into a queue."""

    def __init__(self, model: str, folder: Path, flags: list[str], log_name: str = "mlx-worker.log",
                 interpreter: Path | None = None, script: Path | None = None, size: int | None = None) -> None:
        interpreter = interpreter or python()
        if not interpreter.exists():
            raise ModelError(SETUP if interpreter == python() else PICTURE_SETUP)
        self.model, self.flags = model, flags
        self.size = _size(folder) if size is None else size
        self.log_name = log_name
        store.home().mkdir(parents=True, exist_ok=True)
        log = open(store.home() / log_name, "wb")
        command = [str(interpreter), str(script or worker()), "--model", str(folder)] + flags
        try:
            # A group of its own, so ending it also ends what it started (mflux's command).
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                            text=True, encoding="utf-8", bufsize=1, start_new_session=True)
        except OSError as error:
            raise ModelError(f"The MLX worker could not start: {error}") from error
        finally:
            log.close()
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            try:
                self.lines.put(json.loads(line))
            except ValueError:
                continue
        self.process.stdout.close()
        self.lines.put(None)

    def alive(self) -> bool:
        return self.process.poll() is None

    def send(self, command: dict) -> None:
        try:
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise ModelError(f"The MLX worker stopped: {error}") from error

    def next(self, seconds: float) -> dict | None:
        try:
            return self.lines.get(timeout=seconds)
        except queue.Empty:
            return {"error": "The MLX worker stopped answering."}

    def end(self) -> None:
        """Kill it: it has nothing to save, and macOS takes all of its memory back at once,
        sooner than a Python shutting down would hand it over."""
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            if self.process.poll() is None:
                self.process.kill()
        self.process.wait()
        try:
            self.process.stdin.close()
        except OSError:
            pass


_worker: _Worker | None = None
_lock = threading.Lock()      # held from opening a reply until it is closed


def _died(model: str, log_name: str = "mlx-worker.log") -> str:
    return (f"The MLX worker stopped while running {label(model)}. What it said is in "
            f"{store.home() / log_name}.")


def _ready(model: str, picture: dict | None = None, lora: dict | None = None) -> _Worker:
    """The worker with this model loaded, starting it (and ending any other) if needed.
    A picture model, found by flux.py, runs in the picture venv. A LoRA is baked into the
    model as it loads, so another LoRA, or another strength, loads it again."""
    global _worker
    if picture is not None:
        flags = ["--variant", picture["variant"]]
        if lora:
            flags += ["--lora", lora["path"], "--lora-strength", str(lora["strength"])]
    else:
        folder = _folder(model)
        flags = ["--vision"] if "vision_config" in (_config(folder) or {}) else []
    if _worker is not None and _worker.model == model and _worker.flags == flags and _worker.alive():
        return _worker
    _end()
    if picture is not None:
        started = _Worker(model, picture["folder"], flags, "picture-worker.log",
                          picture_python(), picture_worker(), picture["size"])
    else:
        started = _Worker(model, folder, flags)
    event = started.next(LOAD_SECONDS)
    if not event or not event.get("ready"):
        started.end()
        raise ModelError((event or {}).get("error") or _died(model, started.log_name))
    _worker = started
    return started


def _end() -> None:
    global _worker
    if _worker is not None:
        _worker.end()
        _worker = None


def running() -> list[dict]:
    current = _worker
    if current is None or not current.alive():
        return []
    return [{"name": current.model, "size": current.size, "expires": None}]


def unload(model: str | None = None) -> None:
    """End the worker if it has this model (or any, with None). Waits a little for a reply
    being written; one that goes on longer keeps its model."""
    if not _lock.acquire(timeout=STOP_SECONDS):
        return
    try:
        if _worker is not None and (model is None or _worker.model == model):
            _end()
    finally:
        _lock.release()


class Reply:
    """A reply being written: readline() gives one JSON line at a time and b"" at the end.
    close() stops it if it is not finished."""

    def __init__(self, model: str, process: _Worker) -> None:
        self.model, self.process, self.finished, self.closed = model, process, False, False

    def _line(self, event: dict) -> bytes:
        return (json.dumps({"model": self.model, **event}) + "\n").encode()

    def readline(self) -> bytes:
        if self.finished:
            return b""
        event = self.process.next(WAIT_SECONDS)
        if event is None or "error" in event:
            self.finished = True
            if event is None or not self.process.alive():
                _forget(self.process)
            return self._line({"error": (event or {}).get("error") or _died(self.model)})
        if event.get("done"):
            self.finished = True
            return self._line({
                "message": {"role": "assistant", "content": ""}, "done": True,
                "done_reason": "stop" if event.get("stopped") else event.get("reason") or "stop",
                "prompt_eval_count": event.get("prompt_tokens", 0), "eval_count": event.get("tokens", 0),
                "prompt_tps": event.get("prompt_tps", 0), "eval_tps": event.get("tps", 0)})
        return self._line({"message": {"role": "assistant", "content": event.get("text", "")}, "done": False})

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.finished:
            self.finished = True
            deadline = time.monotonic() + STOP_SECONDS
            try:
                self.process.send({"op": "stop"})
                while True:
                    event = self.process.next(max(0.1, deadline - time.monotonic()))
                    if event is None or "error" in event or event.get("done"):
                        break
            except ModelError:
                pass
            if not self.process.alive() or time.monotonic() > deadline:
                _forget(self.process)
        _lock.release()


def _forget(process: _Worker) -> None:
    global _worker
    process.end()
    if _worker is process:
        _worker = None


def _kv_bits() -> int:
    from helper import context  # only needed once a chat starts

    return context.load_settings()["kv_cache"]["bits"]


def open_chat(model: str, messages: list[dict], keep_alive=None, options: dict | None = None, think=None,
              tools: dict | None = None) -> Reply:
    """Start a reply. The caller closes it; closing it early stops the model. `tools` is
    {"tools": [...], "tools_text": "..."} for a model that may call them."""
    if not _lock.acquire(timeout=WAIT_SECONDS):
        raise ModelError("Another reply is still being written.")
    try:
        process = _ready(model)
        options = options or {}
        process.send({"op": "chat", "messages": [_plain(m) for m in messages], "think": think,
                      "num_ctx": options.get("num_ctx"), "num_predict": options.get("num_predict"),
                      "kv_bits": _kv_bits(), **(tools or {})})
    except BaseException:
        _lock.release()
        raise
    return Reply(model, process)


PICTURE_POLL = 0.5


def make_picture(picture: dict, request: dict, on_progress, stopped) -> bytes | None:
    """Draw one picture with a FLUX.2 model found by flux.py and return the PNG. None if
    stopped() said so first; the worker is ended then, which also frees its memory."""
    if not _lock.acquire(timeout=WAIT_SECONDS):
        raise ModelError("Another reply is still being written.")
    out = None
    references: list[str] = []
    try:
        process = _ready(picture["id"], picture, request.get("lora"))
        on_progress(0, 0)
        fd, out = tempfile.mkstemp(dir=store.home(), prefix=".drawing.", suffix=".png")
        os.close(fd)
        for data in request.get("pictures") or []:
            fd, path = tempfile.mkstemp(dir=store.home(), prefix=".reference.", suffix=".img")
            references.append(path)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        process.send({"op": "picture", "prompt": request["prompt"], "width": request["width"],
                      "height": request["height"], "seed": request["seed"],
                      "steps": request["steps"] or picture["steps"], "guidance": picture["guidance"], "out": out,
                      "references": references})
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            if stopped():
                _forget(process)
                return None
            try:
                event = process.lines.get(timeout=PICTURE_POLL)
            except queue.Empty:
                if time.monotonic() > deadline:
                    _forget(process)
                    raise ModelError(f"{picture['label']} stopped answering.") from None
                continue
            if event is None or "error" in event:
                if event is None or not process.alive():
                    _forget(process)
                raise ModelError((event or {}).get("error") or _died(picture["id"], process.log_name))
            deadline = time.monotonic() + WAIT_SECONDS
            if event.get("done"):
                return Path(out).read_bytes()
            if event.get("loading"):
                on_progress(0, 0)
            if "progress" in event:
                on_progress(int(event["progress"]), int(event.get("total") or 0))
    finally:
        for path in [out, *references]:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        _lock.release()


def _plain(message: dict) -> dict:
    out = {"role": message.get("role", "user"), "content": str(message.get("content") or "")}
    if message.get("images"):
        out["images"] = list(message["images"])
    for key in ("tool_calls", "name", "tool_call_id"):
        if message.get(key):
            out[key] = message[key]
    return out


# --- nomic-embed-text ----------------------------------------------------------------
# A worker of its own, so it sits beside the chat model (0.3 GB) and never makes it leave.

EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5"
EMBED_NAME = "nomic-embed-text"
EMBED_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt")
EMBED_SECONDS = 120
_embedder: _Worker | None = None
_embed_lock = threading.Lock()


def embed_folder() -> Path | None:
    """Where Nomic's files are, once all of them are downloaded."""
    folder = snapshot(hub() / ("models--" + EMBED_REPO.replace("/", "--")))
    if folder and all((folder / name).is_file() for name in EMBED_FILES):
        return folder
    return None


def prepare_embed(say) -> bool:
    """Download Nomic's files into the Hugging Face cache, once. True when they are there."""
    if embed_folder() is not None or os.environ.get("HIKARILM_MLX_PYTHON"):
        return embed_folder() is not None
    if not python().exists():
        return False
    say(f"Getting {EMBED_NAME} for finding earlier chats (about 0.55 GB, once)…")
    script = ("from huggingface_hub import snapshot_download; "
              f"snapshot_download({EMBED_REPO!r}, allow_patterns={list(EMBED_FILES)!r})")
    try:
        subprocess.run([str(python()), "-c", script], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        say(f"  {EMBED_NAME} could not be downloaded ({error}). Earlier chats are not searched until it is; "
            "run  hikarilm restart  to retry.")
        return False
    say(f"  {EMBED_NAME} is ready.")
    return embed_folder() is not None


def embed(texts: list[str]) -> list[list[float]]:
    """Unit vectors for texts that already carry nomic's search_document: or search_query: prefix."""
    global _embedder
    with _embed_lock:
        if _embedder is None or not _embedder.alive():
            folder = embed_folder()
            if folder is None:
                raise ModelError(f"{EMBED_NAME} is not downloaded yet, so earlier chats cannot be searched. "
                                 "Run  hikarilm restart  in Terminal; it downloads it once.")
            started = _Worker(EMBED_NAME, folder, ["--embed"], "embed-worker.log")
            event = started.next(LOAD_SECONDS)
            if not event or not event.get("ready"):
                started.end()
                raise ModelError((event or {}).get("error") or
                                 f"{EMBED_NAME} could not start. What it said is in {store.home() / 'embed-worker.log'}.")
            _embedder = started
        _embedder.send({"op": "embed", "texts": list(texts)})
        event = _embedder.next(EMBED_SECONDS)
        if not event or "error" in event or not isinstance(event.get("vectors"), list) \
                or len(event["vectors"]) != len(texts):
            # A late answer would be taken for the next batch's, so this worker is done.
            _embedder.end()
            _embedder = None
            raise ModelError((event or {}).get("error") or f"{EMBED_NAME} stopped.")
        return event["vectors"]


def embed_running() -> list[dict]:
    current = _embedder
    if current is None or not current.alive():
        return []
    return [{"name": EMBED_NAME, "size": current.size, "expires": None}]


def unload_embed() -> None:
    global _embedder
    with _embed_lock:
        if _embedder is not None:
            _embedder.end()
            _embedder = None


def complete(model: str, messages: list[dict], options: dict, keep_alive=None, think=None) -> str:
    """One whole answer. Used for summaries and reading long files."""
    reply = open_chat(model, messages, keep_alive, options, think)
    text = ""
    try:
        while True:
            line = reply.readline()
            if not line:
                return text
            event = json.loads(line)
            if event.get("error"):
                raise ModelError(event["error"])
            text += (event.get("message") or {}).get("content", "")
    finally:
        reply.close()
