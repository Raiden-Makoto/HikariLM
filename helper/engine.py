"""MLX models, run by helper/mlx_worker.py in a process of their own.

The models are the ones `hf download` put in the Hugging Face cache (and any MLX
model folder in ~/.hikarilm/models). The worker runs with the Python in
~/.hikarilm/venv, which has mlx-lm and mlx-vlm; the helper never imports MLX.
One model is loaded at a time. Unloading ends the worker, which gives all of
its memory back at once.

Answers are handed on as Ollama's chat lines, and a model's details in the shape
of Ollama's /api/show, so the rest of the helper treats both alike.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from helper import store
from helper.ollama import OllamaError

WORKER = Path(__file__).with_name("mlx_worker.py")
LOAD_SECONDS = 600       # a large model read from a slow disk
WAIT_SECONDS = 600       # a long prompt is read before the first piece comes
STOP_SECONDS = 20
DEFAULT_CONTEXT = 8192
SCAN_SECONDS = 5
# Settings a config leaves out because they are the library's defaults.
DEFAULTS = {"gemma3_text": {"head_dim": 256, "max_position_embeddings": 131072}}
# R1 distills write <think> themselves, so their templates do not show it.
THINKERS = re.compile(r"deepseek-r1|qwq", re.IGNORECASE)
SETUP = "MLX is not set up yet. Stop HikariLM and run ./start.sh again; it installs mlx-lm and mlx-vlm once."


class EngineError(OllamaError):
    """Worded for the user."""


PACKAGES = ("mlx-lm", "mlx-vlm")
READY_MARK = ".hikarilm-ready"


def venv() -> Path:
    return store.home() / "venv"


def python() -> Path:
    return Path(os.environ.get("HIKARILM_MLX_PYTHON") or venv() / "bin" / "python")


def prepare(say, base_python: str = sys.executable) -> bool:
    """Make ~/.hikarilm/venv and install the MLX packages into it, once. True when it is ready.
    Nothing is checked when HIKARILM_MLX_PYTHON names another Python."""
    if os.environ.get("HIKARILM_MLX_PYTHON"):
        return True
    mark = venv() / READY_MARK
    if mark.is_file():
        return True
    say(f"Setting up MLX in {venv()} (once, about 1 GB to download)…")
    try:
        if not python().exists():
            subprocess.run([base_python, "-m", "venv", str(venv())], check=True)
        subprocess.run([str(python()), "-m", "pip", "install", "--upgrade", *PACKAGES], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        say(f"  MLX could not be set up ({error}). Ollama's models still work; run ./start.sh again to retry.")
        return False
    mark.write_text(" ".join(PACKAGES) + "\n", encoding="utf-8")
    say("  MLX is ready.")
    return True


def worker() -> Path:
    return Path(os.environ.get("HIKARILM_MLX_WORKER") or WORKER)


def _hub() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _config(folder: Path) -> dict | None:
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _is_mlx(folder: Path) -> bool:
    config = _config(folder)
    return config is not None and "quantization" in config and any(folder.glob("*.safetensors"))


def _snapshot(repo: Path) -> Path | None:
    try:
        return repo / "snapshots" / (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        found = sorted((repo / "snapshots").glob("*"))
        return found[-1] if found else None


_scan: tuple[float, dict[str, Path]] = (0.0, {})
_scan_lock = threading.Lock()


def found() -> dict[str, Path]:
    """Name -> folder for every MLX model downloaded in full. A download still running is left out."""
    global _scan
    with _scan_lock:
        if time.monotonic() - _scan[0] < SCAN_SECONDS:
            return _scan[1]
        out: dict[str, Path] = {}
        hub = _hub()
        for repo in sorted(hub.glob("models--*")) if hub.is_dir() else []:
            if any(repo.glob("blobs/*.incomplete")):
                continue
            snapshot = _snapshot(repo)
            if snapshot and _is_mlx(snapshot):
                out[repo.name[len("models--"):].replace("--", "/", 1)] = snapshot
        own = store.home() / "models"
        for folder in sorted(own.iterdir()) if own.is_dir() else []:
            if folder.is_dir() and _is_mlx(folder):
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
        raise EngineError(f"{model} is not downloaded. Get it with:  ~/.hikarilm/venv/bin/hf download {model}")
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
    """The model's shape as Ollama's /api/show gives it: what budget.py and thinking.py read."""
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

    def __init__(self, model: str, folder: Path, vision: bool) -> None:
        if not python().exists():
            raise EngineError(SETUP)
        self.model = model
        self.size = _size(folder)
        store.home().mkdir(parents=True, exist_ok=True)
        log = open(store.home() / "mlx-worker.log", "wb")
        command = [str(python()), str(worker()), "--model", str(folder)] + (["--vision"] if vision else [])
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                            text=True, encoding="utf-8", bufsize=1)
        except OSError as error:
            raise EngineError(f"The MLX worker could not start: {error}") from error
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
            raise EngineError(f"The MLX worker stopped: {error}") from error

    def next(self, seconds: float) -> dict | None:
        try:
            return self.lines.get(timeout=seconds)
        except queue.Empty:
            return {"error": "The MLX worker stopped answering."}

    def end(self) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


_worker: _Worker | None = None
_lock = threading.Lock()      # held from opening a reply until it is closed


def _died(model: str) -> str:
    return (f"The MLX worker stopped while running {model}. What it said is in "
            f"{store.home() / 'mlx-worker.log'}.")


def _ready(model: str) -> _Worker:
    """The worker with this model loaded, starting it (and ending any other) if needed."""
    global _worker
    if _worker is not None and _worker.model == model and _worker.alive():
        return _worker
    _end()
    folder = _folder(model)
    started = _Worker(model, folder, "vision_config" in (_config(folder) or {}))
    event = started.next(LOAD_SECONDS)
    if not event or not event.get("ready"):
        started.end()
        raise EngineError((event or {}).get("error") or _died(model))
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
    """A reply being written, read like Ollama's streamed /api/chat: readline() gives one JSON
    line at a time and b"" at the end. close() stops it if it is not finished."""

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
            except EngineError:
                pass
            if not self.process.alive() or time.monotonic() > deadline:
                _forget(self.process)
        _lock.release()


def _forget(process: _Worker) -> None:
    global _worker
    process.end()
    if _worker is process:
        _worker = None


def open_chat(model: str, messages: list[dict], keep_alive=None, options: dict | None = None, think=None) -> Reply:
    """Start a reply. The caller closes it; closing it early stops the model."""
    if not _lock.acquire(timeout=WAIT_SECONDS):
        raise EngineError("Another reply is still being written.")
    try:
        process = _ready(model)
        options = options or {}
        process.send({"op": "chat", "messages": [_plain(m) for m in messages], "think": think,
                      "num_ctx": options.get("num_ctx"), "num_predict": options.get("num_predict")})
    except BaseException:
        _lock.release()
        raise
    return Reply(model, process)


def _plain(message: dict) -> dict:
    out = {"role": message.get("role", "user"), "content": str(message.get("content") or "")}
    if message.get("images"):
        out["images"] = list(message["images"])
    return out


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
                raise EngineError(event["error"])
            text += (event.get("message") or {}).get("content", "")
    finally:
        reply.close()
