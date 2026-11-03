"""Run a script a chat model wrote, in macOS's sandbox. Standard library only; the script runs
with the Python in ~/.hikarilm/python-venv, which has numpy, sympy, pandas, scipy and matplotlib.

The sandbox (sandbox-exec, the one macOS's own services use) lets the script reach no network,
start no other program, write only in its scratch folder (deleted after), and read nothing in
the home folder but the Python it runs. Outside it, the helper stops the script after SECONDS
or once it holds more than MEMORY, and keeps the first OUTPUT characters of what it printed.
Plots it leaves open in matplotlib come back as PNG pictures. Off a Mac it does not run at
all, unless the tests ask for it with HIKARILM_TEST_UNSANDBOXED (never honored on a Mac).
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from helper import engine, store

SECONDS = 30
MEMORY = 1250 * 1024 * 1024     # bytes (1.25 GB)
OUTPUT = 25_000                 # characters of output kept, for the model and the page
FLOOD = 4_000_000               # bytes printed after which the script is stopped
FILE_BYTES = 50 * 1024 * 1024   # largest file it may write in its scratch folder
CODE_CHARS = 20_000
PLOTS = 4
PLOT_BYTES = 2_000_000
PACKAGES = ("numpy", "sympy", "pandas", "scipy", "matplotlib")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
PATHS = ".hikarilm-paths.json"  # where the venv's Python lives, written when it is set up
SETUP = "Python for the models is not set up yet. Run  hikarilm restart  in Terminal; it installs it once."

# Read nowhere here but what the profile lets in again (the scratch folder and the Python).
PRIVATE = ("/Users", "/Volumes", "/private/var/folders", "/private/tmp", "/private/var/root")
# Mach services a script needs: logging and user lookups. The rest (opening apps and web
# pages through Launch Services, the pasteboard, the keychain) are refused.
MACH = ("com.apple.system.logger", "com.apple.system.notification_center", "com.apple.system.opendirectoryd.libinfo")


def venv() -> Path:
    return store.home() / "python-venv"


def python() -> Path:
    return Path(os.environ.get("HIKARILM_TOOL_PYTHON") or venv() / "bin" / "python")


def ready() -> bool:
    if os.environ.get("HIKARILM_TOOL_PYTHON"):
        return python().exists()
    try:
        return (venv() / engine.READY_MARK).read_text(encoding="utf-8").split() == list(PACKAGES)
    except OSError:
        return False


def prepare(say, base_python: str = sys.executable) -> bool:
    """Make ~/.hikarilm/python-venv with the packages, once, and note where its Python lives
    (the sandbox lets the script read only that). True when it is ready."""
    if os.environ.get("HIKARILM_TOOL_PYTHON"):
        return True
    if not engine._prepare_venv(venv(), python(), PACKAGES, "Python for the models (" + ", ".join(PACKAGES) + ")",
                                say, base_python):
        return False
    if not (venv() / PATHS).exists():
        script = ("import json, os, sys; print(json.dumps([sys.prefix, sys.base_prefix, sys.executable, "
                  "os.path.realpath(sys.executable)]))")
        try:
            done = subprocess.run([str(python()), "-c", script], capture_output=True, text=True, check=True,
                                  timeout=120)
            # matplotlib makes its font list the first time; done here, outside the sandbox.
            subprocess.run([str(python()), "-c", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"],
                           env={**os.environ, "MPLCONFIGDIR": str(venv() / "mplconfig")}, capture_output=True,
                           timeout=300)
            (venv() / PATHS).write_text(done.stdout, encoding="utf-8")
        except (OSError, subprocess.SubprocessError) as error:
            say(f"  Python for the models could not be checked ({error}). Run  hikarilm restart  to retry.")
            return False
    return True


def _paths() -> list[str]:
    try:
        found = json.loads((venv() / PATHS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        found = []
    paths = {os.path.realpath(p) for p in found if isinstance(p, str)}
    binary = os.path.realpath(python())
    # The installation the binary is in (…/bin/python3.12 → …), which has the standard library.
    return sorted(paths | {os.path.realpath(venv()), binary, os.path.dirname(os.path.dirname(binary))})


def _quote(path: str) -> str:
    return json.dumps(path)


def profile(scratch: str, reads: list[str], programs: list[str]) -> str:
    """The sandbox for one run. Later rules win, so each refusal is followed by what it lets in."""
    home = os.path.realpath(Path.home())
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny process-fork)",
        "(deny signal)",
        "(allow signal (target self))",
        "(deny process-exec)",
        "(allow process-exec " + " ".join(f"(literal {_quote(p)})" for p in programs) + ")",
        "(deny file-write*)",
        f'(allow file-write* (subpath {_quote(scratch)}) (literal "/dev/null") (literal "/dev/zero")'
        ' (literal "/dev/dtracehelper") (regex #"^/dev/fd/") (regex #"^/dev/tty"))',
        "(deny file-read* " + " ".join(f"(subpath {_quote(p)})" for p in (*PRIVATE, home)) + ")",
        "(allow file-read* " + " ".join(f"(subpath {_quote(p)})" for p in (scratch, *reads)) + ")",
        "(deny mach-lookup)",
        "(allow mach-lookup " + " ".join(f'(global-name "{name}")' for name in MACH) + ")",
        "(deny appleevent-send)",
    ]) + "\n"


def sandboxed() -> bool:
    return sys.platform == "darwin"


def _unsandboxed() -> bool:
    return not sandboxed() and os.environ.get("HIKARILM_TEST_UNSANDBOXED") == "1"


def available() -> bool:
    """Whether scripts can run here: Python for the models is set up, on a Mac (or in the tests)."""
    return (sandboxed() or _unsandboxed()) and ready()


# Runs inside the sandbox: limits, the script, then the plots it left open.
BOOTSTRAP = r'''
import ast, os, resource, signal, sys, traceback, warnings
# Past the file size limit a write fails with an error instead of ending the script.
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
for name, value in (("RLIMIT_FSIZE", int(os.environ["HIKARILM_FILE_BYTES"])), ("RLIMIT_CORE", 0),
                    ("RLIMIT_CPU", int(os.environ["HIKARILM_CPU_SECONDS"]))):
    try:
        resource.setrlimit(getattr(resource, name), (value, value))
    except (ValueError, OSError, AttributeError):
        pass
warnings.filterwarnings("ignore", message=".*non-interactive.*")
scratch = os.environ["HIKARILM_SCRATCH"]
with open(os.path.join(scratch, "script.py"), encoding="utf-8") as handle:
    source = handle.read()


def show(value):
    # The last line's value is printed, as in a notebook; matplotlib's own objects are not.
    if value is not None and not type(value).__module__.startswith("matplotlib"):
        print(repr(value))


status = 0
try:
    tree = ast.parse(source, "<script>")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.copy_location(ast.Expr(ast.Call(ast.Name("__show__", ast.Load()), [last.value], [])), last)
        ast.fix_missing_locations(tree)
    exec(compile(tree, "<script>", "exec"), {"__name__": "__main__", "__show__": show})
except SystemExit as stop:
    if stop.code not in (None, 0):
        if not isinstance(stop.code, int):
            print(stop.code, file=sys.stderr)
        status = 1
except BaseException:
    kind, error, frames = sys.exc_info()
    while frames is not None and frames.tb_frame.f_code.co_filename != "<script>":
        frames = frames.tb_next
    sys.stdout.flush()
    traceback.print_exception(kind, error, frames)
    status = 1
pyplot = sys.modules.get("matplotlib.pyplot")
if pyplot is not None:
    for number, figure in enumerate(pyplot.get_fignums()[:int(os.environ["HIKARILM_PLOTS"])]):
        try:
            pyplot.figure(figure).savefig(os.path.join(scratch, "plots", f"plot-{number}.png"), dpi=100,
                                          bbox_inches="tight")
        except Exception as error:
            print(f"[plot {number + 1} could not be saved: {error}]", file=sys.stderr)
sys.stdout.flush()
sys.stderr.flush()
os._exit(status)
'''


class _Usage(ctypes.Structure):
    # rusage_info_v0 from <sys/resource.h>
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]


_libproc = None


def _memory(pid: int) -> int:
    """Bytes the process holds: its footprint on a Mac (what Activity Monitor shows), else its RSS."""
    global _libproc
    if sys.platform == "darwin":
        try:
            if _libproc is None:
                _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            usage = _Usage()
            if _libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
                return int(usage.phys_footprint)
        except (OSError, AttributeError):
            pass
        try:
            done = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2)
            return int(done.stdout.strip() or 0) * 1024
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _kill(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _short_error(output: str) -> str:
    """The last line of a traceback: "ZeroDivisionError: division by zero"."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return lines[-1].strip()[:500] if lines else "the script stopped with an error"


def _plots(folder: Path) -> list[str]:
    out = []
    for path in sorted(folder.glob("plot-*.png"))[:PLOTS]:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) <= PLOT_BYTES:
            out.append("data:image/png;base64," + base64.b64encode(data).decode())
    return out


def run(code: str) -> dict:
    """Run the script: {"name", "code", "output", "pictures", "seconds"}, with "error" when it
    did not finish cleanly (a short line; the whole traceback is in the output)."""
    out = {"name": "python", "code": code if isinstance(code, str) else ""}
    if not isinstance(code, str) or not code.strip():
        return {**out, "output": "", "error": 'give the code to run, as {"code": "..."}'}
    if len(code) > CODE_CHARS:
        return {**out, "output": "", "error": f"the script is longer than {CODE_CHARS:,} characters"}
    if not ready():
        return {**out, "output": "", "error": SETUP}
    unsandboxed = _unsandboxed()
    if not sandboxed() and not unsandboxed:
        return {**out, "output": "", "error": "the Python tool runs only on a Mac, in its sandbox"}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="hikarilm-python-") as made:
        scratch = Path(os.path.realpath(made))
        for name in ("work", "plots", "tmp"):
            (scratch / name).mkdir()
        (scratch / "script.py").write_text(code, encoding="utf-8")
        (scratch / "run.py").write_text(BOOTSTRAP, encoding="utf-8")
        # matplotlib writes in its folder, and builds its font list (seconds) in an empty one: a
        # copy of the one built at setup.
        if (venv() / "mplconfig").is_dir():
            shutil.copytree(venv() / "mplconfig", scratch / "matplotlib")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(scratch / "work"), "TMPDIR": str(scratch / "tmp"),
               "LANG": "en_US.UTF-8", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(scratch / "matplotlib"),
               "OPENBLAS_NUM_THREADS": "4", "HIKARILM_SCRATCH": str(scratch), "HIKARILM_PLOTS": str(PLOTS),
               "HIKARILM_FILE_BYTES": str(FILE_BYTES), "HIKARILM_CPU_SECONDS": str(SECONDS * 2)}
        args = [str(python()), "-I", "-B", "-X", "utf8", str(scratch / "run.py")]
        if not unsandboxed:
            programs = sorted({str(python()), os.path.realpath(python())} | {
                p for p in _paths() if os.path.basename(p).startswith("python")})
            args = [SANDBOX_EXEC, "-p", profile(str(scratch), _paths(), programs), *args]
        try:
            process = subprocess.Popen(args, cwd=scratch / "work", env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            return {**out, "output": "", "error": f"Python could not start ({error})"}
        kept, total = bytearray(), [0]

        def read() -> None:
            while True:
                chunk = process.stdout.read1(65536)
                if not chunk:
                    return
                total[0] += len(chunk)
                if len(kept) < OUTPUT * 4:
                    kept.extend(chunk[:OUTPUT * 4 - len(kept)])
                if total[0] > FLOOD:
                    _kill(process)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        stopped = ""
        while process.poll() is None:
            if time.monotonic() - started > SECONDS:
                stopped = f"stopped after {SECONDS} seconds"
            elif _memory(process.pid) > MEMORY:
                stopped = f"stopped: it used more than {MEMORY / 1024 ** 3:.2f} GB of memory"
            if stopped:
                _kill(process)
                break
            time.sleep(0.05)
        process.wait()
        reader.join(5)
        process.stdout.close()
        if total[0] > FLOOD and not stopped:
            stopped = f"stopped: it printed more than {FLOOD:,} bytes"
        text = kept.decode("utf-8", errors="replace")
        if len(text) > OUTPUT or total[0] > len(kept):
            text = text[:OUTPUT] + f"\n[… output cut at {OUTPUT:,} characters]"
        pictures = _plots(scratch / "plots")
    result = {**out, "output": text, "pictures": pictures, "seconds": round(time.monotonic() - started, 1)}
    if stopped:
        result["error"] = stopped
    elif process.returncode:
        result["error"] = _short_error(text)
    return result


def as_text(answer: dict) -> str:
    """What the model is given back."""
    text = (answer.get("output") or "").rstrip()
    if not text and not answer.get("error"):
        text = "(no output; print what you want to see)"
    if answer.get("error") and answer["error"] not in text:
        text = (text + "\n" if text else "") + "Error: " + answer["error"]
    if answer.get("pictures"):
        made = "1 plot made; the user sees it" if len(answer["pictures"]) == 1 else \
            f"{len(answer['pictures'])} plots made; the user sees them"
        text += f"\n[{made} with your answer]"
    return text
