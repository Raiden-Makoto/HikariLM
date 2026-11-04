"""Try to get out of the Python tool's sandbox, on a Mac. Not part of the test suite.

Runs scripts the way a chat model's are run (python_tool.run, in macOS's sandbox, with the
Python in ~/.hikarilm/python-venv) and reports each as OK or FAIL:
- the packages work inside the sandbox, and a plot comes back as a picture;
- each way out is refused: the internet, your home folder and HikariLM's chats, writing outside
  its scratch folder, other programs (subprocess, os.system, open, ctypes), fork, signals to
  other processes;
- the limits: 30 seconds, 1.25 GB of memory, a flood of output, a file too large.
No model is loaded and nothing is downloaded. Run  hikarilm restart  once first, to set up the venv.

    python3 tests/check_python.py
    python3 tests/check_python.py -v      and what each script printed
"""

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helper import python_tool, store  # noqa: E402

HOME = os.path.realpath(Path.home())
CHATS = os.path.realpath(store.home())
OUTSIDE = os.path.join(HOME, "hikarilm-sandbox-check.txt")
TMP = os.path.join(os.path.realpath(tempfile.gettempdir()), "hikarilm-sandbox-check.txt")

WORKS = [
    ("numpy, sympy, pandas, scipy",
     "import numpy as np, sympy, pandas as pd, scipy.stats as st, scipy.sparse.linalg, scipy.integrate\n"
     "print(np.linalg.solve([[2, 1], [1, 3]], [3, 5]).round(6).tolist())\n"
     "print(sympy.integrate(sympy.sin(sympy.Symbol('x')) ** 2))\n"
     "print(pd.DataFrame({'a': [1, 2, 3]})['a'].mean())\n"
     "print(round(st.norm.cdf(1.96), 4))",
     lambda a: "error" not in a and "0.975" in a["output"] and "2.0" in a["output"]),
    ("a plot comes back as a picture",
     "import matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(-3, 3, 50)\nplt.plot(x, x ** 2)",
     lambda a: "error" not in a and len(a["pictures"]) == 1),
    ("resolving its own paths",
     "import os, sys\nprint(os.path.realpath(sys.executable) != '', os.path.realpath(os.getcwd()) != '')",
     lambda a: "error" not in a and a["output"] == "True True\n"),
    ("files in its own folder",
     "open('a.txt', 'w').write('hi')\nprint(open('a.txt').read())",
     lambda a: a["output"] == "hi\n"),
]

# Each prints ESCAPED only if it got out.
REFUSED = [
    ("the internet (a socket)",
     "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=5)\n    print('ESCAPED')\n"
     "except OSError as e:\n    print('refused:', e)"),
    ("the internet (a web page)",
     "import urllib.request\ntry:\n    urllib.request.urlopen('https://example.com', timeout=5)\n    print('ESCAPED')\n"
     "except Exception as e:\n    print('refused:', e)"),
    ("listing your home folder",
     f"import os\ntry:\n    print('ESCAPED', os.listdir({HOME!r})[:3])\nexcept OSError as e:\n    print('refused:', e)"),
    ("reading ~/.ssh",
     f"import os\ntry:\n    print('ESCAPED', os.listdir({os.path.join(HOME, '.ssh')!r}))\n"
     "except FileNotFoundError:\n    print('ESCAPED (could see it is not there)')\nexcept OSError as e:\n    print('refused:', e)"),
    ("reading HikariLM's chats",
     f"import os\ntry:\n    print('ESCAPED', os.listdir({CHATS!r})[:3])\nexcept OSError as e:\n    print('refused:', e)"),
    ("writing in your home folder",
     f"try:\n    open({OUTSIDE!r}, 'w').write('x')\n    print('ESCAPED')\nexcept OSError as e:\n    print('refused:', e)"),
    ("writing in the temporary folder",
     f"try:\n    open({TMP!r}, 'w').write('x')\n    print('ESCAPED')\nexcept OSError as e:\n    print('refused:', e)"),
    ("starting a program (subprocess)",
     "import subprocess\ntry:\n    print(subprocess.run(['/bin/echo', 'ESCAPED'], capture_output=True, text=True).stdout)\n"
     "except Exception as e:\n    print('refused:', e)"),
    ("starting a program (os.system)",
     "import os\nprint('exit', os.system('/bin/echo ESCAPED'))"),
    ("opening an app",
     "import subprocess\ntry:\n    subprocess.run(['/usr/bin/open', '-a', 'Calculator'], check=True)\n    print('ESCAPED')\n"
     "except Exception as e:\n    print('refused:', e)"),
    ("starting a program (ctypes system)",
     "import ctypes\nprint('exit', ctypes.CDLL(None).system(b'/bin/echo ESCAPED'))"),
    ("fork",
     "import os\ntry:\n    pid = os.fork()\n    if pid == 0:\n        os._exit(0)\n    print('ESCAPED')\n"
     "except OSError as e:\n    print('refused:', e)"),
    ("a signal to the helper",
     "import os\ntry:\n    os.kill(os.getppid(), 0)\n    print('ESCAPED')\nexcept OSError as e:\n    print('refused:', e)"),
]

LIMITS = [
    ("stops after 30 seconds", "while True:\n    pass",
     lambda a: a.get("error") == f"stopped after {python_tool.SECONDS} seconds" and a["seconds"] < python_tool.SECONDS + 5),
    ("stops past 1.25 GB of memory",
     "blocks = []\nwhile True:\n    blocks.append(b'x' * (50 * 1024 * 1024))",
     lambda a: str(a.get("error", "")).startswith("stopped: it used more than")),
    ("stops a flood of output", "while True:\n    print('a' * 1000)",
     lambda a: str(a.get("error", "")).startswith("stopped: it printed more than")
     and len(a["output"]) <= python_tool.OUTPUT + 100),
    ("a file too large is an error", "open('big', 'wb').write(b'x' * (200 * 1024 * 1024))",
     lambda a: "File too large" in str(a.get("error", ""))),
]


def show(answer: dict, verbose: bool) -> None:
    if verbose or "error" in answer and "Traceback" not in answer["output"]:
        for line in (answer["output"].strip() or "(nothing printed)").splitlines()[:12]:
            print("      " + line[:200])
        if answer.get("error"):
            print("      error: " + answer["error"])


def main() -> int:
    verbose = "-v" in sys.argv
    if sys.platform != "darwin":
        print("This checks macOS's sandbox; run it on the Mac.")
        return 1
    if not python_tool.ready():
        print("Python for the models is not set up: run  hikarilm restart  first.")
        return 1
    print("Python:", python_tool.python())
    failed = 0
    print("\nWorks inside the sandbox:")
    for label, code, ok in WORKS:
        answer = python_tool.run(code)
        good = ok(answer)
        failed += not good
        print(f"  {'OK  ' if good else 'FAIL'} {label}  ({answer.get('seconds', 0)} s)")
        show(answer, verbose or not good)
    print("\nRefused:")
    for label, code in REFUSED:
        answer = python_tool.run(code)
        good = "ESCAPED" not in answer["output"]
        failed += not good
        print(f"  {'OK  ' if good else 'FAIL'} {label}")
        show(answer, verbose or not good)
    for path in (OUTSIDE, TMP):
        if os.path.exists(path):
            print(f"  FAIL a file was written outside: {path} (deleted now)")
            os.remove(path)
            failed += 1
    print("\nLimits:")
    for label, code, ok in LIMITS:
        started = time.monotonic()
        answer = python_tool.run(code)
        good = ok(answer)
        failed += not good
        print(f"  {'OK  ' if good else 'FAIL'} {label}  ({time.monotonic() - started:.1f} s)")
        show(answer, verbose or not good)
    print("\nAll OK." if not failed else f"\n{failed} FAILED. Paste this output into the chat.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
