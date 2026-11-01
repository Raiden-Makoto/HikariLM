import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from helper import daemon, server, update

ROOT = Path(__file__).resolve().parent.parent


class HomeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        patcher = mock.patch.dict(os.environ, {"HIKARILM_HOME": str(self.tmp / "home")})
        patcher.start()
        self.addCleanup(patcher.stop)


class CappedLogTest(HomeCase):
    def test_the_log_keeps_only_its_last_lines(self):
        log = daemon.CappedLog(self.tmp / "h.log", max_lines=100)
        self.addCleanup(log.close)
        for number in range(2500):
            log.write(f"line {number}\n".encode())
        lines = (self.tmp / "h.log").read_text().splitlines()
        self.assertLessEqual(len(lines), 100)
        self.assertGreaterEqual(len(lines), 50)
        self.assertEqual(lines[-1], "line 2499")
        self.assertEqual([int(line.split()[1]) for line in lines], list(range(2500 - len(lines), 2500)))

    def test_a_log_already_too_long_is_cut_when_opened(self):
        (self.tmp / "h.log").write_text("".join(f"old {n}\n" for n in range(500)))
        log = daemon.CappedLog(self.tmp / "h.log", max_lines=100)
        log.write(b"new\n")
        log.close()
        lines = (self.tmp / "h.log").read_text().splitlines()
        self.assertLessEqual(len(lines), 100)
        self.assertEqual(lines[-2:], ["old 499", "new"])

    def test_what_the_helper_and_its_children_print_goes_to_the_log(self):
        script = textwrap.dedent(f"""
            import subprocess, sys, time, warnings
            sys.path.insert(0, {str(ROOT)!r})
            from helper import daemon
            from pathlib import Path
            daemon.capture_output(daemon.CappedLog(Path({str(self.tmp / 'h.log')!r}), max_lines=1000))
            print("from python", flush=True)
            print("to stderr", file=sys.stderr, flush=True)
            subprocess.run(["echo", "from a child"])
            time.sleep(0.5)
        """)
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.stdout, "")
        self.assertEqual(run.returncode, 0, run.stderr)
        text = (self.tmp / "h.log").read_text()
        for line in ("from python", "to stderr", "from a child"):
            self.assertIn(line, text)


class ProcessTest(HomeCase):
    def test_stop_ends_a_process_and_a_strangers_pid_is_not_ours(self):
        # Not a child of this process, like the helper in the background: it is reaped once it ends.
        pid = int(subprocess.run(["sh", "-c", "sleep 30 >/dev/null 2>&1 & echo $!"], capture_output=True,
                                 text=True, check=True).stdout)
        self.addCleanup(lambda: daemon.alive(pid) and os.kill(pid, 9))
        daemon.pid_path().parent.mkdir(parents=True)
        daemon.pid_path().write_text(str(pid))
        self.assertIsNone(daemon.saved_pid())
        self.assertTrue(daemon.stop(pid, seconds=5))
        self.assertFalse(daemon.alive(pid))

    def test_the_pid_file_is_only_removed_by_its_own_process(self):
        daemon.pid_path().parent.mkdir(parents=True)
        daemon.write_pid()
        self.assertEqual(daemon.pid_path().read_text(), str(os.getpid()))
        daemon.forget_pid()
        self.assertFalse(daemon.pid_path().exists())
        daemon.pid_path().write_text("1")
        daemon.forget_pid()
        self.assertTrue(daemon.pid_path().exists())


def git(folder: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments], cwd=folder,
                          capture_output=True, text=True, check=True).stdout


class UpdateTest(HomeCase):
    def setUp(self):
        super().setUp()
        origin, theirs, self.mine = self.tmp / "origin.git", self.tmp / "theirs", self.tmp / "mine"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
        subprocess.run(["git", "clone", "-q", str(origin), str(theirs)], check=True, capture_output=True)
        git(theirs, "checkout", "-q", "-b", "main")
        (theirs / "a.txt").write_text("one")
        git(theirs, "add", "a.txt")
        git(theirs, "commit", "-q", "-m", "one")
        git(theirs, "push", "-q", "origin", "main")
        subprocess.run(["git", "clone", "-q", str(origin), str(self.mine)], check=True, capture_output=True)
        (theirs / "a.txt").write_text("two")
        git(theirs, "commit", "-q", "-am", "two")
        git(theirs, "push", "-q", "origin", "main")
        patcher = mock.patch.object(update, "ROOT", self.mine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_it_says_how_far_behind_and_pulls(self):
        self.assertEqual(update.status()["behind"], 1)
        pulled = update.pull()
        self.assertTrue(pulled["updated"])
        self.assertNotEqual(pulled["from"], pulled["to"])
        self.assertEqual((self.mine / "a.txt").read_text(), "two")
        self.assertEqual(update.status()["behind"], 0)
        self.assertFalse(update.pull()["updated"])

    def test_a_copy_without_git_cannot_update(self):
        with mock.patch.object(update, "ROOT", self.tmp / "home"):
            self.assertEqual(update.status(), {"git": False})
            with self.assertRaises(update.UpdateError):
                update.pull()

    def test_the_button_pulls_and_restarts_but_not_while_busy(self):
        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        base = f"http://127.0.0.1:{http.server_address[1]}"

        def post():
            try:
                with urllib.request.urlopen(urllib.request.Request(base + "/api/update", b"", method="POST"),
                                            timeout=60) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                return json.loads(error.read())

        with mock.patch.object(daemon, "restart_later") as restart:
            server.work.chats += 1
            try:
                self.assertIn("still being written", post()["error"])
            finally:
                server.work.chats -= 1
            restart.assert_not_called()
            body = post()
            self.assertTrue(body["restarting"])
            restart.assert_called_once_with(server.HOST, server.PORT)
            self.assertFalse(post()["restarting"])
            self.assertEqual(restart.call_count, 1)
        with urllib.request.urlopen(base + "/api/state", timeout=10) as response:
            state = json.loads(response.read())
        self.assertEqual(state["pid"], os.getpid())
        self.assertFalse(state["background"])


if __name__ == "__main__":
    unittest.main()
