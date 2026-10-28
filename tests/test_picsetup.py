import io
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from helper import context, imagegen, picsetup, sdserver


def program_zip() -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("sd-server", "#!/bin/sh\n")
        archive.writestr("libstable-diffusion.dylib", "lib")
    return data.getvalue()


class FakeDownloads:
    """GitHub's release list, the sd-server zip, and the three model files."""

    def __init__(self, slow: bool = False):
        self.asked: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.asked.append(self.path)
                if self.path == "/releases":
                    names = ["sd-bin-Darwin-macOS-arm64.zip", "sd-bin-win-vulkan-x64.zip", "sd-bin-win-cpu-x64.zip",
                             "sd-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip", "sd-bin-Linux-Ubuntu-24.04-x86_64.zip"]
                    body = json.dumps({"assets": [{"name": name, "size": 1000, "browser_download_url":
                                                   fake.base + "/program.zip"} for name in names]}).encode()
                elif self.path == "/program.zip":
                    body = program_zip()
                else:
                    body = b"x" * (3 << 20)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    for start in range(0, len(body), 1 << 20):
                        self.wfile.write(body[start:start + (1 << 20)])
                        if slow:
                            time.sleep(0.5)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Stop hung up

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class PictureSetupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        self.patch(mock.patch.dict(os.environ, {"LIGHTUSAGI_HOME": str(self.home)}))
        self.patch(mock.patch("shutil.which", return_value=None))
        self.patch(mock.patch.object(picsetup, "KLEIN_BYTES", 0))

    def patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, slow=False):
        fake = FakeDownloads(slow)
        self.addCleanup(fake.stop)
        self.patch(mock.patch.object(picsetup, "RELEASES", fake.base + "/releases"))
        self.patch(mock.patch.object(imagegen, "KLEIN_FILES", tuple(
            (flag, name, f"{fake.base}/{name}") for flag, name, _ in imagegen.KLEIN_FILES)))
        return fake

    def wait(self, setup):
        for _ in range(200):
            if not setup.status()["running"]:
                return setup.status()
            time.sleep(0.05)
        self.fail("setup did not finish")

    def test_downloads_everything_and_adds_the_model(self):
        self.serve()
        setup = picsetup.Setup()
        setup.start()
        status = self.wait(setup)
        self.assertEqual(status["error"], "")
        self.assertTrue(status["finished"])
        self.assertTrue(status["program"].startswith(str(self.home / "sd-server")))
        self.assertTrue(all(item["found"] for item in status["klein"]["files"]))
        self.assertEqual(list((self.home / "models").glob("*.part")), [])
        self.assertEqual(list((self.home / "sd-server").glob("*.zip")), [])
        models, _ = sdserver.parse_models(context.load_settings()["pictures"]["models"])
        self.assertEqual([model["name"] for model in models], ["flux2-klein"])

    def test_files_already_there_are_not_fetched_again(self):
        fake = self.serve()
        (self.home / "models").mkdir()
        (self.home / "models" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"kept")
        setup = picsetup.Setup()
        setup.start()
        self.wait(setup)
        self.assertNotIn("/Qwen3-4B-Q4_K_M.gguf", fake.asked)
        self.assertEqual((self.home / "models" / "Qwen3-4B-Q4_K_M.gguf").read_bytes(), b"kept")

    def test_stop_leaves_no_half_file(self):
        self.serve(slow=True)
        setup = picsetup.Setup()
        setup.start()
        for _ in range(100):
            if setup.status()["file"].endswith(".gguf"):
                break
            time.sleep(0.05)
        setup.stop()
        status = self.wait(setup)
        self.assertFalse(status["finished"])
        self.assertEqual(list((self.home / "models").glob("*.part")), [])
        self.assertFalse(any(item["found"] for item in status["klein"]["files"]))

    def test_release_for_each_computer(self):
        assets = [{"name": name} for name in ("sd-bin-Darwin-macOS-arm64.zip", "sd-bin-win-vulkan-x64.zip",
                                               "sd-bin-Linux-Ubuntu-24.04-x86_64.zip",
                                               "sd-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip")]
        with mock.patch("sys.platform", "darwin"), mock.patch("platform.machine", return_value="arm64"):
            self.assertIn("Darwin", picsetup._release_asset(assets)["name"])
        with mock.patch("sys.platform", "darwin"), mock.patch("platform.machine", return_value="x86_64"):
            with self.assertRaises(picsetup.SetupError):
                picsetup._release_asset(assets)
        with mock.patch("sys.platform", "linux"), mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("ctypes.util.find_library", return_value=None):
            self.assertEqual(picsetup._release_asset(assets)["name"], "sd-bin-Linux-Ubuntu-24.04-x86_64.zip")
        with mock.patch("sys.platform", "linux"), mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("ctypes.util.find_library", return_value="libvulkan.so.1"):
            self.assertEqual(picsetup._release_asset(assets)["name"], "sd-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip")
        with mock.patch("sys.platform", "linux"), mock.patch("platform.machine", return_value="aarch64"):
            with self.assertRaises(picsetup.SetupError):
                picsetup._release_asset(assets)


if __name__ == "__main__":
    unittest.main()
