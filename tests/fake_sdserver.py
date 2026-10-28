"""A stand-in for stable-diffusion.cpp's sd-server, for tests. It answers the same
/sdapi/v1 calls and draws a one-pixel picture slowly enough to report progress.

    python -m tests.fake_sdserver --model NAME --steps-delay 0.05 --listen-ip 127.0.0.1 --listen-port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fake")
    parser.add_argument("--steps-delay", type=float, default=0.05)
    parser.add_argument("--fail", action="store_true", help="exit at once, as a bad model would")
    parser.add_argument("--listen-ip", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    args, _ = parser.parse_known_args()
    if args.fail:
        print("error: failed to load model", flush=True)
        raise SystemExit(1)
    state = {"step": 0, "steps": 0, "stop": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _json(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/sdapi/v1/progress"):
                self._json({"state": {"sampling_step": state["step"], "sampling_steps": state["steps"]}})
            elif self.path == "/sdapi/v1/sd-models":
                self._json([{"title": args.model, "model_name": args.model}])
            elif self.path == "/sdapi/v1/options":
                self._json({"sd_model_checkpoint": args.model})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/sdapi/v1/interrupt":
                state["stop"].set()
                self._json({})
                return
            if self.path != "/sdapi/v1/txt2img":
                self._json({"error": "not found"}, 404)
                return
            state["stop"].clear()
            state["steps"] = int(body.get("steps") or 4)
            for step in range(1, state["steps"] + 1):
                if state["stop"].wait(args.steps_delay):
                    break
                state["step"] = step
            state["step"] = state["steps"] = 0
            self._json({"images": [base64.b64encode(PNG).decode()], "info": json.dumps({"seed": body.get("seed")})})

    ThreadingHTTPServer((args.listen_ip, args.listen_port), Handler).serve_forever()


if __name__ == "__main__":
    main()
