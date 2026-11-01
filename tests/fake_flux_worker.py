"""Stands in for helper/flux_worker.py in tests: no MLX, no mflux.

It says ready, then for each picture writes one progress line a step and a small PNG.
A prompt of "crash" makes it exit, "slow" makes each step take a second, and "fail"
answers with an error and carries on.
"""

import argparse
import json
import sys
import time

PNG = b"\x89PNG\r\n\x1a\n" + b"fake picture"


def say(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    print("loading", args.variant, file=sys.stderr)
    say({"ready": True})
    for line in sys.stdin:
        command = json.loads(line)
        if command["prompt"] == "crash":
            sys.exit(3)
        if command["prompt"] == "fail":
            say({"error": "The picture could not be made: no"})
            continue
        for step in range(1, command["steps"] + 1):
            if command["prompt"] == "slow":
                time.sleep(1)
            say({"progress": step, "total": command["steps"]})
        with open(command["out"], "wb") as handle:
            handle.write(PNG + f" {args.variant} {command['steps']} {command['guidance']}".encode())
        say({"done": True})


if __name__ == "__main__":
    main()
