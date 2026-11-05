"""Draw pictures with a FLUX.2 model through mflux, in a process of its own.

Run by helper/engine.py with the Python in ~/.hikarilm/pictures-venv; the helper never
imports it. One model per process, named the way mflux names it:

    python flux_worker.py --model FOLDER --variant flux2-klein-4b [--lora FILE --lora-strength 1.0]

With a LoRA, each picture is drawn by mflux's own command, the way LoRAs were tried with it:
mflux-generate-flux2 (mflux-generate-flux2-edit with reference pictures), given the picture as
a JSON config with --config-from-conf. It loads the model with the LoRA for every picture.

The first line written is {"ready": true}, or {"error": ...} if the model did not load.
Then one command per line on stdin, {"op": "picture", "prompt", "width", "height",
"seed", "steps", "guidance", "out", "references"}, answered with {"progress": step,
"total": steps} while it draws and {"done": true} once the PNG is at "out". Stopping a
picture ends the process.

"references" are picture files the model draws from as well as the prompt. mflux
draws from them with Flux2KleinEdit, which loads the same weights as Flux2Klein but
cannot draw without a reference, so the worker holds one of the two and swaps when
a picture needs the other ({"loading": true} is written first).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

# mflux and its libraries print as they work; only these lines go to the helper.
OUT = sys.stdout
sys.stdout = sys.stderr


def say(event: dict) -> None:
    OUT.write(json.dumps(event) + "\n")
    OUT.flush()


class Progress:
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        say({"progress": t + 1 - config.init_time_step, "total": config.num_inference_steps - config.init_time_step})


class Drawer:
    """The loaded model: Flux2Klein for a prompt alone, Flux2KleinEdit with references."""

    def __init__(self, folder: str, variant: str, lora: str | None = None, strength: float = 1.0) -> None:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        if not variant.startswith("flux2-") or variant not in AVAILABLE_MODELS:
            raise ValueError(f"This mflux does not know {variant}. Update it: ~/.hikarilm/pictures-venv/bin/pip install -U mflux")
        self.folder, self.config = folder, AVAILABLE_MODELS[variant]
        self.lora = {"lora_paths": [lora], "lora_scales": [strength]} if lora else {}
        self.model, self.edits = None, False
        self.load(edits=False)

    def load(self, edits: bool) -> None:
        import gc

        import mlx.core as mx
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

        self.model = None
        gc.collect()
        mx.clear_cache()
        kind = Flux2KleinEdit if edits else Flux2Klein
        self.model = kind(model_config=self.config, model_path=self.folder, **self.lora)
        self.model.callbacks.register(Progress())
        self.edits = edits

    def draw(self, command: dict) -> None:
        references = [str(path) for path in command.get("references") or []]
        if bool(references) != self.edits:
            say({"loading": True})
            self.load(edits=bool(references))
        options = {"image_paths": references} if references else {}
        image = self.model.generate_image(
            seed=int(command["seed"]), prompt=str(command["prompt"]), num_inference_steps=int(command["steps"]),
            width=int(command["width"]), height=int(command["height"]), guidance=float(command["guidance"]), **options)
        image.image.save(command["out"], format="PNG")
        say({"done": True})


class LoraDrawer:
    """A picture with a LoRA, drawn by mflux's command from a JSON config."""

    STEP = re.compile(r"(\d+)/(\d+) \[")

    def __init__(self, folder: str, variant: str, lora: str, strength: float = 1.0) -> None:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        if not variant.startswith("flux2-") or variant not in AVAILABLE_MODELS:
            raise ValueError(f"This mflux does not know {variant}. Update it: ~/.hikarilm/pictures-venv/bin/pip install -U mflux")
        # The commands mflux installs beside this Python.
        where = Path(sys.executable).parent
        self.commands = {False: where / "mflux-generate-flux2", True: where / "mflux-generate-flux2-edit"}
        missing = [str(path) for path in self.commands.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("mflux's command is not there: " + ", ".join(missing))
        if not Path(lora).is_file():
            raise FileNotFoundError(f"the LoRA file is not there: {lora}")
        self.folder, self.variant, self.lora, self.strength = folder, variant, lora, strength

    def config(self, command: dict, references: list[str]) -> dict:
        config = {"model_path": self.folder, "base_model": self.variant, "prompt": str(command["prompt"]),
                  "width": int(command["width"]), "height": int(command["height"]), "seed": int(command["seed"]),
                  "steps": int(command["steps"]), "guidance": float(command["guidance"]),
                  "lora_paths": [self.lora], "lora_scales": [float(self.strength)]}
        if references:
            config["image_paths"] = references
        return config

    def draw(self, command: dict) -> None:
        references = [str(path) for path in command.get("references") or []]
        fd, conf = tempfile.mkstemp(prefix=".hikarilm-mflux.", suffix=".json", dir=os.path.dirname(command["out"]))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.config(command, references), handle)
            # mflux writes image_1.png beside a file that is already there, never over it.
            Path(command["out"]).unlink(missing_ok=True)
            say({"loading": True})
            run = subprocess.Popen([str(self.commands[bool(references)]), "--config-from-conf", conf,
                                    "--output", str(command["out"])],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            said = []
            reader = threading.Thread(target=self.watch, args=(run.stdout, said), daemon=True)
            reader.start()
            run.wait()
            reader.join(10)
            if run.returncode:
                last = next((line for line in reversed(said) if line.strip()), f"it exited with {run.returncode}")
                raise RuntimeError("mflux-generate-flux2 failed: " + last.strip()[-300:])
            if not Path(command["out"]).stat().st_size:
                raise RuntimeError("mflux-generate-flux2 wrote no picture")
        finally:
            try:
                os.unlink(conf)
            except OSError:
                pass
        say({"done": True})

    def watch(self, stream, said: list[str]) -> None:
        """Pass on mflux's steps (its progress bar, "3/4 [") as progress lines, and keep the rest
        for the log and for saying why it failed."""
        piece = b""
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            sys.stderr.buffer.write(chunk)
            sys.stderr.flush()
            piece += chunk
            parts = re.split(rb"[\r\n]", piece)
            piece = parts.pop()
            for part in parts:
                self.line(part.decode("utf-8", "replace"), said)
        self.line(piece.decode("utf-8", "replace"), said)

    def line(self, text: str, said: list[str]) -> None:
        steps = list(self.STEP.finditer(text))
        if steps:
            say({"progress": int(steps[-1].group(1)), "total": int(steps[-1].group(2))})
            # What is printed right after the bar, on the same line: "...4/4 [00:04<00:00]ValueError: ..."
            text = text[text.find("]", steps[-1].end()) + 1:] if "]" in text[steps[-1].end():] else ""
        if text.strip():
            said.append(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    args = parser.parse_args()
    try:
        drawer = (LoraDrawer(args.model, args.variant, args.lora, args.lora_strength) if args.lora
                  else Drawer(args.model, args.variant))
    except Exception as error:
        traceback.print_exc()
        what = "The picture model could not load" + (" with the LoRA" if args.lora else "")
        say({"error": f"{what}: {type(error).__name__}: {error}"})
        return
    say({"ready": True})
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        if command.get("op") != "picture":
            continue
        try:
            drawer.draw(command)
        except Exception as error:
            traceback.print_exc()
            say({"error": f"The picture could not be made: {type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
