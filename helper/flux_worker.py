"""Draw pictures with a FLUX.2 model through mflux, in a process of its own.

Run by helper/engine.py with the Python in ~/.hikarilm/pictures-venv; the helper never
imports it. One model per process, named the way mflux names it:

    python flux_worker.py --model FOLDER --variant flux2-klein-4b [--lora FILE --lora-strength 1.0]

A LoRA is baked into the weights as the model loads, so it holds for every picture.

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
import sys
import traceback

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    args = parser.parse_args()
    try:
        drawer = Drawer(args.model, args.variant, args.lora, args.lora_strength)
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
