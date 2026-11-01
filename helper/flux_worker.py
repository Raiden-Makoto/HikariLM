"""Draw pictures with a FLUX.2 model through mflux, in a process of its own.

Run by helper/engine.py with the Python in ~/.hikarilm/pictures-venv; the helper never
imports it. One model per process, named the way mflux names it:

    python flux_worker.py --model FOLDER --variant flux2-klein-4b

The first line written is {"ready": true}, or {"error": ...} if the model did not load.
Then one command per line on stdin, {"op": "picture", "prompt", "width", "height",
"seed", "steps", "guidance", "out"}, answered with {"progress": step, "total": steps}
while it draws and {"done": true} once the PNG is at "out". Stopping a picture ends
the process.
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


def load(folder: str, variant: str):
    from mflux.models.common.config.model_config import AVAILABLE_MODELS
    from mflux.models.flux2.variants import Flux2Klein

    if not variant.startswith("flux2-") or variant not in AVAILABLE_MODELS:
        raise ValueError(f"This mflux does not know {variant}. Update it: ~/.hikarilm/pictures-venv/bin/pip install -U mflux")
    model = Flux2Klein(model_config=AVAILABLE_MODELS[variant], model_path=folder)
    model.callbacks.register(Progress())
    return model


def draw(model, command: dict) -> None:
    image = model.generate_image(
        seed=int(command["seed"]), prompt=str(command["prompt"]), num_inference_steps=int(command["steps"]),
        width=int(command["width"]), height=int(command["height"]), guidance=float(command["guidance"]))
    image.image.save(command["out"], format="PNG")
    say({"done": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    try:
        model = load(args.model, args.variant)
    except Exception as error:
        traceback.print_exc()
        say({"error": f"The picture model could not load: {type(error).__name__}: {error}"})
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
            draw(model, command)
        except Exception as error:
            traceback.print_exc()
            say({"error": f"The picture could not be made: {type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
