"""On the Mac: check HikariLM's FLUX.2 table against mflux's, then draw with what is downloaded.

    python3 tests/check_flux.py

It makes ~/.hikarilm/pictures-venv with mflux first if there is none, and runs itself
again with that venv's Python. First, every FLUX.2 model flux.py knows must be one mflux knows, with the same blocks
and the same number of steps. That covers all of them without downloading any. Then
each FLUX.2 model in the Hugging Face cache draws one small picture through the real
worker, saved in /tmp, so you can look at it.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helper import engine, flux, imagegen  # noqa: E402


def check_table() -> bool:
    from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    good = True
    theirs = {key for key in AVAILABLE_MODELS if key.startswith("flux2-")}
    if set(flux.VARIANTS) != theirs:
        print(f"FAIL  mflux's FLUX.2 models are {sorted(theirs)}, HikariLM's are {sorted(flux.VARIANTS)}")
        good = False
    for key, (label, double, single, distilled) in flux.VARIANTS.items():
        config = AVAILABLE_MODELS.get(key)
        if config is None:
            continue
        shape = (config.transformer_overrides.get("num_layers"), config.transformer_overrides.get("num_single_layers"))
        steps = MODEL_INFERENCE_STEPS.get(key)
        ok = shape == (double, single) and steps == flux.STEPS[distilled]
        good = good and ok
        print(f"{'ok  ' if ok else 'FAIL'}  {label}: blocks {shape}, {steps} steps")
    return good


def check_drawing() -> bool:
    found, notes = flux.scan()
    for note in notes:
        print("      " + note)
    if not found:
        print("      No FLUX.2 model is downloaded, so nothing was drawn. Get one with:  " + imagegen.GET_ONE)
        return True
    good = True
    for name, picture in found.items():
        request = imagegen.clean_request({"model": name, "prompt": "a red fox in fresh snow", "size": "small",
                                          "seed": 42})
        started = time.monotonic()
        try:
            image = engine.make_picture(picture, request, lambda step, total: None, lambda: False)
        except engine.ModelError as error:
            print(f"FAIL  {picture['label']}: {error}")
            good = False
            continue
        finally:
            engine.unload()
        out = Path("/tmp") / (name.replace("/", "--") + ".png")
        out.write_bytes(image)
        ok = image.startswith(imagegen.PNG)
        good = good and ok
        print(f"{'ok  ' if ok else 'FAIL'}  {picture['label']}: drew in {time.monotonic() - started:.0f} s, saved {out}")
    return good


def in_picture_venv() -> None:
    """Carry on with the picture venv's Python, making the venv first if it is not there."""
    try:
        import mflux  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("HIKARILM_CHECK_RESTARTED"):
        sys.exit(f"mflux is still missing in {engine.picture_python()}.")
    if not engine.prepare_pictures(print):
        sys.exit(1)
    os.environ["HIKARILM_CHECK_RESTARTED"] = "1"
    python = str(engine.picture_python())
    os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    in_picture_venv()
    table = check_table()
    drawing = check_drawing()
    print("OK" if table and drawing else "FAILED")
    sys.exit(0 if table and drawing else 1)
