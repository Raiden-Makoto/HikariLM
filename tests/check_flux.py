"""On the Mac: check HikariLM's FLUX.2 table against mflux's, then draw with what is downloaded.

    python3 tests/check_flux.py

It makes ~/.hikarilm/pictures-venv with mflux first if there is none, and runs itself
again with that venv's Python. First, every FLUX.2 model flux.py knows must be one mflux knows, with the same blocks
and the same number of steps. That covers all of them without downloading any. Then
each FLUX.2 model in the Hugging Face cache draws one small picture through the real
worker, then a second one from the first as its reference, both saved in /tmp, so you
can look at them: the fox in the meadow should be the fox from the snow. Last, when
FLUX.2 klein base 4B is downloaded, a public LoRA (pixel art, 370 MB) is looked up, added,
downloaded and drawn with, then taken off the list and deleted again.
"""

from __future__ import annotations

import base64
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helper import engine, flux, imagegen, lora  # noqa: E402

TEST_LORA = "adirik/pixel-art-lora-flux.2-klein-4B"
TEST_TRIGGER = "PXART4"


def check_table() -> bool:
    from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    good = True
    theirs = {key for key in AVAILABLE_MODELS if key.startswith("flux2-")}
    if set(flux.VARIANTS) != theirs:
        print(f"FAIL  mflux's FLUX.2 models are {sorted(theirs)}, HikariLM's are {sorted(flux.VARIANTS)}")
        good = False
    for key, (double, single, distilled) in flux.VARIANTS.items():
        config = AVAILABLE_MODELS.get(key)
        if config is None:
            continue
        shape = (config.transformer_overrides.get("num_layers"), config.transformer_overrides.get("num_single_layers"))
        steps = MODEL_INFERENCE_STEPS.get(key)
        ok = shape == (double, single) and steps == flux.STEPS[distilled]
        good = good and ok
        print(f"{'ok  ' if ok else 'FAIL'}  {key}: blocks {shape}, {steps} steps")
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
        # The second picture is drawn from the first, so the worker swaps to Flux2KleinEdit.
        image = b""
        try:
            for kind, prompt in (("", "a red fox in fresh snow"),
                                 ("-from-reference", "the same fox, now sitting in a sunny green meadow")):
                payload = {"model": name, "prompt": prompt, "size": "small", "seed": 42}
                if image:
                    payload["references"] = [base64.b64encode(image).decode()]
                request = imagegen.clean_request(payload)
                started = time.monotonic()
                try:
                    image = engine.make_picture(picture, request, lambda step, total: None, lambda: False)
                except engine.ModelError as error:
                    print(f"FAIL  {picture['label']}{kind}: {error}")
                    good = False
                    break
                out = Path("/tmp") / (name.replace("/", "--") + kind + ".png")
                out.write_bytes(image)
                ok = image.startswith(imagegen.PNG)
                good = good and ok
                print(f"{'ok  ' if ok else 'FAIL'}  {picture['label']}{kind}: drew in "
                      f"{time.monotonic() - started:.0f} s, saved {out}")
        finally:
            engine.unload()
    return good


def base_4b_report(found_models: dict) -> str:
    """Why no base 4B was found: what the scan did find, and each base 4B download it passed over."""
    named = [name + " (" + picture["variant"] + ")" for name, picture in found_models.items()]
    lines = ["Found: " + (", ".join(named) or "none") + "."]
    hub = engine.hub()
    lines.append(f"Hugging Face cache: {hub}.")
    for repo in sorted(hub.glob("models--*")) if hub.is_dir() else []:
        name = repo.name[len("models--"):].replace("--", "/", 1)
        if (flux.variant_of(name) or ("",))[0] != lora.VARIANT:
            continue
        snapshot = engine.snapshot(repo)
        if any(repo.glob("blobs/*.incomplete")):
            why = "a download is unfinished (an .incomplete file in blobs)"
        elif not snapshot:
            why = "it has no snapshot (refs/main)"
        else:
            why = flux.identify(name, snapshot)[1] or "it looks whole"
        lines.append(f"{name}: {why.rstrip('.')}.")
    return " ".join(lines)


def check_loras() -> bool:
    """Look up, add and download a public LoRA, draw with it on the base 4B (beside the same
    seed without it, and from a reference picture with it, so Flux2KleinEdit loads it too),
    then take it off the list and delete it. What was there before the check is left alone."""
    flux.forget_scan()
    found_models = flux.scan()[0]
    bases = [(name, picture) for name, picture in found_models.items() if picture["variant"] == lora.VARIANT]
    if not bases:
        print("      No LoRA was drawn: no FLUX.2 klein base 4B was found. " + base_4b_report(found_models))
        return True
    name, picture = bases[0]
    try:
        found = lora.look_up(TEST_LORA)
    except lora.LoraError as error:
        print(f"FAIL  looking up {TEST_LORA}: {error}")
        return False
    if found.get("error") or found["trigger"] != TEST_TRIGGER:
        print(f"FAIL  {TEST_LORA}: {found.get('error') or 'trigger word ' + repr(found['trigger'])}")
        return False
    print(f"ok    looked up {TEST_LORA}: {found['file']}, {found['size'] / 1e6:.0f} MB, trigger {found['trigger']}")
    listed_before = any(item["id"] == found["id"] for item in lora.saved())
    folder = engine.hub() / ("models--" + TEST_LORA.replace("/", "--"))
    cached_before = folder.exists()
    good = True
    try:
        lora.add(found)
        from huggingface_hub import hf_hub_download

        started = time.monotonic()
        hf_hub_download(TEST_LORA, found["file"], cache_dir=str(engine.hub()))
        item = lora.find(found["id"])
        if not item:
            print(f"FAIL  {TEST_LORA} is downloaded but not ready: {lora.listing()}")
            return False
        print(f"ok    downloaded it in {time.monotonic() - started:.0f} s")
        image = b""
        for kind, extra in (("", {"lora": item["id"]}), ("-without", {}), ("-from-reference", {"lora": item["id"]})):
            payload = {"model": name, "prompt": "a red fox in fresh snow", "size": "small", "seed": 42, **extra}
            if kind == "-from-reference":
                payload["references"] = [base64.b64encode(image).decode()]
            request = imagegen.clean_request(payload)
            imagegen.attach_lora(request, picture)
            started = time.monotonic()
            try:
                drawn = engine.make_picture(picture, request, lambda step, total: None, lambda: False)
            except engine.ModelError as error:
                print(f"FAIL  LoRA{kind}: {error}")
                good = False
                break
            image = image or drawn
            out = Path("/tmp") / ("lora-pixel-art" + kind + ".png")
            out.write_bytes(drawn)
            ok = drawn.startswith(imagegen.PNG)
            good = good and ok
            print(f"{'ok  ' if ok else 'FAIL'}  LoRA{kind} ({request['prompt']}): drew in "
                  f"{time.monotonic() - started:.0f} s, saved {out}")
    finally:
        engine.unload()
        if not listed_before:
            lora.remove(found["id"])
        if not cached_before:
            shutil.rmtree(folder, ignore_errors=True)
        print(f"      took {TEST_LORA} off the list and deleted it" if not (listed_before or cached_before)
              else f"      left {TEST_LORA} as it was before the check")
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
    loras = check_loras()
    print("OK" if table and drawing and loras else "FAILED")
    sys.exit(0 if table and drawing and loras else 1)
