"""Writes down what Kokoro said, with Whisper, for tests/check_voice.py. Runs with the Python
in ~/.hikarilm/voice-venv (mlx-audio has Whisper too):

    voice-venv/bin/python tests/voice_hear.py --whisper <Whisper's folder> < clips.json

clips.json is a list of {"pcm": "<file of 16-bit mono samples>", "rate": 24000}; it prints a
list of the texts heard, in the same order, as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", required=True)
    args = parser.parse_args()
    clips = json.load(sys.stdin)
    out = sys.stdout
    sys.stdout = sys.stderr  # mlx-audio prints as it works
    import numpy as np
    from mlx_audio.stt.utils import load_model
    from scipy.signal import resample_poly

    model = load_model(Path(args.whisper))
    heard = []
    for clip in clips:
        samples = np.fromfile(clip["pcm"], dtype="<i2").astype(np.float32) / 32768
        rate = int(clip.get("rate") or 24000)
        audio = resample_poly(samples, 16000, rate).astype(np.float32) if rate != 16000 else samples
        result = model.generate(audio, language="en", temperature=0.0, condition_on_previous_text=False)
        heard.append(result.text.strip())
    out.write(json.dumps(heard) + "\n")


if __name__ == "__main__":
    main()
