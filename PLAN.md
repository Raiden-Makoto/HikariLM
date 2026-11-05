# HikariLM plan

What is coming, in order. What is already in is in the README and the release notes (5.5.0 is
the latest).

- **v6**: LoRA files from this Mac, and LoRAs on the distilled 4B if a check shows they work.

## Recommended models

Any MLX model works; these are the ones to start with. Get one with
`~/.hikarilm/venv/bin/hf download <repo>`. Sizes are the download.

| For | Model | Repo | Download |
|---|---|---|---|
| Everyday chat, thinking or not | Qwen3 8B | `mlx-community/Qwen3-8B-4bit` | 4.6 GB |
| Dense reasoning, math, step-by-step analysis | DeepSeek R1 8B | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Code | Qwen2.5 Coder 7B | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| Reading pictures | Gemma 3 12B | `mlx-community/gemma-3-12b-it-4bit` | 8.1 GB |
| Drawing pictures | FLUX.2 klein 4B | `mlx-community/FLUX.2-Klein-4B-4bit` | 4.6 GB |
| Drawing, more detail | FLUX.2 klein 9B | `mlx-community/flux2-klein-9b-4bit` | 9.5 GB |
| Drawing with LoRAs | FLUX.2 klein base 4B | `mflux-community/flux-2-klein-base-4b-mflux-q4` | 4.6 GB |
| Speech to text | Whisper large-v3 turbo | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 1.6 GB |
| Speech to text, smaller | Whisper small | `mlx-community/whisper-small-asr-fp16` | 0.5 GB |
| Reading aloud | Kokoro 82M | `mlx-community/Kokoro-82M-bf16` | 0.4 GB |
| Optional: BitNet | Any BitNet model | `<bitnet model>` | varies |

- **Qwen3 8B** follows *Let models think* in Settings: off, it answers straight away (quick
  questions, "what is the square root of 4761"); on, it works things out first. One download for both.
- **DeepSeek R1** is the one for dense reasoning: hard math, proofs, logic, and analysis that
  has to be worked through step by step. It always thinks, at length, and that long chain of
  thought is what makes it better than Qwen3 there. On simple questions it thinks far longer
  than it needs to; use Qwen3 with thinking off for those. It is not given the Python tool:
  given the calculator, it thought about it and never called it, so it works the math out in
  its reasoning.
- **Qwen2.5 Coder** for the code editor and code questions. It does not think, and is not
  given the Python tool.
- **Gemma 3 12B** reads attached pictures and reference pictures, and chats too. It and Qwen3
  8B call the Python tool.
- **FLUX.2 klein**: the 4B is the one to take with less than 18 GB of RAM, the 9B with more. The
  base 4B is slower (50 steps) and is the one LoRAs draw on.
- **Kokoro** reads answers aloud, beside the chat model. **Whisper** writes down what you say;
  only mlx-community's `-asr-` Whisper repos work (the others have no tokenizer).
- **BitNet** models are optional: 1.58-bit weights, small for their size class. None is
  recommended over the others; any one HikariLM runs will do (the Falcon3 and Falcon-E
  1.58-bit family, Microsoft's `microsoft/bitnet-b1.58-2B-4T`). They get no tools.
- Nomic's `nomic-embed-text-v1.5` (about 0.55 GB) is downloaded by HikariLM itself, for finding
  earlier chats; nothing to do.
- **Not recommended: gpt-oss.** gpt-oss is not recommended due to outdated and not tested.

## Rules that carry over

Macs with Apple Silicon only, everything in MLX, on the Mac. The models from their makers'
official weights (or mlx-community's and mflux-community's MLX copies), not fine-tunes. One large
model in memory at a time, RAM first, nothing written to disk until Save, and chats, notes and
settings keep working across each upgrade. No tests folder: what is checked is checked on the Mac.

## Open checks

- [ ] In the page: a question that wants a plot ("plot y = x² from -3 to 3") shows *Ran Python*
  with the code, the output and the picture, and the plot is still there after reopening the chat.
  An answer from 4.8 still shows its *Calculated:* line.
- [ ] Any new recommended chat model: try the Python tool with it first; one that does not call
  it well goes in `tools.POOR_CALLERS`.

## Notes from earlier releases

Found while building them, and worth knowing before changing them:

- **macOS 27 and scipy.** macOS 27's loader refuses scipy 1.15 and older (`__DATA/__thread_bss has
  a zero-fill section type, but offset field is not zero`,
  [scipy#25635](https://github.com/scipy/scipy/issues/25635)); on Python 3.10 that is the newest
  scipy pip can get. So `python-venv` (the Python tool) is made with Python 3.11 or newer and
  asks for `scipy>=1.16.3`, and so is `voice-venv`.
- **The Python tool's sandbox.** Homebrew and python.org Pythons are framework builds: their
  `python3.x` is a launcher that resolves its own folder and starts `Python.app` with
  posix_spawn, so the sandbox lets the folders above the venv be looked up and lets that
  interpreter start. Gemma 3 writes its call as a ```` ```tool_call ```` block of JSON.
- **Kokoro** (mlx-audio 0.6.4, misaki 0.9.4). mlx-audio pins `mlx==0.32.4`, wants transformers
  5.14 or newer, and does not install misaki. `misaki[en]` would bring torch, so misaki's English
  parts are installed one by one, with spaCy's `en_core_web_sm` 3.8.0 (misaki would otherwise
  download it on first use). misaki needs Python 3.12 or older: `voice-venv` is made with 3.11 or
  3.12. Kokoro's config has no model_type, so it is loaded with `model_type="kokoro"`; 24 kHz;
  the 28 American and British English voices are offered.
- **Whisper** (mlx-audio 0.6.4) reads its tokenizer with transformers' `WhisperProcessor` from the
  model's folder, so only mlx-community's `-asr-` repos work; `whisper-large-v3-turbo` and
  `whisper-small-mlx` are weights only. It hears 30 seconds at a time; parts that are silence are
  skipped, since Whisper makes words up for them.

## More LoRAs (v6)

LoRA files from this Mac, and whether LoRAs work on the distilled 4B. Hugging Face LoRAs on the base 4B are already in. A picture with a LoRA is drawn by
mflux's own command, `mflux-generate-flux2` (`mflux-generate-flux2-edit` with reference pictures),
from a JSON config passed with `--config-from-conf`, the way LoRAs were tried with mflux; it loads
the model with the LoRA for each picture. mflux bakes a LoRA into the model when it loads, and on
a saved q4 model the baked layers become 8-bit; DoRA files do not load.

### Already in, before 6.0 (not released)

- [x] **No trigger word needed.** A LoRA's trigger word is optional: empty puts nothing in front of
  the prompt, and the LoRA list and look-up say so.
- [x] **Prompt with tags**, for LoRAs trained on booru tags (`1girl, blue_hair, school_uniform`)
  rather than sentences, for whoever has a second model that turns sentences into tags. Each LoRA
  has the switch in Settings; with it on, the button by Options says *Make tags*, and the model
  the user picked under Settings, *Make tags with*, turns the box into tags to read and edit
  before drawing, the way Rewrite works (undo puts the words back). It is asked "Convert this to
  danbooru tags: …" (the form such taggers take), with what the vision model saw in a reference
  picture added, and its answer is cleaned to one comma-separated list. The trigger word, if any,
  still goes in front. Nothing is recommended or picked for it; it is the user's own model.
- [x] A tagger has to be an MLX model HikariLM lists, like any chat model; one converted by the
  user goes in `~/.hikarilm/models`. (BooruNL, for one, is a Qwen3.5 fine-tune: that is converted
  with mlx-vlm, since mlx-lm's converter drops Qwen3.5's vision part and HikariLM runs a model
  whose config has one with mlx-vlm.)
- [x] **LoRA pictures through mflux's command**, `mflux-generate-flux2 --config-from-conf <json>`,
  as above. Stopping a picture ends the command too (the worker runs in a process group of its own).
- [ ] On the Mac: draw with a LoRA (with and without a reference picture), and with a tag LoRA.

### v6.0.0: your own LoRA files, on the base 4B

- [ ] *Add a LoRA…* in the LoRAs part of Settings: pick a `.safetensors` (from the `/setup/lora` page, or
  anywhere); it is copied to `~/.hikarilm/loras/`, and the same checks as for Hugging Face ones refuse what cannot
  work on the base 4B. Its name is the file name; Remove deletes the copy.
- [ ] The trigger word comes from the file when its trainer saved one (kohya's `ss_tag_frequency`
  and the like in the header). A LoRA does not always have one (many style LoRAs need none), so
  with none found there is none, and the user can type one.
- [ ] Listed in the same picker as the Hugging Face ones; everything else as for those.
- [ ] README, `__version__ = "6.0.0"`, tag `v6.0.0`.

### v6.0.0-alpha: does a LoRA work on the distilled 4B? (a check, not a release)

- [ ] On the Mac: the same LoRAs (a Hugging Face one and a local one), prompts and seeds on the base
  4B (50 steps) and the distilled 4B (4 steps), at strengths 0.5, 0.8 and 1.0; look whether the
  distilled pictures show the LoRA as well as the base ones and stay clean (no burnt colors,
  no mush). Try a q4 and a q8 distilled 4B, since baking makes q4 layers 8-bit.
- [ ] Write down what was found here. If it works: v6.0.1. If not: LoRAs stay on the base 4B, and
  the line under the picker says why.

### v6.0.1: LoRAs on the distilled 4B (only if the check passes)

- [ ] The picker shows for the distilled 4B too, with the strength the check found best as the
  start. Still not on the 9B.
- [ ] README, `__version__ = "6.0.1"`, tag `v6.0.1`.
