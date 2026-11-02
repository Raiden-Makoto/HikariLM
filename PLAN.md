# HikariLM plan

What is coming, in order. What is already in is in the README and the release notes.

- **After 4.8**: find out on the Mac which models call the calculator.
- **v5**: voice, on the Mac: speech to text with Whisper (5.0), then answers read aloud with
  Kokoro (5.5).
- **v6**: LoRA files from this Mac, and LoRAs on the distilled 4B if a check shows they work.

## Recommended models

Any MLX model works; these are the ones to start with. Get one with
`~/.hikarilm/venv/bin/hf download <repo>`. Sizes are the download.

| For | Model | Repo | Download |
|---|---|---|---|
| Everyday chat, thinking or not | Qwen3 8B | `mlx-community/Qwen3-8B-4bit` | 4.6 GB |
| Harder problems, always thinks | DeepSeek R1 8B | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Code | Qwen2.5 Coder 7B | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| Reading pictures | Gemma 3 12B | `mlx-community/gemma-3-12b-it-4bit` | 8.1 GB |
| Drawing pictures | FLUX.2 klein 4B | `mlx-community/FLUX.2-Klein-4B-4bit` | 4.6 GB |
| Drawing, more detail | FLUX.2 klein 9B | `mlx-community/flux2-klein-9b-4bit` | 9.5 GB |
| Drawing with LoRAs | FLUX.2 klein base 4B | `mflux-community/flux-2-klein-base-4b-mflux-q4` | 4.6 GB |
| Speech to text (from 5.0) | Whisper large-v3 turbo | `mlx-community/whisper-large-v3-turbo` | 1.6 GB |
| Speech to text, smaller (from 5.0) | Whisper small | `mlx-community/whisper-small-mlx` | 0.5 GB |
| Reading aloud (from 5.5) | Kokoro 82M | `mlx-community/Kokoro-82M-bf16` | 0.4 GB |
| Optional: BitNet | Falcon3 10B 1.58-bit | `tiiuae/Falcon3-10B-Instruct-1.58bit` | 4.0 GB |

- **Qwen3 8B** follows *Let models think* in Settings: off, it answers straight away (quick
  questions, "what is the square root of 4761"); on, it works things out first. One download for both.
- **DeepSeek R1** always thinks, which is what it is for: math, logic and puzzles. On trivial
  questions it thinks far longer than it needs to; use Qwen3 with thinking off for those, or the
  calculator (4.8) if it calls it.
- **Qwen2.5 Coder** for the code editor and code questions. It does not think.
- **Gemma 3 12B** reads attached pictures and reference pictures, and chats too.
- **FLUX.2 klein**: the 4B is the one to take with less than 18 GB of RAM, the 9B with more. The
  base 4B is slower (50 steps) and is the one LoRAs draw on.
- **Whisper** and **Kokoro** are listed for when voice comes in; HikariLM does not use them yet.
- **BitNet** models are optional: 1.58-bit weights, small for their size class. The rest of the
  Falcon3 and Falcon-E 1.58-bit family and Microsoft's `microsoft/bitnet-b1.58-2B-4T` work too.
- Nomic's `nomic-embed-text-v1.5` (about 0.55 GB) is downloaded by HikariLM itself, for finding
  earlier chats; nothing to do.

## Rules that carry over

Macs with Apple Silicon only, everything in MLX, on the Mac. The models from their makers'
official weights (or mlx-community's and mflux-community's MLX copies), not fine-tunes. One large
model in memory at a time, RAM first, nothing written to disk until Save, and chats, notes and
settings keep working across each upgrade. Real-model checks run on the Mac in `tests/check_*.py`;
a check downloads what it needs and deletes it after, unless it was there before. Unit tests
download nothing.

## After 4.8.0: which models call the calculator

4.8.0 gives the chat models a scientific calculator. Models whose chat template takes `tools`
get it there; the rest get it described in the system message, in the `<tool_call>` format. The
helper reads calls in Qwen's, Mistral's, DeepSeek's and Gemma's formats. Whether each model
actually calls it is not known yet.

- [ ] Run `python3 tests/check_tools.py` on the Mac with the suggested chat models (Qwen3 8B,
  DeepSeek R1 8B, Qwen2.5 Coder, Gemma 3 12B, Falcon3, and gpt-oss if it is there). It asks
  each one four sums and says whether it called the calculator and got them right. Write down
  what it found here.
- [ ] For a model that writes calls in a format the helper does not read yet (gpt-oss's
  Harmony is the likely one), read it too.
- [ ] A model that does not call it reliably, or gets worse answers with it, is left without
  it: a list in the helper, so it never sees the description.
- [ ] Recommended models: note which ones use the calculator.

### Later: full Python (no version yet)

The model writes a whole script (statistics, dates, a table of data) and the helper runs it. This
needs a real sandbox on the Mac: no network, no files outside a scratch folder, and limits on
time and memory. Planned once the calculator works and it is clear which models call tools well.

## Voice

Speak a prompt instead of typing it (5.0), and have answers read aloud (5.5). All of it runs on
the Mac, like the rest: the browser's own speech recognition is not used, because Chrome's sends
the audio to Google.

**Packages.** mlx-audio does both directions: speech to text (Whisper) and text to speech
(Kokoro), so 5.0 and 5.5 share it. When last looked at (0.6.3) it pinned its own MLX
(`mlx==0.32.4`) and wanted transformers 5.14 or later, so it gets a venv of its own,
`~/.hikarilm/voice-venv`, made by `hikarilm restart` once a voice model is downloaded, the way
`pictures-venv` is made for mflux. mlx-whisper is the fallback for speech to text if
mlx-audio's Whisper falls short; it pulls in torch and numba.

**Models.**

| For | Version | Suggested | Notes |
|---|---|---|---|
| Speech to text | v5.0 | `mlx-community/whisper-large-v3-turbo` | Many languages. Any MLX Whisper works (`whisper-small-mlx` for less memory). |
| Reading aloud | v5.5 | `mlx-community/Kokoro-82M-bf16` | Small, Apache 2.0, several voices. |

Only the models themselves, as with pictures: Whisper from mlx-community (OpenAI's weights
converted), Kokoro from mlx-community or hexgrad, the way the start window already says what it
leaves out.

### v5.0.0: speech to text

- [ ] Check on the Mac before building on it: mlx-audio's newest version and what it pins, the
  exact Whisper repos it loads, its text against mlx-whisper's on the same clip, how long the
  turbo takes for a minute of speech, and whether it can share the chat models' venv after all.
  Write down what was found here.
- [ ] **Microphone button** by Attach, in Chat, the code editor and picture mode. Click to start,
  click to stop (Esc cancels); a level meter and a timer while it listens. The words go into
  the message box, added where the cursor is, to read and fix; Send stays yours. Microphone
  access works on 127.0.0.1 without HTTPS. When it is refused, the page says how to allow it.
- [ ] **No ffmpeg.** The page records with MediaRecorder, decodes with Web Audio and resamples to
  16 kHz mono with an OfflineAudioContext, and sends 16-bit PCM. Whisper reads it as a number
  array. Limit: 10 minutes a recording.
- [ ] **`/api/transcribe`**: takes the PCM, the model and the language (or none, to detect it),
  streams `{"text"}` as it is written, then `{"done"}`. Runs in a voice worker
  (`helper/voice_worker.py`) in the voice venv, started like the picture worker, with a fake
  worker for the tests. Stop ends it and keeps what was written so far.
- [ ] **Memory.** Find out on the Mac whether Whisper turbo can stay loaded beside the chat model,
  like the notes model, or must take its place, like FLUX.2 klein. If it takes its place, the
  page says so each time, the way reading pictures says what was loaded and what left.
- [ ] **Setup page**, `/setup/voice`, like the picture one: Whisper turbo first, the smaller
  Whisper under a fold, each with its command and Copy, and where it stands. *Get a voice
  model…* in the microphone button's place until one is there.
- [ ] **Settings**: *Listen with* (the speech-to-text model) and the language (or detect).
- [ ] `tests/check_voice.py` on the Mac: macOS's own `say` speaks a known sentence to a file and
  `afconvert` makes it 16 kHz, Whisper writes it back through the real worker, and the words are
  compared. It downloads the small Whisper if no Whisper is there, and deletes it after (unless it
  was there before), like the LoRA in `check_flux.py`. Nothing to download for the unit tests.
- [ ] README, `__version__ = "5.0.0"`, tag `v5.0.0`.

### v5.5.0: reading answers aloud

- [ ] Check on the Mac what Kokoro needs besides its weights: mlx-audio turns text into sounds
  with misaki, which may download a spaCy model or want espeak-ng for some languages. The setup
  page lists whatever that turns out to be, and `hikarilm restart` installs it in the voice venv.
- [ ] **Listen** button on every answer, beside Copy, and a setting to read each answer when it
  finishes. `/api/speak` takes text, a voice and a speed and streams audio a sentence at a time,
  so it starts before the whole answer is spoken; Stop ends it. One answer is read at a time:
  Listen on another stops the first.
- [ ] **What is read**: the answer's words. Code blocks, tables and LaTeX are skipped or said
  briefly ("a Python code block"), links by their text, and thinking is never read. Pictures are
  not read; their prompt is not either.
- [ ] **Memory.** Kokoro (82M parameters) stays loaded beside the chat model, like the notes model:
  it never counts as the one large model, and it is never unloaded for one. It loads the first
  time something is read aloud (at start when *Read answers aloud* is on), in the voice worker,
  which keeps running for it.
- [ ] **Setup page**: Kokoro added to `/setup/voice` under *Reading aloud*, with its command, Copy
  and what else it needs.
- [ ] **Settings**: *Read aloud with*, the voice (with a short sample to hear it), the speed, and
  *Read answers aloud when they finish*.
- [ ] `tests/check_voice.py` on the Mac also speaks a sentence with Kokoro and writes it back with
  Whisper, and compares. Kokoro is downloaded for the check and deleted after, unless it was
  there before.
- [ ] README, `__version__ = "5.5.0"`, tag `v5.5.0`.

## More LoRAs (v6)

The rest of the LoRA work, after voice: LoRA files from this Mac, and whether LoRAs work on the
distilled 4B. Hugging Face LoRAs on the base 4B are already in. mflux bakes a LoRA into the model
when it loads, and on a saved q4 model the baked layers become 8-bit; DoRA files do not load.

### v6.0.0: your own LoRA files, on the base 4B

- [ ] *Add a LoRA…* in the LoRAs part of Settings: pick a `.safetensors` (from the `/setup/lora` page, or
  anywhere); it is copied to `~/.hikarilm/loras/`, and the same checks as for Hugging Face ones refuse what cannot
  work on the base 4B. Its name is the file name; Remove deletes the copy.
- [ ] The trigger word comes from the file when its trainer saved one (kohya's `ss_tag_frequency`
  and the like in the header); otherwise the user types it once.
- [ ] Listed in the same picker as the Hugging Face ones; everything else as for those.
- [ ] Tests, README, `__version__ = "6.0.0"`, tag `v6.0.0`.

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
- [ ] Tests, README, `__version__ = "6.0.1"`, tag `v6.0.1`.
