# HikariLM plan

What is coming, in order. What is already in is in the README and the release notes.

- **v5**: voice, on the Mac: answers read aloud with Kokoro (5.0), then speech to text with
  Whisper (5.5).
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
| Speech to text (from 5.5) | Whisper large-v3 turbo | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 1.6 GB |
| Speech to text, smaller (from 5.5) | Whisper small | `mlx-community/whisper-small-asr-fp16` | 0.5 GB |
| Reading aloud (from 5.0) | Kokoro 82M | `mlx-community/Kokoro-82M-bf16` | 0.4 GB |
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
- **Kokoro** reads answers aloud from 5.0, beside the chat model. **Whisper** writes down what you
  say from 5.5; only mlx-community's `-asr-` Whisper repos work (the others have no tokenizer).
- **BitNet** models are optional: 1.58-bit weights, small for their size class. None is
  recommended over the others; any one HikariLM runs will do (the Falcon3 and Falcon-E
  1.58-bit family, Microsoft's `microsoft/bitnet-b1.58-2B-4T`). They get no tools.
- Nomic's `nomic-embed-text-v1.5` (about 0.55 GB) is downloaded by HikariLM itself, for finding
  earlier chats; nothing to do.

## Rules that carry over

Macs with Apple Silicon only, everything in MLX, on the Mac. The models from their makers'
official weights (or mlx-community's and mflux-community's MLX copies), not fine-tunes. One large
model in memory at a time, RAM first, nothing written to disk until Save, and chats, notes and
settings keep working across each upgrade. No tests folder: what is checked is checked on the Mac.

## v4.9.0: Python in place of the calculator

4.8.0's calculator showed who calls tools: Qwen3 8B and Gemma 3 12B (4 of 4 sums, all right);
coders, DeepSeek R1 and BitNet models are left without it (`tools.offered`). 4.9.0 gives the
same models Python instead (`helper/python_tool.py`), with numpy, sympy, pandas, scipy and
matplotlib in `~/.hikarilm/python-venv`, in macOS's sandbox, 30 seconds, 1.25 GB, 25,000
characters of output, plots as pictures.

The venv is made with Python 3.11 or newer and asks for `scipy>=1.16.3`. macOS 27's loader
refuses scipy 1.15 and older (`__DATA/__thread_bss has a zero-fill section type, but offset
field is not zero`, [scipy#25635](https://github.com/scipy/scipy/issues/25635)), and on Python 3.10
that is the newest scipy pip can get. Setup imports the compiled parts once and says so if they
do not load; a venv made with an older Python is made again.

- [x] On the Mac: the packages and a plot work inside the sandbox, every way out is refused, and
  the limits hold.
- [x] Qwen3 8B and Gemma 3 12B call the tool for the 4 sums and get them right, now by writing code. Gemma 3 writes its call as a
  ```` ```tool_call ```` block of JSON, which is read as a call since issue 10.
- [ ] In the page: a question that wants a plot ("plot y = x² from -3 to 3") shows *Ran Python*
  with the code, the output and the picture, and the plot is still there after reopening the chat.
  An answer from 4.8 still shows its *Calculated:* line.
- [ ] gpt-oss 20B: its calls (Harmony, `to=functions.python`) are read, but its thinking
  channels are not split off yet.
- [ ] Any new recommended chat model: try the Python tool with it first; one that does not call
  it well goes in `tools.POOR_CALLERS`.
- [x] README, `__version__ = "4.9.0"`, tag `v4.9.0`.

## Voice

Have answers read aloud (5.0), and speak a prompt instead of typing it (5.5). All of it runs on
the Mac, like the rest: the browser's own speech recognition is not used, because Chrome's sends
the audio to Google.

**Packages.** mlx-audio does both directions: speech to text (Whisper) and text to speech
(Kokoro), so 5.0 and 5.5 share it. It pins its own MLX (`mlx==0.32.4` in 0.6.4) and wants
transformers 5.14 or later, so it has a venv of its own, `~/.hikarilm/voice-venv`, made by
`hikarilm restart` once a voice model is downloaded, the way `pictures-venv` is made for mflux. mlx-whisper is the fallback for speech to text if
mlx-audio's Whisper falls short; it pulls in torch and numba.

**Models.**

| For | Version | Suggested | Notes |
|---|---|---|---|
| Reading aloud | v5.0 | `mlx-community/Kokoro-82M-bf16` | Small, Apache 2.0; 28 English voices. |
| Speech to text | v5.5 | `mlx-community/whisper-large-v3-turbo-asr-fp16` | Many languages. mlx-audio reads Whisper's tokenizer from the model's folder, so only the `-asr-` repos work (they have it; `whisper-large-v3-turbo` and `whisper-small-mlx` are weights only). `whisper-small-asr-fp16` or a 4-bit one for less memory. |

Only the models themselves, as with pictures: Whisper from mlx-community (OpenAI's weights
converted), Kokoro from mlx-community or hexgrad, the way the start window already says what it
leaves out.

### v5.0.0: reading answers aloud

In `helper/voice.py`, `helper/voice_worker.py`, `web/js/spoken.js` and `web/js/listen.js`. What was found, from mlx-audio 0.6.4 and misaki 0.9.4:

- mlx-audio pins `mlx==0.32.4` and wants transformers 5.14 or newer, and does not install misaki,
  which its Kokoro needs for English. `misaki[en]` would bring torch (for a transformer parser
  that `trf=False` does not use), so misaki's English parts are installed one by one: misaki,
  num2words, spacy, phonemizer-fork, espeakng-loader (espeak-ng's library and data, for words
  misaki does not know), and spaCy's `en_core_web_sm` 3.8.0, which misaki would otherwise
  download the first time it reads. Plus `scipy>=1.16.3` for macOS 27.
- misaki needs Python older than 3.13, and scipy that macOS 27 can load needs 3.11 or newer:
  the voice venv is made with 3.11 or 3.12 (found in Homebrew, python.org or the PATH), or the
  start window says to `brew install python@3.12`.
- Kokoro's config has no model_type, so the worker loads it with `model_type="kokoro"`. It
  makes 24 kHz sound, and its voices are in the repo's `voices/` folder; the American (a) and
  British (b) English ones are offered (28 of them), since the others need misaki's Japanese
  or Chinese parts or espeak-ng's rules for their language.

- [x] Works on the Mac, and says variables right.
- [x] **Listen** button under every answer with words, and a setting to read each answer when it
  finishes. The page sends `/api/speak` the lines to say, a voice and a speed; the helper streams
  each line's sound as Kokoro makes it, so the first is heard while the rest are made. Stop (or
  Esc) ends it. One answer is read at a time: Listen on another stops the first.
- [x] **What is read** (`spoken.js`): the answer's words. Code blocks are named ("There is a
  Python code block here, 3 lines long."), small tables are read row by row and big ones named,
  math is said in words ("x squared plus 1 equals 0", "the integral from 0 to 4 of…") or named
  when it is too much to say ("There is an equation here"), links by their text and bare ones by
  their site, pictures in an answer by their alt text, plots from the Python tool as "There is a
  plot above.", symbols in words (°, %, ≤, →), and emoji, markup and thinking never. Answers that
  are only a picture get no Listen.
- [x] **Memory.** Kokoro (82M parameters) stays loaded beside the chat model, like the notes model:
  it never counts as the one large model, and it is never unloaded for one. It loads the first
  time something is read aloud, in the voice worker, which keeps running for it.
- [x] **Setup**: Kokoro on the welcome page under *Voice*, with its command and Copy; the
  start window says how to get it, and `hikarilm restart` sets it up once it is there.
- [x] **Settings**: *Reading aloud*: the voice, a sample to hear it, the speed, and *Read answers
  aloud when they finish*.
- [x] README, `__version__ = "5.0.0"`, tag `v5.0.0`.

### v5.5.0: speech to text

In `helper/voice.py`, `helper/voice_worker.py --listen` and `web/js/dictate.js`; not released.
What was found, from mlx-audio 0.6.4: its Whisper loads the tokenizer with transformers'
`WhisperProcessor` from the model's own folder, so only repos that have `tokenizer.json` and
`preprocessor_config.json` work: mlx-community's `-asr-` ones (`whisper-large-v3-turbo-asr-fp16`,
1.6 GB; `whisper-small-asr-fp16`, 0.5 GB; 4-, 5-, 6- and 8-bit ones too). The older
`whisper-large-v3-turbo` and `whisper-small-mlx` are weights only; the start window names any it
finds and says which to get instead. It takes 16 kHz sound as a number array, and goes in the
voice venv 5.0 made.

- [x] **Microphone button** by Attach, in Chat, the code editor and picture mode. Click to start,
  click to stop; Esc cancels a recording, and stops the writing down (keeping what was written).
  A level meter and a timer while it listens. The words go into the message box where the cursor
  was, with spaces around them, to read and fix; Send stays yours.
- [x] **Asking for the microphone.** The page asks only when the microphone button is pressed.
  The first time, it says the browser will ask and that the sound stays on the Mac. Already
  blocked: it says how to allow it in that browser (Safari, Chrome, Edge, Firefox), and in macOS
  (System Settings > Privacy & Security > Microphone), without asking again. Refused by macOS,
  no microphone, or one another app holds: each says what to do. Opened at an address other than
  127.0.0.1 or localhost, where browsers refuse the microphone over http: it says so.
- [x] **No ffmpeg.** The page records with MediaRecorder, decodes with Web Audio and resamples to
  16 kHz mono with an OfflineAudioContext, and sends 16-bit PCM. Limit: 10 minutes a recording;
  it stops itself there.
- [x] **`/api/transcribe`**: the PCM as the body, `?model=` and `?language=` (none to detect).
  The worker writes it down 30 seconds at a time, cut at the quietest tenth of a second near
  each 30, skips parts that are silence (Whisper makes words up for them), and streams `{"text"}`
  for each part, then `{"done", "language"}`. Stop ends it and keeps what was written so far.
- [x] **Memory.** Whisper stays loaded beside the chat model, like Kokoro, and leaves after 5
  minutes unused.
- [x] **Setup**: Whisper turbo and the smaller Whisper on the welcome page under *Voice*, beside
  Kokoro, each with its command and Copy. Until one is there, the microphone button is dimmed
  and says *Get a voice model…*, with the command and a link to the welcome page.
- [x] **Settings**: *Speech to text*: *Listen with* (the Whisper) and the language (or detect).
- [ ] On the Mac: speak a sentence and a minute of speech into it in Safari and in Chrome, and
  allow, refuse and allow the microphone again.
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
