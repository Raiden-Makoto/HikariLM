# HikariLM plan

- **v2**: Macs with Apple Silicon only. Released.
- **v3**: every model inside the HikariLM helper with Apple's MLX. Any MLX model works; the
  big three below are only suggestions for Macs with less than 18 GB of RAM. No Ollama, no LM Studio.
- **v3.1**: nomic-embed-text in MLX, and with it RAG: earlier chats and pictures found again.
- **v3.5**: Falcon3 1.58-bit from TII's own files, checked against TII's code. Released.
- **v4**: the models from their makers' official weights, and pictures with FLUX.2 klein through mflux.
  Only the models themselves: no LoRA, no fine-tunes. Released.

| Model | Version | Runs with | Weights | About |
|---|---|---|---|---|
| Qwen2.5-Coder 7B | v3 | mlx-lm | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| DeepSeek-R1 8B | v3 | mlx-lm | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Gemma 3 12B (reads pictures) | v3 | mlx-vlm | `mlx-community/gemma-3-12b-it-4bit` | 8 GB |
| Falcon3 10B BitNet | v3.5 | mlx-lm | `tiiuae/Falcon3-10B-Instruct-1.58bit`, TII's own files | 4 GB |
| nomic-embed-text | v3.1 | our own MLX port | `nomic-ai/nomic-embed-text-v1.5` | 0.3 GB |
| FLUX.2 klein 4B | v4 | mflux | `mlx-community/FLUX.2-Klein-4B-4bit` (any FLUX.2 klein works) | 4.6 GB |

Rules that carry over: one large model in memory at a time, RAM first, nothing
written to disk until Save, chats and notes keep working across each upgrade.

## v2.0.0: Mac only

- [x] Delete the Linux and WSL paths; `start.py` refuses anything but Apple Silicon.
- [x] `__version__ = "2.0.0"`, tag `v2.0.0`.
- [x] v2.0.1: LM Studio (llmster) support removed.

## v3.0.0: MLX, with the big three

### 1. Download the big three again (on the Mac)

Ollama's files are GGUF and MLX cannot use them, so these come again from
Hugging Face. About 17 GB.

- [x] `python3 -m venv ~/.hikarilm/venv && ~/.hikarilm/venv/bin/pip install mlx-lm mlx-vlm`
- [x] `~/.hikarilm/venv/bin/hf download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
- [x] `~/.hikarilm/venv/bin/hf download mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit`
- [x] `~/.hikarilm/venv/bin/hf download mlx-community/gemma-3-12b-it-4bit`
- [x] Each one answers, and Gemma reads a picture.
- [x] Speed: replies are fast on the Mac; no numbers needed.

Keep the Ollama copies until v3 is released; `ollama rm` them after.

### 2. The engine

`helper/engine.py` finds the MLX models `hf download` put in the Hugging Face
cache and runs them in `helper/mlx_worker.py`, a process started with the venv's
Python (the helper itself stays standard library only). One model at a time;
unloading ends the worker, which frees all of its memory. Replies come back as
Ollama's chat lines, so the rest of the helper is unchanged.

- [x] `engine.py` and `mlx_worker.py`: load, stream (chat template, stop, R1's thinking on and off), unload.
- [x] The last chat's prompt cache is kept, so a follow-up only reads the new message (mlx-lm models).
- [x] `models.py` stays the one door: MLX names go to `engine.py`, the rest to Ollama until step 3.
- [x] `budget.py`: the model's shape from its `config.json` (with Gemma 3's left-out defaults).
- [x] `keepalive.py` and `router.py`: unchanged; they unload MLX models through `models.py`.
- [x] Gemma's picture reading through mlx-vlm (pictures go to the worker as files).
- [x] Model list in the page: MLX models first, with vision and thinking from their configs.
- [x] Tests with a fake worker (`tests/fake_mlx_worker.py`) and the three real `config.json` files.
- [x] A run on the Mac with the real three: chat, stop, follow-up, thinking cut. (A picture read works.)
- [x] `start.sh`: make `~/.hikarilm/venv` on the first run and install the packages, once.
- [x] Speed check: fast enough. Speculative decoding is not needed.
- [x] KV cache quantization, a Settings choice (8-bit by default, 4-bit, or full): mlx-lm's `kv_bits`,
  so the same memory holds a longer context. Gemma 3 (mlx-vlm) and sliding-window caches stay full.
  8-bit checked on the Mac. Settings warns that 4-bit may cost some accuracy.
- [x] Model names are shown short everywhere ("DeepSeek R1"), from one table in `models.js` and `helper/names.py`.

### 3. Remove the old way

- [x] Delete `ollama.py`, and the Ollama parts of `imagegen.py`, `budget.py`, `keepalive.py`, `router.py`,
  `models.py`, `server.py` and `start.py`. Errors are `engine.ModelError`.
- [x] Settings that only mattered for Ollama go: "Use earlier chats and pictures" (nomic ran in Ollama; back in
  v3.1), Ollama's context defaults, Ollama's picture models.
- [x] README: what `start.sh` installs, the three suggested models and their sizes.
- [x] v3.0.1: any MLX model is listed, full precision too (mlx-community, a repo named for MLX, or
  `~/.hikarilm/models`); the docs say the three are suggestions for under 18 GB of RAM.

### 4. Release v3.0.0

- [x] All tests pass, and the models have been used on the Mac.
- [x] `__version__ = "3.0.0"`, tag `v3.0.0`, release notes saying v2 users download the models again.

## v3.1.0: nomic-embed-text and RAG

`notes.py` (the index of earlier chats and picture descriptions) and `documents.excerpts` (the passages of a
long file closest to the question) take an `embed` function; in v3.0 there was none, so notes were off
and long files fell back to shared words.

- [x] `helper/nomic_mlx.py`: nomic-embed-text v1.5 in MLX from Nomic's own weights, run by the worker with
  `--embed` in a process of its own. Small (0.3 GB), so it sits beside the large model and never makes it
  leave; it is held to the same keepalive. `start.py` downloads its files once (no ONNX).
- [x] Fake-worker tests for the embed path; notes and document vectors carry an embedder stamp.
- [x] On the Mac: `tests/check_nomic.py` matches Nomic's PyTorch vectors (cosine at least 0.999). Nomic's
  code needs transformers 4 ([#5](https://github.com/Raiden-Makoto/HikariLM/issues/5)).
- [x] Turn the notes back on: the indexer at start and on save, `/api/search`, `/api/notes`, and the Settings
  switch. Notes made with Ollama's nomic are indexed again once.
- [x] Long files: the closest passages by meaning again.
- [x] `__version__ = "3.1.0"`, tag `v3.1.0`, once the Mac check passes.

## v4.0.0: official weights only

Each one is checked against its makers' own code before it is used.

- [x] Falcon3 BitNet needs no conversion: TII ships ternary weights packed four to a byte with one scale per
  matrix, the layout mlx-lm's BitLinear reads (`quant_method: bitnet`). HikariLM lists that repo as it is.
- [x] mlx-lm multiplies a layer's inputs unrounded; TII's reference (transformers' BitLinear) rounds each
  token's inputs to 8-bit levels first, as the model was trained. The worker does the same
  (`round_bitnet_activations`).
- [x] On the Mac: `tests/check_falcon.py` (now `check_bitnet.py`) matches TII's reference (one layer, then whole-prompt logits).
  MLX is closer to TII's code in float32 than TII's own bf16 run is; the few next tokens it picks
  differently are near ties ([#6](https://github.com/Raiden-Makoto/HikariLM/issues/6)).
- [x] Add it to the suggested models once it passes.
- [x] Released on its own as v3.5.0, ahead of the rest of v4.

### Every BitNet model, not just Falcon3

- [x] TII's BitLinear (all Falcon3 and Falcon-E 1.58-bit, Llama3-8B-1.58): as above.
- [x] Microsoft's AutoBitLinear, offline (`microsoft/bitnet-b1.58-2B-4T`): inputs rounded in float32,
  the output multiplied by the weight scale, as transformers does.
- [x] AutoBitLinear, online (`-bf16` master weights): made ternary and packed once when loaded, the
  same rounding transformers' WeightQuant does on every call.
- [x] mlx-community's 4- and 8-bit copies: mlx-lm skips its BitNet layers when a config has a
  `quantization` block, so llama-shaped ones did not load; `load_bitnet` puts them in first.
- [x] On the Mac: `tests/check_bitnet_forms.py` passes. Released in v3.5.1, with a "Report a problem" button in Settings. It needs no downloads: it writes a tiny random
  model in each form and compares MLX with transformers on it. (`tests/check_bitnet.py` does the same
  for a real repo, for anyone who has one.)

### Pictures with mflux

- [x] Every FLUX.2 klein mflux runs (4B, 9B, 9B-KV, base 4B, base 9B), from Black Forest Labs or
  mlx-community's copies at any bit width; no conversion of our own. `flux.py` names each one the
  way mflux does, from the repo's name, checked against the transformer's blocks in the weights.
- [x] Only the models themselves: fine-tunes, LoRA adapters and other makers' re-uploads are left out,
  with a note saying why.
- [x] `flux_worker.py` runs in `~/.hikarilm/pictures-venv` (mflux pins its own MLX and wants torch and
  transformers 5), installed by `start.py` once a FLUX.2 model is downloaded. It shares the one worker
  slot with the chat models, so the one-large-model rule holds, and the keepalive frees it.
- [x] Delete `sdserver.py` and `picsetup.py`, the setup window, the picture server address, and the
  picture settings.
- [x] Tests with a fake picture worker.
- [x] On the Mac: FLUX.2 klein makes pictures in the page. (`python3 tests/check_flux.py` checks mflux's
  table and draws with each downloaded model; it makes the picture venv if needed.)
- [x] `__version__ = "4.0.0"`, tag `v4.0.0`. Released.

## v4.0.1: chat works again

- [x] 4.0.0 dropped the `KEEP_ALIVE` import from `server.py`, so every chat and picture description
  stopped with "The helper hit a bug". Fixed, with `tests/test_server.py`: a chat and a description
  through the server, and a check that every name the helper uses is defined. Released.

## v4.1.0: prompt rewriting

FLUX.2 klein draws what a prompt describes far better than what it names: "Usagi from Chiikawa"
gets a generic rabbit. Black Forest Labs suggests rewriting prompts for FLUX.2.

- [x] A Settings choice, "Rewrite prompts with": a chat model (a general one before a coder by
  default) or "Don't rewrite". `/api/rewrite` and `helper/rewrite.py`; thinking is off for it.
  Models that think before every answer (DeepSeek R1, QwQ, gpt-oss, Magistral, "reasoning" and
  "thinking" models) are never offered and the helper refuses them (`thinking.always_thinks`).
  With a picture attached, the vision model describes how it looks (`/api/describe` with
  `purpose: "look"`, kept apart from the description written for code) and that goes into the rewrite.
- [x] The rewrite streams into the message box, with a note above it and "Use my words"; the next
  send draws what is in the box. The picture record keeps `request` and the drawn `prompt`, and
  the picture shows the drawn prompt under "Drawn from a rewritten prompt". Vary keeps both.
- [x] The chat model leaves memory before the picture model loads (one large model at a time):
  tested, the picture's loading line names the chat model it unloads.
- [x] Settings, "Read pictures with": which model that can see reads pictures. Reading and
  rewriting say which model is loaded and which one left (`{"hikarilm": "loaded"}` lines), and
  what the vision model saw shows on the note and stays with the picture (`reference`).
- [x] Tried on the Mac: the rewrite plus the picture read draws closer to the character than
  without, but not exactly. Text cannot carry a face; 4.2.0 gives the picture model the picture.
- [x] A short warning under the message box in picture mode, and in the picture setting: "Picture
  models may not recognize niche characters, people or places. Describe how they look or attach a
  reference image." Picture mode shows the attach button again for this (4.0.0 hides it). Rewriting helps only if the chat model knows them; it can be wrong too, so the warning
  stays with rewriting on.
- [x] `__version__ = "4.1.0"`, tag `v4.1.0`. Released.

## v4.2.0: drawing from the reference picture itself

mflux's `Flux2KleinEdit` takes the same FLUX.2 klein weights plus `image_paths` and conditions on the
reference pictures' pixels (9B-KV caches them for the later steps). The read and the rewrite stay:
the picture model gets the rewritten prompt and the pictures.

- [x] `flux_worker.py`: with references, `Flux2KleinEdit.generate_image(..., image_paths=...)`;
  without, `Flux2Klein` as now. `Flux2KleinEdit` cannot draw without a reference and the two
  cannot share loaded weights, so the worker holds one and reloads when a picture needs the other.
  The pictures go to the worker as temporary files and are deleted after.
- [x] `/api/generate` takes `references` (at most 4 PNG, JPEG, WebP or GIF, 10 MB each); the
  record keeps which were used (by hash, not a copy), so Vary draws from them again while they
  are still in the chat. Chats are saved without pictures, so after a reload Vary says to attach
  them again. A reference no longer needs a model that can see: without one, the read is skipped.
- [x] The rewrite is told the picture model sees the reference too (`shown`), so it describes the
  scene and what changes rather than repeating every detail of the character.
- [x] Tried on the Mac: the reference carries the character, but new things in the request can go
  missing ("Usagi eating a burger" drew Usagi with no burger). Drawing from a reference works like
  editing it, so the model leans to copying the picture as it is. With a reference, the rewrite
  now starts with the action and every object asked for, leaves out the character's look (the
  picture has it) and does not carry over the reference's pose or background. The general
  instructions also say every object named must be in the prompt, shown being used.
- [x] `tests/check_flux.py` draws a fox, then the same fox in a meadow from the first picture.
  Run on the Mac.
- [x] Edit and Delete on every message you sent (`web/js/prompt-actions.js`). Edit opens it in
  place; sending replaces it and everything after it, asking first when that removes more than
  its own reply. Delete removes it and its reply, and the picture file if the reply was a picture;
  an empty chat is deleted. Tried in the browser with the fake workers.
- [x] Rewriting only when asked: a Rewrite button by Options. Send draws the words in the box as
  they are; a rewrite done first is drawn as edited. Hidden when Settings says Don't rewrite.
- [x] Changing the picture above: after a picture, a message that reads like a change ("add a
  sunset to the background", "make it night", "remove the hat") sends `change: <picture id>`.
  The helper gives that picture to FLUX.2 klein first, ahead of any reference, at its own size; the
  record keeps `changed`, and Vary changes the same picture again. A note above the message box
  says "Changes the picture above" or "Draws a new picture" and turns it the other way. No
  Rewrite for a change: it is said in the user's own words.
- [x] Deleting a chat deletes what was read from its pictures: saved chats keep each picture's
  hash (never the picture), and the descriptions, the look descriptions and their notes go too,
  unless another chat has the same picture. Chats saved before 4.2.0 have no hashes, so theirs stay.
- [x] Looks of a reference ("-look") were notes in 4.1, so they could turn up as context in any
  chat. They are not notes any more, and the helper removes the old ones when it starts.
- [x] `__version__ = "4.2.0"`, tag `v4.2.0`. Released.

## v4.2.1: a setup page for pictures

- [x] *Get a picture model…* (and *How to set it up* in Settings) goes to a page of its own,
  `/setup/pictures` (`web/setup-pictures.html`, `js/setup.js`, `css/setup.css`): what draws and
  where, then three steps, each ticked when done. It looks again every 5 seconds until it is
  ready, so it follows the download, and then *Start drawing* goes back to Chat with FLUX.2 klein
  picked. Download a model: the 4B (recommended under 18 GB of RAM)
  and the 9B (`mlx-community/flux2-klein-9b-4bit`, more detail), each with its command and a
  Copy button, and the other FLUX.2 klein models under a fold. Run ./start.sh again for mflux.
  Draw: Options, Rewrite, references, changing a picture, the niche-character warning. A line at
  the top says where it stands (nothing yet, downloaded but no mflux, ready), and Check again
  looks again and refreshes the model list.
- [x] The start window and the "not downloaded" error name the 9B too, so the 4B command does not
  read as the only one.
- [x] With the 4B and the 9B both downloaded, the list says "FLUX.2 klein 4B" and "FLUX.2 klein 9B".
- [x] `__version__ = "4.2.1"`, tag `v4.2.1`. Released.

## v5.0.0: voice

Speak a prompt instead of typing it, and have answers read aloud. All of it runs on the Mac, like
the rest: the browser's own speech recognition is not used, because Chrome's sends the audio to
Google.

**Packages.** mlx-audio (0.6.3) does both directions: speech to text (Whisper) and text
to speech (Kokoro). It pins its own MLX (`mlx==0.32.4`) and wants transformers 5.14 or later, so
it gets a venv of its own, `~/.hikarilm/voice-venv`, made by `./start.sh` once a voice model is
downloaded, the way `pictures-venv` is made for mflux. mlx-whisper is the fallback for speech to
text if mlx-audio's Whisper falls short; it pulls in torch and numba.

**Models.**

| For | Suggested | Notes |
|---|---|---|
| Speech to text | `mlx-community/whisper-large-v3-turbo` | Many languages. Any MLX Whisper works (`whisper-small-mlx` for less memory). |
| Reading aloud | `mlx-community/Kokoro-82M-bf16` | Small, Apache 2.0, several voices. |

Check on the Mac before building on them: the exact repos mlx-audio loads, the Whisper output
against mlx-whisper's on the same clip, and whether either can share the chat models' venv after
all.

- [ ] **Microphone button** by Attach, in Chat, the code editor and picture mode. Click to start,
  click to stop (Esc cancels); a level meter and a timer while it listens. The words go into
  the message box, added where the cursor is, to read and fix; Send stays yours. Microphone
  access works on 127.0.0.1 without HTTPS. When it is refused, the page says how to allow it.
- [ ] **No ffmpeg.** The page records with MediaRecorder, decodes with Web Audio and resamples to
  16 kHz mono with an OfflineAudioContext, and sends 16-bit PCM. Whisper reads it as a number
  array. Limit: 10 minutes a recording.
- [ ] **`/api/transcribe`**: takes the PCM and the model, streams `{"text"}` as it is written, then
  `{"done"}`. Runs in a voice worker (`helper/voice_worker.py`) in the voice venv, started like
  the picture worker, with a fake worker for the tests.
- [ ] **Read aloud**: a Listen button on every answer, and a setting to read each answer when it
  finishes. `/api/speak` takes text and a voice and streams audio a sentence at a time, so it
  starts before the whole answer is spoken; Stop ends it. Code blocks, tables and LaTeX are
  skipped or said briefly ("a Python code block"), and thinking is never read.
- [ ] **Memory.** Kokoro (82M parameters) stays loaded beside the chat model, like the notes model:
  it never counts as the one large model, and it is never unloaded for one. It loads the first
  time something is read aloud (at start when *Read answers aloud* is on), in the voice worker,
  which keeps running for it. Whisper turbo is bigger; find out on the Mac whether it can stay
  too or must take the chat model's place, like FLUX.2 klein. If it takes its place, the page
  says so each time, the way reading pictures says what was loaded and what left.
- [ ] Check on the Mac what Kokoro needs besides its weights: mlx-audio turns text into sounds
  with misaki, which may download a spaCy model or want espeak-ng for some languages. The setup
  page lists whatever that turns out to be, and `./start.sh` installs it with the voice venv.
- [ ] **Setup page**, `/setup/voice`, like the picture one: Whisper turbo first, the smaller
  Whisper under a fold, Kokoro for reading aloud, each with its command and Copy, and
  where it stands. *Get a voice model…* in the microphone button's place until one is there.
- [ ] **Settings**: *Listen with* (the speech-to-text model) and language (or detect), *Read
  aloud with*, the voice, the speed, and *Read answers aloud when they finish*.
- [ ] `tests/check_voice.py` on the Mac: speaks a sentence with Kokoro, writes it back with Whisper,
  and compares. Nothing to download for the unit tests.
- [ ] README, `__version__ = "5.0.0"`, tag `v5.0.0`.
