# HikariLM plan

- **v2**: Macs with Apple Silicon only. Released.
- **v3**: every model inside the HikariLM helper with Apple's MLX. Any MLX model works; the
  big three below are only suggestions for Macs with less than 18 GB of RAM. No Ollama, no LM Studio.
- **v3.1**: nomic-embed-text in MLX, and with it RAG: earlier chats and pictures found again.
- **v4**: the models we convert ourselves, from their makers' official weights.

| Model | Version | Runs with | Weights | About |
|---|---|---|---|---|
| Qwen2.5-Coder 7B | v3 | mlx-lm | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| DeepSeek-R1 8B | v3 | mlx-lm | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Gemma 3 12B (reads pictures) | v3 | mlx-vlm | `mlx-community/gemma-3-12b-it-4bit` | 8 GB |
| Falcon3 10B BitNet | v4 | mlx-lm | converted by us from `tiiuae/Falcon3-10B-Instruct-1.58bit` | 4 GB |
| nomic-embed-text | v3.1 | our own MLX port | `nomic-ai/nomic-embed-text-v1.5` | 0.3 GB |
| FLUX.2 klein 4B | v4 | mflux | `black-forest-labs/FLUX.2-klein-4B`, made 4-bit by us | 15 GB download, 4.6 GB kept |

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
- [ ] A run on the Mac with the real three: chat, stop, follow-up, thinking cut. (A picture read works.)
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

## v4.0.0: our own conversions, from official weights only

Each one is checked against its makers' own code before it is used.

- [ ] Falcon3 BitNet: `mlx_lm.convert --hf-path tiiuae/Falcon3-10B-Instruct-1.58bit`, kept in `~/.hikarilm/models/`.
  Compare a few answers with TII's reference (transformers) once. Add it to the model list.
- [ ] FLUX.2 klein: mflux with Black Forest Labs' weights, made 4-bit and saved once (`mflux-save`).
  Picture setup in the page downloads and converts it, with progress.
- [ ] Delete `sdserver.py` and `picsetup.py`, and the settings that only mattered for sd-server.
- [ ] All tests pass, and a picture made on the Mac.
- [ ] `__version__ = "4.0.0"`, tag `v4.0.0`.
