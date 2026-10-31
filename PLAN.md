# HikariLM plan

- **v2**: Macs with Apple Silicon only. Released.
- **v3**: every model inside the HikariLM helper with Apple's MLX, starting with the
  big three from ready-made MLX weights. No Ollama, no sd-server, no LM Studio.
- **v4**: the models we convert ourselves, from their makers' official weights.

| Model | Version | Runs with | Weights | About |
|---|---|---|---|---|
| Qwen2.5-Coder 7B | v3 | mlx-lm | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| DeepSeek-R1 8B | v3 | mlx-lm | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Gemma 3 12B (reads pictures) | v3 | mlx-vlm | `mlx-community/gemma-3-12b-it-4bit` | 8 GB |
| Falcon3 10B BitNet | v4 | mlx-lm | converted by us from `tiiuae/Falcon3-10B-Instruct-1.58bit` | 4 GB |
| nomic-embed-text | v4 | our own MLX port | `nomic-ai/nomic-embed-text-v1.5` | 0.3 GB |
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
- [ ] Each one answers: `~/.hikarilm/venv/bin/mlx_lm.generate --model <repo> --prompt "hi"`
  (Gemma: `mlx_vlm.generate --model <repo> --image <a picture> --prompt "what is this"`)
- [ ] Note tokens per second for each, to compare with Ollama later.

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
- [ ] A run on the Mac with the real three: chat, stop, follow-up, thinking cut, a picture read.
- [x] `start.sh`: make `~/.hikarilm/venv` on the first run and install the packages, once.
- [ ] Speed check against the step 1 numbers. Try speculative decoding and KV cache quantization only if it is slow.

### 3. Remove the old way

- [ ] Delete `ollama.py`, and the Ollama parts of `imagegen.py`, `budget.py`, `keepalive.py`.
- [ ] Settings that only mattered for Ollama go.
- [ ] README: what `start.sh` installs, the three models and their sizes.

### 4. Release v3.0.0

- [ ] All tests pass, and each model has been used on the Mac: chat, code editor Apply, a picture read.
- [ ] `__version__ = "3.0.0"`, tag `v3.0.0`, release notes saying v2 users download the models again.

## v4.0.0: our own conversions, from official weights only

Each one is checked against its makers' own code before it is used.

- [ ] Falcon3 BitNet: `mlx_lm.convert --hf-path tiiuae/Falcon3-10B-Instruct-1.58bit`, kept in `~/.hikarilm/models/`.
  Compare a few answers with TII's reference (transformers) once. Add it to the model list.
- [ ] nomic-embed-text: `helper/embed.py`, a small NomicBERT in MLX loading Nomic's own weights.
  A test checks its vectors against Nomic's published ones. Notes are indexed again once, since the numbers change slightly.
- [ ] FLUX.2 klein: mflux with Black Forest Labs' weights, made 4-bit and saved once (`mflux-save`).
  Picture setup in the page downloads and converts it, with progress.
- [ ] Delete `sdserver.py` and `picsetup.py`, and the settings that only mattered for sd-server.
- [ ] All tests pass, and a picture made and a note found on the Mac.
- [ ] `__version__ = "4.0.0"`, tag `v4.0.0`.
