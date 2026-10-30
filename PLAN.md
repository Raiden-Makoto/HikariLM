# HikariLM v2 plan

v2 runs every model inside the HikariLM helper with Apple's MLX. No Ollama, no
sd-server, no LM Studio. Macs with Apple Silicon only.

| Model | Runs with | Weights | About |
|---|---|---|---|
| Qwen2.5-Coder 7B | mlx-lm | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| DeepSeek-R1 8B | mlx-lm | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Gemma 3 12B (reads pictures) | mlx-vlm | `mlx-community/gemma-3-12b-it-4bit` | 8 GB |
| Falcon3 10B BitNet | mlx-lm | converted by us from `tiiuae/Falcon3-10B-Instruct-1.58bit` | 4 GB |
| nomic-embed-text | our own MLX port | `nomic-ai/nomic-embed-text-v1.5` | 0.3 GB |
| FLUX.2 klein 4B | mflux | `black-forest-labs/FLUX.2-klein-4B`, made 4-bit by us | 15 GB download, 4.6 GB kept |

Rules that carry over: one large model in memory at a time, RAM first, nothing
written to disk until Save, chats and notes keep working across the upgrade.

## 0. Mac only

- [x] Delete the Linux and WSL paths; `start.py` refuses anything but Apple Silicon.

## 1. Download the big three again (on the Mac)

Ollama's files are GGUF and MLX cannot use them, so these come again from
Hugging Face. About 17 GB.

- [ ] `python3 -m venv ~/.hikarilm/venv && ~/.hikarilm/venv/bin/pip install mlx-lm mlx-vlm`
- [ ] `~/.hikarilm/venv/bin/hf download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
- [ ] `~/.hikarilm/venv/bin/hf download mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit`
- [ ] `~/.hikarilm/venv/bin/hf download mlx-community/gemma-3-12b-it-4bit`
- [ ] Each one answers: `~/.hikarilm/venv/bin/mlx_lm.generate --model <repo> --prompt "hi"`
  (Gemma: `mlx_vlm.generate --model <repo> --image <a picture> --prompt "what is this"`)
- [ ] Note tokens per second for each, to compare with Ollama later.

Keep the Ollama copies until step 4 is done; `ollama rm` them after.

## 2. The engine, with the big three

New `helper/engine.py`: loads a model with mlx-lm or mlx-vlm, streams a chat
reply, stops on request, and frees the memory when the model is swapped or idle.

- [ ] `engine.py`: load, stream (chat template, stop, thinking split for R1), unload.
- [ ] Keep each chat's prompt cache between turns, so a follow-up only reads the new message.
- [ ] `models.py` stays the one door the rest of the helper uses; it calls `engine.py` instead of Ollama and LM Studio.
- [ ] `budget.py`: context length from free unified memory and the model's own size, not Ollama's numbers.
- [ ] `keepalive.py` and `router.py`: one big model at a time, unloaded after it sits idle.
- [ ] Gemma's picture reading (`descriptions.py`) through mlx-vlm.
- [ ] Model list in the page: the installed MLX models, with which can see and which think.
- [ ] `start.sh`: make `~/.hikarilm/venv` on the first run and install the packages, once.
- [ ] Tests with a fake engine, as the fake Ollama does today. Then a run on the Mac with the real three.
- [ ] Speed check against the step 1 numbers. Try speculative decoding and KV cache quantization only if it is slow.

## 3. The tricky three, from official weights only

Each one is checked against its makers' own code before it is used.

- [ ] Falcon3 BitNet: `mlx_lm.convert --hf-path tiiuae/Falcon3-10B-Instruct-1.58bit`, kept in `~/.hikarilm/models/`.
  Compare a few answers with TII's reference (transformers) once. Add it to the model list.
- [ ] nomic-embed-text: `helper/embed.py`, a small NomicBERT in MLX loading Nomic's own weights.
  A test checks its vectors against Nomic's published ones. Notes are indexed again once, since the numbers change slightly.
- [ ] FLUX.2 klein: mflux with Black Forest Labs' weights, made 4-bit and saved once (`mflux-save`).
  Picture setup in the page downloads and converts it, with progress.

## 4. Remove the old way

- [ ] Delete `ollama.py`, `llmster.py`, `sdserver.py`, `picsetup.py` and the Ollama parts of `imagegen.py`, `budget.py`, `keepalive.py`.
- [ ] Settings that only mattered for Ollama or sd-server go.
- [ ] README: Mac only, what `start.sh` installs, the six models and their sizes.

## 5. Release v2.0.0

- [ ] All tests pass, and every model has been used on the Mac: chat, code editor Apply, a picture read, a picture made, a note found.
- [ ] `__version__ = "2.0.0"`, tag `v2.0.0`, release notes saying v1 users download the models again.
