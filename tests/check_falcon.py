"""Check Falcon3 1.58-bit in MLX against TII's reference (transformers), on a Mac. Not part of the test suite.

HikariLM runs TII's own files (tiiuae/Falcon3-10B-Instruct-1.58bit) with mlx-lm, rounding each
layer's inputs the way transformers does (mlx_worker.round_bitnet_activations). This checks one
layer on random inputs, then the logits of whole prompts, MLX against PyTorch on the CPU.

    ~/.hikarilm/venv/bin/hf download tiiuae/Falcon3-10B-Instruct-1.58bit
    python3 -m venv /tmp/falcon-check
    /tmp/falcon-check/bin/pip install mlx mlx-lm "transformers>=4.48,<5" torch accelerate numpy
    /tmp/falcon-check/bin/python tests/check_falcon.py

It needs about 9 GB free: each side holds the 4 GB model, one after the other. PyTorch on the CPU
takes a minute or two.
"""

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # transformers' BitLinear is torch.compile'd; not needed here

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))

from helper import engine  # noqa: E402

REPO = "tiiuae/Falcon3-10B-Instruct-1.58bit"
MIN_LAYER = 0.9999     # cosine of one layer's outputs
MIN_COSINE = 0.99      # lowest cosine of the logits at any position
MIN_TOP1 = 0.95        # share of positions where both pick the same next token
PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that reverses a linked list.",
    "Explain in two sentences why the sky is blue.",
]


def cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    import numpy as np
    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        print(f"transformers {transformers.__version__} is too new for this check. Install 4.x:\n"
              f"    {sys.executable} -m pip install \"transformers>=4.48,<5\"")
        return 2
    engine.forget_scan()
    folder = engine.found().get(REPO)
    if folder is None:
        print(f"{REPO} is not downloaded, or HikariLM does not list it. Download it:\n"
              f"    ~/.hikarilm/venv/bin/hf download {REPO}")
        return 2

    import mlx.core as mx
    from mlx_lm import load
    from mlx_worker import round_bitnet_activations

    round_bitnet_activations()
    model, tokenizer = load(str(folder))
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
               for text in PROMPTS]

    layer = model.model.layers[0].self_attn.q_proj
    packed = np.array(layer.weight)
    weight_scale = np.array(layer.weight_scale.astype(mx.float32))
    x = np.random.default_rng(0).standard_normal((4, layer.in_features)).astype(np.float32)
    ours_layer = np.array(layer(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))
    ours = [np.array(model(mx.array(ids)[None])[0].astype(mx.float32)) for ids in prompts]
    del model, layer
    gc.collect()
    mx.clear_cache()

    import torch
    from transformers import AutoModelForCausalLM
    from transformers.integrations.bitnet import BitLinear

    reference = BitLinear(packed.shape[1], packed.shape[0] * 4, bias=False, dtype=torch.bfloat16)
    reference.weight.copy_(torch.from_numpy(packed))
    reference.weight_scale.copy_(torch.from_numpy(weight_scale).to(torch.bfloat16))
    with torch.no_grad():
        theirs_layer = reference(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
    layer_cos = min(cosine(a, b) for a, b in zip(ours_layer, theirs_layer))
    print(f"layer 0 q_proj, random inputs: lowest cosine {layer_cos:.6f}")

    hf = AutoModelForCausalLM.from_pretrained(str(folder), torch_dtype=torch.bfloat16)
    hf.eval()
    worst, agree, positions = 1.0, 0, 0
    for text, ids, mine in zip(PROMPTS, prompts, ours):
        with torch.no_grad():
            theirs = hf(torch.tensor([ids])).logits[0].float().numpy()
        cosines = [cosine(a, b) for a, b in zip(mine, theirs)]
        same = int((mine.argmax(-1) == theirs.argmax(-1)).sum())
        worst, agree, positions = min(worst, min(cosines)), agree + same, positions + len(ids)
        top_mine = [tokenizer.decode([t]) for t in np.argsort(-mine[-1])[:5]]
        top_theirs = [tokenizer.decode([t]) for t in np.argsort(-theirs[-1])[:5]]
        print(f"{text[:50]:50}  lowest cosine {min(cosines):.5f}, same next token {same}/{len(ids)}")
        print(f"    first words, MLX: {top_mine}\n    first words, TII: {top_theirs}")
    top1 = agree / positions
    ok = layer_cos >= MIN_LAYER and worst >= MIN_COSINE and top1 >= MIN_TOP1
    print(f"lowest cosine {worst:.5f}, same next token at {top1:.1%} of positions")
    print("OK: MLX matches TII's reference." if ok else "FAILED: MLX does not match TII's reference closely enough.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
