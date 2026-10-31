"""Check Falcon3 1.58-bit in MLX against TII's reference (transformers), on a Mac. Not part of the test suite.

HikariLM runs TII's own files (tiiuae/Falcon3-10B-Instruct-1.58bit) with mlx-lm, rounding each
layer's inputs the way transformers does (mlx_worker.round_bitnet_activations). This checks one
layer on random inputs, then the logits of whole prompts, MLX against PyTorch on the CPU.

    ~/.hikarilm/venv/bin/hf download tiiuae/Falcon3-10B-Instruct-1.58bit
    python3 -m venv /tmp/falcon-check
    /tmp/falcon-check/bin/pip install mlx mlx-lm "transformers>=4.48,<5" torch accelerate numpy
    /tmp/falcon-check/bin/python tests/check_falcon.py

It needs about 6 GB free: the MLX model is freed before PyTorch loads its own, so only one
model is in memory at a time (the float32 run makes PyTorch's embeddings twice as large).
PyTorch on the CPU takes a few minutes.

Passing means MLX is within 0.99 cosine and 95% same next tokens of TII's bf16 run, or no further
from exact (TII's own code in float32) than TII's bf16 run is itself.
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
    tii = [run(hf, ids) for ids in prompts]

    # TII's own code in float32: how far its bf16 run is from exact, the most MLX can be expected to match.
    hf.float()
    for module in hf.modules():
        if isinstance(module, BitLinear):
            module.dtype = torch.float32
    exact = [run(hf, ids) for ids in prompts]

    print("\nMLX against TII (bf16):")
    worst, top1 = compare(prompts, ours, tii, tokenizer, "MLX")
    print("\nTII (bf16) against TII (float32), the yardstick:")
    floor_worst, floor_top1 = compare(prompts, tii, exact, tokenizer, "TII bf16")
    print("\nMLX against TII (float32):")
    exact_worst, exact_top1 = compare(prompts, ours, exact, tokenizer, "MLX")

    close = worst >= MIN_COSINE and top1 >= MIN_TOP1
    within = exact_worst >= floor_worst - 0.01 and exact_top1 >= floor_top1 - 0.02
    ok = layer_cos >= MIN_LAYER and (close or within)
    print(f"\nMLX to TII bf16: lowest cosine {worst:.5f}, same next token {top1:.1%}")
    print(f"MLX to exact: lowest cosine {exact_worst:.5f}, same next token {exact_top1:.1%}; "
          f"TII's own bf16 to exact: {floor_worst:.5f}, {floor_top1:.1%}")
    print("OK: MLX matches TII's reference." if ok else "FAILED: MLX does not match TII's reference closely enough.")
    return 0 if ok else 1


def run(hf, ids):
    import torch

    with torch.no_grad():
        return hf(torch.tensor([ids])).logits[0].float().numpy()


def compare(prompts, first, second, tokenizer, name) -> tuple[float, float]:
    """Lowest cosine of the logits at any position, and the share of positions with the same next token."""
    import numpy as np

    worst, agree, positions = 1.0, 0, 0
    for text, ids, a, b in zip(PROMPTS, prompts, first, second):
        cosines = [cosine(x, y) for x, y in zip(a, b)]
        low = int(np.argmin(cosines))
        same = int((a.argmax(-1) == b.argmax(-1)).sum())
        worst, agree, positions = min(worst, cosines[low]), agree + same, positions + len(ids)
        print(f"  {text[:46]:46}  lowest cosine {cosines[low]:.5f} at token {low} "
              f"({tokenizer.decode([ids[low]])!r}), same next token {same}/{len(ids)}")
        print(f"      next words, {name}: {[tokenizer.decode([t]) for t in np.argsort(-a[-1])[:5]]}")
        print(f"      next words, other: {[tokenizer.decode([t]) for t in np.argsort(-b[-1])[:5]]}")
    return worst, agree / positions


if __name__ == "__main__":
    sys.exit(main())
