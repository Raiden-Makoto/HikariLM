"""Check a BitNet model in MLX against its makers' reference (transformers), on a Mac. Not part of the test suite.

HikariLM runs BitNet models with mlx-lm, loading and computing each form the way transformers
does (mlx_worker.load_bitnet). This checks one layer on random inputs, then the logits of whole
prompts, MLX against PyTorch on the CPU.

    ~/.hikarilm/venv/bin/hf download tiiuae/Falcon3-10B-Instruct-1.58bit
    python3 -m venv /tmp/falcon-check
    /tmp/falcon-check/bin/pip install mlx mlx-lm "transformers>=4.52,<5" torch accelerate numpy
    /tmp/falcon-check/bin/python tests/check_bitnet.py tiiuae/Falcon3-10B-Instruct-1.58bit

Any BitNet repo works the same way, for example microsoft/bitnet-b1.58-2B-4T,
microsoft/bitnet-b1.58-2B-4T-bf16, tiiuae/Falcon-E-3B-Instruct or HF1BitLLM/Llama3-8B-1.58-100B-tokens.
mlx-community's copies with 4- or 8-bit embeddings cannot be read by transformers, so name the
repo they were made from second, downloaded too:

    /tmp/falcon-check/bin/python tests/check_bitnet.py mlx-community/bitnet-b1.58-2B-4T-4bit microsoft/bitnet-b1.58-2B-4T

Only one model is in memory at a time: the MLX model is freed before PyTorch loads its own (the
float32 run makes PyTorch's embeddings twice as large). PyTorch on the CPU takes a few minutes.

Passing means MLX is within 0.99 cosine and 95% same next tokens of the reference's bf16 run, or
no further from exact (the reference's own code in float32) than its bf16 run is itself, with
every next token it picks differently a near tie in the exact run (within 10 points of probability).
"""

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # transformers' BitNet layers are torch.compile'd; not needed here

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))

from helper import engine  # noqa: E402

DEFAULT = "tiiuae/Falcon3-10B-Instruct-1.58bit"
LAYER = "model.layers.0.self_attn.q_proj"
MIN_LAYER = 0.9999     # cosine of one layer's outputs
MIN_COSINE = 0.99      # lowest cosine of the logits at any position
MIN_TOP1 = 0.95        # share of positions where both pick the same next token
MAX_TIE = 0.10         # where MLX picks another token than exact, exact may favour its own by this much at most
PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that reverses a linked list.",
    "Explain in two sentences why the sky is blue.",
]


def cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def downloaded(repo: str) -> Path | None:
    engine.forget_scan()
    folder = engine.found().get(repo)
    if folder is None:
        print(f"{repo} is not downloaded, or HikariLM does not list it. Download it:\n"
              f"    ~/.hikarilm/venv/bin/hf download {repo}")
    return folder


def file_weight(folder: Path, name: str):
    """One tensor as the repo stores it, read with torch."""
    from safetensors import safe_open

    for path in sorted(folder.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(name)


def reference_layer(folder: Path, settings: dict):
    """The transformers layer the reference runs for LAYER, with the repo's own weights."""
    import torch
    from transformers.integrations.bitnet import AutoBitLinear, BitLinear, unpack_weights

    weight = file_weight(folder, LAYER + ".weight")
    packed = weight.dtype == torch.uint8
    out_features = weight.shape[0] * 4 if packed else weight.shape[0]
    if settings.get("linear_class") != "autobitlinear":
        layer = BitLinear(weight.shape[1], out_features, bias=False, dtype=torch.bfloat16)
        layer.weight.copy_(weight)
        layer.weight_scale.copy_(file_weight(folder, LAYER + ".weight_scale").to(torch.bfloat16))
        return layer
    online = settings.get("quantization_mode") == "online"
    layer = AutoBitLinear(weight.shape[1], out_features, bias=False, dtype=torch.bfloat16, online_quant=online)
    with torch.no_grad():
        layer.weight.copy_(unpack_weights(weight, torch.bfloat16) if packed else weight.to(torch.bfloat16))
        if not online:
            layer.weight_scale.copy_(file_weight(folder, LAYER + ".weight_scale").to(torch.bfloat16))
    return layer.to(torch.bfloat16)


def main() -> int:
    import numpy as np
    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        print(f"transformers {transformers.__version__} is too new for this check. Install 4.x:\n"
              f"    {sys.executable} -m pip install \"transformers>=4.52,<5\"")
        return 2
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    folder = downloaded(repo)
    if folder is None:
        return 2

    from mlx_worker import bitnet_config, load_bitnet

    settings = bitnet_config(str(folder))
    if settings is None:
        print(f"{repo} is not a BitNet model.")
        return 2
    if "group_size" in settings and len(sys.argv) < 3:
        print(f"{repo} keeps its embeddings in MLX's own 4- or 8-bit form, which transformers cannot read.\n"
              f"Name the repo it was made from second, for example:\n"
              f"    {sys.executable} {sys.argv[0]} {repo} microsoft/bitnet-b1.58-2B-4T")
        return 2
    reference = downloaded(sys.argv[2]) if len(sys.argv) > 2 else folder
    if reference is None:
        return 2
    form = ("Microsoft's AutoBitLinear, " + settings.get("quantization_mode", "offline")
            if settings.get("linear_class") == "autobitlinear" else "TII's BitLinear")
    print(f"{repo}: {form}")

    import mlx.core as mx

    model, tokenizer = load_bitnet(str(folder), settings)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
               for text in PROMPTS]
    layer = model.model.layers[0].self_attn.q_proj
    x = np.random.default_rng(0).standard_normal((4, layer.in_features)).astype(np.float32)
    ours_layer = np.array(layer(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))
    ours = [np.array(model(mx.array(ids)[None])[0].astype(mx.float32)) for ids in prompts]
    del model, layer
    gc.collect()
    mx.clear_cache()

    import torch
    from transformers import AutoModelForCausalLM
    from transformers.integrations.bitnet import BitLinear

    with torch.no_grad():
        theirs_layer = reference_layer(folder, settings)(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
    layer_cos = min(cosine(a, b) for a, b in zip(ours_layer, theirs_layer))
    print(f"layer 0 q_proj, random inputs: lowest cosine {layer_cos:.6f}")

    hf = AutoModelForCausalLM.from_pretrained(str(reference), torch_dtype=torch.bfloat16)
    hf.eval()
    bf16 = [run(hf, ids) for ids in prompts]

    # The reference in float32: how far its bf16 run is from exact, the most MLX can be expected to match.
    # transformers refuses .float() on quantized models; torch's own leaves the packed uint8 weights as they are.
    torch.nn.Module.float(hf)
    for module in hf.modules():
        if isinstance(module, BitLinear):
            module.dtype = torch.float32
    exact = [run(hf, ids) for ids in prompts]

    print("\nMLX against the reference (bf16):")
    worst, top1, _ = compare(prompts, ours, bf16, tokenizer, "MLX")
    print("\nThe reference in bf16 against float32, the yardstick:")
    floor_worst, floor_top1, _ = compare(prompts, bf16, exact, tokenizer, "bf16")
    print("\nMLX against the reference (float32):")
    exact_worst, exact_top1, gap = compare(prompts, ours, exact, tokenizer, "MLX")

    close = worst >= MIN_COSINE and top1 >= MIN_TOP1
    within = exact_worst >= floor_worst - 0.01 and gap <= MAX_TIE
    ok = layer_cos >= MIN_LAYER and (close or within)
    print(f"\nMLX to the reference in bf16: lowest cosine {worst:.5f}, same next token {top1:.1%}")
    print(f"MLX to exact: lowest cosine {exact_worst:.5f}, same next token {exact_top1:.1%}, "
          f"widest gap where they differ {gap:.1%}; the reference's own bf16 to exact: "
          f"{floor_worst:.5f}, {floor_top1:.1%}")
    print("OK: MLX matches the reference." if ok else "FAILED: MLX does not match the reference closely enough.")
    return 0 if ok else 1


def run(hf, ids):
    import torch

    with torch.no_grad():
        return hf(torch.tensor([ids])).logits[0].float().numpy()


def compare(prompts, first, second, tokenizer, name) -> tuple[float, float, float]:
    """Lowest cosine of the logits at any position, the share of positions with the same next token,
    and the widest gap where they differ: how much likelier `second` found its own pick than
    `first`'s. A small gap is a near tie, which any change in rounding can flip."""
    import numpy as np

    worst, agree, positions, widest = 1.0, 0, 0, 0.0
    for text, ids, a, b in zip(PROMPTS, prompts, first, second):
        cosines = [cosine(x, y) for x, y in zip(a, b)]
        low = int(np.argmin(cosines))
        picks_a, picks_b = a.argmax(-1), b.argmax(-1)
        same = int((picks_a == picks_b).sum())
        worst, agree, positions = min(worst, cosines[low]), agree + same, positions + len(ids)
        print(f"  {text[:46]:46}  lowest cosine {cosines[low]:.5f} at token {low} "
              f"({tokenizer.decode([ids[low]])!r}), same next token {same}/{len(ids)}")
        for at in np.nonzero(picks_a != picks_b)[0]:
            p = np.exp(b[at] - b[at].max())
            p /= p.sum()
            gap = float(p[picks_b[at]] - p[picks_a[at]])
            widest = max(widest, gap)
            print(f"      after token {at} ({tokenizer.decode([ids[at]])!r}): other picks "
                  f"{tokenizer.decode([picks_b[at]])!r} at {p[picks_b[at]]:.1%}, {name} picks "
                  f"{tokenizer.decode([picks_a[at]])!r}, which other gives {p[picks_a[at]]:.1%}")
        print(f"      next words, {name}: {[tokenizer.decode([t]) for t in np.argsort(-a[-1])[:5]]}")
        print(f"      next words, other: {[tokenizer.decode([t]) for t in np.argsort(-b[-1])[:5]]}")
    return worst, agree / positions, widest


if __name__ == "__main__":
    sys.exit(main())
