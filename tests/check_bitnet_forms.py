"""Check every BitNet form HikariLM loads against transformers, on a Mac, with no downloads. Not part of the test suite.

It writes tiny random models (about 100 KB each) in each form a BitNet repo comes in, then runs
each through HikariLM's MLX loader (mlx_worker.load_bitnet) and through transformers, and compares
the logits:

- TII's BitLinear, llama-shaped (Falcon3, Falcon-E, Llama3-8B-1.58);
- Microsoft's AutoBitLinear, packed (bitnet-b1.58-2B-4T);
- AutoBitLinear with full bf16 weights, made ternary as it runs (bitnet-b1.58-2B-4T-bf16);
- an mlx-community copy with 4-bit embeddings, compared with the same model whose embeddings
  are those 4-bit values in bf16 (transformers cannot read MLX's 4-bit form).

    /tmp/falcon-check/bin/pip install "transformers>=4.52,<5"
    /tmp/falcon-check/bin/python tests/check_bitnet_forms.py

A form passes when MLX's logits are within 0.995 cosine of transformers' bf16 run, or no further
from its float32 run than transformers' own bf16 run is.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # transformers' BitNet layers are torch.compile'd; not needed here

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "helper"))

HIDDEN, HEADS, KV_HEADS, INNER, LAYERS, VOCAB = 64, 4, 2, 128, 2, 512
MIN_COSINE = 0.995
COMMON = {"hidden_size": HIDDEN, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
          "head_dim": HIDDEN // HEADS, "intermediate_size": INNER, "num_hidden_layers": LAYERS,
          "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
          "attention_bias": False, "mlp_bias": False, "torch_dtype": "bfloat16"}
PROJECTIONS = {"self_attn.q_proj": (HIDDEN, HIDDEN), "self_attn.k_proj": (KV_HEADS * HIDDEN // HEADS, HIDDEN),
               "self_attn.v_proj": (KV_HEADS * HIDDEN // HEADS, HIDDEN), "self_attn.o_proj": (HIDDEN, HIDDEN),
               "mlp.gate_proj": (INNER, HIDDEN), "mlp.up_proj": (INNER, HIDDEN), "mlp.down_proj": (HIDDEN, INNER)}


def packed_bytes(rng, out_features, in_features):
    """Random ternary weights as TII and Microsoft ship them: four 2-bit fields (0, 1, 2) a byte."""
    import numpy as np

    fields = rng.integers(0, 3, size=(4, out_features // 4, in_features)).astype(np.uint8)
    return fields[0] | (fields[1] << 2) | (fields[2] << 4) | (fields[3] << 6)


def tensors(rng, form):
    import numpy as np

    out = {"model.embed_tokens.weight": rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32),
           "model.norm.weight": (1 + 0.1 * rng.standard_normal(HIDDEN)).astype(np.float32)}
    if form == "tii":
        out["lm_head.weight"] = (0.05 * rng.standard_normal((VOCAB, HIDDEN))).astype(np.float32)
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            out[prefix + norm + ".weight"] = (1 + 0.1 * rng.standard_normal(HIDDEN)).astype(np.float32)
        if form != "tii":
            out[prefix + "self_attn.attn_sub_norm.weight"] = (1 + 0.1 * rng.standard_normal(HIDDEN)).astype(np.float32)
            out[prefix + "mlp.ffn_sub_norm.weight"] = (1 + 0.1 * rng.standard_normal(INNER)).astype(np.float32)
        for name, (rows, cols) in PROJECTIONS.items():
            if form == "microsoft-bf16":
                out[prefix + name + ".weight"] = (0.05 * rng.standard_normal((rows, cols))).astype(np.float32)
                continue
            out[prefix + name + ".weight"] = packed_bytes(rng, rows, cols)
            # TII's scale divides the output, Microsoft's multiplies it; both near what real models use.
            scale = rng.uniform(15, 40) if form == "tii" else rng.uniform(0.02, 0.06)
            out[prefix + name + ".weight_scale"] = np.array([scale], np.float32)
    return out


def config(form):
    if form == "tii":
        return {**COMMON, "model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_act": "silu",
                "tie_word_embeddings": False,
                "quantization_config": {"modules_to_not_convert": ["lm_head"], "quant_method": "bitnet"}}
    return {**COMMON, "model_type": "bitnet", "architectures": ["BitNetForCausalLM"], "hidden_act": "relu2",
            "tie_word_embeddings": True,
            "quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear",
                                    "quantization_mode": "online" if form == "microsoft-bf16" else "offline"}}


def write_torch(folder: Path, settings: dict, weights: dict) -> Path:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(settings))
    save_file({name: torch.from_numpy(value) if value.dtype == np.uint8 else torch.from_numpy(value).to(torch.bfloat16)
               for name, value in weights.items()}, str(folder / "model.safetensors"), metadata={"format": "pt"})
    return folder


def mlx_copy(root: Path, settings: dict, weights: dict) -> tuple[Path, Path]:
    """An mlx-community style copy with 4-bit embeddings, and the same model in bf16 for transformers."""
    import mlx.core as mx
    import numpy as np

    quantized, plain = {}, dict(weights)
    for name in ("model.embed_tokens", "lm_head"):
        value = mx.array(weights[name + ".weight"]).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(value, group_size=64, bits=4)
        quantized.update({name + ".weight": packed, name + ".scales": scales, name + ".biases": biases})
        plain[name + ".weight"] = np.array(mx.dequantize(packed, scales, biases, group_size=64, bits=4).astype(mx.float32))
    for name, value in weights.items():
        if name not in quantized:
            quantized[name] = mx.array(value) if value.dtype == np.uint8 else mx.array(value).astype(mx.bfloat16)
    four_bit = {"group_size": 64, "bits": 4, **settings["quantization_config"]}
    folder = root / "mlx-4bit"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({**settings, "quantization": four_bit, "quantization_config": four_bit}))
    mx.save_safetensors(str(folder / "model.safetensors"), quantized, metadata={"format": "mlx"})
    return folder, write_torch(root / "mlx-4bit-reference", settings, plain)


def cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def lowest(first, second) -> float:
    return min(cosine(a, b) for a, b in zip(first, second))


def check(name: str, folder: Path, reference: Path, ids) -> bool:
    import gc

    import mlx.core as mx
    import numpy as np
    import torch
    from mlx_worker import bitnet_config, load_bitnet
    from transformers import AutoModelForCausalLM
    from transformers.integrations.bitnet import BitLinear

    model, _ = load_bitnet(str(folder), bitnet_config(str(folder)), tokenizer=False)
    ours = np.array(model(mx.array(ids)[None])[0].astype(mx.float32))
    del model
    gc.collect()

    hf = AutoModelForCausalLM.from_pretrained(str(reference), torch_dtype=torch.bfloat16)
    hf.eval()
    with torch.no_grad():
        bf16 = hf(torch.tensor([ids])).logits[0].float().numpy()
        torch.nn.Module.float(hf)
        for module in hf.modules():
            if isinstance(module, BitLinear):
                module.dtype = torch.float32
        exact = hf(torch.tensor([ids])).logits[0].float().numpy()
    to_bf16, to_exact, floor = lowest(ours, bf16), lowest(ours, exact), lowest(bf16, exact)
    same = float((ours.argmax(-1) == bf16.argmax(-1)).mean())
    ok = to_bf16 >= MIN_COSINE or to_exact >= floor - 0.01
    print(f"{'OK    ' if ok else 'FAILED'} {name:15}  MLX to transformers bf16 {to_bf16:.5f} (same next token {same:.0%}), "
          f"MLX to float32 {to_exact:.5f}, transformers bf16 to float32 {floor:.5f}")
    return ok


def main() -> int:
    import numpy as np
    import transformers

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    if major >= 5 or (major, minor) < (4, 52):
        print(f"transformers {transformers.__version__} cannot run every BitNet form. Install 4.52 to 4.x:\n"
              f"    {sys.executable} -m pip install \"transformers>=4.52,<5\"")
        return 2
    rng = np.random.default_rng(0)
    ids = rng.integers(0, VOCAB, size=24).tolist()
    results = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for form in ("tii", "microsoft", "microsoft-bf16"):
            folder = write_torch(root / form, config(form), tensors(rng, form))
            results.append(check(form, folder, folder, ids))
        four_bit, reference = mlx_copy(root, config("tii"), tensors(rng, "tii"))
        results.append(check("mlx-4bit (tii)", four_bit, reference, ids))
    print("OK: every BitNet form matches transformers." if all(results) else "FAILED: see the forms above.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
