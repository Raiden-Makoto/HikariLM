"""nomic-embed-text v1.5 in MLX, loading Nomic's own weights (nomic-ai/nomic-embed-text-v1.5).

Only the worker imports this; the helper never imports MLX. It follows Nomic's
NomicBertModel (modeling_hf_nomic_bert.py in nomic-ai/nomic-bert-2048) for the
settings v1.5 uses: post-norm blocks, one Wqkv matrix laid out (q, k, v) x heads
x head_dim, rotary positions on the whole head (base 1000, halves rotated, not
pairs), a SiLU-gated MLP (fc11 * silu(fc12), then fc2), token type 0 added to
every token, and no biases. Then the model card's pooling: the mean over real
tokens, a layer norm without weights, and unit length.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import mlx.nn as nn

MAX_TOKENS = 2048    # what v1.5 was trained on; past it Nomic scales the rotary base, which is not done here
POOL_EPS = 1e-5      # torch.nn.functional.layer_norm's default, as the model card calls it
WEIGHTS = mx.float16


class _Embeddings(nn.Module):
    def __init__(self, vocab: int, types: int, dim: int) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, dim)
        self.token_type_embeddings = nn.Embedding(types, dim)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.Wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class _MLP(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.fc11 = nn.Linear(dim, inner, bias=False)
        self.fc12 = nn.Linear(dim, inner, bias=False)
        self.fc2 = nn.Linear(inner, dim, bias=False)


class _LayerNorm(nn.Module):
    """Computed in float32: the weights are float16, and eps 1e-12 is below what float16 holds."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.fast.layer_norm(x.astype(mx.float32), self.weight.astype(mx.float32),
                                 self.bias.astype(mx.float32), self.eps)
        return out.astype(x.dtype)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, inner: int, eps: float, base: float) -> None:
        super().__init__()
        self.heads, self.head_dim, self.base = heads, dim // heads, base
        self.attn = _Attention(dim)
        self.mlp = _MLP(dim, inner)
        self.norm1 = _LayerNorm(dim, eps)
        self.norm2 = _LayerNorm(dim, eps)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch, length, dim = x.shape
        qkv = self.attn.Wqkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        x = self.norm1(x + self.attn.out_proj(out))
        return self.norm2(x + self.mlp.fc2(self.mlp.fc11(x) * nn.silu(self.mlp.fc12(x))))


class _Encoder(nn.Module):
    def __init__(self, blocks: list[_Block]) -> None:
        super().__init__()
        self.layers = blocks


class NomicBert(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim, heads = config["n_embd"], config["n_head"]
        inner, eps = config["n_inner"], config["layer_norm_epsilon"]
        if config.get("activation_function") != "swiglu" or config.get("prenorm") or config.get("rotary_emb_interleaved") \
                or float(config.get("rotary_emb_fraction", 1.0)) != 1.0 or config.get("rotary_scaling_factor"):
            raise ValueError("this is not the nomic-embed-text v1.5 layout")
        self.embeddings = _Embeddings(config["vocab_size"], config["type_vocab_size"], dim)
        self.emb_ln = _LayerNorm(dim, eps)
        self.encoder = _Encoder([_Block(dim, heads, inner, eps, float(config["rotary_emb_base"]))
                                 for _ in range(config["n_layer"])])

    def __call__(self, ids: mx.array, keep: mx.array) -> mx.array:
        x = self.embeddings.word_embeddings(ids) + self.embeddings.token_type_embeddings.weight[0]
        x = self.emb_ln(x)
        mask = keep[:, None, None, :]
        for block in self.encoder.layers:
            x = block(x, mask)
        weights = keep[..., None].astype(mx.float32)
        pooled = (x.astype(mx.float32) * weights).sum(axis=1) / mx.maximum(weights.sum(axis=1), 1e-9)
        pooled = mx.fast.layer_norm(pooled, None, None, POOL_EPS)
        return pooled / mx.maximum(mx.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)


class Embedder:
    """Loads the model once; embed() turns texts (with their search_ prefix) into unit vectors."""

    def __init__(self, path: str) -> None:
        from transformers import BertTokenizerFast

        with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.model = NomicBert(config)
        weights = mx.load(os.path.join(path, "model.safetensors"))
        self.model.load_weights([(name, value.astype(WEIGHTS)) for name, value in weights.items()], strict=True)
        mx.eval(self.model.parameters())
        self.tokenizer = BertTokenizerFast.from_pretrained(path)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS)
        ids = mx.array(encoded["input_ids"], dtype=mx.int32)
        keep = mx.array(encoded["attention_mask"]).astype(mx.bool_)
        vectors = self.model(ids, keep)
        mx.eval(vectors)
        return vectors.tolist()
