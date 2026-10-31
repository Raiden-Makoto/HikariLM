"""Check helper/nomic_mlx.py against Nomic's own code, on a Mac. Not part of the test suite.

It embeds the same texts twice: with HikariLM's MLX port, and with Nomic's
PyTorch model (their remote code, as the model card runs it), then compares.
Run ./start.sh once first so the weights are in the Hugging Face cache, then:

    python3 -m venv /tmp/nomic-check
    /tmp/nomic-check/bin/pip install mlx transformers torch einops
    /tmp/nomic-check/bin/python tests/check_nomic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))

from helper import engine  # noqa: E402

MIN_COSINE = 0.999
TEXTS = [
    "search_query: What is TSNE?",
    "search_query: Who is Laurens van der Maaten?",
    "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van der Maaten",
    "search_document: Boil the water, add salt, then the pasta, and stir now and then.",
    "search_document: " + " ".join(f"Line {n}: an AVL tree rotates left or right after an insert." for n in range(150)),
]


def nomic_reference(folder: Path) -> list[list[float]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModel.from_pretrained(engine.EMBED_REPO, trust_remote_code=True, safe_serialization=True)
    model.eval()
    encoded = tokenizer(TEXTS, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    with torch.no_grad():
        tokens = model(**encoded)[0]
    mask = encoded["attention_mask"].unsqueeze(-1).expand(tokens.size()).float()
    pooled = torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
    return F.normalize(pooled, p=2, dim=1).tolist()


def main() -> int:
    folder = engine.embed_folder()
    if folder is None:
        print("nomic-embed-text is not downloaded. Run ./start.sh once, then this again.")
        return 2
    from nomic_mlx import Embedder

    ours = Embedder(str(folder)).embed(TEXTS)
    theirs = nomic_reference(folder)
    worst = 1.0
    for text, a, b in zip(TEXTS, ours, theirs):
        cosine = sum(x * y for x, y in zip(a, b))
        worst = min(worst, cosine)
        print(f"{cosine:.6f}  max diff {max(abs(x - y) for x, y in zip(a, b)):.5f}  {text[:60]}")
    tsne = [sum(x * y for x, y in zip(ours[0], doc)) for doc in ours[2:4]]
    print(f"'What is TSNE?' against the TSNE text {tsne[0]:.3f}, against the pasta one {tsne[1]:.3f}")
    ok = worst >= MIN_COSINE and tsne[0] > tsne[1]
    print("OK: the MLX port matches Nomic's vectors." if ok else f"FAILED: the lowest cosine is {worst:.6f}.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
