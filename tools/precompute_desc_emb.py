# tools/precompute_desc_emb.py
import os
import sys
import json
import re
import argparse
from typing import Dict

import torch

# Make repo root importable (so `from clip import clip` works)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from clip import clip as clip_lib  # your local CLIP wrapper


def shorten_desc(text: str, max_words: int = 60) -> str:
    """Keep first ~2 sentences, then cap by words (helps CLIP token budget)."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    s = " ".join(parts[:2]).strip()
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).strip()
    return s


def wrap_desc(text: str) -> str:
    # Stable wrapper that matches your remote-sensing style descriptions
    return f"a satellite image described as: {text}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/bing_rgb")
    ap.add_argument("--desc_json", type=str, default="description.json")
    ap.add_argument("--model", type=str, default="ViT-B/16")
    ap.add_argument("--out", type=str, default="desc_emb_bank.pt")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    root = args.root
    desc_path = os.path.join(root, args.desc_json)
    out_path = os.path.join(root, args.out)

    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Description JSON not found: {desc_path}")

    with open(desc_path, "r", encoding="utf-8") as f:
        desc_dict: Dict[str, str] = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    url = clip_lib._MODELS[args.model]
    model_path = clip_lib._download(url)

    try:
        # JIT archive
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        jit_model = None
        state_dict = torch.load(model_path, map_location="cpu")

    # This matches your repo's "original CLIP" construction (no prompt tuning)
    design_details = {
        "trainer": "IVLP",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }

    model = clip_lib.build_model(state_dict or jit_model.state_dict(), design_details).to(device)
    model.eval()

    items = list(desc_dict.items())
    print(f"[DESC] Loaded {len(items)} descriptions from {desc_path}")
    print(f"[DESC] Encoding with CLIP {args.model} on {device}")
    print(f"[DESC] Output bank: {out_path}")

    bank: Dict[str, torch.Tensor] = {}

    bs = int(args.batch_size)
    for start in range(0, len(items), bs):
        chunk = items[start : start + bs]
        keys = [k for k, _ in chunk]
        texts = [wrap_desc(shorten_desc(t)) for _, t in chunk]

        tokens = clip_lib.tokenize(texts, context_length=77, truncate=True).to(device)

        feats = model.encode_text(tokens)
        feats = feats.float()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

        for k, e in zip(keys, feats):
            bank[k] = e.detach().cpu()

        if (start // bs) % 10 == 0:
            print(f"  encoded {min(start+bs, len(items))}/{len(items)}")

    torch.save(bank, out_path)
    print(f"[DESC] Saved: {out_path} (entries={len(bank)})")


if __name__ == "__main__":
    main()
