# tools/precompute_meta_token_bank.py
import os
import sys
import json
import re
import argparse
from typing import Dict, Any

import torch

# repo root on path so "from clip import clip" works
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from clip import clip as clip_lib


def _num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.2f}" if isinstance(x, float) else str(x)
    return str(x).strip()


def _yn(x: Any) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    s = str(x).strip().lower()
    return "yes" if s in {"true", "1", "yes", "y", "t"} else "no"


def _km(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return ""
    return f"{m.group(1)} km"


def meta_to_text(m: Dict[str, Any]) -> str:
    gdp = str(m.get("GDP (in billion yuan)", "")).strip()
    pop = str(m.get("Population", "")).strip()
    area = str(m.get("Area (in sq km)", "")).strip()
    lit = m.get("Literacy Rate", "")
    dens = str(m.get("Density", "")).strip()

    parts = []
    if gdp:
        parts.append(f"GDP {gdp} billion yuan.")
    if pop:
        parts.append(f"Population {pop}.")
    if area:
        parts.append(f"Area {area} sq km.")
    if lit != "" and lit is not None:
        parts.append(f"Literacy {float(lit):.2f} percent.")
    if dens:
        parts.append(f"Population density {dens} per sq km.")
    return " ".join(parts).strip()


def wrap_meta(s: str) -> str:
    return f"metadata: {s}"


def add_ctx_prefix(text: str, n_ctx: int) -> str:
    """
    HiCroPL inserts n_ctx learned tokens right after SOS, overwriting the first n_ctx
    tokens of the original text stream. Add n_ctx dummy words so real metadata survives.
    """
    if n_ctx <= 0:
        return text
    dummy = " ".join(["X"] * n_ctx)  # keep it short; typically tokenizes compactly
    return f"{dummy} {text}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="gid")
    ap.add_argument("--meta_json", type=str, default="geographical_metadata.json")
    ap.add_argument("--out", type=str, default="meta_token_bank.pt")
    ap.add_argument("--n_ctx", type=int, default=16, help="Must match TRAINER.HICROPL.N_CTX")
    args = ap.parse_args()

    meta_path = os.path.join(args.root, args.meta_json)
    out_path = os.path.join(args.root, args.out)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict: Dict[str, Dict[str, Any]] = json.load(f)

    bank: Dict[str, torch.Tensor] = {}

    # default/fallback token (for missing keys)
    default_text = add_ctx_prefix(wrap_meta("unknown."), args.n_ctx)
    default_tok = clip_lib.tokenize([default_text], context_length=77, truncate=True)[0].cpu()
    bank["__default__"] = default_tok

    for k, m in meta_dict.items():
        text = add_ctx_prefix(wrap_meta(meta_to_text(m)), args.n_ctx)
        tok = clip_lib.tokenize([text], context_length=77, truncate=True)[0].cpu()
        bank[str(k)] = tok

    torch.save(bank, out_path)
    print(f"[META-TOK] n_ctx={args.n_ctx} | Saved {len(bank)-1} tokens + __default__ to {out_path}")


if __name__ == "__main__":
    main()