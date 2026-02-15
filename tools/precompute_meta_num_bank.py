import argparse
import json
import math
import os
import os.path as osp
import re
from typing import Dict, Any, List

import torch


def _parse_km(x) -> float:
    """Accepts '30.47 km' or 30.47 or None."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root, e.g., data/bing_rgb")
    ap.add_argument(
        "--in_json",
        type=str,
        default="metadata.json",
        help="Metadata json filename under root (default: metadata.json)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="meta_num_bank.pt",
        help="Output bank filename under root (default: meta_num_bank.pt)",
    )
    args = ap.parse_args()

    root = args.root
    in_path = osp.join(root, args.in_json)
    out_path = osp.join(root, args.out)

    if not osp.exists(in_path):
        raise FileNotFoundError(f"metadata json not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Dict[str, Any]] = json.load(f)

    keys = sorted(list(meta.keys()))
    if len(keys) == 0:
        raise RuntimeError("metadata.json is empty")

    # Collect raw fields
    region_types: List[str] = []
    dens_list: List[float] = []
    lit_list: List[float] = []
    dist_d_list: List[float] = []
    dist_u_list: List[float] = []

    for k in keys:
        m = meta[k]
        region_types.append(str(m.get("Region Type", "")).strip().lower())
        dens_list.append(float(m.get("Population Density", 0.0)))
        lit_list.append(float(m.get("Literacy", 0.0)))
        dist_d_list.append(_parse_km(m.get("distance_to_district_sadar", 0.0)))
        dist_u_list.append(_parse_km(m.get("distance_to_upazila_sadar", 0.0)))

    # Build region type vocab (one-hot)
    uniq_region = sorted(list({rt for rt in region_types if rt != ""}))
    if len(uniq_region) == 0:
        uniq_region = [""]  # fallback
    r2id = {rt: i for i, rt in enumerate(uniq_region)}
    R = len(uniq_region)

    # Normalization stats (simple + stable)
    # density: log1p then divide by max
    dens_log = [math.log1p(max(0.0, d)) for d in dens_list]
    dens_log_max = max(dens_log) if len(dens_log) else 1.0
    dens_log_max = dens_log_max if dens_log_max > 0 else 1.0

    # literacy: percentage -> [0,1] by /100
    # distances: divide by max
    dist_d_max = max(dist_d_list) if len(dist_d_list) else 1.0
    dist_u_max = max(dist_u_list) if len(dist_u_list) else 1.0
    dist_d_max = dist_d_max if dist_d_max > 0 else 1.0
    dist_u_max = dist_u_max if dist_u_max > 0 else 1.0

    bank: Dict[str, Any] = {}
    fields = [
        "log1p_pop_density_norm",
        "literacy_norm",
        "dist_district_km_norm",
        "dist_upazila_km_norm",
        "inside_district_sadar",
        "inside_upazila_sadar",
        f"region_type_onehot[{R}]",
    ]

    # Optional meta header (won't be used during training lookup)
    bank["__meta__"] = {
        "feature_dim": 6 + R,
        "region_types": uniq_region,
        "region_type_to_id": r2id,
        "fields": fields,
        "norm": {
            "dens_log_max": dens_log_max,
            "dist_d_max": dist_d_max,
            "dist_u_max": dist_u_max,
        },
    }

    for i, k in enumerate(keys):
        m = meta[k]

        d = float(m.get("Population Density", 0.0))
        d = max(0.0, d)
        d = math.log1p(d) / dens_log_max

        l = float(m.get("Literacy", 0.0)) / 100.0
        l = max(0.0, min(1.0, l))

        dd = _parse_km(m.get("distance_to_district_sadar", 0.0)) / dist_d_max
        du = _parse_km(m.get("distance_to_upazila_sadar", 0.0)) / dist_u_max

        inside_d = 1.0 if bool(m.get("inside_district_sadar", False)) else 0.0
        inside_u = 1.0 if bool(m.get("inside_upazila_sadar", False)) else 0.0

        rt = str(m.get("Region Type", "")).strip().lower()
        onehot = torch.zeros((R,), dtype=torch.float32)
        if rt in r2id:
            onehot[r2id[rt]] = 1.0

        vec = torch.tensor([d, l, dd, du, inside_d, inside_u], dtype=torch.float32)
        vec = torch.cat([vec, onehot], dim=0)  # [6+R]

        bank[k] = vec

    os.makedirs(root, exist_ok=True)
    torch.save(bank, out_path)

    print(f"[META] Loaded metadata entries: {len(keys)}")
    print(f"[META] Region types: {uniq_region} (R={R})")
    print(f"[META] Saved meta bank: {out_path}")
    print(f"[META] Feature dim: {int(bank['__meta__']['feature_dim'])}")
    print(f"[META] Fields: {fields}")


if __name__ == "__main__":
    main()
