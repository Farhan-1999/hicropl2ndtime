# tools/compute_class_counts.py
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

CLASSNAMES = ["Unrecognized", "Farmland", "Water", "Forest", "Built-Up", "Meadow"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="e.g., data/bing_rgb")
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--num-classes", type=int, default=6)
    ap.add_argument("--ignore-index", type=int, default=255)
    args = ap.parse_args()

    masks_dir = Path(args.root) / args.split / "masks"
    assert masks_dir.exists(), f"Missing masks dir: {masks_dir}"

    mask_files = sorted([p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"]])
    assert len(mask_files) > 0, f"No mask files found in {masks_dir}"

    total = np.zeros(args.num_classes, dtype=np.int64)
    ignored = 0

    for mp in mask_files:
        m = Image.open(mp)
        if m.mode not in {"L", "I"}:
            m = m.convert("L")
        arr = np.array(m, dtype=np.int64).ravel()

        if args.ignore_index is not None:
            ignored += int((arr == args.ignore_index).sum())
            arr = arr[arr != args.ignore_index]

        # safety: if any out-of-range labels exist, warn
        if arr.size > 0:
            mx, mn = int(arr.max()), int(arr.min())
            if mn < 0 or mx >= args.num_classes:
                print(f"[WARN] {mp.name}: label range [{mn},{mx}] outside [0,{args.num_classes-1}]")

        total += np.bincount(arr, minlength=args.num_classes)

    total_pixels = int(total.sum())
    print("=== Pixel counts (split:", args.split, ") ===")
    for i in range(args.num_classes):
        name = CLASSNAMES[i] if i < len(CLASSNAMES) else f"class_{i}"
        frac = (total[i] / max(1, total_pixels)) * 100.0
        print(f"{i:02d} {name:12s}: {total[i]:12d}  ({frac:6.2f}%)")
    print(f"ignored_pixels: {ignored}")
    print(f"valid_pixels  : {total_pixels}")

if __name__ == "__main__":
    main()