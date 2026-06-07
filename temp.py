from pathlib import Path
import numpy as np
from PIL import Image

mask_dir = Path("data/gid/train/masks")  # adjust split if needed
vals = set()
has_255 = False

for p in list(mask_dir.glob("*.png"))[:200]:  # scan first 200
    m = Image.open(p)
    arr = np.array(m)
    u = np.unique(arr)
    vals.update(u.tolist())
    if 255 in u:
        has_255 = True

print("Unique values found (sorted, first 60):", sorted(vals)[:60])
print("Contains 255 ignore?:", has_255)
print("Min/Max:", min(vals), max(vals))