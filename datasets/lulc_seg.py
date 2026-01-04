# datasets/lulc_seg.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASSNAMES_LULC_6 = [
    "Unrecognized",
    "Farmland",
    "Water",
    "Forest",
    "Built-Up",
    "Meadow",
]


@dataclass(frozen=True)
class LULCSample:
    key: str
    img_path: str
    mask_path: str
    meta_raw: Optional[Dict[str, Any]]
    meta_num: Optional[torch.Tensor]


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _parse_km(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return float("nan")
    val = float(m.group(1))
    # defensive: convert meters to km if it looks like meters
    if (" m" in s or s.endswith("m")) and ("km" not in s):
        val = val / 1000.0
    return val


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return float("nan")


def _to_bool01(x: Any) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if x is None:
        return 0.0
    s = str(x).strip().lower()
    return 1.0 if s in {"true", "1", "yes", "y", "t"} else 0.0


def encode_metadata_numeric(meta: Dict[str, Any]) -> torch.Tensor:
    """
    Fast numeric encoding (for conditioning). You can expand this later.

    Order:
      0: Population Density
      1: Literacy
      2: distance_to_district_sadar (km)
      3: distance_to_upazila_sadar (km)
      4: inside_district_sadar (0/1)
      5: inside_upazila_sadar (0/1)
    """
    pop = _to_float(meta.get("Population Density"))
    lit = _to_float(meta.get("Literacy"))
    d_dist = _parse_km(meta.get("distance_to_district_sadar"))
    d_upz = _parse_km(meta.get("distance_to_upazila_sadar"))
    in_dist = _to_bool01(meta.get("inside_district_sadar"))
    in_upz = _to_bool01(meta.get("inside_upazila_sadar"))
    return torch.tensor([pop, lit, d_dist, d_upz, in_dist, in_upz], dtype=torch.float32)

@DATASET_REGISTRY.register()
class LULCSegDataset(Dataset):
    """
    Layout (your confirmed structure):
      root/
        train/
          images/
          masks/
        val/
          images/
          masks/
        metadata.json

    Masks:
      Supports either:
        - same filename in masks dir: masks/patch_00000.png
        - suffix-based: masks/patch_00000_gt.png (mask_suffix default "_gt")
    """

    def __init__(
        self,
        root: Union[str, Path] = "data/bing_rgb",
        split: str = "train",
        metadata_json: Union[str, Path] = "metadata.json",
        # If you want to override the default split dirs:
        images_dir: Optional[Union[str, Path]] = None,
        masks_dir: Optional[Union[str, Path]] = None,
        classnames: Sequence[str] = CLASSNAMES_LULC_6,
        transforms: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_suffix: str = "_gt",
        validate_labels: bool = False,
        return_raw_meta: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.classnames = list(classnames)
        self.num_classes = len(self.classnames)
        self.transforms = transforms
        self.mask_suffix = mask_suffix
        self.return_raw_meta = return_raw_meta

        # Default dirs from your structure
        if images_dir is None:
            self.images_dir = (self.root / self.split / "images").resolve()
        else:
            p = Path(images_dir)
            self.images_dir = p.resolve() if p.is_absolute() else (self.root / p).resolve()

        if masks_dir is None:
            self.masks_dir = (self.root / self.split / "masks").resolve()
        else:
            p = Path(masks_dir)
            self.masks_dir = p.resolve() if p.is_absolute() else (self.root / p).resolve()

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")

        # Metadata
        meta_path = Path(metadata_json)
        meta_path = meta_path.resolve() if meta_path.is_absolute() else (self.root / meta_path).resolve()
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        # Collect image keys
        keys = sorted([p.name for p in self.images_dir.iterdir() if p.is_file() and _is_image_file(p)])
        if max_samples is not None:
            keys = keys[: int(max_samples)]
        if len(keys) == 0:
            raise RuntimeError(f"No images found in: {self.images_dir}")

        # Build samples (pre-encode numeric metadata once for speed)
        samples: List[LULCSample] = []
        for key in keys:
            img_path = (self.images_dir / key).resolve()
            mask_path = self._resolve_mask_path(key)
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for '{key}'. Tried:\n"
                    f"  1) {self.masks_dir / key}\n"
                    f"  2) {self.masks_dir / (Path(key).stem + self.mask_suffix + Path(key).suffix)}\n"
                    f"Check your mask filenames or adjust mask_suffix."
                )

            meta_raw = self._metadata.get(key)
            meta_num = encode_metadata_numeric(meta_raw) if meta_raw is not None else None

            samples.append(
                LULCSample(
                    key=key,
                    img_path=str(img_path),
                    mask_path=str(mask_path),
                    meta_raw=meta_raw,
                    meta_num=meta_num,
                )
            )

        self.samples = samples

        if validate_labels:
            self._quick_validate_masks(n_checks=min(20, len(self.samples)))

    def _resolve_mask_path(self, img_filename: str) -> Path:
        """
        Mask naming conventions supported:
          - masks_dir/patch_00000.png
          - masks_dir/patch_00000_gt.png   (mask_suffix default "_gt")
        """
        p1 = (self.masks_dir / img_filename).resolve()
        if p1.exists():
            return p1
        stem = Path(img_filename).stem
        ext = Path(img_filename).suffix
        p2 = (self.masks_dir / f"{stem}{self.mask_suffix}{ext}").resolve()
        return p2

    def _quick_validate_masks(self, n_checks: int = 10) -> None:
        idxs = np.linspace(0, len(self.samples) - 1, num=n_checks, dtype=int).tolist()
        bad = []
        for i in idxs:
            mp = self.samples[i].mask_path
            m = Image.open(mp)
            if m.mode not in {"L", "I"}:
                m = m.convert("L")
            arr = np.array(m)
            uniq = np.unique(arr).astype(int).tolist()
            if any((u < 0 or u >= self.num_classes) for u in uniq):
                bad.append((Path(mp).name, uniq[:50]))
        if bad:
            msg = "\n".join([f"  - {name}: unique={uniq}" for name, uniq in bad[:5]])
            raise ValueError(
                "Mask label validation failed: values outside [0, num_classes-1].\n"
                f"num_classes={self.num_classes}\nExamples:\n{msg}\n"
                "If your masks are color-coded RGB, you must add a color->id mapping step."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        img = Image.open(s.img_path).convert("RGB")
        mask = Image.open(s.mask_path)
        if mask.mode not in {"L", "I"}:
            mask = mask.convert("L")

        if self.transforms is not None:
            img_t, mask_t = self.transforms(img, mask)
        else:
            # Fallback: minimal tensor conversion (no augmentation)
            img_arr = np.array(img).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()

            mask_arr = np.array(mask).astype(np.int64)
            mask_t = torch.from_numpy(mask_arr)

        meta_out: Optional[Dict[str, Any]] = None
        if s.meta_raw is not None:
            meta_out = {
                "numeric": s.meta_num if s.meta_num is not None else encode_metadata_numeric(s.meta_raw),
                "Location": s.meta_raw.get("Location", ""),
                "Upazila": s.meta_raw.get("Upazila", ""),
                "District": s.meta_raw.get("District", ""),
                "Region Type": s.meta_raw.get("Region Type", ""),
                "description": s.meta_raw.get("description", ""),
            }
            if self.return_raw_meta:
                meta_out["raw"] = s.meta_raw

        return {
            "img": img_t,
            "mask": mask_t,
            "key": s.key,
            "impath": s.img_path,
            "maskpath": s.mask_path,
            "meta": meta_out,
        }
