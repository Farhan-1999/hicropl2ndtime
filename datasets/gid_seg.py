# datasets/gid.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dassl.data.datasets import DATASET_REGISTRY


# Default placeholder names (override via cfg.DATASET.CLASSNAMES)
CLASSNAMES_GID_24 = [f"Class_{i}" for i in range(6)]


@dataclass(frozen=True)
class GIDSample:
    key: str
    img_path: str
    mask_path: str


def _is_tiff(p: Path) -> bool:
    return p.suffix.lower() in {".tif", ".tiff"}


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Normalize TIFF array layout to HxWxC when possible.
    Handles common cases:
      - HxW
      - HxWxC
      - CxHxW
    """
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        # If it's CxHxW, convert to HxWxC
        if arr.shape[0] in (1, 3, 4) and (arr.shape[1] > 16 and arr.shape[2] > 16):
            return np.transpose(arr, (1, 2, 0))
        return arr
    raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")


def _scale_to_uint8(x: np.ndarray, scale_mode: str = "auto") -> np.ndarray:
    """
    Convert arbitrary numeric array to uint8 [0,255].
    - auto: if max<=1 -> *255; elif max<=255 -> cast; else scale by global max
    - max: scale by global max
    - none: just clip/cast
    """
    x = x.astype(np.float32)

    if scale_mode not in {"auto", "max", "none"}:
        raise ValueError(f"scale_mode must be one of auto/max/none, got: {scale_mode}")

    if scale_mode == "none":
        x = np.clip(x, 0, 255)
        return x.astype(np.uint8)

    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 0:
        return np.zeros_like(x, dtype=np.uint8)

    if scale_mode == "auto":
        if mx <= 1.0:
            x = x * 255.0
        elif mx <= 255.0:
            # already in 0..255-ish
            pass
        else:
            x = x * (255.0 / mx)
    else:  # "max"
        x = x * (255.0 / mx)

    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def read_tiff_rgb(
    path: Union[str, Path],
    rgb_bands: Tuple[int, int, int] = (2, 1, 0),  # for B,G,R,NIR -> RGB uses (R,G,B)=(2,1,0)
    scale_mode: str = "auto",
) -> Image.Image:
    """
    Read a 4-band TIFF with band order (B, G, R, NIR) and return an RGB PIL image.

    rgb_bands:
      Indices into the channel dimension AFTER converting to HxWxC.
      Default (2,1,0) maps B,G,R,NIR -> RGB.

    scale_mode:
      auto/max/none (see _scale_to_uint8)
    """
    path = Path(path)

    arr = None
    # Prefer tifffile for multi-band geotiffs / unusual layouts
    try:
        import tifffile  # type: ignore

        arr = tifffile.imread(str(path))
    except Exception:
        # Fallback to PIL (works for many TIFFs if Pillow has libtiff)
        with Image.open(str(path)) as im:
            arr = np.array(im)

    arr = _to_hwc(np.array(arr))

    # If fewer than 3 channels, replicate
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    # If we have at least 3 channels, pick RGB bands
    if arr.shape[2] >= 3:
        r, g, b = rgb_bands
        # Defensive bounds check
        if max(r, g, b) >= arr.shape[2]:
            raise ValueError(
                f"rgb_bands={rgb_bands} out of range for TIFF with C={arr.shape[2]} at {path}"
            )
        rgb = arr[:, :, [r, g, b]]
    else:
        # should not happen due to replication above, but keep safe
        rgb = np.repeat(arr, 3, axis=2)

    rgb8 = _scale_to_uint8(rgb, scale_mode=scale_mode)
    return Image.fromarray(rgb8, mode="RGB")


@DATASET_REGISTRY.register()
class GIDSegDataset(Dataset):
    """
    GID/GF-2 segmentation dataset.

    Expected layout:
      root/
        train/
          images/   (TIFF: .tif/.tiff, 4-band B,G,R,NIR)
          masks/    (PNG masks: *_24label.png)
        val/
          images/
          masks/
        (optional) test/
          images/
          masks/

    Image example:
      GF2_PMS1__L1A0000564539-MSS1.tiff

    Mask example:
      GF2_PMS1__L1A0000564539-MSS1_24label.png
    """

    def __init__(
        self,
        root: Union[str, Path] = "gid_patch",
        split: str = "train",
        images_dir: Optional[Union[str, Path]] = None,
        masks_dir: Optional[Union[str, Path]] = None,
        classnames: Sequence[str] = CLASSNAMES_GID_24,
        transforms: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_suffix: str = "",
        mask_ext: str = ".png",
        ignore_index: int = 255,
        # TIFF handling
        rgb_bands: Tuple[int, int, int] = (2, 1, 0),  # B,G,R,NIR -> RGB
        tiff_scale_mode: str = "auto",
        # Label handling (useful if your masks are 1..24 instead of 0..23)
        label_offset: int = 0,  # set to -1 if masks are 1..24
        validate_labels: bool = False,
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
        self.mask_ext = mask_ext
        self.ignore_index = int(ignore_index)

        self.rgb_bands = tuple(rgb_bands)
        self.tiff_scale_mode = tiff_scale_mode
        self.label_offset = int(label_offset)

        # Default dirs from your confirmed structure
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

        # Collect TIFF image filenames
        keys = sorted([p.name for p in self.images_dir.iterdir() if p.is_file() and _is_tiff(p)])
        if max_samples is not None:
            keys = keys[: int(max_samples)]
        if len(keys) == 0:
            raise RuntimeError(f"No TIFF images found in: {self.images_dir}")

        samples: List[GIDSample] = []
        for key in keys:
            img_path = (self.images_dir / key).resolve()
            mask_path = self._resolve_mask_path(key)
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for '{key}'. Tried:\n"
                    f"  1) {self.masks_dir / (Path(key).stem + self.mask_suffix + self.mask_ext)}\n"
                    f"  2) {self.masks_dir / (Path(key).stem + self.mask_suffix + Path(key).suffix)}\n"
                    f"Check your mask_suffix/mask_ext or filenames."
                )

            samples.append(
                GIDSample(
                    key=key,
                    img_path=str(img_path),
                    mask_path=str(mask_path),
                )
            )

        self.samples = samples

        if validate_labels:
            self._quick_validate_masks(n_checks=min(20, len(self.samples)))

    def _resolve_mask_path(self, img_filename: str) -> Path:
        """
        Default convention:
          image: <stem>.tiff
          mask : <stem>_24label.png
        """
        stem = Path(img_filename).stem

        # Primary: suffix + explicit mask_ext (png)
        p1 = (self.masks_dir / f"{stem}{self.mask_suffix}{self.mask_ext}").resolve()
        if p1.exists():
            return p1

        # Secondary: suffix + same ext as image (rare, but cheap to check)
        ext_img = Path(img_filename).suffix
        p2 = (self.masks_dir / f"{stem}{self.mask_suffix}{ext_img}").resolve()
        return p2

    def _quick_validate_masks(self, n_checks: int = 10) -> None:
        idxs = np.linspace(0, len(self.samples) - 1, num=n_checks, dtype=int).tolist()
        bad: List[Tuple[str, List[int]]] = []
        for i in idxs:
            mp = self.samples[i].mask_path
            m = Image.open(mp)
            if m.mode not in {"L", "I"}:
                m = m.convert("L")
            arr = np.array(m).astype(np.int64)

            # Apply offset (without touching ignore_index)
            if self.label_offset != 0:
                valid = arr != self.ignore_index
                arr2 = arr.copy()
                arr2[valid] = arr2[valid] + self.label_offset
                arr = arr2

            uniq = np.unique(arr)
            # Allow ignore_index plus [0..num_classes-1]
            allowed_min, allowed_max = 0, self.num_classes - 1
            ok = np.all((uniq == self.ignore_index) | ((uniq >= allowed_min) & (uniq <= allowed_max)))
            if not ok:
                bad.append((mp, uniq.tolist()))

        if bad:
            msg = "Some masks contain unexpected label IDs (after label_offset applied):\n"
            for mp, uniq in bad[:5]:
                msg += f"  - {mp}: {uniq}\n"
            msg += (
                f"Expected labels in [0..{self.num_classes-1}] plus ignore_index={self.ignore_index}. "
                f"If your masks are 1..{self.num_classes}, set label_offset=-1."
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        # 4-band TIFF (B,G,R,NIR) -> RGB PIL
        img = read_tiff_rgb(
            s.img_path,
            rgb_bands=self.rgb_bands,
            scale_mode=self.tiff_scale_mode,
        )

        mask = Image.open(s.mask_path)

        # Keep paletted masks as-is (mode "P") so np.array(mask) returns class IDs
        if mask.mode in {"P", "L", "I"}:
            pass
        else:
            mask = mask.convert("L")

        if self.transforms is not None:
            img_t, mask_t = self.transforms(img, mask)
        else:
            # Minimal fallback conversion (no augmentation)
            img_arr = np.array(img).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()

            mask_arr = np.array(mask).astype(np.int64)
            # Apply offset without shifting ignore_index
            if self.label_offset != 0:
                valid = mask_arr != self.ignore_index
                mask_arr2 = mask_arr.copy()
                mask_arr2[valid] = mask_arr2[valid] + self.label_offset
                mask_arr = mask_arr2

            mask_t = torch.from_numpy(mask_arr)
        
        # --- GID label remap: handle masks with values {0..24}
        # interpret as 0=unlabeled/background, 1..24=classes
        if mask_t.min().item() == 0 and mask_t.max().item() == self.num_classes:
            mask_t = mask_t.clone()
            mask_t[mask_t == 0] = self.ignore_index  # ignore background
            valid = mask_t != self.ignore_index
            mask_t[valid] = mask_t[valid] - 1        # shift 1..24 -> 0..23

        # If transforms path was used, apply label_offset afterward too
        if self.transforms is not None and self.label_offset != 0:
            mi = mask_t
            valid = mi != self.ignore_index
            mi2 = mi.clone()
            mi2[valid] = mi2[valid] + self.label_offset
            mask_t = mi2

        return {
            "img": img_t,
            "mask": mask_t,
            "key": s.key,
            "impath": s.img_path,
            "maskpath": s.mask_path,
        }