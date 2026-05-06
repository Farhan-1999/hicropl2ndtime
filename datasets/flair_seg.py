# datasets/flair_seg.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dassl.data.datasets import DATASET_REGISTRY


# Default FLAIR 12 classes
CLASSNAMES_FLAIR_12 = [
    "building",
    "pervious surface",
    "impervious surface",
    "bare soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous vegetation",
    "agricultural land",
    "plowed land",
]


@dataclass(frozen=True)
class FLAIRSample:
    key: str           # relative image path under split/images (unique)
    img_path: str
    mask_path: str


def _is_tiff(p: Path) -> bool:
    return p.suffix.lower() in {".tif", ".tiff"}


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array layout to HxWxC when possible.
    Handles:
      - HxW
      - HxWxC
      - CxHxW
    """
    if arr.ndim == 2:
        return arr[:, :, None]

    if arr.ndim == 3:
        # If it looks like CxHxW, convert to HxWxC
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return np.transpose(arr, (1, 2, 0))
        return arr

    raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")


def _scale_to_uint8(x: np.ndarray, scale_mode: str = "auto") -> np.ndarray:
    """
    Convert numeric array to uint8 [0,255].

    scale_mode:
      - auto: if max<=1 -> *255; elif max<=255 -> cast; else scale by max
      - max : always scale by max
      - none: clip/cast only
    """
    x = x.astype(np.float32)

    if scale_mode not in {"auto", "max", "none"}:
        raise ValueError(f"scale_mode must be one of auto/max/none, got: {scale_mode}")

    if x.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    if scale_mode == "none":
        x = np.clip(x, 0, 255)
        return x.astype(np.uint8)

    mx = float(np.nanmax(x))
    if mx <= 0:
        return np.zeros_like(x, dtype=np.uint8)

    if scale_mode == "auto":
        if mx <= 1.0:
            x = x * 255.0
        elif mx <= 255.0:
            pass
        else:
            x = x * (255.0 / mx)
    else:  # max
        x = x * (255.0 / mx)

    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def _read_tiff_array(path: Union[str, Path]) -> np.ndarray:
    """
    Read TIFF robustly. Prefer tifffile, fallback to PIL.
    """
    path = Path(path)

    try:
        import tifffile  # type: ignore
        arr = tifffile.imread(str(path))
    except Exception:
        with Image.open(str(path)) as im:
            arr = np.array(im)

    return np.asarray(arr)


def read_flair_rgb(
    path: Union[str, Path],
    rgb_bands: Tuple[int, int, int] = (0, 1, 2),
    scale_mode: str = "auto",
) -> Image.Image:
    """
    Read a FLAIR TIFF patch and return a PIL RGB image.

    Since you did not specify the exact band semantics/order for FLAIR images,
    this loader uses the first 3 channels by default. Change rgb_bands if needed.
    """
    arr = _read_tiff_array(path)
    arr = _to_hwc(arr)

    # If single-channel, replicate
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.shape[2] < 3:
        raise ValueError(f"FLAIR image has fewer than 3 channels: {path}, shape={arr.shape}")

    r, g, b = rgb_bands
    if max(r, g, b) >= arr.shape[2]:
        raise ValueError(
            f"rgb_bands={rgb_bands} out of range for image with C={arr.shape[2]} at {path}"
        )

    rgb = arr[:, :, [r, g, b]]
    rgb8 = _scale_to_uint8(rgb, scale_mode=scale_mode)
    return Image.fromarray(rgb8, mode="RGB")


def read_flair_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Read a single-channel TIFF mask and return raw integer labels.

    Expected raw labels:
      1..12 = real classes
      13    = other
    """
    arr = _read_tiff_array(path)

    if arr.ndim == 3:
        # If it somehow comes as HxWx1 or CxHxW with one channel, squeeze it
        arr = _to_hwc(arr)
        if arr.shape[2] != 1:
            raise ValueError(f"Expected single-channel mask, got shape={arr.shape} at {path}")
        arr = arr[:, :, 0]

    return arr.astype(np.int64)


def remap_flair_mask_np(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Remap raw FLAIR labels to 0-based labels for training.

    Raw:
      1..12 -> 0..11
      13    -> ignore_index
    """
    out = np.full(mask.shape, fill_value=ignore_index, dtype=np.int64)

    valid = (mask >= 1) & (mask <= 12)
    out[valid] = mask[valid] - 1

    # raw 13 ("other") stays ignore
    return out


def remap_flair_mask_tensor(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Torch version of remap_flair_mask_np.
    """
    out = torch.full_like(mask, fill_value=int(ignore_index))
    valid = (mask >= 1) & (mask <= 12)
    out[valid] = mask[valid] - 1
    return out


@DATASET_REGISTRY.register()
class FLAIRSegDataset(Dataset):
    """
    FLAIR segmentation dataset.

    Expected structure:
      root/
        train/
          images/
            <folder>/<folder>/IMG_x.tif
          masks/
            <folder>/<folder>/MSK_x.tif
        val/
          images/
            <folder>/<folder>/IMG_x.tif
          masks/
            <folder>/<folder>/MSK_x.tif
        (optional) test/
          images/
          masks/

    Matching rule:
      IMG_x.tif -> MSK_x.tif
      with the same relative subfolder structure.
    """

    def __init__(
        self,
        root: Union[str, Path] = "flair",
        split: str = "train",
        images_dir: Optional[Union[str, Path]] = None,
        masks_dir: Optional[Union[str, Path]] = None,
        classnames: Sequence[str] = CLASSNAMES_FLAIR_12,
        transforms: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_suffix: str = "",  # unused, kept for compatibility with your builder
        ignore_index: int = 255,
        validate_labels: bool = False,
        # TIFF / channel handling
        rgb_bands: Tuple[int, int, int] = (0, 1, 2),
        tiff_scale_mode: str = "auto",
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
        self.ignore_index = int(ignore_index)

        self.rgb_bands = tuple(rgb_bands)
        self.tiff_scale_mode = tiff_scale_mode

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

        # Recursively collect all TIFF images under split/images
        img_paths = sorted([
            p for p in self.images_dir.rglob("*")
            if p.is_file() and _is_tiff(p)
        ])

        if max_samples is not None:
            img_paths = img_paths[: int(max_samples)]

        if len(img_paths) == 0:
            raise RuntimeError(f"No TIFF images found under: {self.images_dir}")

        samples: List[FLAIRSample] = []
        for img_path in img_paths:
            mask_path = self._resolve_mask_path(img_path)
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image:\n"
                    f"  image = {img_path}\n"
                    f"  expected mask = {mask_path}"
                )

            # Use relative image path as key so it's unique even with nested folders
            rel_key = img_path.relative_to(self.images_dir).as_posix()

            samples.append(
                FLAIRSample(
                    key=rel_key,
                    img_path=str(img_path.resolve()),
                    mask_path=str(mask_path.resolve()),
                )
            )

        self.samples = samples

        if validate_labels:
            self._quick_validate_masks(n_checks=min(20, len(self.samples)))

    def _resolve_mask_path(self, img_path: Path) -> Path:
        """
        Map:
        split/images/.../img/IMG_x.tif
        to:
        split/masks/.../msk/MSK_x.tif
        """
        rel = img_path.relative_to(self.images_dir)

        parts = list(rel.parts)

        # Replace final folder "img" -> "msk" if present
        if len(parts) >= 2 and parts[-2].lower() == "img":
            parts[-2] = "msk"

        # Replace filename prefix IMG_ -> MSK_
        fname = parts[-1]
        if fname.startswith("IMG_"):
            fname = "MSK_" + fname[len("IMG_"):]
        else:
            fname = fname.replace("IMG_", "MSK_", 1)
        parts[-1] = fname

        return (self.masks_dir / Path(*parts)).resolve()

    def _quick_validate_masks(self, n_checks: int = 10) -> None:
        idxs = np.linspace(0, len(self.samples) - 1, num=n_checks, dtype=int).tolist()
        bad: List[Tuple[str, List[int]]] = []

        for i in idxs:
            mp = self.samples[i].mask_path
            raw = read_flair_mask(mp)
            remapped = remap_flair_mask_np(raw, ignore_index=self.ignore_index)
            uniq = np.unique(remapped)

            ok = np.all(
                (uniq == self.ignore_index) |
                ((uniq >= 0) & (uniq <= self.num_classes - 1))
            )

            if not ok:
                bad.append((mp, uniq.tolist()))

        if bad:
            msg = "Some FLAIR masks contain unexpected remapped labels:\n"
            for mp, uniq in bad[:5]:
                msg += f"  - {mp}: {uniq}\n"
            msg += (
                f"Expected labels in [0..{self.num_classes - 1}] "
                f"plus ignore_index={self.ignore_index}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        img = read_flair_rgb(
            s.img_path,
            rgb_bands=self.rgb_bands,
            scale_mode=self.tiff_scale_mode,
        )

        mask_raw = read_flair_mask(s.mask_path)
        mask_pil = Image.fromarray(mask_raw.astype(np.uint8), mode="L")

        if self.transforms is not None:
            img_t, mask_t = self.transforms(img, mask_pil)
            mask_t = mask_t.long()
            mask_t = remap_flair_mask_tensor(mask_t, ignore_index=self.ignore_index)
        else:
            img_arr = np.array(img).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()

            mask_arr = remap_flair_mask_np(mask_raw, ignore_index=self.ignore_index)
            mask_t = torch.from_numpy(mask_arr).long()

        return {
            "img": img_t,
            "mask": mask_t,
            "key": s.key,          # unique relative path under split/images
            "impath": s.img_path,
            "maskpath": s.mask_path,
        }