# dassl/data/transforms/seg_transforms.py
#
# Paired (image, mask) transforms for semantic segmentation.
# Key properties:
#   - Every geometric transform is applied identically to image and mask
#   - Image uses bilinear/bicubic interpolation; mask ALWAYS uses nearest
#   - Mask is returned as torch.long (class IDs), image as float tensor (CHW)
#
# Designed to integrate smoothly with the repo's cfg style:
#   cfg.INPUT.SIZE, cfg.INPUT.INTERPOLATION, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD, cfg.INPUT.TRANSFORMS
#
# Typical usage in your seg trainer:
#   from dassl.data.transforms.seg_transforms import build_seg_transform
#   tfm_train = build_seg_transform(cfg, is_train=True, ignore_index=255)
#   img_t, mask_t = tfm_train(img_pil, mask_pil)

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception as e:
    raise ImportError(
        "torchvision is required for seg_transforms.py. Please install torchvision."
    ) from e


Pair = Tuple[Image.Image, Image.Image]
TensorPair = Tuple[torch.Tensor, torch.Tensor]


def _to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if len(x) != 2:
        raise ValueError(f"Expected int or 2-tuple, got: {x}")
    return (int(x[0]), int(x[1]))


def _interp_mode(name: str) -> InterpolationMode:
    n = str(name).lower()
    if n in {"bilinear", "linear"}:
        return InterpolationMode.BILINEAR
    if n in {"bicubic", "cubic"}:
        return InterpolationMode.BICUBIC
    if n in {"nearest"}:
        return InterpolationMode.NEAREST
    # default
    return InterpolationMode.BICUBIC


class Compose:
    """Compose paired transforms: (img, mask) -> (img, mask)."""

    def __init__(self, transforms: List[Callable[[Image.Image, Image.Image], Pair]]):
        self.transforms = transforms

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        if random.random() < self.p:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask


class Resize:
    """
    Resize image to `size` using `img_interp`, and mask using nearest.
    `size` can be int or (H, W).
    """

    def __init__(self, size: Union[int, Tuple[int, int]], img_interp: InterpolationMode):
        self.size = _to_2tuple(size)
        self.img_interp = img_interp

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        img = TF.resize(img, self.size, interpolation=self.img_interp)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return img, mask


class RandomResize:
    """
    Randomly rescale the shorter side by a factor in [min_scale, max_scale].
    Keeps aspect ratio.
    """

    def __init__(self, min_scale: float, max_scale: float, img_interp: InterpolationMode):
        assert min_scale > 0 and max_scale >= min_scale
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.img_interp = img_interp

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = img.size  # PIL gives (W, H)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = TF.resize(img, (new_h, new_w), interpolation=self.img_interp)
        mask = TF.resize(mask, (new_h, new_w), interpolation=InterpolationMode.NEAREST)
        return img, mask


class RandomResizedCrop:
    """
    RandomResizedCrop applied consistently to image & mask.
    This is closer to the repo's "random_resized_crop" behavior.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.5, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        img_interp: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = _to_2tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.img_interp = img_interp

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        # Torchvision's internal get_params is not exposed via TF; implement equivalent logic
        width, height = img.size
        area = height * width

        log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))

        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = np.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)

                img = TF.resized_crop(img, i, j, h, w, self.size, interpolation=self.img_interp)
                mask = TF.resized_crop(mask, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST)
                return img, mask

        # Fallback: center crop after resize
        img = TF.resize(img, self.size, interpolation=self.img_interp)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return img, mask


class RandomCrop:
    """
    Random crop to `size`. If image is smaller, pad first.
    `fill_mask` should typically be ignore_index (e.g., 255) or 0 if you use class 0 as background.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        pad_if_needed: bool = True,
        fill_img: int = 0,
        fill_mask: int = 255,
    ):
        self.size = _to_2tuple(size)
        self.pad_if_needed = bool(pad_if_needed)
        self.fill_img = int(fill_img)
        self.fill_mask = int(fill_mask)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Pair:
        th, tw = self.size  # (H, W)
        w, h = img.size

        if self.pad_if_needed:
            pad_left = pad_top = pad_right = pad_bottom = 0
            if w < tw:
                diff = tw - w
                pad_left = diff // 2
                pad_right = diff - pad_left
            if h < th:
                diff = th - h
                pad_top = diff // 2
                pad_bottom = diff - pad_top

            if pad_left or pad_top or pad_right or pad_bottom:
                img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill_img)
                mask = TF.pad(mask, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill_mask)
                w, h = img.size

        if w == tw and h == th:
            return img, mask

        if w < tw or h < th:
            # After padding, this shouldn't happen, but keep safe:
            img = TF.resize(img, (th, tw), interpolation=InterpolationMode.BICUBIC)
            mask = TF.resize(mask, (th, tw), interpolation=InterpolationMode.NEAREST)
            return img, mask

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img = TF.crop(img, i, j, th, tw)
        mask = TF.crop(mask, i, j, th, tw)
        return img, mask


class ToTensor:
    """Convert PIL image/mask to tensors. Mask becomes LongTensor of IDs."""

    def __call__(self, img: Image.Image, mask: Image.Image) -> TensorPair:
        img_t = TF.to_tensor(img)  # float32, [0,1], CHW

        # For mask: keep integer IDs (no scaling)
        if mask.mode not in {"L", "I"}:
            mask = mask.convert("L")
        mask_np = np.array(mask, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np).long()
        return img_t, mask_t


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))

    def __call__(self, img_t: torch.Tensor, mask_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = TF.normalize(img_t, mean=self.mean, std=self.std)
        return img_t, mask_t


class PairTensorCompose:
    """
    Compose for tensor-stage transforms (after ToTensor).
    """

    def __init__(self, transforms: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]):
        self.transforms = transforms

    def __call__(self, img_t: torch.Tensor, mask_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            img_t, mask_t = t(img_t, mask_t)
        return img_t, mask_t


@dataclass
class SegTransform:
    """
    Full paired transform pipeline callable:
      (PIL img, PIL mask) -> (tensor img, tensor mask)
    """
    pil_ops: Compose
    tensor_ops: PairTensorCompose
    to_tensor: ToTensor

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.pil_ops(img, mask)
        img_t, mask_t = self.to_tensor(img, mask)
        img_t, mask_t = self.tensor_ops(img_t, mask_t)
        return img_t, mask_t


def build_seg_transform(cfg, is_train: bool, ignore_index: int = 255) -> Callable[[Image.Image, Image.Image], TensorPair]:
    """
    Build a segmentation transform pipeline from cfg.INPUT.* and cfg.INPUT.TRANSFORMS.

    Supported keywords in cfg.INPUT.TRANSFORMS:
      - "random_resized_crop"
      - "random_resize" (uses cfg.INPUT.SCALE_RANGE if present; else default (0.5, 2.0))
      - "random_crop"   (uses cfg.INPUT.CROP_SIZE if present; else cfg.INPUT.SIZE)
      - "resize"
      - "random_flip"   (hflip, p=0.5)
      - "random_vflip"  (vflip, p=0.5)
      - "normalize"
    """
    img_interp = _interp_mode(getattr(cfg.INPUT, "INTERPOLATION", "bicubic"))

    # Common sizes
    size = getattr(cfg.INPUT, "SIZE", (224, 224))
    size = _to_2tuple(size)

    crop_size = getattr(cfg.INPUT, "CROP_SIZE", None)
    if crop_size is None:
        crop_size = size
    crop_size = _to_2tuple(crop_size)

    # RandomResizedCrop specific knobs (if provided)
    scale_range = getattr(cfg.INPUT, "SCALE_RANGE", None)
    if scale_range is None:
        scale_range = (0.5, 2.0)

    mean = getattr(cfg.INPUT, "PIXEL_MEAN", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(cfg.INPUT, "PIXEL_STD", [0.26862954, 0.26130258, 0.27577711])

    tnames = list(getattr(cfg.INPUT, "TRANSFORMS", []))

    # Build PIL-stage ops (geometric)
    pil_ops: List[Callable[[Image.Image, Image.Image], Pair]] = []

    if is_train:
        for name in tnames:
            n = str(name).lower()

            if n == "random_resized_crop":
                pil_ops.append(RandomResizedCrop(size=crop_size, img_interp=img_interp))

            elif n == "random_resize":
                pil_ops.append(RandomResize(min_scale=scale_range[0], max_scale=scale_range[1], img_interp=img_interp))

            elif n == "random_crop":
                pil_ops.append(RandomCrop(size=crop_size, pad_if_needed=True, fill_img=0, fill_mask=ignore_index))

            elif n == "resize":
                pil_ops.append(Resize(size=size, img_interp=img_interp))

            elif n == "random_flip":
                pil_ops.append(RandomHorizontalFlip(p=0.5))

            elif n == "random_vflip":
                pil_ops.append(RandomVerticalFlip(p=0.5))

            elif n == "normalize":
                # handled in tensor stage
                continue

            else:
                raise ValueError(f"Unsupported seg transform: {name}")

        # If neither crop nor resized-crop is specified, but you want fixed training size,
        # it's common to resize then random crop. We do NOT enforce it automatically.
    else:
        # Eval: deterministic, usually resize only (unless user wants original size)
        # If "resize" appears, apply it; otherwise leave as-is.
        do_resize = any(str(x).lower() == "resize" for x in tnames)
        if do_resize:
            pil_ops.append(Resize(size=size, img_interp=img_interp))

    # Tensor-stage ops
    tensor_ops: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = []
    if any(str(x).lower() == "normalize" for x in tnames):
        tensor_ops.append(Normalize(mean=mean, std=std))

    return SegTransform(pil_ops=Compose(pil_ops), to_tensor=ToTensor(), tensor_ops=PairTensorCompose(tensor_ops))
