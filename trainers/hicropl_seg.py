# trainers/hicropl_seg.py
#
# HiCroPL for semantic segmentation (6-class LULC) with:
#   - CrossModalPromptLearner + hierarchical cross-modal transfer (kept intact via import from trainers/hicropl.py)
#   - LKP behavior (kept intact inside CrossModalPromptLearner.forward() in trainers/hicropl.py)
#   - DeepLabV3-style decoder head (ASPP) to produce dense pixel embeddings
#   - CLIP semantics (Option A): per-pixel logits via cosine similarity with prompted CLIP text embeddings
#
# IMPORTANT ASSUMPTION (minimal change elsewhere):
#   Your CLIP visual forward should return patch tokens (or a tuple including patch tokens).
#   This trainer supports common return formats:
#       - tuple: (global_feat, patch_tokens) or (patch_tokens, ...)
#       - dict:  {"patch_tokens": ...} or {"tokens": ...}
#       - tensor: patch_tokens directly
#   patch_tokens can be either:
#       - [B, 1+N, D] (includes class token)  -> trainer drops first token
#       - [B, N, D]
#
# Dataset expected output (from datasets/lulc_seg.py):
#   batch["img"]  : float tensor [B,3,H,W]
#   batch["mask"] : long  tensor [B,H,W] with values 0..C-1 (or ignore_index)
#   batch["meta"]["numeric"] optional: float tensor [B,F]

import math
import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from dassl.data.transforms import build_seg_transform

from datasets.lulc_seg import LULCSegDataset, CLASSNAMES_LULC_6



# Reuse HiCroPL core components exactly (keeps hierarchical transfer + LKP intact)
from .hicropl import load_clip_to_cpu, CrossModalPromptLearner, TextEncoder
import wandb

# -------------------------
# DeepLabV3-style decoder
# -------------------------

class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
        # 3x3 dilated
        for r in atrous_rates:
            self.branches.append(ASPPConv(in_ch, out_ch, dilation=r))
        # image pooling
        self.branches.append(ASPPPooling(in_ch, out_ch))

        proj_in = out_ch * len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(proj_in, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3EmbeddingHead(nn.Module):
    """
    Produces dense pixel embeddings (not class logits).
    We keep it light: ASPP -> 3x3 conv -> 1x1 conv to embed_dim.
    """
    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        aspp_ch: int = 256,
        atrous_rates=(6, 12, 18),
    ):
        super().__init__()
        self.aspp = ASPP(in_ch, aspp_ch, atrous_rates=atrous_rates)
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch, aspp_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.to_embed = nn.Conv2d(aspp_ch, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        x = self.fuse(x)
        x = self.to_embed(x)
        return x


# -------------------------
# CLIP-semantics segmentation model
# -------------------------

def _infer_grid_hw(num_tokens: int, H: int, W: int, patch: int = 16) -> Tuple[int, int]:
    """
    Prefer H//patch, W//patch; fallback to sqrt(num_tokens) if needed.
    """
    gh = max(1, H // patch)
    gw = max(1, W // patch)
    if gh * gw == num_tokens:
        return gh, gw
    s = int(math.sqrt(num_tokens))
    if s * s == num_tokens:
        return s, s
    # last resort: keep gh,gw but allow reshape error to be explicit
    return gh, gw


def _extract_patch_tokens(visual_out: Any) -> torch.Tensor:
    """
    Supports:
      - tensor [B,N,D]
      - tuple with a [B,N,D] element
      - dict with key "patch_tokens" or "tokens"
    """
    if isinstance(visual_out, torch.Tensor):
        return visual_out

    if isinstance(visual_out, dict):
        if "patch_tokens" in visual_out:
            return visual_out["patch_tokens"]
        if "tokens" in visual_out:
            return visual_out["tokens"]
        raise RuntimeError(f"visual_out dict keys={list(visual_out.keys())}, expected 'patch_tokens' or 'tokens'")

    if isinstance(visual_out, (tuple, list)):
        # choose the first 3D tensor we can find
        for x in visual_out:
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                return x
        raise RuntimeError("visual_out tuple/list did not contain a 3D tensor [B,N,D]")

    raise RuntimeError(f"Unsupported visual_out type: {type(visual_out)}")


class CustomCLIPSeg(nn.Module):
    """
    Prompted CLIP -> patch tokens -> DeepLabV3 embedding head -> cosine similarity with prompted text.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames

        self.prompt_learner = CrossModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # [C,77]
        self.text_encoder = TextEncoder(clip_model)

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # CLIP text embedding dim is the output of text_encoder (after projection)
        # In OpenAI CLIP ViT-B/16, this is 512.
        embed_dim = int(clip_model.text_projection.shape[1]) if clip_model.text_projection.dim() == 2 else int(clip_model.text_projection.shape[0])

        # ViT-B/16 patch token width is typically 768
        in_ch = getattr(cfg.MODEL, "VIT_WIDTH", 768)
        # If not specified, keep a safe default:
        if not isinstance(in_ch, int):
            in_ch = 768

        # Optional config knobs
        seg_cfg = getattr(cfg.TRAINER, "HICROPL_SEG", None)
        aspp_ch = getattr(seg_cfg, "ASPP_CHANNELS", 256) if seg_cfg is not None else 256
        # atrous rates can be tuned; keep standard DeepLabV3 defaults
        atrous_rates = getattr(seg_cfg, "ATROUS_RATES", (6, 12, 18)) if seg_cfg is not None else (6, 12, 18)

        self.decoder = DeepLabV3EmbeddingHead(in_ch=in_ch, embed_dim=embed_dim, aspp_ch=aspp_ch, atrous_rates=atrous_rates)

        # Patch size for grid reshape (ViT-B/16 => 16)
        self.patch_size = getattr(seg_cfg, "PATCH_SIZE", 16) if seg_cfg is not None else 16

    def forward(
        self,
        image: torch.Tensor,
        meta_num: Optional[torch.Tensor] = None,  # kept for later (metadata-conditioned prompts), safe to ignore now
    ) -> torch.Tensor:
        """
        Returns:
          logits: [B, C, H, W]
        """
        B, _, H, W = image.shape

        # Prompts (keeps HiCroPL hierarchical transfer + LKP intact)
        text_input, visual_ctx0, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner()

        # Text features: [C, embed_dim]
        text_features = self.text_encoder(text_input, self.tokenized_prompts, cross_prompts_text_deeper)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)

        # Visual forward: must provide patch tokens
        # (Your CLIP visual forward should be extended to return patch tokens.)
        visual_out = self.image_encoder(image.type(self.dtype), visual_ctx0, cross_prompts_visual_deeper, return_patch_tokens=True)
        patch_tokens = _extract_patch_tokens(visual_out)  # [B, N or 1+N, D]

        # Drop class token if present
        # If token count matches (H//p * W//p + 1), we assume first token is CLS.
        gh_guess = max(1, H // self.patch_size)
        gw_guess = max(1, W // self.patch_size)
        if patch_tokens.shape[1] == gh_guess * gw_guess + 1:
            patch_tokens = patch_tokens[:, 1:, :]

        # Reshape tokens -> feature map
        gh, gw = _infer_grid_hw(patch_tokens.shape[1], H, W, patch=self.patch_size)
        feat = patch_tokens.transpose(1, 2).contiguous().view(B, patch_tokens.shape[2], gh, gw)  # [B, D, gh, gw]

        # Dense pixel embeddings at low-res grid
        pix_emb = self.decoder(feat)  # [B, embed_dim, gh, gw]

        # Upsample to input resolution
        pix_emb = F.interpolate(pix_emb, size=(H, W), mode="bilinear", align_corners=False)

        # Normalize pixel embeddings
        pix_emb = pix_emb / (pix_emb.norm(dim=1, keepdim=True) + 1e-12)

        # Similarity logits
        logit_scale = self.logit_scale.exp()
        # logits: [B, C, H, W]
        logits = logit_scale * torch.einsum("bchw,kc->bkhw", pix_emb, text_features)

        return logits


# -------------------------
# Segmentation losses
# -------------------------

def _soft_dice_loss_from_logits(
        logits: torch.Tensor,          # [B,C,H,W]
        target: torch.Tensor,          # [B,H,W]
        ignore_index: int = 255,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Soft Dice loss for multi-class segmentation.
        - Ignores ignore_index pixels
        - Averages Dice over classes present in GT in this batch (more stable)
        """
        logits = logits.float()
        B, C, H, W = logits.shape

        valid = target.ne(ignore_index)
        if int(valid.sum().item()) == 0:
            return logits.new_tensor(0.0)

        # one_hot cannot handle values like 255 -> make a safe target
        target_safe = target.clone()
        target_safe[~valid] = 0  # any valid class id is fine here because we mask later

        target_1h = F.one_hot(target_safe, num_classes=C).permute(0, 3, 1, 2).float()  # [B,C,H,W]

        valid_f = valid.unsqueeze(1).float()  # [B,1,H,W]
        probs = torch.softmax(logits, dim=1) * valid_f
        target_1h = target_1h * valid_f

        inter = (probs * target_1h).sum(dim=(0, 2, 3))       # [C]
        p_sum = probs.sum(dim=(0, 2, 3))                     # [C]
        t_sum = target_1h.sum(dim=(0, 2, 3))                 # [C]

        dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)   # [C]

        present = t_sum > 0
        if present.any():
            return 1.0 - dice[present].mean()

        return logits.new_tensor(0.0)


def _focal_loss_from_logits(
    logits: torch.Tensor,          # [B,C,H,W]
    target: torch.Tensor,          # [B,H,W]
    ignore_index: int = 255,
    gamma: float = 2.0,
    class_weight: Optional[torch.Tensor] = None,  # [C] on same device
) -> torch.Tensor:
    """
    Multi-class focal loss:
    FL = (1 - pt)^gamma * (-log pt)
    If class_weight is provided, it multiplies each pixel loss by class_weight[y].
    """
    logits = logits.float()
    B, C, H, W = logits.shape

    target_flat = target.view(-1)                                  # [N]
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)        # [N,C]

    valid = target_flat.ne(ignore_index)
    if int(valid.sum().item()) == 0:
        return logits.new_tensor(0.0)

    tgt = target_flat[valid]                                       # [Nv]
    log_probs = F.log_softmax(logits_flat[valid], dim=1)           # [Nv,C]
    log_pt = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)      # [Nv]
    pt = log_pt.exp()

    loss = ((1.0 - pt) ** float(gamma)) * (-log_pt)                # [Nv]

    if class_weight is not None:
        loss = loss * class_weight[tgt]

    return loss.mean()


# -------------------------
# Segmentation metrics
# -------------------------


@torch.no_grad()
def _confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    pred:   [B,H,W] int64
    target: [B,H,W] int64
    returns: [C,C] where rows=gt, cols=pred
    """
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    if ignore_index is not None:
        mask = target != int(ignore_index)
        pred = pred[mask]
        target = target[mask]

    k = target * num_classes + pred
    cm = torch.bincount(k, minlength=num_classes * num_classes)
    cm = cm.view(num_classes, num_classes)
    return cm


@torch.no_grad()
def _miou_from_cm(cm: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    Returns:
      miou (float), per_class_iou (Tensor[C])
    """
    cm = cm.to(torch.float64)
    inter = torch.diag(cm)
    gt = cm.sum(dim=1)
    pd = cm.sum(dim=0)
    union = gt + pd - inter
    iou = inter / (union + 1e-12)
    valid = union > 0
    miou = float(iou[valid].mean().item()) if valid.any() else 0.0
    return miou, iou.to(torch.float32)


# -------------------------
# Trainer
# -------------------------

@TRAINER_REGISTRY.register()
class HiCroPL_Seg(TrainerX):
    """
    Segmentation trainer.

    NOTES:
      - We set train_loader_u = train_loader_x to satisfy TrainerX.run_epoch.
      - We override after_epoch() and test() to avoid classification evaluator.
      - We still register model with self.register_model(...) for checkpointing.
    """

    def check_cfg(self, cfg):
        prec = cfg.TRAINER.HICROPL.PREC
        assert prec in ["fp16", "fp32", "amp"], "cfg.TRAINER.HICROPL.PREC must be one of fp16/fp32/amp"

    def _get_root_dir(self) -> str:
        # For this dataset, pass --root data/bing_rgb
        root_dir = getattr(self.cfg.DATASET, "ROOT", "")
        if root_dir is None or str(root_dir).strip() == "":
            root_dir = "data/bing_rgb"
        return str(root_dir)

    def _get_classnames(self):
        # Allow config override; otherwise fallback to fixed list
        cn = getattr(self.cfg.DATASET, "CLASSNAMES", None)
        if cn is None:
            return CLASSNAMES_LULC_6
        return list(cn)

    def build_data_loader(self):
        cfg = self.cfg
        root_dir = self._get_root_dir()
        classnames = self._get_classnames()

        self.num_classes = len(classnames)
        self.classnames = classnames
        self.lab2cname = {i: n for i, n in enumerate(classnames)}

        ignore_index = getattr(cfg.DATASET, "IGNORE_INDEX", 255)

        tfm_train = build_seg_transform(cfg, is_train=True, ignore_index=int(ignore_index))
        tfm_test = build_seg_transform(cfg, is_train=False, ignore_index=int(ignore_index))

        # Dataset uses fixed structure: root/{split}/images and root/{split}/masks
        train_set = LULCSegDataset(
            root=root_dir,
            split="train",
            metadata_json="metadata.json",
            transforms=tfm_train,
            mask_suffix=getattr(cfg.DATASET, "MASK_SUFFIX", "_gt"),
            validate_labels=getattr(cfg.DATASET, "VALIDATE_LABELS", False),
            return_raw_meta=getattr(cfg.DATASET, "RETURN_RAW_META", False),
        )

        val_set = LULCSegDataset(
            root=root_dir,
            split="val",
            metadata_json="metadata.json",
            transforms=tfm_test,
            mask_suffix=getattr(cfg.DATASET, "MASK_SUFFIX", "_gt"),
            validate_labels=False,
            return_raw_meta=getattr(cfg.DATASET, "RETURN_RAW_META", False),
        )

        # Dataloader params
        bs_train = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        bs_test = cfg.DATALOADER.TEST.BATCH_SIZE
        nw = cfg.DATALOADER.NUM_WORKERS

        self.train_loader_x = DataLoader(
            train_set,
            batch_size=bs_train,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            drop_last=True,
        )

        # TrainerX requires train_loader_u; use same loader (fully supervised).
        self.train_loader_u = self.train_loader_x

        self.val_loader = DataLoader(
            val_set,
            batch_size=bs_test,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            drop_last=False,
        )

        # If you don't have a separate test split, point test_loader to val_loader.
        self.test_loader = self.val_loader

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # CLIP default is fp16; lift to fp32 if needed
        if cfg.TRAINER.HICROPL.PREC in ("fp32", "amp"):
            clip_model.float()

        print("Building segmentation model (prompted CLIP + DeepLabV3 head)")
        self.model = CustomCLIPSeg(cfg, classnames, clip_model)

        print("Turning off gradients in the CLIP encoders; training prompts + decoder (and optional meta encoder later)")
        train_keys = ("prompt_learner", "decoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(any(k in name for k in train_keys))

        enabled = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Parameters to be updated ({len(enabled)}):")
        for n in enabled:
            print(f"  - {n}")

        self.model.to(self.device)

        self._init_seg_loss_cfg()

        # ----------------------------
        # Optimizer with prompt-only LR
        # ----------------------------
        base_lr = float(cfg.OPTIM.LR)
        mult = float(getattr(cfg.OPTIM, "PROMPT_LR_MULT", 1.0))
        prompt_lr = base_lr * mult

        prompt_params = []
        other_params = []

        def _is_prompt_vector(name: str, p: torch.nn.Parameter) -> bool:
            # "prompt vectors only" heuristic:
            # - must be inside prompt_learner
            # - name contains ctx/prompt
            # - tensor has >=2 dims (vectors/tables), avoids tiny scalars/biases
            if "prompt_learner" not in name:
                return False
            if not any(k in name for k in ["ctx", "prompt"]):
                return False
            return p.ndim >= 2

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_prompt_vector(n, p):
                prompt_params.append(p)
            else:
                other_params.append(p)

        print(f"[OPT] base_lr={base_lr} prompt_lr={prompt_lr} (x{mult})")
        print(f"[OPT] prompt_params={len(prompt_params)} other_params={len(other_params)}")

        # Build optimizer manually (keeps your cfg knobs)
        optim_name = cfg.OPTIM.NAME.lower()
        wd = float(cfg.OPTIM.WEIGHT_DECAY)
        eps = float(cfg.OPTIM.EPS)

        param_groups = [
            {"params": other_params, "lr": base_lr, "weight_decay": wd},
            {"params": prompt_params, "lr": prompt_lr, "weight_decay": wd},
        ]

        if optim_name == "adam":
            betas = (float(cfg.OPTIM.ADAM_BETA1), float(cfg.OPTIM.ADAM_BETA2))
            self.optim = torch.optim.Adam(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=wd)

        elif optim_name == "adamw":
            betas = (float(cfg.OPTIM.ADAM_BETA1), float(cfg.OPTIM.ADAM_BETA2))
            self.optim = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=wd)

        elif optim_name == "sgd":
            self.optim = torch.optim.SGD(
                param_groups,
                lr=base_lr,
                momentum=float(cfg.OPTIM.MOMENTUM),
                dampening=float(cfg.OPTIM.SGD_DAMPNING),
                nesterov=bool(cfg.OPTIM.SGD_NESTEROV),
                weight_decay=wd,
            )

        elif optim_name == "rmsprop":
            self.optim = torch.optim.RMSprop(
                param_groups,
                lr=base_lr,
                momentum=float(cfg.OPTIM.MOMENTUM),
                alpha=float(cfg.OPTIM.RMSPROP_ALPHA),
                eps=eps,
                weight_decay=wd,
            )

        else:
            raise ValueError(f"Unsupported optimizer: {cfg.OPTIM.NAME}")

        # Scheduler will preserve group-wise ratios automatically
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("HiCroPL_Seg", self.model, self.optim, self.sched)


        self.scaler = GradScaler() if cfg.TRAINER.HICROPL.PREC == "amp" else None

        self._prev_prompt_snapshot = None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), using DataParallel")
            self.model = nn.DataParallel(self.model)

        self._wandb_on = (wandb is not None) and (os.environ.get("WANDB_MODE", "").lower() != "disabled")
        self._wandb_run = None
        if self._wandb_on and (wandb.run is None):
            self._wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "HiCroPLSeg"),
                name=os.environ.get("WANDB_RUN_NAME", osp.basename(self.output_dir)),
                config=self.cfg,
                dir=self.output_dir,
            )
        else:
            self._wandb_run = wandb.run if (wandb is not None) else None

        # train-epoch accumulators
        self._train_cm = None
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self._last_eval_stats = {}  # e.g., {"val": {...}, "test": {...}}

    @torch.no_grad()
    def _get_prompt_learner(self):
        m = self.model
        if isinstance(m, nn.DataParallel):
            m = m.module
        return m.prompt_learner

    @torch.no_grad()
    def _snapshot_prompts(self):
        """
        Returns CPU float32 copies of prompt tensors:
        - cross_prompts_text: list of [n_ctx, ctx_dim] (or [n_cls,n_ctx,ctx_dim] depending on impl)
        - cross_prompts_visual: list of [n_ctx, v_dim]
        """
        pl = self._get_prompt_learner()

        text_layers = [p.detach().float().cpu().clone() for p in pl.cross_prompts_text]
        vis_layers  = [p.detach().float().cpu().clone() for p in pl.cross_prompts_visual]

        return {"text": text_layers, "visual": vis_layers}

    @torch.no_grad()
    def _log_and_dump_prompts(self):
        snap = self._snapshot_prompts()

        dump_dir = osp.join(self.output_dir, "prompt_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = osp.join(dump_dir, f"epoch_{self.epoch+1:04d}.pt")
        torch.save(snap, dump_path)

        # ---- print small slices (not the whole huge tensors) ----
        def _stats(x: torch.Tensor):
            return {
                "l2": float(x.norm().item()),
                "mean": float(x.mean().item()),
                "std": float(x.std(unbiased=False).item()),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
            }

        print(f"\n[PromptDump] epoch={self.epoch+1} saved={dump_path}")

        # Print first token vector slice from layer 0 (most informative)
        t0 = snap["text"][0]
        v0 = snap["visual"][0]

        # If text layer is [n_cls, n_ctx, dim], show class 0 token 0
        if t0.dim() == 3:
            t0_vec = t0[0, 0]  # [dim]
        else:
            t0_vec = t0[0]     # [dim]

        v0_vec = v0[0]         # [dim]

        print(f"  Text prompt L0: shape={tuple(t0.shape)} stats={_stats(t0)}")
        print(f"    Text L0 token[0] first 12 dims: {t0_vec[:12].tolist()}")
        print(f"  Visual prompt L0: shape={tuple(v0.shape)} stats={_stats(v0)}")
        print(f"    Visual L0 token[0] first 12 dims: {v0_vec[:12].tolist()}")

        # ---- log L2 norms (and deltas) to tensorboard ----
        for i, t in enumerate(snap["text"]):
            self.write_scalar(f"prompts/text_L{i}/l2", float(t.norm().item()), self.epoch)
        for i, v in enumerate(snap["visual"]):
            self.write_scalar(f"prompts/visual_L{i}/l2", float(v.norm().item()), self.epoch)

        if self._prev_prompt_snapshot is not None:
            prev = self._prev_prompt_snapshot

            for i, (t, pt) in enumerate(zip(snap["text"], prev["text"])):
                dt = (t - pt)
                self.write_scalar(f"prompts/text_L{i}/delta_l2", float(dt.norm().item()), self.epoch)
            for i, (v, pv) in enumerate(zip(snap["visual"], prev["visual"])):
                dv = (v - pv)
                self.write_scalar(f"prompts/visual_L{i}/delta_l2", float(dv.norm().item()), self.epoch)

        self._prev_prompt_snapshot = snap
        

    def parse_batch_train(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        img = batch["img"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True).long()

        meta_num = None
        # Default PyTorch collate will collate nested dicts if present for all samples.
        if "meta" in batch and batch["meta"] is not None:
            meta = batch["meta"]
            if isinstance(meta, dict) and "numeric" in meta and meta["numeric"] is not None:
                meta_num = meta["numeric"].to(self.device, non_blocking=True).float()

        return img, mask, meta_num

    def parse_batch_test(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.parse_batch_train(batch)

    def _init_seg_loss_cfg(self):
        """
        Reads loss settings from cfg.TRAINER.HICROPL_SEG (with safe defaults).
        Supported:
            LOSS: "ce", "ce_dice", "focal", "focal_dice"
            CE_WEIGHT, DICE_WEIGHT
            FOCAL_GAMMA
            CLASS_WEIGHTS (list/tuple length C)  [optional]
            DICE_EPS
        """
        seg_cfg = getattr(self.cfg.TRAINER, "HICROPL_SEG", None)

        self._loss_name = str(getattr(seg_cfg, "LOSS", "ce_dice")).lower()
        self._loss_ce_w = float(getattr(seg_cfg, "CE_WEIGHT", 1.0))
        self._loss_dice_w = float(getattr(seg_cfg, "DICE_WEIGHT", 1.0))
        self._loss_focal_gamma = float(getattr(seg_cfg, "FOCAL_GAMMA", 2.0))
        self._loss_dice_eps = float(getattr(seg_cfg, "DICE_EPS", 1e-6))

        cw = getattr(seg_cfg, "CLASS_WEIGHTS", None)
        self._loss_class_weight = None
        if cw is not None and isinstance(cw, (list, tuple)) and len(cw) > 0:
            self._loss_class_weight = torch.tensor(list(cw), device=self.device, dtype=torch.float32)

        valid_modes = {"ce", "ce_dice", "focal", "focal_dice"}
        if self._loss_name not in valid_modes:
            print(f"[WARN] Unknown HICROPL_SEG.LOSS='{self._loss_name}', falling back to 'ce_dice'")
            self._loss_name = "ce_dice"

        print(
            f"[LOSS] mode={self._loss_name} ce_w={self._loss_ce_w} "
            f"dice_w={self._loss_dice_w} gamma={self._loss_focal_gamma}"
        )

    def _compute_seg_loss(self, logits: torch.Tensor, mask: torch.Tensor, ignore_index: int):
        """
        Returns:
            total_loss (Tensor scalar),
            comps (dict of floats for logging)
        """
        if not hasattr(self, "_loss_name"):
            self._init_seg_loss_cfg()

        logits_fp32 = logits.float()

        valid = mask.ne(ignore_index)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            z = logits_fp32.sum() * 0.0
            return z, {"total": 0.0, "ce": 0.0, "focal": 0.0, "dice": 0.0}

        # --- CE (pixel-normalized) ---
        ce_sum = F.cross_entropy(
            logits_fp32,
            mask,
            ignore_index=ignore_index,
            reduction="sum",
            weight=self._loss_class_weight,
        )
        ce = ce_sum / float(n_valid)

        # --- focal (optional) ---
        if self._loss_name.startswith("focal"):
            focal = _focal_loss_from_logits(
                logits_fp32,
                mask,
                ignore_index=ignore_index,
                gamma=self._loss_focal_gamma,
                class_weight=self._loss_class_weight,
            )
            cls_loss = focal
        else:
            focal = logits_fp32.new_tensor(0.0)
            cls_loss = ce

        # --- dice (optional) ---
        if self._loss_name.endswith("dice"):
            dice = _soft_dice_loss_from_logits(
                logits_fp32,
                mask,
                ignore_index=ignore_index,
                eps=self._loss_dice_eps,
            )
        else:
            dice = logits_fp32.new_tensor(0.0)

        total = self._loss_ce_w * cls_loss + self._loss_dice_w * dice

        comps = {
            "total": float(total.detach().item()),
            "ce": float(ce.detach().item()),
            "focal": float(focal.detach().item()),
            "dice": float(dice.detach().item()),
        }
        return total, comps

    @torch.no_grad()
    def _sliding_window_logits(
        self,
        img: torch.Tensor,                    # [1,3,H,W]
        meta_num: Optional[torch.Tensor],     # [1,M] or None
        tile: int = 224,
        stride: int = 112,
    ) -> torch.Tensor:
        """Run sliding-window inference and stitch logits back to [1,C,H,W]."""
        assert img.dim() == 4 and img.size(0) == 1, "Pass a single image [1,3,H,W]"
        device = img.device
        _, _, H, W = img.shape
        C = self.num_classes

        # Pad if needed (for small images)
        pad_h = max(tile - H, 0)
        pad_w = max(tile - W, 0)
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)

        Hp, Wp = img.shape[-2], img.shape[-1]

        # Make sure last tile covers the border
        ys = list(range(0, max(Hp - tile + 1, 1), stride))
        xs = list(range(0, max(Wp - tile + 1, 1), stride))
        if ys[-1] != Hp - tile:
            ys.append(Hp - tile)
        if xs[-1] != Wp - tile:
            xs.append(Wp - tile)

        out = torch.zeros((1, C, Hp, Wp), device=device, dtype=torch.float32)
        cnt = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.float32)

        prec = self.cfg.TRAINER.HICROPL.PREC

        for y in ys:
            for x in xs:
                patch = img[:, :, y:y + tile, x:x + tile]

                if prec == "amp":
                    with autocast():
                        logits_patch = self.model(patch, meta_num=meta_num)
                else:
                    logits_patch = self.model(patch, meta_num=meta_num)

                logits_patch = logits_patch.float()

                out[:, :, y:y + tile, x:x + tile] += logits_patch
                cnt[:, :, y:y + tile, x:x + tile] += 1.0

        out = out / cnt.clamp_min(1.0)
        out = out[:, :, :H, :W]  # remove padding
        return out


    def forward_backward(self, batch_x, batch_u=None):
        """
        TrainerX.run_epoch calls forward_backward(batch_x, batch_u).
        We ignore batch_u (fully supervised).
        """
        img, mask, meta_num = self.parse_batch_train(batch_x)

        prec = self.cfg.TRAINER.HICROPL.PREC
        ignore_index = int(getattr(self.cfg.DATASET, "IGNORE_INDEX", 255))

        # If the whole crop is ignore pixels, skip update to avoid NaNs
        valid = mask.ne(ignore_index)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            loss_summary = {"loss": 0.0}
            if self.batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0:
                self.write_scalar("train/loss", 0.0, self.epoch * self.num_batches + self.batch_idx)
            return loss_summary

        self.optim.zero_grad(set_to_none=True)

        if prec == "amp":
            with autocast():
                logits = self.model(img, meta_num=meta_num)  # [B,C,H,W]

            loss, comps = self._compute_seg_loss(logits, mask, ignore_index)

            if not torch.isfinite(loss):
                return {"loss": float("nan")}

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            logits = self.model(img, meta_num=meta_num)
            loss, comps = self._compute_seg_loss(logits, mask, ignore_index)

            if not torch.isfinite(loss):
                return {"loss": float("nan")}

            loss.backward()
            self.optim.step()

        loss_value = float(loss.item())
        loss_summary = {"loss": loss_value, **comps}

        if self.batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0:
            step = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", loss_value, step)
            self.write_scalar("train/loss_ce", float(comps["ce"]), step)
            self.write_scalar("train/loss_focal", float(comps["focal"]), step)
            self.write_scalar("train/loss_dice", float(comps["dice"]), step)

        return loss_summary

    def after_epoch(self):
        """
        Same logic as SimpleTrainer.after_epoch, but metric = mIoU.
        """
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")  # mIoU
            is_best = curr_result >= self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar",
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

  

        self._log_and_dump_prompts()
        
    @torch.no_grad()
    def test(self, split: Optional[str] = None) -> float:
        """
        Computes mIoU on val/test and returns it as the main score.
        Also logs per-class IoU and a numerically safe val/test loss.
        """
        self.set_model_mode("eval")

        if split is None:
            split = "test"

        if split == "val" and self.val_loader is not None:
            loader = self.val_loader
        else:
            loader = self.test_loader

        num_classes = self.num_classes
        ignore_index = int(getattr(self.cfg.DATASET, "IGNORE_INDEX", 255))

        # Confusion matrix for mIoU
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

        # Safe loss accumulation (pixel-weighted)
        total_loss_sum = 0.0     # sum of CE over valid pixels
        total_valid_px = 0       # number of valid pixels
        skipped_batches = 0
        nonfinite_batches = 0

        for batch in loader:
            img, mask, meta_num = self.parse_batch_test(batch)

            tile = int(self.cfg.INPUT.SIZE[0])  # 224
            stride = tile // 2                 # 112

            logits_list = []
            B = img.size(0)

            for i in range(B):
                img_i = img[i:i+1]
                meta_i = meta_num[i:i+1] if meta_num is not None else None

                # If test images are larger than tile, use sliding window
                if img_i.size(-1) > tile or img_i.size(-2) > tile:
                    logits_i = self._sliding_window_logits(img_i, meta_i, tile=tile, stride=stride)
                else:
                    prec = self.cfg.TRAINER.HICROPL.PREC
                    if prec == "amp":
                        with autocast():
                            logits_i = self.model(img_i, meta_num=meta_i)
                    else:
                        logits_i = self.model(img_i, meta_num=meta_i)

                    logits_i = logits_i.float()

                logits_list.append(logits_i)

            logits = torch.cat(logits_list, dim=0)  # [B,C,H,W]


            # If logits contain NaN/Inf, sanitize to avoid NaN loss logging
            if not torch.isfinite(logits).all():
                nonfinite_batches += 1
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            # mIoU confusion matrix update (CPU)
            pred = torch.argmax(logits, dim=1)  # [B,H,W]
            cm += _confusion_matrix(
                pred.detach().cpu(),
                mask.detach().cpu(),
                num_classes,
                ignore_index=ignore_index
            )

            # Safe loss: skip if this batch has 0 valid pixels
            valid = mask.ne(ignore_index)
            n_valid = int(valid.sum().item())

            if n_valid == 0:
                skipped_batches += 1
                continue

            loss_sum = F.cross_entropy(
                logits.float(),          # compute CE in fp32 for stability
                mask,
                ignore_index=ignore_index,
                reduction="sum",         # sum over valid pixels (never 0/0)
            )

            total_loss_sum += float(loss_sum.item())
            total_valid_px += n_valid

        # Compute IoU
        miou, per_iou = _miou_from_cm(cm)

        # Safe average loss
        avg_loss = (total_loss_sum / float(total_valid_px)) if total_valid_px > 0 else 0.0

        # Logging (tensorboard)
        self.write_scalar(f"{split}/loss", float(avg_loss), self.epoch)
        self.write_scalar(f"{split}/mIoU", float(miou), self.epoch)
        for i, cname in enumerate(self.classnames):
            self.write_scalar(f"{split}/IoU_{cname}", float(per_iou[i].item()), self.epoch)

        # Console prints
        print(f"[{split}] val_loss: {avg_loss:.4f} (valid_px={total_valid_px}, skipped_batches={skipped_batches}, nonfinite_batches={nonfinite_batches})")
        print(f"[{split}] mIoU: {miou:.4f}")
        for i, cname in enumerate(self.classnames):
            print(f"  - IoU[{i:02d}] {cname}: {float(per_iou[i].item()):.4f}")

        # wandb logging (fixes the old IoU[i] bug by using per_iou)
        if wandb.run is not None:
            log_dict = {
                f"{split}/loss": float(avg_loss),
                f"{split}/mIoU": float(miou),
                "epoch": self.epoch + 1,
            }
            for i in range(num_classes):
                log_dict[f"{split}/IoU_class_{i}"] = float(per_iou[i].item())
            wandb.log(log_dict)

        return float(miou)

