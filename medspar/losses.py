"""Losses for LLM-MedSAM."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from experiment_protocol import common_mask_loss


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Mask logits/targets must match Bx1xHxW exactly, got {logits.shape}, {targets.shape}")
    logits = logits.float()
    targets = targets.float()
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def mask_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return dice_loss_with_logits(logits, targets) + F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def binary_dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Mask metric inputs must be identical Bx1xHxW shapes; broadcasting is forbidden")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Mask probability threshold must be in [0,1]")
    probs = torch.sigmoid(logits.float())
    preds = (probs > threshold).float()
    targets = targets.float()
    dims = tuple(range(1, preds.ndim))
    inter = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    tgt_sum = targets.sum(dim=dims)
    dice = (2 * inter + eps) / (pred_sum + tgt_sum + eps)
    union = pred_sum + tgt_sum - inter
    iou = (inter + eps) / (union + eps)
    return dice, iou


def compute_llm_medsam_loss(
    lm_loss: Optional[torch.Tensor],
    pred_masks: Optional[torch.Tensor],
    gt_masks: torch.Tensor,
    lambda_lm: float = 1.0,
    lambda_mask: float = 1.0,
    *,
    spatial_logits: Optional[torch.Tensor] = None,
    pred_box_norm: Optional[torch.Tensor] = None,
    gt_box_norm: Optional[torch.Tensor] = None,
    pred_box_embeddings: Optional[torch.Tensor] = None,
    gt_box_embeddings: Optional[torch.Tensor] = None,
    lambda_attn: float = 0.0,
    lambda_box: float = 0.0,
    lambda_prompt: float = 0.0,
) -> Dict[str, torch.Tensor]:
    zero = gt_masks.new_zeros((), dtype=torch.float32)
    weights = (lambda_lm, lambda_mask, lambda_attn, lambda_box, lambda_prompt)
    if any(w < 0 for w in weights):
        raise ValueError("Loss weights must be nonnegative")
    if lambda_mask and pred_masks is None:
        raise ValueError("Active mask loss requires decoded masks")
    if lambda_lm and lm_loss is None:
        raise ValueError("Active LM loss requires a language-model loss")
    if lambda_attn and spatial_logits is None:
        raise ValueError("Active spatial prior loss requires spatial_logits")
    if lambda_box and (pred_box_norm is None or gt_box_norm is None):
        raise ValueError("Active box loss requires predicted and GT normalized XYXY boxes")
    if lambda_prompt and (pred_box_embeddings is None or gt_box_embeddings is None):
        raise ValueError("Active prompt distillation requires box-only student and teacher embeddings")
    components = {
        "loss_lm": lm_loss.float() if lambda_lm else zero,
        "loss_mask": common_mask_loss(pred_masks, gt_masks) if lambda_mask else zero,
        "loss_attn": spatial_attention_loss(spatial_logits, gt_masks) if lambda_attn else zero,
        "loss_box": box_localization_loss(pred_box_norm, gt_box_norm) if lambda_box else zero,
        "loss_prompt": prompt_distillation_loss(pred_box_embeddings, gt_box_embeddings) if lambda_prompt else zero,
    }
    total = sum(float(w) * value for w, value in zip(weights, components.values()))
    return {"loss": total, "loss_total": total.detach(), **{k: v.detach() for k, v in components.items()}}


def spatial_attention_loss(spatial_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    target = F.interpolate(gt_masks.float(), size=spatial_logits.shape[-2:], mode="area")
    return mask_loss_with_logits(spatial_logits, target)


@torch.no_grad()
def masks_to_boxes(masks: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """Derive XYXY edges from binary masks; right/bottom edges are exclusive.

    Empty masks map to a zero-area box, never a fabricated full-image target.
    """
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("Masks must have shape Bx1xHxW")
    foreground = masks[:, 0] > 0.5
    b, h, w = foreground.shape
    rows, cols = foreground.any(2), foreground.any(1)
    xs = torch.arange(w, device=masks.device)[None, :].expand(b, -1)
    ys = torch.arange(h, device=masks.device)[None, :].expand(b, -1)
    x1 = xs.masked_fill(~cols, w).amin(1)
    y1 = ys.masked_fill(~rows, h).amin(1)
    x2 = xs.masked_fill(~cols, -1).amax(1) + 1
    y2 = ys.masked_fill(~rows, -1).amax(1) + 1
    boxes = torch.stack([x1, y1, x2, y2], 1).float()
    boxes[~foreground.flatten(1).any(1)] = 0
    if normalized:
        boxes = boxes / boxes.new_tensor([w, h, w, h])
    return boxes


def box_iou_giou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """Aligned (not pairwise) B-vector IoU/GIoU without torchvision dependency."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected matching Bx4 XYXY boxes")
    pred, target = pred.float(), target.float()
    inter_wh = (torch.minimum(pred[:, 2:], target[:, 2:]) -
                torch.maximum(pred[:, :2], target[:, :2])).clamp_min(0)
    intersection = inter_wh.prod(1)
    area_pred = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(1)
    area_target = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(1)
    union = area_pred + area_target - intersection
    iou = intersection / union.clamp_min(eps)
    enclosing = (torch.maximum(pred[:, 2:], target[:, 2:]) -
                 torch.minimum(pred[:, :2], target[:, :2])).clamp_min(0).prod(1)
    giou = iou - (enclosing - union) / enclosing.clamp_min(eps)
    return iou, giou


def box_localization_loss(pred_box_norm: torch.Tensor, gt_box_norm: torch.Tensor) -> torch.Tensor:
    target = gt_box_norm.detach().float()
    _, giou = box_iou_giou(pred_box_norm, target)
    return F.smooth_l1_loss(pred_box_norm.float(), target) + (1 - giou).mean()


def prompt_distillation_loss(pred_box_embeddings: torch.Tensor, gt_box_embeddings: torch.Tensor) -> torch.Tensor:
    if pred_box_embeddings.shape != gt_box_embeddings.shape or pred_box_embeddings.shape[1] != 2:
        raise ValueError("Prompt distillation compares exactly two box-corner embeddings per image")
    student, teacher = pred_box_embeddings.float(), gt_box_embeddings.detach().float()
    cosine = 1 - F.cosine_similarity(student, teacher, dim=-1).mean()
    return cosine + 0.5 * F.smooth_l1_loss(student, teacher)


@torch.no_grad()
def localization_metrics(outputs, gt_masks: torch.Tensor, input_image_size=None):
    """Per-sample diagnostics in a single documented coordinate system."""
    spatial = outputs["spatial_logits"]
    target = F.interpolate(gt_masks.float(), size=spatial.shape[-2:], mode="area")
    dice, iou = binary_dice_iou_from_logits(spatial, (target > 0.5).float())
    gt_boxes = masks_to_boxes(gt_masks, normalized=True)
    box_iou, box_giou = box_iou_giou(outputs["pred_box_norm"], gt_boxes)
    coords, labels = outputs["point_coords"], outputs["point_labels"]
    input_h, input_w = input_image_size or outputs.get("prompt_input_size", gt_masks.shape[-2:])
    h, w = gt_masks.shape[-2:]
    x = ((coords[..., 0] + 0.5) * w / input_w).floor().long().clamp(0, w - 1)
    y = ((coords[..., 1] + 0.5) * h / input_h).floor().long().clamp(0, h - 1)
    batch = torch.arange(gt_masks.shape[0], device=coords.device)[:, None]
    hits = gt_masks[batch, 0, y, x] > 0.5
    pos, neg = labels.eq(1), labels.eq(0)
    pos_rate = (hits & pos).sum(1).float() / pos.sum(1).clamp_min(1)
    neg_rate = (~hits & neg).sum(1).float() / neg.sum(1).clamp_min(1)
    return {"spatial_prior_dice": dice, "spatial_prior_iou": iou,
            "box_iou": box_iou, "box_giou": box_giou,
            "positive_point_hit_rate": pos_rate, "negative_point_correct_rate": neg_rate}
