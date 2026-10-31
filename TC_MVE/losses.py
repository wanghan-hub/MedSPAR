"""Losses for Steer-MedSAM."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def mask_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def make_patch_targets(mask: torch.Tensor, patch_logits: torch.Tensor) -> torch.Tensor:
    """Downsample a binary mask to the patch-logit grid.

    Uses area interpolation, equivalent to a soft foreground fraction per patch.
    """
    return F.interpolate(mask.float(), size=patch_logits.shape[-2:], mode="area")


def patch_bce_loss(patch_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    patch_targets = make_patch_targets(mask, patch_logits)
    return F.binary_cross_entropy_with_logits(patch_logits, patch_targets)


def compute_total_loss(
    pred_mask_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    patch_logits: torch.Tensor | None = None,
    lambda_patch: float = 0.2,
) -> Dict[str, torch.Tensor]:
    loss_dice = dice_loss_with_logits(pred_mask_logits, gt_mask)
    loss_bce = mask_bce_loss(pred_mask_logits, gt_mask)
    loss_mask = loss_dice + loss_bce

    if patch_logits is not None and lambda_patch > 0:
        loss_patch = patch_bce_loss(patch_logits, gt_mask)
    else:
        loss_patch = pred_mask_logits.new_tensor(0.0)

    loss_total = loss_mask + float(lambda_patch) * loss_patch
    return {
        "loss": loss_total,
        "loss_mask": loss_mask.detach(),
        "loss_dice": loss_dice.detach(),
        "loss_bce": loss_bce.detach(),
        "loss_patch": loss_patch.detach(),
    }


@torch.no_grad()
def binary_dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, preds.ndim))
    inter = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    union = pred_sum + target_sum - inter
    dice = (2 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice, iou
