"""Bounding-box utilities for MedSAM-style box prompts."""

from __future__ import annotations

import random
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


Box = Tuple[float, float, float, float]


def _as_numpy_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    return mask


def mask_to_bbox(mask, empty_ok: bool = False) -> Box:
    """Return xyxy bbox from a binary mask.

    The returned coordinates are inclusive-style pixel coordinates and can be fed
    to MedSAM/SAM prompt encoder after conversion to float tensor.
    """
    mask_np = _as_numpy_mask(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        if empty_ok:
            h, w = mask_np.shape[-2:]
            return (0.0, 0.0, float(w - 1), float(h - 1))
        raise ValueError("Cannot compute bbox from an empty mask.")
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def clip_box(box: Iterable[float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def jitter_box(
    box: Iterable[float],
    width: int,
    height: int,
    shift: int = 20,
    rng: Optional[random.Random] = None,
) -> Box:
    if rng is None:
        rng = random
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 -= rng.randint(0, shift)
    y1 -= rng.randint(0, shift)
    x2 += rng.randint(0, shift)
    y2 += rng.randint(0, shift)
    return clip_box((x1, y1, x2, y2), width, height)


def enlarge_box(
    box: Iterable[float],
    width: int,
    height: int,
    ratio: float = 1.5,
) -> Box:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1 + 1.0) * ratio
    bh = max(1.0, y2 - y1 + 1.0) * ratio
    return clip_box((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), width, height)


def full_image_box(width: int, height: int) -> Box:
    return (0.0, 0.0, float(width - 1), float(height - 1))


def scale_box(
    box: Iterable[float],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Box:
    """Scale xyxy box from (H, W) to (H, W)."""
    from_h, from_w = from_size
    to_h, to_w = to_size
    x1, y1, x2, y2 = [float(v) for v in box]
    sx = float(to_w) / float(from_w)
    sy = float(to_h) / float(from_h)
    return clip_box((x1 * sx, y1 * sy, x2 * sx, y2 * sy), to_w, to_h)


def choose_train_box(
    mask,
    width: int,
    height: int,
    mode: str = "mixed",
    bbox_shift: int = 20,
    enlarge_ratio: float = 1.5,
    p_tight: float = 0.7,
    p_enlarge: float = 0.2,
    rng: Optional[random.Random] = None,
) -> Box:
    """Choose training box.

    Default mixed recipe:
    - 70% tight bbox + jitter
    - 20% enlarged bbox
    - 10% full-image bbox
    """
    if rng is None:
        rng = random
    mode = mode.lower()
    tight = mask_to_bbox(mask)

    if mode in {"tight", "standard"}:
        return jitter_box(tight, width, height, shift=bbox_shift, rng=rng)
    if mode in {"enlarged", "enlarge", "weak"}:
        return enlarge_box(tight, width, height, ratio=enlarge_ratio)
    if mode in {"full", "full_image", "full-image"}:
        return full_image_box(width, height)
    if mode != "mixed":
        raise ValueError(f"Unknown box mode: {mode}")

    r = rng.random()
    if r < p_tight:
        return jitter_box(tight, width, height, shift=bbox_shift, rng=rng)
    if r < p_tight + p_enlarge:
        return enlarge_box(tight, width, height, ratio=enlarge_ratio)
    return full_image_box(width, height)


def choose_eval_box(
    mask,
    width: int,
    height: int,
    mode: str = "tight",
    bbox_shift: int = 0,
    enlarge_ratio: float = 1.5,
) -> Box:
    mode = mode.lower()
    tight = mask_to_bbox(mask)
    if mode in {"tight", "standard"}:
        return jitter_box(tight, width, height, shift=bbox_shift)
    if mode in {"enlarged", "enlarge", "weak"}:
        return enlarge_box(tight, width, height, ratio=enlarge_ratio)
    if mode in {"full", "full_image", "full-image"}:
        return full_image_box(width, height)
    raise ValueError(f"Unknown eval box mode: {mode}")
