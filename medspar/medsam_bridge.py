"""Shared SAM prompt encoding and mask decoding, including historical SEG prompts.

Freezing PromptEncoder parameters does not disable differentiation with respect
to predicted box coordinates or coarse masks. Only the box teacher uses no_grad.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_geometric_prompts(
    prompt_encoder: nn.Module,
    points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    boxes: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode predicted geometry using the original, differentiable SAM adapter.

    Coordinates must already be in prompt_encoder.input_image_size. No spatial
    resizing, GT substitution or detach is performed here.
    """
    parameter = next(prompt_encoder.parameters())
    device, dtype = parameter.device, parameter.dtype
    sizes = []
    if points is not None:
        coords, labels = points
        if coords.ndim != 3 or coords.shape[-1] != 2 or labels.shape != coords.shape[:2]:
            raise ValueError("points must be (BxKx2 coordinates, BxK labels)")
        points = (coords.to(device=device, dtype=torch.float32), labels.to(device=device, dtype=torch.long))
        sizes.append(coords.shape[0])
    if boxes is not None:
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape Bx4 in SAM input XYXY coordinates")
        boxes = boxes.to(device=device, dtype=torch.float32)
        sizes.append(boxes.shape[0])
    if masks is not None:
        if masks.ndim != 4 or masks.shape[1] != 1 or tuple(masks.shape[-2:]) != tuple(prompt_encoder.mask_input_size):
            raise ValueError(f"masks must be Bx1x{tuple(prompt_encoder.mask_input_size)}, got {tuple(masks.shape)}")
        masks = masks.to(device=device, dtype=dtype)
        sizes.append(masks.shape[0])
    if batch_size is not None:
        sizes.append(int(batch_size))
    if sizes and (min(sizes) < 1 or len(set(sizes)) != 1):
        raise ValueError(f"Prompt batch sizes must match: {sizes}")
    sparse, dense = prompt_encoder(points=points, boxes=boxes, masks=masks)
    if points is None and boxes is None and masks is None and batch_size is not None:
        sparse = sparse.expand(batch_size, -1, -1)
        dense = dense.expand(batch_size, -1, -1, -1)
    return sparse, dense


def encode_box_prompt_pair(
    prompt_encoder: nn.Module,
    pred_box_xyxy: torch.Tensor,
    gt_box_xyxy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (student, teacher) box-only embeddings from the same SAM encoder."""
    if pred_box_xyxy.shape != gt_box_xyxy.shape:
        raise ValueError("Teacher and student boxes must have identical Bx4 shapes")
    student, _ = encode_geometric_prompts(prompt_encoder, boxes=pred_box_xyxy)
    with torch.no_grad():
        teacher, _ = encode_geometric_prompts(prompt_encoder, boxes=gt_box_xyxy)
    return student, teacher.detach()


def decode_masks_from_embeddings(
    prompt_encoder: nn.Module,
    mask_decoder: nn.Module,
    image_embeddings: torch.Tensor,
    sparse_embeddings: torch.Tensor,
    dense_embeddings: torch.Tensor,
    multimask_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode one set of fused prompts per image, preserving input gradients."""
    if image_embeddings.shape[0] != sparse_embeddings.shape[0] or image_embeddings.shape != dense_embeddings.shape:
        raise ValueError("Expected one sparse/dense prompt set per image and matching dense spatial shape")
    device, dtype = image_embeddings.device, image_embeddings.dtype
    return mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=prompt_encoder.get_dense_pe().to(device=device, dtype=dtype),
        sparse_prompt_embeddings=sparse_embeddings.to(device=device, dtype=dtype),
        dense_prompt_embeddings=dense_embeddings.to(device=device, dtype=dtype),
        multimask_output=multimask_output,
    )


def get_no_mask_dense_embeddings(prompt_encoder: nn.Module, batch_size: int, device, dtype) -> torch.Tensor:
    """Return SAM's dense no-mask embedding expanded to batch size."""
    if not hasattr(prompt_encoder, "no_mask_embed"):
        # Fallback: call prompt encoder with no prompts; official SAM usually returns B=1.
        _, dense = prompt_encoder(points=None, boxes=None, masks=None)
        return dense.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1).contiguous()
    emb = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
    h, w = prompt_encoder.image_embedding_size
    return emb.expand(batch_size, -1, h, w).to(device=device, dtype=dtype).contiguous()


def decode_masks_from_seg_prompt(
    prompt_encoder: nn.Module,
    mask_decoder: nn.Module,
    image_embeddings: torch.Tensor,
    seg_sparse_embeddings: torch.Tensor,
    boxes: Optional[torch.Tensor] = None,
    use_box: Optional[torch.Tensor | bool] = True,
    multimask_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode masks with optional box prompt and mandatory [SEG] prompt.

    Args:
        seg_sparse_embeddings: B x 1 x 256, projected [SEG] token.
        boxes: B x 4 boxes in 1024 image coordinates.
        use_box: bool or B bool tensor. If mixed per sample, this function splits
            the batch internally and recombines outputs.
    """
    b = image_embeddings.shape[0]
    device = image_embeddings.device
    dtype = image_embeddings.dtype
    if isinstance(use_box, bool):
        use_box_tensor = torch.full((b,), bool(use_box), device=device, dtype=torch.bool)
    elif use_box is None:
        use_box_tensor = torch.ones((b,), device=device, dtype=torch.bool)
    else:
        use_box_tensor = use_box.to(device=device, dtype=torch.bool).flatten()
        if use_box_tensor.numel() != b:
            raise ValueError(f"use_box must have B elements, got {use_box_tensor.numel()} vs B={b}")

    low_res_all = None
    iou_all = None

    for flag in [True, False]:
        idx = torch.nonzero(use_box_tensor.eq(flag), as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        img_g = image_embeddings.index_select(0, idx)
        seg_g = seg_sparse_embeddings.index_select(0, idx).to(dtype=img_g.dtype)

        if flag:
            if boxes is None:
                raise ValueError("use_box=True but boxes is None")
            boxes_g = boxes.index_select(0, idx).to(device=device, dtype=torch.float32)
            box_sparse, dense_g = prompt_encoder(points=None, boxes=boxes_g, masks=None)
            box_sparse = box_sparse.to(dtype=img_g.dtype)
            dense_g = dense_g.to(dtype=img_g.dtype)
            sparse_g = torch.cat([box_sparse, seg_g], dim=1)
        else:
            dense_g = get_no_mask_dense_embeddings(prompt_encoder, idx.numel(), device=device, dtype=img_g.dtype)
            sparse_g = seg_g

        low_res_g, iou_g = mask_decoder(
            image_embeddings=img_g,
            image_pe=prompt_encoder.get_dense_pe().to(device=device, dtype=img_g.dtype),
            sparse_prompt_embeddings=sparse_g,
            dense_prompt_embeddings=dense_g,
            multimask_output=multimask_output,
        )

        if low_res_all is None:
            low_res_shape = (b,) + tuple(low_res_g.shape[1:])
            iou_shape = (b,) + tuple(iou_g.shape[1:])
            low_res_all = torch.empty(low_res_shape, device=device, dtype=low_res_g.dtype)
            iou_all = torch.empty(iou_shape, device=device, dtype=iou_g.dtype)
        low_res_all.index_copy_(0, idx, low_res_g)
        iou_all.index_copy_(0, idx, iou_g)

    return low_res_all, iou_all
