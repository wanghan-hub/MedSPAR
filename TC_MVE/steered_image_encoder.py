"""Monkey-patch/wrapper utilities to inject text steering into MedSAM ImageEncoderViT.

This file intentionally avoids editing `segment_anything/modeling/image_encoder.py`.
It attaches GatedTextCrossAttention modules to selected blocks and replaces the
image encoder forward method so it can accept text features.
"""

from __future__ import annotations

import types
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .gated_cross_attention import GatedTextCrossAttention


def _infer_image_encoder_dim(image_encoder: nn.Module, fallback: int = 768) -> int:
    # Official SAM ViT-B blocks have norm1.normalized_shape == (768,).
    try:
        return int(image_encoder.blocks[0].norm1.normalized_shape[0])
    except Exception:
        return int(fallback)


def _steered_image_encoder_forward(
    self: nn.Module,
    x: torch.Tensor,
    text_feats: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    return_patch_tokens: bool = False,
):
    """Forward replacement for SAM/MedSAM ImageEncoderViT.

    Compatible with the official SAM image encoder structure:
        patch_embed -> pos_embed -> blocks -> neck

    Args:
        x: image tensor, (B, 3, H, W), normally 1024x1024.
        text_feats: projected text tokens, (B, L, C), optional.
        text_mask: text token mask, (B, L), optional.
        return_patch_tokens: if True, also return Bx64x64xC tokens before neck.
    """
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + self.pos_embed

    for blk in self.blocks:
        gca = getattr(blk, "gated_cross_attn", None)
        if gca is not None and text_feats is not None:
            b, h, w, c = x.shape
            x_flat = x.reshape(b, h * w, c)
            x_flat = gca(x_flat, text_feats, text_mask)
            x = x_flat.reshape(b, h, w, c)
        x = blk(x)

    patch_tokens = x
    x = self.neck(x.permute(0, 3, 1, 2))

    if return_patch_tokens:
        return x, patch_tokens
    return x


def inject_text_steering_to_image_encoder(
    image_encoder: nn.Module,
    cross_attn_layers: Sequence[int] = (2, 5, 6, 11),
    dim: int | None = None,
    num_heads: int = 12,
    dropout: float = 0.0,
    use_ffn: bool = False,
    ffn_mult: float = 4.0,
) -> nn.Module:
    """Attach Gated Cross-Attention to selected MedSAM ViT blocks.

    Args:
        cross_attn_layers: 0-indexed layer ids. For MedSAM Global Attention
            layers 3, 6, 7, 12, pass [2, 5, 6, 11].
    """
    if not hasattr(image_encoder, "blocks"):
        raise AttributeError("image_encoder must have a `blocks` attribute like SAM ImageEncoderViT.")

    if dim is None:
        dim = _infer_image_encoder_dim(image_encoder)

    layer_set = set(int(i) for i in cross_attn_layers)
    for idx, blk in enumerate(image_encoder.blocks):
        if idx in layer_set:
            blk.gated_cross_attn = GatedTextCrossAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_ffn=use_ffn,
                ffn_mult=ffn_mult,
            )
        else:
            blk.gated_cross_attn = None

    image_encoder.forward = types.MethodType(_steered_image_encoder_forward, image_encoder)
    image_encoder.text_steering_layers = tuple(sorted(layer_set))
    image_encoder.text_steering_dim = int(dim)
    return image_encoder


def iter_gated_cross_attention_modules(image_encoder: nn.Module):
    for blk in getattr(image_encoder, "blocks", []):
        gca = getattr(blk, "gated_cross_attn", None)
        if gca is not None:
            yield gca
