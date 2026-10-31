"""Vision backbone builders for raw MedSAM and Stage-1 Steer-MedSAM."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn


def _plain_image_encoder_forward(self, x: torch.Tensor, return_patch_tokens: bool = False, **kwargs):
    """Forward for original SAM/MedSAM image encoder with optional patch tokens."""
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    for blk in self.blocks:
        x = blk(x)
    patch_tokens = x
    x = self.neck(x.permute(0, 3, 1, 2))
    if return_patch_tokens:
        return x, patch_tokens
    return x


def patch_plain_image_encoder(image_encoder: nn.Module) -> nn.Module:
    image_encoder.forward = types.MethodType(_plain_image_encoder_forward, image_encoder)
    return image_encoder








class MedSAMVisionWrapper(nn.Module):
    """Unified interface over raw MedSAM or Steer-MedSAM.

    encode(images, steer_texts) returns image_embeddings and patch_tokens.
    The underlying SAM/MedSAM model is exposed as .medsam for prompt/mask decoder.
    """

    def __init__(self, vision_type: str, medsam: nn.Module, steer_model: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.vision_type = vision_type
        self.medsam = medsam
        self.steer_model = steer_model

    def encode(self, images: torch.Tensor, steer_texts: Optional[Sequence[str]] = None):
        if self.vision_type == "TC_MVE":
            if steer_texts is None:
                raise ValueError("steer_medsam vision requires steer_texts for its text-conditioned image encoder.")
            return self.steer_model.visual_encoder.encode(images, steer_texts)
        return self.medsam.image_encoder(images, return_patch_tokens=True)

    def encode_with_spatial(self, images: torch.Tensor, steer_texts: Optional[Sequence[str]] = None):
        """Return the uncompressed grid explicitly while retaining encode's API."""
        image_embeddings, patch_tokens = self.encode(images, steer_texts=steer_texts)
        if patch_tokens.ndim != 4:
            raise ValueError(f"Vision encoder must return BxHxWxC patch tokens, got {patch_tokens.shape}")
        spatial_hw = (int(patch_tokens.shape[1]), int(patch_tokens.shape[2]))
        if tuple(image_embeddings.shape[-2:]) != spatial_hw:
            raise ValueError("Image embeddings and high-resolution patch tokens must share their spatial grid.")
        return image_embeddings, patch_tokens, spatial_hw


def build_medsam_vision(vision_type='TC_MVE', medsam_checkpoint='', model_type='vit_b',
                       stage0_checkpoint='', stage1_checkpoint='',
                       steer_text_encoder='bert-base-uncased', steer_cross_attn_layers=(2,5,6,11),
                       steer_text_max_length=64, local_files_only=True):
    from TC_MVE.model import build_steer_medsam, build_stage0_medsam
    from TC_MVE.train_utils import load_visual_checkpoint
    if vision_type == 'medsam':
        if stage1_checkpoint: raise ValueError('medsam_vision must not load Stage1 components')
        stage0 = build_stage0_medsam(medsam_checkpoint, model_type)
        if stage0_checkpoint: load_visual_checkpoint(stage0_checkpoint, stage0)
        patch_plain_image_encoder(stage0.medsam.image_encoder)
        return MedSAMVisionWrapper('medsam', stage0.medsam)
    if vision_type != 'TC_MVE' or stage0_checkpoint:
        raise ValueError('TC_MVE requires Stage1; Stage0 belongs only to medsam_vision')
    steer = build_steer_medsam(checkpoint=medsam_checkpoint, model_type=model_type,
        text_encoder_name=steer_text_encoder, cross_attn_layers=steer_cross_attn_layers,
        text_max_length=steer_text_max_length, local_files_only=local_files_only)
    if stage1_checkpoint: load_visual_checkpoint(stage1_checkpoint, steer)
    return MedSAMVisionWrapper('TC_MVE', steer.medsam, steer)
