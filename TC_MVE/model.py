"""Steer-MedSAM model wrapper."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_head import PatchSegHead
from .steered_image_encoder import inject_text_steering_to_image_encoder, iter_gated_cross_attention_modules
from .text_encoder import MedicalTextEncoder
from .encoder import TC_MVE


class SteerMedSAM(nn.Module):
    """MedSAM with SteerViT-style text conditioning inside the image encoder."""

    def __init__(
        self,
        medsam_model: nn.Module,
        text_encoder_name: str = "bert-base-uncased",
        cross_attn_layers: Sequence[int] = (2, 5, 6, 11),
        vision_dim: int = 768,
        cross_attn_heads: int = 12,
        text_max_length: int = 64,
        freeze_text_encoder: bool = True,
        use_patch_head: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.medsam = medsam_model
        self.vision_dim = int(vision_dim)
        self.use_patch_head = bool(use_patch_head)

        inject_text_steering_to_image_encoder(
            self.medsam.image_encoder,
            cross_attn_layers=cross_attn_layers,
            dim=self.vision_dim,
            num_heads=cross_attn_heads,
        )

        self.text_encoder = MedicalTextEncoder(
            model_name=text_encoder_name,
            vision_dim=self.vision_dim,
            max_length=text_max_length,
            freeze_text_encoder=freeze_text_encoder,
            local_files_only=local_files_only,
        )
        self.patch_head = PatchSegHead(dim=self.vision_dim) if self.use_patch_head else None

        self.visual_encoder = TC_MVE(self.text_encoder, self.medsam.image_encoder)
        self.training_stage = "stage1"
        self.freeze_base_and_enable_trainable_parts()

    def freeze_base_and_enable_trainable_parts(self) -> None:
        """Freeze original MedSAM/text backbone; train connector/GCA/mask decoder/patch head."""
        for p in self.medsam.parameters():
            p.requires_grad = False

        # Train mask decoder.
        for p in self.medsam.mask_decoder.parameters():
            p.requires_grad = True

        # The SAM/MedSAM mask decoder also contains an IoU prediction head.
        # Our current objective uses Dice + BCE (+ optional patch BCE) and does
        # not supervise iou_predictions, so this head receives no gradient.
        # In DDP, leaving it trainable triggers "unused parameter" errors.
        # Freeze it unless you explicitly add an IoU-head loss.
        iou_head = getattr(self.medsam.mask_decoder, "iou_prediction_head", None)
        if iou_head is not None:
            for p in iou_head.parameters():
                p.requires_grad = False

        # Train Gated Cross-Attention modules inserted inside image encoder.
        for gca in iter_gated_cross_attention_modules(self.medsam.image_encoder):
            for p in gca.parameters():
                p.requires_grad = True

        # Keep text backbone frozen, train text-to-vision connector.
        for p in self.text_encoder.text_model.parameters():
            p.requires_grad = False
        for p in self.text_encoder.connector.parameters():
            p.requires_grad = True

        if self.patch_head is not None:
            for p in self.patch_head.parameters():
                p.requires_grad = True

        # Prompt encoder stays frozen.
        for p in self.medsam.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor, boxes: torch.Tensor, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        device = images.device
        boxes = boxes.to(device).float()

        image_embeddings, patch_tokens = self.visual_encoder.encode(images, texts)

        sparse_embeddings, dense_embeddings = self.medsam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.medsam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.medsam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        pred_masks = F.interpolate(
            low_res_masks,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        patch_logits = self.patch_head(patch_tokens) if self.patch_head is not None else None
        return {
            "pred_masks": pred_masks,
            "low_res_masks": low_res_masks,
            "patch_logits": patch_logits,
            "iou_predictions": iou_predictions,
        }


def build_steer_medsam(
    checkpoint: str,
    model_type: str = "vit_b",
    text_encoder_name: str = "bert-base-uncased",
    cross_attn_layers: Sequence[int] = (2, 5, 6, 11),
    vision_dim: int = 768,
    cross_attn_heads: int = 12,
    text_max_length: int = 64,
    use_patch_head: bool = True,
    local_files_only: bool = False,
) -> SteerMedSAM:
    """Build from the original MedSAM/SAM registry.

    Assumes you run from the MedSAM project root where `segment_anything` is
    importable.
    """
    from segment_anything import sam_model_registry

    medsam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SteerMedSAM(
        medsam_model=medsam,
        text_encoder_name=text_encoder_name,
        cross_attn_layers=cross_attn_layers,
        vision_dim=vision_dim,
        cross_attn_heads=cross_attn_heads,
        text_max_length=text_max_length,
        use_patch_head=use_patch_head,
        local_files_only=local_files_only,
    )


class MedSAMStage0(nn.Module):
    """Plain MedSAM with the same shared-parameter policy as Stage1."""
    def __init__(self, medsam):
        super().__init__()
        self.medsam, self.patch_head, self.text_encoder = medsam, None, None
        self.training_stage = 'stage0'
        self.freeze_base_and_enable_trainable_parts()
    def freeze_base_and_enable_trainable_parts(self):
        self.medsam.requires_grad_(False)
        self.medsam.mask_decoder.requires_grad_(True)
        self.medsam.mask_decoder.iou_prediction_head.requires_grad_(False)
    def forward(self, images, boxes, texts=None):
        image_embeddings = self.medsam.image_encoder(images)
        sparse, dense = self.medsam.prompt_encoder(points=None, boxes=boxes.float(), masks=None)
        low_res, iou = self.medsam.mask_decoder(image_embeddings=image_embeddings,
            image_pe=self.medsam.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, multimask_output=False)
        pred = F.interpolate(low_res, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return dict(pred_masks=pred, low_res_masks=low_res, patch_logits=None, iou_predictions=iou)

def build_stage0_medsam(checkpoint, model_type='vit_b', **unused_stage1_options):
    from segment_anything import sam_model_registry
    return MedSAMStage0(sam_model_registry[model_type](checkpoint=checkpoint))
