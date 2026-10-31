"""Unified reasoning-grounded automatic MedSAM segmentation and controlled ablations."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llm_backbone import load_llm_backbone, install_seg_row_adapters
from .medsam_bridge import (
    decode_masks_from_seg_prompt, decode_masks_from_embeddings,
    encode_geometric_prompts, encode_box_prompt_pair,
)
from .projectors import SEGProjector, VisualProjector, VisualTokenCompressor
from .seg_token import DEFAULT_SEG_TOKEN, gather_seg_hidden
from .stage_policy import is_grounding_stage, apply_stage_freeze_policy, keep_frozen_modules_eval
from .experiments import ExperimentConfig, UNIFIED_VERSION
from .spatial_grounding import (
    ReasoningQueryPooler, SEGAnchorProjector, SemanticSpatialGrounder,
    AutomaticGeometricPromptGenerator, SemanticPromptProjector, PromptFusionGate,
)
from .vision_backbones import build_medsam_vision
from experiment_protocol import check_context


def validate_cached_visual_features(features, images, vision_dim, sam_prompt_dim):
    """Validate batch-local eval features without casting or detaching them."""
    if not isinstance(features, tuple) or len(features) != 3:
        raise ValueError("cached_visual_features must be (image_embeddings, patch_tokens, spatial_hw)")
    embeddings, tokens, spatial_hw = features
    if not isinstance(embeddings, torch.Tensor) or not isinstance(tokens, torch.Tensor):
        raise TypeError("Cached image embeddings and patch tokens must be tensors")
    if (not isinstance(spatial_hw, tuple) or len(spatial_hw) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in spatial_hw)):
        raise ValueError("Cached spatial_hw must contain two positive integers")
    batch = images.shape[0]
    if (images.ndim != 4 or batch < 1 or embeddings.ndim != 4 or tokens.ndim != 4
            or tuple(embeddings.shape) != (batch, sam_prompt_dim, *spatial_hw)
            or tuple(tokens.shape) != (batch, *spatial_hw, vision_dim)):
        raise ValueError("Cached visual feature batch, grid or channel dimensions disagree")
    for value in (embeddings, tokens):
        if value.device != images.device or not value.is_floating_point() or value.requires_grad:
            raise ValueError("Cached features must be floating, gradient-free and on the image device")
    return embeddings, tokens, spatial_hw


def verify_single_seg_token(tokenizer, seg_token: str = DEFAULT_SEG_TOKEN) -> int:
    """Require [SEG] to be a real single tokenizer token."""
    ids = tokenizer.encode(seg_token, add_special_tokens=False)
    token_id = tokenizer.convert_tokens_to_ids(seg_token)

    if len(ids) != 1:
        raise RuntimeError(
            f"{seg_token} must be a single token, but tokenizer.encode returned {ids}. "
            "Use a permanent LLM directory where [SEG] has already been added."
        )

    if int(ids[0]) != int(token_id):
        raise RuntimeError(
            f"{seg_token} tokenizer mismatch: encode={ids}, convert_tokens_to_ids={token_id}."
        )

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and int(token_id) == int(unk_id):
        raise RuntimeError(
            f"{seg_token} maps to unk_token_id={unk_id}. "
            "Please add [SEG] as an additional_special_token."
        )

    return int(token_id)


def verify_llm_vocab_size(llm: nn.Module, tokenizer) -> None:
    """Ensure tokenizer length matches input embedding and output LM head size."""
    vocab_size = len(tokenizer)

    input_emb = llm.get_input_embeddings()
    if input_emb is None or not hasattr(input_emb, "weight"):
        raise RuntimeError("LLM has no valid input embeddings.")

    input_rows = int(input_emb.weight.shape[0])
    if input_rows != vocab_size:
        raise RuntimeError(
            f"Tokenizer length ({vocab_size}) does not match input embedding rows ({input_rows}). "
            "Please use the permanent SEG-token model directory and resize embeddings before training."
        )

    output_emb = llm.get_output_embeddings()
    if output_emb is not None and hasattr(output_emb, "weight"):
        output_rows = int(output_emb.weight.shape[0])
        if output_rows != vocab_size:
            raise RuntimeError(
                f"Tokenizer length ({vocab_size}) does not match LM head rows ({output_rows}). "
                "Please resize the output embeddings / LM head before training."
            )


def require_permanent_seg_token_dir(llm_name_or_path: str, seg_token: str = DEFAULT_SEG_TOKEN) -> None:
    """Avoid hidden dynamic token insertion by requiring seg_token_info.json in local model dir."""
    path = Path(llm_name_or_path)
    info_path = path / "seg_token_info.json"

    if not path.exists() or not path.is_dir():
        raise RuntimeError(
            f"llm_name_or_path must point to a local permanent SEG-token model directory, got: {llm_name_or_path}"
        )

    if not info_path.exists():
        raise RuntimeError(
            f"Missing {info_path}. This usually means you are using the original LLM directory, "
            f"not the permanent {seg_token}-augmented directory. "
            "Run tools/add_seg_token_to_llm.py first and point --llm_name_or_path to the new -SEG directory."
        )

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        saved_seg = info.get("seg_token", seg_token)
        if saved_seg != seg_token:
            raise RuntimeError(
                f"seg_token_info.json contains seg_token={saved_seg!r}, but current seg_token={seg_token!r}."
            )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse {info_path}: {e}") from e


def weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seg_token_id: int,
    seg_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Language modeling loss with extra weight on the [SEG] target token.

    logits: [B, L, V]
    labels: [B, L], where ignored positions are -100.
    """
    if labels is None:
        return logits.sum() * 0.0

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.shape[-1]
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    valid_mask = shift_labels.ne(-100).float()
    weights = torch.ones_like(shift_labels, dtype=torch.float32, device=shift_labels.device)
    weights[shift_labels.eq(int(seg_token_id))] = float(seg_loss_weight)

    denom = (valid_mask * weights).sum().clamp_min(1.0)
    loss = (loss_per_token.float() * valid_mask * weights).sum() / denom
    return loss


class MedSPAR(nn.Module):
    def __init__(
        self,
        vision_wrapper: nn.Module,
        llm: nn.Module,
        tokenizer,
        seg_token_id: int,
        llm_hidden_size: int,
        vision_dim: int = 768,
        sam_prompt_dim: int = 256,
        visual_grid: int = 16,
        visual_compress_mode: str = "avg",
        visual_projector_type: str = "mlp2x_gelu",
        seg_projector_type: str = "mlp2x_gelu",
        train_mask_decoder: bool = True,
        freeze_vision_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        seg_token: str = DEFAULT_SEG_TOKEN,
        seg_loss_weight: float = 1.0,
        stage: str = "unified",
        experiment_id: str = "main",
        num_reasoning_queries: int = 4,
        num_pos_points: int = 4,
        num_neg_points: int = 4,
        grounder_layers: int = 2,
        grounder_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vision = vision_wrapper
        self.medsam = vision_wrapper.medsam
        self.llm = llm
        self.tokenizer = tokenizer
        self.seg_token = seg_token
        self.seg_token_id = int(seg_token_id)
        if is_grounding_stage(stage):
            self.pipeline_version = UNIFIED_VERSION
        self.llm_hidden_size = int(llm_hidden_size)
        self.vision_dim = int(vision_dim)
        self.sam_prompt_dim = int(sam_prompt_dim)
        self.seg_loss_weight = float(seg_loss_weight)
        self.stage = stage.lower()
        self.experiment_id = experiment_id
        self.ablation_config = ExperimentConfig(experiment_id)
        self.vision_backbone = self.ablation_config.vision_backbone
        actual_vision = getattr(vision_wrapper, "vision_type", self.vision_backbone)
        if actual_vision != self.vision_backbone:
            raise ValueError("Ablation and actual vision backbone disagree")

        checked_seg_id = verify_single_seg_token(self.tokenizer, self.seg_token)
        if checked_seg_id != self.seg_token_id:
            raise RuntimeError(
                f"seg_token_id mismatch: llm_pack={self.seg_token_id}, tokenizer_check={checked_seg_id}."
            )
        verify_llm_vocab_size(self.llm, self.tokenizer)

        self.visual_compressor = VisualTokenCompressor(
            out_grid=visual_grid,
            mode=visual_compress_mode,
            vision_dim=vision_dim,
            trainable_norm=True,
        )
        self.visual_projector = VisualProjector(
            vision_dim=vision_dim,
            llm_dim=llm_hidden_size,
            projector_type=visual_projector_type,
        )
        self.seg_projector = None
        self.reasoning_query_pooler = ReasoningQueryPooler(
            llm_dim=llm_hidden_size, grounder_dim=grounder_dim, num_queries=num_reasoning_queries,
        )
        self.seg_anchor_projector = SEGAnchorProjector(llm_hidden_size, grounder_dim)
        self.semantic_spatial_grounder = SemanticSpatialGrounder(
            vision_dim=vision_dim, grounder_dim=grounder_dim, num_layers=grounder_layers,
        )
        self.automatic_geometric_prompt_generator = AutomaticGeometricPromptGenerator(
            grounder_dim=grounder_dim, num_pos_points=num_pos_points, num_neg_points=num_neg_points,
        )
        self.semantic_prompt_projector = SemanticPromptProjector(grounder_dim, sam_prompt_dim)
        self.prompt_fusion_gate = PromptFusionGate(grounder_dim)

        self.configure_trainable_parts(
            train_mask_decoder=train_mask_decoder,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
        )
        apply_stage_freeze_policy(self, self.stage)

    def train(self, mode: bool = True):
        """Frozen backbones must not introduce dropout into fixed reasoning."""
        super().train(mode)
        if mode:
            keep_frozen_modules_eval(self)
        return self

    def configure_trainable_parts(
        self,
        train_mask_decoder: bool = True,
        freeze_vision_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
    ) -> None:
        # Vision encoder policy. Stage 2 first version should freeze it.
        if freeze_vision_encoder:
            for p in self.vision.parameters():
                p.requires_grad = False

        # Prompt encoder normally remains frozen; it supplies box embeddings / dense no-mask embeddings.
        if freeze_prompt_encoder:
            for p in self.medsam.prompt_encoder.parameters():
                p.requires_grad = False

        # Mask decoder may be trained; freeze IoU head unless explicitly supervised.
        for p in self.medsam.mask_decoder.parameters():
            p.requires_grad = bool(train_mask_decoder)

        iou_head = getattr(self.medsam.mask_decoder, "iou_prediction_head", None)
        if iou_head is not None:
            for p in iou_head.parameters():
                p.requires_grad = False

        # Visual prefix modules and SEG projector are always trainable.
        for m in [self.visual_compressor, self.visual_projector, self.seg_projector]:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True

    def enable_lm_embedding_and_head_training(
        self,
        train_lm_embeddings: bool = True,
        train_lm_head: bool = True,
    ) -> None:
        """Enable training for LLM token embeddings and output LM head.

        This is necessary for the newly added [SEG] token because LoRA alone may
        not update its input embedding row or its output prediction row.
        """
        if train_lm_embeddings:
            emb = self.llm.get_input_embeddings()
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad = True

        if train_lm_head:
            head = self.llm.get_output_embeddings()
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True

    def _build_llm_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        visual_embeds: torch.Tensor,
    ):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # Important: visual prefix must match LLM embedding dtype to avoid dtype mismatch.
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype, device=text_embeds.device)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        b, nv, _ = visual_embeds.shape
        visual_mask = torch.ones((b, nv), device=attention_mask.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            visual_labels = torch.full((b, nv), -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([visual_labels, labels], dim=1)

        check_context(self.llm, inputs_embeds.shape[1])
        return inputs_embeds, full_attention_mask, full_labels, nv

    def encode_image(self, images: torch.Tensor, steer_texts: Optional[Sequence[str]] = None):
        return self.vision.encode(images, steer_texts=steer_texts)

    def _encode_for_grounding(self, images, steer_texts):
        # The vision wrapper also owns the mask decoder. Only the image path
        # determines whether encoding needs a graph (P4 connector/GCA/ViT).
        trainable = any(p.requires_grad for p in self.medsam.image_encoder.parameters())
        steer = getattr(self.vision, "steer_model", None)
        if steer is not None:
            trainable |= any(p.requires_grad for p in steer.text_encoder.parameters())
        with nullcontext() if trainable else torch.no_grad():
            if hasattr(self.vision, "encode_with_spatial"):
                result = self.vision.encode_with_spatial(images, steer_texts=steer_texts)
            else:
                embeddings, tokens = self.encode_image(images, steer_texts)
                if tokens.ndim != 4:
                    raise ValueError("Vision must expose an explicit BxHxWxC uncompressed grid.")
                result = embeddings, tokens, tuple(tokens.shape[1:3])
        if result[1].shape[-1] != self.vision_dim:
            raise ValueError(f"--vision_dim={self.vision_dim} differs from actual patch dimension {result[1].shape[-1]}")
        return result

    def _llm_forward_hidden(self, inputs_embeds, attention_mask, compute_lm_loss):
        kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                      output_hidden_states=True, return_dict=True, use_cache=False)
        if not compute_lm_loss:
            # Use the causal LM's registered decoder when available to avoid
            # allocating BxTxVocabulary logits for a frozen language model.
            causal = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
            decoder = getattr(causal, "base_model", causal)
            if decoder is not causal:
                output = decoder(**kwargs)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    hidden = output.hidden_states[-1]
                return hidden, None
        output = self.llm(**kwargs)
        return output.hidden_states[-1], output.logits if compute_lm_loss else None

    def extract_assistant_hidden(self, hidden, input_ids, attention_mask,
                                 visual_prefix_len, labels=None,
                                 assistant_mask=None, reasoning_mask=None):
        """Select only assistant answer states, with SEG separately anchored.

        Masks are in text-token coordinates (no visual-prefix offset). Padding,
        system/user tokens, SEG itself, tokens after SEG and tokenizer specials
        can never enter the reasoning pool. Empty reasoning is a valid empty
        semantic context, handled explicitly by the pooler.
        """
        valid = attention_mask.bool()
        if assistant_mask is None:
            if labels is None:
                raise ValueError("Supply assistant_mask for generated answers, or labels for teacher forcing.")
            assistant_mask = labels.ne(-100)
        assistant_mask = assistant_mask.to(device=input_ids.device, dtype=torch.bool)
        if assistant_mask.shape != input_ids.shape or (assistant_mask & ~valid).any():
            raise ValueError("assistant_mask must match input_ids and exclude padding.")
        seg_mask = assistant_mask & valid & input_ids.eq(self.seg_token_id)
        counts = seg_mask.sum(1)
        if not counts.eq(1).all():
            raise ValueError(f"Each assistant answer must contain exactly one {self.seg_token}; counts={counts.tolist()}")
        seg_positions = seg_mask.long().argmax(1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        allowed = assistant_mask & valid & (positions < seg_positions[:, None])
        for special_id in getattr(self.tokenizer, "all_special_ids", []):
            allowed &= input_ids.ne(int(special_id))
        if reasoning_mask is not None:
            reasoning_mask = reasoning_mask.to(device=input_ids.device, dtype=torch.bool)
            if reasoning_mask.shape != allowed.shape or (reasoning_mask & ~allowed).any():
                raise ValueError("reasoning_mask contains a prompt/padding/special/post-SEG token.")
            allowed &= reasoning_mask
        text_hidden = hidden[:, visual_prefix_len:visual_prefix_len + input_ids.shape[1], :]
        batch_ids = torch.arange(input_ids.shape[0], device=hidden.device)
        seg_hidden = text_hidden[batch_ids, seg_positions, :]
        return text_hidden, seg_hidden, allowed

    def _detach_diagnostics(self, outputs):
        # DDP examines all returned tensors as potential backward roots. Keep
        # only actual loss inputs attached; otherwise P1's unsupervised coarse
        # mask and P2's all-teacher batches appear falsely "used" to DDP.
        live = getattr(self, "loss_graph_keys", {
            "loss_lm", "pred_masks", "spatial_logits", "pred_box_norm", "pred_box_embeddings",
        })
        def detached(value):
            if torch.is_tensor(value):
                return value.detach()
            if isinstance(value, list):
                return [detached(v) for v in value]
            if isinstance(value, tuple):
                return tuple(detached(v) for v in value)
            return value
        return {key: value if key in live else detached(value) for key, value in outputs.items()}

    def _forward_grounding(self, images, input_ids, attention_mask, labels, boxes,
                           steer_texts, stage, assistant_mask, reasoning_mask,
                           auto_prompt_probability, compute_lm_loss, decode_masks,
                           prompt_source, cached_visual_features=None):
        if cached_visual_features is None:
            image_embeddings, patch_tokens, spatial_hw = self._encode_for_grounding(images, steer_texts)
        else:
            if self.training or torch.is_grad_enabled() or labels is not None or compute_lm_loss:
                raise ValueError("Visual feature reuse is restricted to eval/no_grad segmentation inference")
            image_embeddings, patch_tokens, spatial_hw = validate_cached_visual_features(
                cached_visual_features, images, self.vision_dim, self.sam_prompt_dim)
        prompt_hw = tuple(int(v) for v in self.medsam.prompt_encoder.input_image_size)
        if tuple(images.shape[-2:]) != prompt_hw:
            raise ValueError(f"Image size {tuple(images.shape[-2:])} must match SAM prompt coordinates {prompt_hw}.")

        language_trainable = any(p.requires_grad for p in self.llm.parameters())
        language_trainable |= any(p.requires_grad for p in self.visual_compressor.parameters())
        language_trainable |= any(p.requires_grad for p in self.visual_projector.parameters())
        with nullcontext() if language_trainable else torch.no_grad():
            # Joint training retains the language and optional vision gradients.
            language_tokens = patch_tokens if language_trainable else patch_tokens.detach()
            language_tokens = language_tokens.to(next(self.visual_compressor.parameters()).dtype)
            visual_embeds = self.visual_projector(self.visual_compressor(language_tokens))
            inputs, full_mask, full_labels, prefix_len = self._build_llm_inputs(
                input_ids, attention_mask, labels, visual_embeds,
            )
            hidden, logits = self._llm_forward_hidden(inputs, full_mask, compute_lm_loss)
            loss_lm = None
            if compute_lm_loss and full_labels is not None:
                loss_lm = weighted_lm_loss(logits.float(), full_labels, self.seg_token_id, self.seg_loss_weight)
        reasoning_hidden, seg_hidden, selected_mask = self.extract_assistant_hidden(
            hidden, input_ids, attention_mask, prefix_len, labels, assistant_mask, reasoning_mask,
        )
        policy = self.ablation_config
        reasoning_queries = None
        seg_anchor = self.seg_anchor_projector(seg_hidden)
        if policy.reasoning_queries:
            reasoning_queries = self.reasoning_query_pooler(reasoning_hidden, selected_mask)
            query_bank = torch.cat([reasoning_queries, seg_anchor], dim=1)
        else:
            query_bank = seg_anchor
        grounded = self.semantic_spatial_grounder(query_bank, patch_tokens, spatial_hw=spatial_hw)
        geometry = self.automatic_geometric_prompt_generator(
            grounded["spatial_logits"], grounded["grounded_queries"], grounded["high_res_tokens"],
            input_image_size=prompt_hw, mask_input_size=self.medsam.prompt_encoder.mask_input_size,
        )
        geo_sparse, geo_dense = encode_geometric_prompts(
            self.medsam.prompt_encoder,
            points=(geometry["point_coords"], geometry["point_labels"]) if policy.point_prompts else None,
            boxes=geometry["pred_box_xyxy"] if policy.box_prompt else None,
            batch_size=images.shape[0],
            masks=geometry["pred_coarse_mask_logits"] if policy.coarse_mask_prompt else None,
        )
        semantic = gates = None
        fused_auto = geo_sparse
        if policy.semantic_prompt:
            semantic = self.semantic_prompt_projector(grounded["grounded_queries"])
            fused_auto, gates = self.prompt_fusion_gate(grounded["grounded_queries"], semantic, geo_sparse)
        if prompt_source not in {None, "auto"}:
            raise ValueError("Unified model uses predicted prompts only; GT boxes are loss targets, never decoder inputs.")
        if auto_prompt_probability not in {None, 1.0}:
            raise ValueError("Unified training has no prompt curriculum")
        b = images.shape[0]
        use_auto = torch.ones(b, device=images.device, dtype=torch.bool)

        pred_box_embeddings = gt_box_embeddings = None
        if boxes is not None:
            pred_box_embeddings, gt_box_embeddings = encode_box_prompt_pair(
                self.medsam.prompt_encoder, geometry["pred_box_xyxy"], boxes,
            )

        low_res = iou_predictions = pred_masks = None
        if decode_masks:
            low_res, iou_predictions = decode_masks_from_embeddings(
                self.medsam.prompt_encoder, self.medsam.mask_decoder, image_embeddings,
                fused_auto, geo_dense, multimask_output=False)
            pred_masks = F.interpolate(low_res, size=images.shape[-2:], mode="bilinear", align_corners=False)
        if not getattr(self, "_sanity_printed", False):
            def shape(value):
                return list(value.shape) if value is not None else None
            self.runtime_sanity = {
                "experiment_id": self.experiment_id, "vision_backbone": self.vision_backbone,
                "reasoning_queries_shape": shape(reasoning_queries), "seg_anchor_shape": shape(seg_anchor),
                "high_res_tokens_shape": shape(patch_tokens), "spatial_hw": list(spatial_hw),
                "grounded_queries_shape": shape(grounded["grounded_queries"]),
                "spatial_logits_shape": shape(grounded["spatial_logits"]),
                "points_enabled": policy.point_prompts, "box_enabled": policy.box_prompt,
                "coarse_mask_enabled": policy.coarse_mask_prompt, "semantic_prompt_enabled": policy.semantic_prompt,
                "spatial_supervision_enabled": policy.spatial_supervision,
                "final_sparse_shape": shape(fused_auto), "final_dense_shape": shape(geo_dense),
                "pred_mask_shape": shape(pred_masks),
            }
            print("First forward: " + json.dumps(self.runtime_sanity))
            self._sanity_printed = True

        outputs = dict(grounded)
        outputs.update(geometry)
        outputs.update({
            "loss_lm": loss_lm, "logits": logits,
            "reasoning_queries": reasoning_queries, "seg_anchor": seg_anchor,
            "reasoning_mask": selected_mask,
            "geo_sparse_embeddings": geo_sparse, "geo_dense_embeddings": geo_dense,
            "semantic_prompt_embeddings": semantic, "semantic_gates": gates,
            "fused_sparse_embeddings": fused_auto,
            "selected_dense_embeddings": geo_dense,
            "pred_box_embeddings": pred_box_embeddings, "gt_box_embeddings": gt_box_embeddings,
            "pred_masks": pred_masks, "low_res_masks": low_res, "iou_predictions": iou_predictions,
            "prompt_source": ["auto" if flag else "gt_box" for flag in use_auto.tolist()],
            "use_auto": use_auto, "use_box": torch.full_like(use_auto, policy.box_prompt), "prompt_input_size": prompt_hw,
        })
        return self._detach_diagnostics(outputs) if self.training else outputs

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        steer_texts: Optional[Sequence[str]] = None,
        stage: Optional[str] = None,
        use_box: Optional[torch.Tensor] = None,
        box_dropout_prob: Optional[float] = None,
        assistant_mask: Optional[torch.Tensor] = None,
        reasoning_mask: Optional[torch.Tensor] = None,
        auto_prompt_probability: Optional[float] = None,
        compute_lm_loss: Optional[bool] = None,
        decode_masks: Optional[bool] = None,
        prompt_source: Optional[str] = None,
        cached_visual_features=None,
    ) -> Dict[str, torch.Tensor]:
        device = images.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device) if labels is not None else None
        boxes = boxes.to(device).float() if boxes is not None else None
        stage = (stage or self.stage).lower()
        if is_grounding_stage(stage):
            if use_box is not None or box_dropout_prob is not None:
                raise ValueError("Unified training uses automatic geometry, not use_box/bbox-dropout controls.")
            return self._forward_grounding(
                images, input_ids, attention_mask, labels, boxes, steer_texts, stage,
                assistant_mask, reasoning_mask, auto_prompt_probability,
                bool(compute_lm_loss) if compute_lm_loss is not None else labels is not None,
                bool(decode_masks) if decode_masks is not None else True,
                prompt_source, cached_visual_features,
            )

        raise ValueError("Only unified Stage2 is supported")


def build_medspar(
    vision_type: str,
    medsam_checkpoint: str,
    llm_name_or_path: str,
    model_type: str = "vit_b",
    stage0_checkpoint: str = "",
    stage1_checkpoint: str = "",
    steer_text_encoder: str = "bert-base-uncased",
    steer_cross_attn_layers: Sequence[int] = (2, 5, 6, 11),
    steer_text_max_length: int = 64,
    vision_dim: int = 768,
    visual_grid: int = 16,
    llm_torch_dtype: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    seg_token: str = DEFAULT_SEG_TOKEN,
    train_mask_decoder: bool = True,
    freeze_vision_encoder: bool = True,
    freeze_prompt_encoder: bool = True,
    seg_loss_weight: float = 1.0,
    require_permanent_seg_token: bool = True,
    stage: str = "unified",
    experiment_id: str = "main",
    num_reasoning_queries: int = 4,
    num_pos_points: int = 4,
    num_neg_points: int = 4,
    grounder_layers: int = 2,
    grounder_dim: int = 256,
) -> MedSPAR:
    if require_permanent_seg_token:
        require_permanent_seg_token_dir(llm_name_or_path, seg_token=seg_token)

    policy = ExperimentConfig(experiment_id)
    if vision_type != policy.vision_backbone:
        raise ValueError("vision_type must match experiment_id")
    # Vision constructors consume different amounts of RNG. Isolate them so
    # LoRA/projector/grounder initialization is identical across vision ablations.
    with torch.random.fork_rng(devices=[]):
        vision = build_medsam_vision(
            vision_type=vision_type,
            medsam_checkpoint=medsam_checkpoint,
            model_type=model_type,
            stage0_checkpoint=stage0_checkpoint,
            stage1_checkpoint=stage1_checkpoint,
            steer_text_encoder=steer_text_encoder,
            steer_cross_attn_layers=steer_cross_attn_layers,
            steer_text_max_length=steer_text_max_length,
            local_files_only=local_files_only,
        )

    llm_pack = load_llm_backbone(
        model_name_or_path=llm_name_or_path,
        seg_token=seg_token,
        torch_dtype=llm_torch_dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        freeze_llm_base=True,
    )

    checked_seg_id = verify_single_seg_token(llm_pack.tokenizer, seg_token)
    if int(checked_seg_id) != int(llm_pack.seg_token_id):
        raise RuntimeError(
            f"SEG token id mismatch after loading LLM: checked={checked_seg_id}, pack={llm_pack.seg_token_id}."
        )
    verify_llm_vocab_size(llm_pack.model, llm_pack.tokenizer)
    info_path = Path(llm_name_or_path) / "seg_token_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if int(info.get("seg_token_id", checked_seg_id)) != checked_seg_id:
            raise RuntimeError("seg_token_info.json records a different SEG id from the loaded tokenizer")
        if int(info.get("new_tokenizer_len", len(llm_pack.tokenizer))) != len(llm_pack.tokenizer):
            raise RuntimeError("seg_token_info.json and tokenizer vocabulary lengths differ")

    return MedSPAR(
        vision_wrapper=vision,
        llm=llm_pack.model,
        tokenizer=llm_pack.tokenizer,
        seg_token_id=llm_pack.seg_token_id,
        llm_hidden_size=llm_pack.hidden_size,
        vision_dim=vision_dim,
        visual_grid=visual_grid,
        train_mask_decoder=train_mask_decoder,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder,
        seg_token=seg_token,
        seg_loss_weight=seg_loss_weight,
        stage=stage, experiment_id=experiment_id,
        num_reasoning_queries=num_reasoning_queries,
        num_pos_points=num_pos_points,
        num_neg_points=num_neg_points,
        grounder_layers=grounder_layers,
        grounder_dim=grounder_dim,
    )
