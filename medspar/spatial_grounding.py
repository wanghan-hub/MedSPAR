"""Reasoning-grounded spatial priors and automatic SAM geometric prompts.

The grounder consumes the uncompressed, channels-last vision patch grid. Boxes
use normalized XYXY edges; points use SAM pixel-index coordinates, whose pixel
centres are shifted by +0.5 inside the existing SAM PromptEncoder.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ReasoningQueryPooler(nn.Module):
    """Compress only masked assistant reasoning into a small semantic query bank."""

    def __init__(self, llm_dim: int, grounder_dim: int = 256,
                 num_queries: int = 4, num_heads: int = 8) -> None:
        super().__init__()
        if num_queries < 1 or grounder_dim % num_heads:
            raise ValueError("num_queries must be positive and num_heads must divide grounder_dim")
        self.input_projection = nn.Sequential(nn.Linear(llm_dim, grounder_dim), nn.LayerNorm(grounder_dim))
        self.queries = nn.Parameter(torch.empty(1, num_queries, grounder_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(grounder_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(grounder_dim)
        self.ffn = nn.Sequential(nn.Linear(grounder_dim, 4 * grounder_dim), nn.GELU(),
                                 nn.Linear(4 * grounder_dim, grounder_dim))
        self.norm2 = nn.LayerNorm(grounder_dim)

    def forward(self, hidden: torch.Tensor, reasoning_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or reasoning_mask.shape != hidden.shape[:2]:
            raise ValueError("Expected hidden BxTxD and reasoning_mask BxT")
        mask = reasoning_mask.to(device=hidden.device, dtype=torch.bool)
        hidden = hidden.to(dtype=self.input_projection[0].weight.dtype)
        memory = self.input_projection(hidden)
        # No system/user/padding state can leak through even on empty reasoning.
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        if memory.shape[1] == 0:
            memory = memory.new_zeros(memory.shape[0], 1, memory.shape[-1])
            mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        else:
            mask = mask.clone()
            mask[:, 0] |= ~mask.any(dim=1)
        queries = self.queries.expand(hidden.shape[0], -1, -1)
        attended, _ = self.attention(queries, memory, memory, key_padding_mask=~mask, need_weights=False)
        queries = self.norm1(queries + attended)
        return self.norm2(queries + self.ffn(queries))


class SEGAnchorProjector(nn.Module):
    """Project h_SEG to a spatial controller query, not a final SAM prompt."""

    def __init__(self, llm_dim: int, grounder_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, grounder_dim),
                                  nn.GELU(), nn.Linear(grounder_dim, grounder_dim))

    def forward(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        if seg_hidden.ndim != 2:
            raise ValueError("seg_hidden must have shape BxD")
        return self.proj(seg_hidden.to(dtype=self.proj[0].weight.dtype)).unsqueeze(1)


class _GroundingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attention(queries, queries, queries, need_weights=False)
        queries = self.norm1(queries + attended)
        attended, _ = self.cross_attention(queries, image_tokens, image_tokens, need_weights=False)
        queries = self.norm2(queries + attended)
        return self.norm3(queries + self.ffn(queries))


class SemanticSpatialGrounder(nn.Module):
    """Ground semantic/anchor queries against the full vision patch grid."""

    def __init__(self, vision_dim: int, grounder_dim: int = 256,
                 num_layers: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        if num_layers < 1 or grounder_dim % num_heads:
            raise ValueError("num_layers must be positive and num_heads must divide grounder_dim")
        self.vision_dim = int(vision_dim)
        self.image_projection = nn.Sequential(nn.Linear(vision_dim, grounder_dim), nn.LayerNorm(grounder_dim))
        self.layers = nn.ModuleList([_GroundingBlock(grounder_dim, num_heads) for _ in range(num_layers)])
        self.query_localization = nn.Linear(grounder_dim, grounder_dim)
        self.key_localization = nn.Linear(grounder_dim, grounder_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, query_bank: torch.Tensor, patch_tokens: torch.Tensor,
                spatial_hw: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        if patch_tokens.ndim == 4:
            actual_hw = tuple(patch_tokens.shape[1:3])
            if spatial_hw is not None and tuple(spatial_hw) != actual_hw:
                raise ValueError(f"spatial_hw={spatial_hw} disagrees with patch grid {actual_hw}")
            spatial_hw = actual_hw
            tokens = patch_tokens.reshape(patch_tokens.shape[0], -1, patch_tokens.shape[-1])
        elif patch_tokens.ndim == 3:
            tokens = patch_tokens
            if spatial_hw is None:
                side = math.isqrt(tokens.shape[1])
                if side * side != tokens.shape[1]:
                    raise ValueError("Flattened non-square patch tokens require explicit spatial_hw")
                spatial_hw = (side, side)
        else:
            raise ValueError("patch_tokens must be BxHxWxC or BxNxC")
        h, w = (int(v) for v in spatial_hw)
        if h <= 0 or w <= 0 or h * w != tokens.shape[1] or tokens.shape[-1] != self.vision_dim:
            raise ValueError(f"Invalid patch grid or vision dimension: {tuple(tokens.shape)}, {(h, w)}")
        if query_bank.ndim != 3 or query_bank.shape[0] != tokens.shape[0] or query_bank.shape[1] < 1:
            raise ValueError("query_bank must be BxQxD with an anchor in the final slot")
        image_tokens = self.image_projection(tokens.to(dtype=self.image_projection[0].weight.dtype))
        queries = query_bank.to(dtype=self.image_projection[0].weight.dtype)
        for layer in self.layers:
            queries = layer(queries, image_tokens)
        q_loc = F.normalize(self.query_localization(queries[:, -1:, :]).float(), dim=-1)
        k_loc = F.normalize(self.key_localization(image_tokens).float(), dim=-1)
        # FP32 avoids low-precision normalization/temperature instability.
        temperature = self.log_temperature.float().exp().clamp(max=100.0)
        spatial_logits = (temperature * (q_loc @ k_loc.transpose(1, 2))).reshape(tokens.shape[0], 1, h, w)
        return {"grounded_queries": queries, "high_res_tokens": image_tokens,
                "spatial_logits": spatial_logits, "spatial_prob": spatial_logits.sigmoid(),
                "spatial_hw": (h, w)}


class AutomaticGeometricPromptGenerator(nn.Module):
    """Generate positive/negative points, differentiable boxes and coarse masks."""

    def __init__(self, grounder_dim: int = 256, num_pos_points: int = 4,
                 num_neg_points: int = 4) -> None:
        super().__init__()
        if num_pos_points < 0 or num_neg_points < 0:
            raise ValueError("Point counts cannot be negative")
        self.num_pos_points, self.num_neg_points = int(num_pos_points), int(num_neg_points)
        self.box_head = nn.Sequential(nn.LayerNorm(2 * grounder_dim), nn.Linear(2 * grounder_dim, grounder_dim),
                                      nn.GELU(), nn.Linear(grounder_dim, 4))
        self.coarse_mask_head = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))

    @torch.no_grad()
    def _select_points(self, spatial_prob: torch.Tensor, input_image_size: Tuple[int, int]):
        b, _, h, w = spatial_prob.shape
        count = self.num_pos_points + self.num_neg_points
        if count > h * w:
            raise ValueError(f"Need at least {count} spatial cells for distinct positive/negative points")
        flat = spatial_prob.flatten(1)
        positive = flat.topk(self.num_pos_points, dim=1, largest=True).indices
        # Exclude positive locations so a tie never assigns conflicting labels.
        negative_scores = flat.scatter(1, positive, float("inf"))
        negative = negative_scores.topk(self.num_neg_points, dim=1, largest=False).indices
        indices = torch.cat((positive, negative), dim=1)
        input_h, input_w = (int(v) for v in input_image_size)
        x = (indices.remainder(w).float() + 0.5) * (input_w / w) - 0.5
        y = (torch.div(indices, w, rounding_mode="floor").float() + 0.5) * (input_h / h) - 0.5
        coords = torch.stack((x, y), dim=-1)
        labels = torch.cat((torch.ones(b, self.num_pos_points, device=flat.device, dtype=torch.long),
                            torch.zeros(b, self.num_neg_points, device=flat.device, dtype=torch.long)), dim=1)
        return coords, labels

    def forward(self, spatial_logits: torch.Tensor, grounded_queries: torch.Tensor,
                high_res_tokens: torch.Tensor, input_image_size: Tuple[int, int],
                mask_input_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        if spatial_logits.ndim != 4 or spatial_logits.shape[1] != 1:
            raise ValueError("spatial_logits must have shape Bx1xHxW")
        b, _, h, w = spatial_logits.shape
        if high_res_tokens.shape[:2] != (b, h * w) or grounded_queries.shape[0] != b:
            raise ValueError("Spatial prior, image tokens and grounded queries must have matching grids/batches")
        if len(input_image_size) != 2 or len(mask_input_size) != 2 or min(*input_image_size, *mask_input_size) <= 0:
            raise ValueError("Image and mask input sizes must be positive (H,W) pairs")
        prob = spatial_logits.float().sigmoid()
        weights = prob.flatten(2)
        pooled = (weights @ high_res_tokens.float()).squeeze(1) / weights.sum(dim=-1).clamp_min(1e-6)
        box_features = torch.cat((grounded_queries[:, -1, :].float(), pooled), dim=-1)
        cxcywh = self.box_head(box_features.to(dtype=self.box_head[0].weight.dtype)).float().sigmoid()
        centre, extent = cxcywh[:, :2], cxcywh[:, 2:]
        box_norm = torch.cat((centre - extent / 2, centre + extent / 2), dim=-1).clamp(0.0, 1.0)
        input_h, input_w = input_image_size
        scale = box_norm.new_tensor([input_w, input_h, input_w, input_h])
        point_coords, point_labels = self._select_points(prob, input_image_size)
        coarse_logits = self.coarse_mask_head(spatial_logits.to(dtype=self.coarse_mask_head[0].weight.dtype))
        coarse_logits = F.interpolate(coarse_logits, size=tuple(mask_input_size), mode="bilinear", align_corners=False)
        return {"point_coords": point_coords, "point_labels": point_labels,
                "pred_box_cxcywh": cxcywh, "pred_box_norm": box_norm, "pred_box_xyxy": box_norm * scale,
                "pred_coarse_mask_logits": coarse_logits}


class SemanticPromptProjector(nn.Module):
    """Produce compact semantic sparse prompts outside SAM's PromptEncoder."""

    def __init__(self, grounder_dim: int = 256, sam_prompt_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(grounder_dim), nn.Linear(grounder_dim, sam_prompt_dim),
                                  nn.GELU(), nn.Linear(sam_prompt_dim, sam_prompt_dim))

    def forward(self, grounded_queries: torch.Tensor) -> torch.Tensor:
        return self.proj(grounded_queries.to(dtype=self.proj[0].weight.dtype))


class PromptFusionGate(nn.Module):
    """Gate each semantic query before concatenating it with geometric tokens."""

    def __init__(self, grounder_dim: int = 256) -> None:
        super().__init__()
        self.gate = nn.Linear(grounder_dim, 1)

    def forward(self, grounded_queries: torch.Tensor, semantic_prompts: torch.Tensor,
                geo_sparse: torch.Tensor):
        if semantic_prompts.shape[:2] != grounded_queries.shape[:2]:
            raise ValueError("There must be one semantic prompt per grounded query")
        gates = self.gate(grounded_queries.to(dtype=self.gate.weight.dtype)).sigmoid()
        semantic_gated = gates * semantic_prompts
        return torch.cat((geo_sparse.to(dtype=semantic_gated.dtype), semantic_gated), dim=1), gates
