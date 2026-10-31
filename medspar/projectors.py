"""Projection modules bridging MedSAM visual tokens and LLM / SAM prompt space."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualTokenCompressor(nn.Module):
    """Compress dense MedSAM/SAM patch tokens before sending them to an LLM.

    Input patch tokens are expected to be B x H x W x C, usually B x 64 x 64 x 768
    for MedSAM ViT-B. Feeding all 4096 tokens to an LLM is expensive, so this
    module reduces the grid to out_grid x out_grid, e.g. 16x16=256 tokens or
    8x8=64 tokens.
    """

    def __init__(
        self,
        out_grid: int = 16,
        mode: Literal["avg", "max"] = "avg",
        vision_dim: int = 768,
        trainable_norm: bool = True,
    ) -> None:
        super().__init__()
        self.out_grid = int(out_grid)
        self.vision_dim = int(vision_dim)
        if mode not in {"avg", "max"}:
            raise ValueError(f"Unsupported compression mode: {mode}")
        self.mode = mode
        # Compression itself is adaptive pooling. This LayerNorm gives the compressor
        # a small trainable part and stabilizes tokens before the visual projector.
        self.norm = nn.LayerNorm(self.vision_dim) if trainable_norm else nn.Identity()

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError(f"patch_tokens must be BxHxWxC, got shape={tuple(patch_tokens.shape)}")
        x = patch_tokens.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        if self.mode == "avg":
            x = F.adaptive_avg_pool2d(x, (self.out_grid, self.out_grid))
        else:
            x = F.adaptive_max_pool2d(x, (self.out_grid, self.out_grid))
        x = x.permute(0, 2, 3, 1).contiguous()  # B, g, g, C
        x = x.view(x.shape[0], self.out_grid * self.out_grid, x.shape[-1])
        return self.norm(x)


class VisualProjector(nn.Module):
    """Project compressed MedSAM patch tokens into the LLM hidden dimension."""

    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 2048,
        projector_type: Literal["mlp2x_gelu", "linear"] = "mlp2x_gelu",
    ) -> None:
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.llm_dim = int(llm_dim)
        self.norm = nn.LayerNorm(self.vision_dim)
        if projector_type == "linear":
            self.proj = nn.Linear(self.vision_dim, self.llm_dim)
        elif projector_type == "mlp2x_gelu":
            self.proj = nn.Sequential(
                nn.Linear(self.vision_dim, self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, self.llm_dim),
            )
        else:
            raise ValueError(f"Unsupported projector_type={projector_type}")

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # Keep LayerNorm/projection numerically safe; caller may autocast.
        return self.proj(self.norm(visual_tokens))


class SEGProjector(nn.Module):
    """Project the LLM [SEG] hidden state to SAM sparse prompt embedding space."""

    def __init__(
        self,
        llm_dim: int,
        sam_prompt_dim: int = 256,
        projector_type: Literal["mlp2x_gelu", "linear"] = "mlp2x_gelu",
    ) -> None:
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.sam_prompt_dim = int(sam_prompt_dim)
        self.norm = nn.LayerNorm(self.llm_dim)
        if projector_type == "linear":
            self.proj = nn.Linear(self.llm_dim, self.sam_prompt_dim)
        elif projector_type == "mlp2x_gelu":
            self.proj = nn.Sequential(
                nn.Linear(self.llm_dim, self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, self.sam_prompt_dim),
            )
        else:
            raise ValueError(f"Unsupported projector_type={projector_type}")

    def forward(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        if seg_hidden.ndim != 2:
            raise ValueError(f"seg_hidden must be BxD, got shape={tuple(seg_hidden.shape)}")
        seg_prompt = self.proj(self.norm(seg_hidden))
        return seg_prompt.unsqueeze(1)  # B, 1, 256
