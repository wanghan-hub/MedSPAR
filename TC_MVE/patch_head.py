"""Training-only auxiliary patch segmentation head."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchSegHead(nn.Module):
    """Predict a coarse segmentation logit from each image-encoder patch token.

    This head is used only during training to encourage text-conditioned patch
    tokens. It is not required during inference.
    """

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.head = nn.Linear(dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            patch_tokens: (B, H, W, C)
        Returns:
            patch_logits: (B, 1, H, W)
        """
        logits = self.head(patch_tokens)  # B,H,W,1
        return logits.permute(0, 3, 1, 2).contiguous()
