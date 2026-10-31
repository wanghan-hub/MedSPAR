"""SteerViT-style gated text cross-attention for SAM/MedSAM image tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedTextCrossAttention(nn.Module):
    """Let image tokens attend to text tokens through a zero-initialized tanh gate.

    image_tokens are queries; text_tokens are keys and values. The gate is
    initialized to zero, so the module initially behaves like an identity residual
    branch, matching the SteerViT stabilizing design.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        use_ffn: bool = False,
        ffn_mult: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.norm = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))

        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            hidden = int(self.dim * ffn_mult)
            self.ffn = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.dim),
            )
            self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            image_tokens: (B, N, C)
            text_tokens:  (B, L, C)
            text_mask:    (B, L), True for valid text tokens.
        """
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()

        q = self.norm(image_tokens)
        out, _ = self.attn(
            query=q,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        image_tokens = image_tokens + torch.tanh(self.attn_gate) * out

        if self.use_ffn:
            image_tokens = image_tokens + torch.tanh(self.ffn_gate) * self.ffn(image_tokens)
        return image_tokens
