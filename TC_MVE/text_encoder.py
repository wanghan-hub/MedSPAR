"""Configurable frozen text encoder plus text-to-vision MLP connector."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class MedicalTextEncoder(nn.Module):
    """AutoModel-based text encoder for Steer-MedSAM.

    The text backbone is frozen by default. The connector remains trainable and
    maps token-level text features to the SAM image-encoder token dimension.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        vision_dim: int = 768,
        max_length: int = 64,
        freeze_text_encoder: bool = True,
        normalize_text: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = int(max_length)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.normalize_text = bool(normalize_text)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.text_model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        text_dim = int(self.text_model.config.hidden_size)
        self.text_dim = text_dim
        self.vision_dim = int(vision_dim)

        self.connector = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, self.vision_dim),
        )

        if self.freeze_text_encoder:
            for p in self.text_model.parameters():
                p.requires_grad = False
            self.text_model.eval()

    def train(self, mode: bool = True):  # keep frozen text backbone in eval mode
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"BERT input exceeds text_max_length={self.max_length}; no truncation")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if self.freeze_text_encoder:
            with torch.no_grad():
                text_feats = self.text_model(**encoded).last_hidden_state
        else:
            text_feats = self.text_model(**encoded).last_hidden_state

        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=-1)

        text_feats = self.connector(text_feats)
        text_mask = encoded["attention_mask"].bool()
        return text_feats, text_mask
