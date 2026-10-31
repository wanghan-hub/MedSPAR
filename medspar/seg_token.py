"""Utilities for [SEG] token handling."""

from __future__ import annotations

from typing import Iterable

import torch

DEFAULT_SEG_TOKEN = "[SEG]"


def resolve_existing_seg_token(tokenizer, seg_token: str = DEFAULT_SEG_TOKEN) -> int:
    """Validate the permanent token without changing tokenizer/model state.

    The final design calls this action token ``<SEG>`` conceptually. Existing
    project checkpoints spell it ``[SEG]``; keep that spelling and id intact.
    """
    ids = tokenizer.encode(seg_token, add_special_tokens=False)
    token_id = tokenizer.convert_tokens_to_ids(seg_token)
    if token_id is None or len(ids) != 1 or int(ids[0]) != int(token_id):
        raise RuntimeError(f"{seg_token!r} must already exist as one permanent tokenizer token; got {ids}.")
    if int(token_id) < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"{seg_token!r} is missing from the permanent tokenizer vocabulary.")
    return int(token_id)


def add_seg_token(tokenizer, seg_token: str = DEFAULT_SEG_TOKEN) -> int:
    """Add [SEG] as an additional special token if needed and return token id."""
    if seg_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
    seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
    if seg_token_id is None or seg_token_id < 0:
        raise RuntimeError(f"Failed to add/resolve seg token: {seg_token}")
    return int(seg_token_id)


def ensure_single_seg_token(text: str, seg_token: str = DEFAULT_SEG_TOKEN, append_if_missing: bool = True) -> str:
    count = text.count(seg_token)
    if count == 0 and append_if_missing:
        text = text.rstrip() + " " + seg_token
        count = 1
    if count != 1:
        raise ValueError(f"Each output must contain exactly one {seg_token}; got {count}. text={text[:200]!r}")
    return text


def find_seg_positions(input_ids: torch.Tensor, seg_token_id: int) -> torch.Tensor:
    """Return one [SEG] position per batch item.

    Args:
        input_ids: B x T text token ids, without visual prefix.
    Returns:
        positions: B tensor. Raises if any sample does not contain exactly one [SEG].
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be BxT, got {tuple(input_ids.shape)}")
    mask = input_ids.eq(int(seg_token_id))
    counts = mask.sum(dim=1)
    bad = torch.nonzero(counts.ne(1), as_tuple=False).flatten()
    if bad.numel() > 0:
        raise RuntimeError(f"Each sequence must contain exactly one [SEG]. Bad batch indices: {bad.tolist()}, counts={counts.tolist()}")
    return mask.float().argmax(dim=1).long()


def gather_seg_hidden(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    seg_token_id: int,
    visual_prefix_len: int = 0,
) -> torch.Tensor:
    """Gather hidden state at the [SEG] position from an LLM sequence.

    hidden_states includes visual prefix if inputs_embeds were built as
    [visual_embeds, text_embeds], while input_ids only correspond to text tokens.
    Therefore the [SEG] position must be offset by visual_prefix_len.
    """
    seg_pos_text = find_seg_positions(input_ids, seg_token_id)
    seg_pos_full = seg_pos_text + int(visual_prefix_len)
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, seg_pos_full, :]
