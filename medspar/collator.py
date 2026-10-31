"""Conversation collation with explicit assistant and reasoning token masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from experiment_protocol import conversation_ids


def format_prompt(question: str, system_prompt: str) -> str:
    return (
        f"### System:\n{system_prompt}\n\n"
        f"### User:\n<image>\n{question.strip()}\n\n"
        f"### Assistant:\n"
    )


def encode_prompt(tokenizer, question, system_prompt, max_length):
    return conversation_ids(tokenizer, question, max_length=max_length, system_prompt=system_prompt)[0]


def pad_token_id(tokenizer) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer needs an existing pad or EOS token.")


@dataclass
class LLMReasonSegCollator:
    tokenizer: Any
    max_length: int = 1024
    system_prompt: str = "You are a helpful medical image segmentation assistant."
    seg_token: str = "[SEG]"
    require_single_seg: bool = True

    def _format_prompt(self, question: str) -> str:
        return format_prompt(question, self.system_prompt)

    def _get_seg_token_id(self) -> int:
        ids = self.tokenizer.encode(self.seg_token, add_special_tokens=False)
        token_id = self.tokenizer.convert_tokens_to_ids(self.seg_token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if len(ids) != 1 or token_id is None or int(ids[0]) != int(token_id) or token_id == unk_id:
            raise RuntimeError(f"{self.seg_token} must already be a permanent single tokenizer token: {ids}.")
        return int(token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seg_id = self._get_seg_token_id()
        sequences, prompt_lengths = [], []
        for item in batch:
            ids, plen = conversation_ids(self.tokenizer, item["question"], item["answer"],
                                         self.max_length, self.system_prompt)
            sequences.append(ids)
            prompt_lengths.append(plen)
        length = max(map(len, sequences))
        input_ids = torch.full((len(batch), length), pad_token_id(self.tokenizer), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, (ids, plen) in enumerate(zip(sequences, prompt_lengths)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            assistant_mask[i, plen:len(ids)] = True
        labels = input_ids.masked_fill(~assistant_mask, -100)
        seg_positions = input_ids.eq(seg_id) & assistant_mask
        seg_counts = seg_positions.sum(-1)
        if self.require_single_seg and seg_counts.ne(1).any():
            bad = torch.where(seg_counts.ne(1))[0].tolist()
            raise RuntimeError(f"Assistant answers must retain exactly one {self.seg_token}; "
                               f"bad batch indices={bad}, counts={seg_counts.tolist()}. "
                               "Check annotations and increase max_length if truncated.")
        positions = torch.arange(length).unsqueeze(0)
        first_seg = positions.expand_as(input_ids).masked_fill(~seg_positions, length).min(-1).values
        reasoning_mask = assistant_mask & positions.lt(first_seg.unsqueeze(1))
        for special_id in getattr(self.tokenizer, "all_special_ids", []):
            reasoning_mask &= input_ids.ne(int(special_id))
        out = {
            "image": torch.stack([x["image"] for x in batch]),
            "mask": torch.stack([x["mask"] for x in batch]),
            "box": torch.stack([x["box"] for x in batch]),
            "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "assistant_mask": assistant_mask, "reasoning_mask": reasoning_mask,
            "seg_counts": seg_counts,
        }
        for key in ("question", "answer", "steer_text", "dataset", "target", "image_path", "mask_path"):
            out[key] = [item.get(key, "") for item in batch]
        return out


class LLMGenerationCollator:
    """Build image/question model inputs; metadata never enters tokenization."""

    def __init__(self, tokenizer, max_length=1024,
                 system_prompt="You are a helpful medical image segmentation assistant."):
        self.tokenizer, self.max_length, self.system_prompt = tokenizer, int(max_length), system_prompt

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [encode_prompt(self.tokenizer, x["question"], self.system_prompt, self.max_length) for x in batch]
        width = max(map(len, ids))
        input_ids = torch.full((len(batch), width), pad_token_id(self.tokenizer), dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for i, tokens in enumerate(ids):
            input_ids[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long)
            mask[i, -len(tokens):] = 1
        out = {"image": torch.stack([x["image"] for x in batch]),
               "input_ids": input_ids, "attention_mask": mask,
               "prompt_text": [format_prompt(x["question"], self.system_prompt) for x in batch]}
        for key in ("question", "dataset", "image_path", "mask_path", "record_index", "target",
                    "target_phrase", "body_part", "modality"):
            out[key] = [x.get(key, "") for x in batch]
        out["steer_text"] = list(out["question"])
        out["reference_answer"] = [x.get("reference_answer", x.get("answer", "")) for x in batch]
        return out

