"""LLM loader with optional LoRA/QLoRA support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from .seg_token import DEFAULT_SEG_TOKEN, resolve_existing_seg_token


class SEGRowEmbedding(nn.Module):
    """Train a single FP32 token row without a dense vocabulary gradient/state."""
    def __init__(self, base, token_id, row=None):
        super().__init__()
        self.base = base
        self.token_id = int(token_id)
        self.seg_row = row if row is not None else nn.Parameter(base.weight[token_id].detach().float().clone())
        self.num_embeddings, self.embedding_dim = base.weight.shape
        self.padding_idx = getattr(base, "padding_idx", None)
        self.base.requires_grad_(False)

    @property
    def weight(self):
        # HF uses this for device/dtype/vocabulary inspection, not the forward row.
        return self.base.weight

    def forward(self, input_ids):
        original = self.base(input_ids)
        return torch.where(input_ids.eq(self.token_id).unsqueeze(-1),
                           self.seg_row.to(original.dtype), original)


class SEGRowLMHead(nn.Module):
    """Keep the original vocabulary head frozen, replacing only the SEG logit."""
    def __init__(self, base, token_id, row=None):
        super().__init__()
        self.base = base
        self.token_id = int(token_id)
        self.seg_row = row if row is not None else nn.Parameter(base.weight[token_id].detach().float().clone())
        self.out_features, self.in_features = base.weight.shape
        self.base.requires_grad_(False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        logits = self.base(hidden)
        seg_logit = torch.nn.functional.linear(hidden, self.seg_row.to(hidden.dtype))
        bias = getattr(self.base, "bias", None)
        if bias is not None:
            seg_logit = seg_logit + bias[self.token_id]
        # CopySlices retains gradients to other logits and to the replacement row.
        logits[..., self.token_id] = seg_logit.to(logits.dtype)
        return logits


def install_seg_row_adapters(llm, seg_token_id):
    """All independent P stages use identical checkpoint keys, frozen or trained."""
    causal = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
    embedding, head = causal.get_input_embeddings(), causal.get_output_embeddings()
    if isinstance(embedding, SEGRowEmbedding):
        if embedding.token_id != seg_token_id or not isinstance(head, SEGRowLMHead):
            raise RuntimeError("Inconsistent SEG row adapters")
        return
    if head is None:
        raise RuntimeError("Independent P training requires a causal LM output head")
    tied = embedding.weight is head.weight
    wrapped_embedding = SEGRowEmbedding(embedding, seg_token_id)
    wrapped_head = SEGRowLMHead(head, seg_token_id, wrapped_embedding.seg_row if tied else None)
    causal.set_input_embeddings(wrapped_embedding)
    causal.set_output_embeddings(wrapped_head)


def resolve_torch_dtype(dtype: str):
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@dataclass
class LLMBackboneOutput:
    model: nn.Module
    tokenizer: object
    seg_token_id: int
    hidden_size: int


def _maybe_enable_lora(
    model: nn.Module,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
):
    if not use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise ImportError("LoRA requested but `peft` is not installed. Install peft or disable --use_lora.") from exc

    # k-bit preparation casts ordinary half-precision weights to fp32; only
    # apply it to genuinely quantized models, preserving the requested dtype.
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    targets = [x.strip() for x in lora_target_modules.split(",") if x.strip()]
    cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return get_peft_model(model, cfg)


def load_llm_backbone(
    model_name_or_path: str,
    seg_token: str = DEFAULT_SEG_TOKEN,
    torch_dtype: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    freeze_llm_base: bool = True,
) -> LLMBackboneOutput:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The permanent local checkpoint owns the vocabulary. Training/evaluation
    # must never add a token or silently initialize a fresh embedding row.
    seg_token_id = resolve_existing_seg_token(tokenizer, seg_token=seg_token)

    dtype = resolve_torch_dtype(torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
        if embedding is not None and embedding.weight.shape[0] != len(tokenizer):
            raise RuntimeError("Permanent SEG tokenizer and LLM vocabulary sizes differ; refusing to resize.")

    if freeze_llm_base:
        for p in model.parameters():
            p.requires_grad = False

    model = _maybe_enable_lora(
        model,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    if hidden_size <= 0:
        raise RuntimeError("Cannot infer LLM hidden size from config.")
    return LLMBackboneOutput(model=model, tokenizer=tokenizer, seg_token_id=seg_token_id, hidden_size=hidden_size)
