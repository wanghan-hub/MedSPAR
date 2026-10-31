#!/usr/bin/env python3
"""Permanently add [SEG] as a special token to a local Hugging Face causal LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Original local HF model directory.")
    p.add_argument("--output_dir", type=str, required=True, help="New output directory to save the model with [SEG].")
    p.add_argument("--seg_token", type=str, default="[SEG]")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda. cpu is safer but slower.")
    p.add_argument("--init_text", type=str, default="segmentation mask target region lesion tumor object")
    p.add_argument("--max_shard_size", type=str, default="2GB")
    p.add_argument("--overwrite_output_dir", action="store_true")
    return p.parse_args()


def dtype_from_str(dtype: str):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def is_same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.data_ptr() == b.data_ptr()


def encode_init_ids(tokenizer, init_text: str, old_vocab_size: int) -> List[int]:
    ids = tokenizer.encode(init_text, add_special_tokens=False)
    ids = [int(i) for i in ids if 0 <= int(i) < old_vocab_size]
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


@torch.no_grad()
def init_new_token_rows(model, tokenizer, seg_token: str, old_vocab_size: int, init_text: str):
    seg_id = tokenizer.convert_tokens_to_ids(seg_token)
    input_emb = model.get_input_embeddings()
    if input_emb is None:
        raise RuntimeError("model.get_input_embeddings() returned None.")

    input_weight = input_emb.weight
    if int(seg_id) < int(old_vocab_size):
        output_emb = model.get_output_embeddings()
        return {"seg_token_id": int(seg_id), "init_source": "preserved_existing_SEG_row",
                "lm_head_present": output_emb is not None,
                "lm_head_tied_to_input_embeddings": output_emb is not None and is_same_storage(input_weight, output_emb.weight),
                "lm_head_row_initialized": False, "init_ids": []}
    init_ids = encode_init_ids(tokenizer, init_text, old_vocab_size)

    if init_ids:
        init_vec = input_weight[init_ids].detach().float().mean(dim=0).to(dtype=input_weight.dtype, device=input_weight.device)
        init_source = f"mean(init_text_tokens={init_ids})"
    else:
        init_vec = input_weight[:old_vocab_size].detach().float().mean(dim=0).to(dtype=input_weight.dtype, device=input_weight.device)
        init_source = "mean(all_old_input_embeddings)"

    input_weight[seg_id].copy_(init_vec)

    output_emb = model.get_output_embeddings()
    output_initialized = False
    output_tied = False

    if output_emb is not None and hasattr(output_emb, "weight"):
        output_weight = output_emb.weight
        output_tied = is_same_storage(input_weight, output_weight)

        if output_weight.shape[0] > seg_id:
            if output_tied:
                output_initialized = True
            else:
                if init_ids and max(init_ids) < output_weight.shape[0]:
                    out_vec = output_weight[init_ids].detach().float().mean(dim=0).to(dtype=output_weight.dtype, device=output_weight.device)
                else:
                    out_vec = output_weight[:old_vocab_size].detach().float().mean(dim=0).to(dtype=output_weight.dtype, device=output_weight.device)
                output_weight[seg_id].copy_(out_vec)
                output_initialized = True

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    return {
        "seg_token_id": int(seg_id),
        "init_source": init_source,
        "lm_head_present": output_emb is not None,
        "lm_head_tied_to_input_embeddings": bool(output_tied),
        "lm_head_row_initialized": bool(output_initialized),
        "init_ids": init_ids,
    }


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken
    model_path = Path(args.model_path)
    out_dir = Path(args.output_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_dir}\n"
            "Use --overwrite_output_dir only if you intentionally want to overwrite it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = dtype_from_str(args.dtype)

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=True,
    )

    old_tokenizer_len = len(tokenizer)
    before_ids = tokenizer.encode(args.seg_token, add_special_tokens=False)
    before_token_id = tokenizer.convert_tokens_to_ids(args.seg_token)

    print("Before adding:")
    print(f"  len(tokenizer): {old_tokenizer_len}")
    print(f"  encode({args.seg_token!r}): {before_ids}")
    print(f"  convert_tokens_to_ids: {before_token_id}")

    added_token = AddedToken(args.seg_token, special=True, normalized=False, single_word=False, lstrip=False, rstrip=False)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [added_token]})

    after_ids = tokenizer.encode(args.seg_token, add_special_tokens=False)
    seg_id = tokenizer.convert_tokens_to_ids(args.seg_token)

    if len(after_ids) != 1 or int(after_ids[0]) != int(seg_id):
        raise RuntimeError(
            f"Failed to add {args.seg_token} as a single token. "
            f"encode={after_ids}, token_id={seg_id}"
        )

    print("After adding tokenizer token:")
    print(f"  num_added: {num_added}")
    print(f"  len(tokenizer): {len(tokenizer)}")
    print(f"  seg_token_id: {seg_id}")
    print(f"  encode({args.seg_token!r}): {after_ids}")

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        device_map=None,
    )
    model.to(torch.device(args.device))

    old_embedding_size = int(model.get_input_embeddings().weight.shape[0])
    print(f"Old input embedding size: {old_embedding_size}")
    print(f"Tokenizer length after adding token: {len(tokenizer)}")

    if len(tokenizer) != old_embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model.config, "text_config"):
            model.config.text_config.vocab_size = len(tokenizer)
        print(f"Resized token embeddings: {old_embedding_size} -> {len(tokenizer)}")
    else:
        print("No resize needed: tokenizer length already equals embedding size.")

    init_info = init_new_token_rows(
        model=model,
        tokenizer=tokenizer,
        seg_token=args.seg_token,
        old_vocab_size=min(old_embedding_size, old_tokenizer_len),
        init_text=args.init_text,
    )

    final_ids = tokenizer.encode(args.seg_token, add_special_tokens=False)
    final_seg_id = tokenizer.convert_tokens_to_ids(args.seg_token)
    final_emb_size = int(model.get_input_embeddings().weight.shape[0])
    assert len(final_ids) == 1
    assert int(final_ids[0]) == int(final_seg_id)
    assert final_emb_size == len(tokenizer)

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Saving tokenizer and model to: {out_dir}")
    tokenizer.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=args.max_shard_size)

    info = {
        "source_model_path": str(model_path),
        "output_dir": str(out_dir),
        "seg_token": args.seg_token,
        "num_added": int(num_added),
        "old_tokenizer_len": int(old_tokenizer_len),
        "new_tokenizer_len": int(len(tokenizer)),
        "old_embedding_size": int(old_embedding_size),
        "new_embedding_size": int(final_emb_size),
        "seg_token_id": int(final_seg_id),
        "seg_encode": [int(x) for x in final_ids],
        "dtype": args.dtype,
        "init_text": args.init_text,
        **init_info,
    }

    with open(out_dir / "seg_token_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
