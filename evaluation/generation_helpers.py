"""Generation CLI and batch helper used by official MedSPAR evaluation."""
from __future__ import annotations
import argparse
from typing import Any, Dict
import torch
from medspar.experiments import EXPERIMENTS
from medspar.generation import generate_reasoning
from medspar.train_utils import autocast_context
def build_parser():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_root", type=str, default="/data/ReaSeg/data")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--eval_box_mode", type=str, default="tight", choices=["tight", "enlarged", "full"])
    p.add_argument("--enlarge_ratio", type=float, default=1.5)

    # Vision backbone selection, kept consistent with eval_medspar.py.
    p.add_argument("--vision_type", type=str, default=None, choices=["medsam", "TC_MVE"])
    p.add_argument("--medsam_checkpoint", type=str, required=True)
    p.add_argument("--model_type", type=str, default="vit_b")
    p.add_argument("--stage0_checkpoint", type=str, default="")
    p.add_argument("--stage1_checkpoint", type=str, default="")
    p.add_argument("--steer_text_encoder", type=str, default=None, help="Default: training checkpoint's text tokenizer/model path; override only to relocate the same resource")
    p.add_argument("--steer_cross_attn_layers", nargs="+", type=int, default=[2, 5, 6, 11])
    p.add_argument("--steer_text_max_length", type=int, default=64)
    p.add_argument("--vision_dim", type=int, default=768)

    # LLM
    p.add_argument("--llm_name_or_path", type=str, required=True)
    p.add_argument("--llm_torch_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--seg_token", type=str, default="[SEG]")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--visual_grid", type=int, default=16)

    # Stage selection. Text generation itself does not use the box policy, but the
    # stage is recorded and used by the bash script to select checkpoints.
    p.add_argument("--stage", type=str, default="unified", choices=["unified"])
    p.add_argument("--num_reasoning_queries", type=int, default=4)
    p.add_argument("--num_pos_points", type=int, default=4)
    p.add_argument("--num_neg_points", type=int, default=4)
    p.add_argument("--grounder_layers", type=int, default=2)
    p.add_argument("--grounder_dim", type=int, default=256)

    p.add_argument("--experiment_id", choices=EXPERIMENTS, default=None)

    # Checkpoint / runtime
    p.add_argument("--checkpoint", type=str, required=True, help="LLM-MedSAM checkpoint: best.pth/last.pth")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=0, help="First N samples per dataset for server verification; 0 evaluates all.")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--output_dir", type=str, default="./output/eval_medspar_generation")
    p.add_argument("--rescore_csv", type=str, default="", help="Re-score an existing generated_texts.csv without running generation again.")

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=4)

    # Text metric switches
    p.add_argument("--system_prompt", type=str, default="You are a helpful medical image segmentation assistant.")
    p.add_argument("--save_jsonl", action="store_true", help="Also save generated_texts.jsonl. Enabled by default in code output.")
    p.add_argument("--disable_stop_on_seg", action="store_true", help="Do not use [SEG] as an EOS/stop token during generation.")
    return p


def generate_batch(model, batch: Dict[str, Any], device: torch.device, args) -> Dict[str, Any]:
    with autocast_context(device, args.precision):
        return generate_reasoning(
            model, batch["image"], batch["input_ids"], batch["attention_mask"], batch["question"],
            max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
            temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop_on_seg=not args.disable_stop_on_seg,
        )

