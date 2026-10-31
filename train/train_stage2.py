#!/usr/bin/env python3
"""Train Full or one experiment_id in a single joint Stage2 from Stage1 + base LLM."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from medspar.collator import LLMReasonSegCollator, LLMGenerationCollator
from medspar.dataset import LLMReasonSegDataset
from medspar.losses import binary_dice_iou_from_logits, compute_llm_medsam_loss, masks_to_boxes, localization_metrics
from medspar.model import build_medspar
from medspar.experiments import EXPERIMENTS, ExperimentConfig, MODEL_CONFIG_FIELDS
from medspar.stage_policy import (get_stage_config, is_grounding_stage,
    apply_stage_freeze_policy, steer_trainable_modules, GROUNDING_MODULE_NAMES, keep_frozen_modules_eval)
from medspar.train_utils import (
    autocast_context,
    load_checkpoint,
    make_grad_scaler,
    move_batch_to_device,
    print_parameter_statistics,
    print_precision_policy,
    print_trainable_parameter_names,
    save_checkpoint,
    save_json, parameter_statistics, unwrap_model, optimizer_coverage, prepare_trainable_precision,
)
from experiment_protocol import native_scores, native_metrics, PROTOCOL_VERSION
from experiment_protocol import training_budget
from experiment_reporting import TrainingReporter


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_root", type=str, default="/data/ReaSeg/data")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--bbox_shift", type=int, default=20)
    p.add_argument("--enlarge_ratio", type=float, default=1.5)
    p.add_argument("--train_box_mode", type=str, default="mixed", choices=["mixed", "tight", "enlarged", "full"])
    p.add_argument("--eval_box_mode", type=str, default="tight", choices=["tight", "enlarged", "full"])

    # Vision backbone selection
    p.add_argument("--vision_type", type=str, default=None, choices=["medsam", "TC_MVE"])
    p.add_argument("--medsam_checkpoint", type=str, required=True)
    p.add_argument("--model_type", type=str, default="vit_b")
    p.add_argument("--stage0_checkpoint", type=str, default="", help="Optional Stage0/Baseline MedSAM checkpoint for vision_type=medsam.")
    p.add_argument("--stage1_checkpoint", type=str, default="", help="Stage1 Steer-MedSAM checkpoint for vision_type=steer_medsam.")
    p.add_argument("--steer_text_encoder", type=str, default="/data/ReaSeg/pretrained/bert-base-uncased")
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

    # LoRA
    p.add_argument("--use_lora", action="store_true", help="Enable LLM LoRA fine-tuning.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Projectors / stage
    p.add_argument("--visual_grid", type=int, default=16, help="16 => 256 visual tokens; 8 => 64 visual tokens.")
    p.add_argument("--stage", type=str, default="unified", choices=["unified"])
    p.add_argument("--experiment_id", choices=EXPERIMENTS, default="main")
    p.add_argument("--lambda_lm", type=float, default=None)
    p.add_argument("--lambda_mask", type=float, default=None)

    # SEG-token training controls
    p.add_argument("--seg_loss_weight", type=float, default=1.0, help="LM loss weight for the [SEG] target token.")
    p.add_argument("--train_lm_embeddings", dest="train_lm_embeddings", action="store_true", default=None)
    p.add_argument("--no_train_lm_embeddings", dest="train_lm_embeddings", action="store_false")
    p.add_argument("--train_lm_head", dest="train_lm_head", action="store_true", default=None)
    p.add_argument("--no_train_lm_head", dest="train_lm_head", action="store_false")
    p.add_argument("--lr_lm_embedding", type=float, default=1e-5)
    p.add_argument("--lr_lm_head", type=float, default=1e-5)
    p.add_argument(
        "--allow_non_permanent_seg_token",
        action="store_true",
        help="Disable seg_token_info.json requirement. Not recommended; use only for debugging.",
    )

    # Optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size.")
    p.add_argument("--val_batch_size", type=int, default=2, help="Per-GPU validation batch; kept at 2 for the common protocol.")
    p.add_argument("--profile_compute", action=argparse.BooleanOptionalAction, default=True,
                   help="Record partial supported-operator FLOPs for the first real training batch.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_projector", type=float, default=1e-4)
    p.add_argument("--lr_mask_decoder", type=float, default=5e-5)
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", type=str, default="", help="Resume the SAME training run: loads model+optimizer and continues epoch count.")
    p.add_argument("--init_from", type=str, default="", help="Unsupported for formal ablations: start each run from the same Stage1 + base LLM.")
    p.add_argument("--output_dir", type=str, default="./work_dir/stage2_MedSPAR")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--print_trainable_names", action="store_true")

    # DDP
    p.add_argument("--gpu_ids", type=str, default="")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--ddp_find_unused_parameters", action="store_true")
    p.add_argument("--lambda_attn", type=float, default=None)
    p.add_argument("--lambda_box", type=float, default=None)
    p.add_argument("--lambda_prompt", type=float, default=None)
    p.add_argument("--num_reasoning_queries", type=int, default=4)
    p.add_argument("--num_pos_points", type=int, default=4)
    p.add_argument("--num_neg_points", type=int, default=4)
    p.add_argument("--grounder_layers", type=int, default=2)
    p.add_argument("--grounder_dim", type=int, default=256)
    p.add_argument("--unfreeze_steer_modules", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--unfreeze_last_vit_blocks", type=int, default=0, choices=[0, 2, 3, 4])
    p.add_argument("--lr_grounding", type=float, default=1e-4)
    p.add_argument("--lr_steer", type=float, default=1e-5)
    p.add_argument("--lr_vision", type=float, default=2e-6)
    p.add_argument("--system_prompt", default="You are a helpful medical image segmentation assistant.")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--smoke_batches", type=int, default=0, help="Real-data structural verification only; no trainable checkpoint is saved.")
    args = p.parse_args(argv)
    cfg = get_stage_config(args.stage)
    policy = ExperimentConfig(args.experiment_id)
    if args.vision_type is not None and args.vision_type != policy.vision_backbone:
        p.error("vision_type is determined by experiment_id; only medsam_vision uses Original MedSAM")
    args.vision_type = args.vision_backbone = policy.vision_backbone
    for name in ("lambda_lm", "lambda_mask", "lambda_attn", "lambda_box", "lambda_prompt"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(cfg, name))
    if not policy.spatial_supervision:
        args.lambda_attn = 0.0
    elif args.lambda_attn <= 0:
        p.error("Only no_spatial_heatmap_loss may disable spatial supervision")
    for name in ("train_lm_embeddings", "train_lm_head"):
        if getattr(args, name) is None:
            setattr(args, name, True)
    if args.unfreeze_steer_modules is None:
        args.unfreeze_steer_modules = False
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = True
    if args.init_from:
        p.error("Do not initialize from a trained Full/P/L checkpoint. Every experiment starts independently from Stage1 + base LLM.")
    if not args.resume:
        if args.experiment_id == 'medsam_vision':
            if not args.stage0_checkpoint or args.stage1_checkpoint:
                p.error('medsam_vision requires new Stage0 and forbids Stage1 initialization')
        elif not args.stage1_checkpoint or args.stage0_checkpoint:
            p.error('TC_MVE experiments require new Stage1 and forbid Stage0 initialization')
    if not args.use_lora or not args.train_lm_embeddings or not args.train_lm_head:
        p.error("Unified training requires LoRA and trainable input/output embedding matrices")
    if min(args.lambda_lm, args.lambda_mask, args.lambda_box, args.lambda_prompt) <= 0:
        p.error("Unified LM/mask/box/prompt losses must all remain positive")
    if min(args.epochs, args.batch_size, args.val_every, args.max_new_tokens) < 1 or args.smoke_batches < 0 or args.val_batch_size < 0:
        p.error("Invalid epochs, batch size, validation frequency, generation length or smoke limit")
    args.best_metric_name = cfg.best_metric_name
    args.validation_mode = "image_question_free_generation_auto_geometry"
    args.ddp_find_unused_parameters = True
    args.effective_flags = policy.flags()
    args.scheduler = "none_constant_lr"
    return args


MODEL_BUILD_ARG_NAMES = (
    "vision_type", "medsam_checkpoint", "llm_name_or_path", "model_type",
    "stage0_checkpoint", "stage1_checkpoint", "steer_text_encoder",
    "steer_cross_attn_layers", "steer_text_max_length", "vision_dim", "visual_grid",
    "llm_torch_dtype", "local_files_only", "trust_remote_code", "use_lora", "lora_r",
    "lora_alpha", "lora_dropout", "lora_target_modules", "seg_token", "seg_loss_weight",
    "stage", "experiment_id", "num_reasoning_queries", "num_pos_points", "num_neg_points", "grounder_layers", "grounder_dim",
)


def build_model_from_args(args):
    model = build_medspar(**{key: getattr(args, key) for key in MODEL_BUILD_ARG_NAMES},
                                       require_permanent_seg_token=not args.allow_non_permanent_seg_token)
    model.model_config = {key: getattr(args, key) for key in MODEL_CONFIG_FIELDS}
    return model



def configure_training_model(model, args):
    apply_stage_freeze_policy(model, args.stage, args.unfreeze_steer_modules, args.unfreeze_last_vit_blocks)
    model.loss_graph_keys = {key for key, weight in (
        ("loss_lm", args.lambda_lm), ("pred_masks", args.lambda_mask),
        ("spatial_logits", args.lambda_attn), ("pred_box_norm", args.lambda_box),
        ("pred_box_embeddings", args.lambda_prompt)) if weight > 0}
    if args.gradient_checkpointing:
        enable = getattr(model.llm, "gradient_checkpointing_enable", None)
        if enable is None:
            raise RuntimeError("LLM does not support gradient checkpointing; explicitly pass --no-gradient_checkpointing.")
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.llm.config.use_cache = False


def setup_distributed(args):
    if args.gpu_ids:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_ids)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0
        if args.num_gpus > 1:
            raise RuntimeError("--num_gpus > 1 requires torchrun or scripts/stage2_*.sh")

    return distributed, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _add_param_group(groups, params, lr, name, used_param_ids):
    params = [p for p in params if p is not None and p.requires_grad and id(p) not in used_param_ids]
    if len(params) > 0:
        groups.append({"params": params, "lr": lr, "name": name})
        for p in params:
            used_param_ids.add(id(p))


def build_optimizer(model, args):
    raw = model.module if hasattr(model, "module") else model
    prepare_trainable_precision(raw)

    groups = []
    used_param_ids = set()
    if is_grounding_stage(args.stage):
        _add_param_group(groups, raw.visual_compressor.parameters(), args.lr_projector, "visual_compressor", used_param_ids)
        _add_param_group(groups, raw.visual_projector.parameters(), args.lr_projector, "visual_projector", used_param_ids)
        _add_param_group(groups, raw.llm.get_input_embeddings().parameters(), args.lr_lm_embedding, "lm_embedding", used_param_ids)
        _add_param_group(groups, raw.llm.get_output_embeddings().parameters(), args.lr_lm_head, "lm_head", used_param_ids)
        _add_param_group(groups, (p for n, p in raw.llm.named_parameters() if "lora_" in n), args.lr_lora, "llm_lora", used_param_ids)
        for name in GROUNDING_MODULE_NAMES:
            _add_param_group(groups, getattr(raw, name).parameters(), args.lr_grounding, name, used_param_ids)
        _add_param_group(groups, raw.medsam.mask_decoder.parameters(), args.lr_mask_decoder, "mask_decoder", used_param_ids)
        for index, module in enumerate(steer_trainable_modules(raw)):
            _add_param_group(groups, module.parameters(), args.lr_steer, f"steer_{index}", used_param_ids)
        _add_param_group(groups, raw.medsam.image_encoder.parameters(), args.lr_vision, "vision_last_blocks", used_param_ids)
        uncovered = [n for n, parameter in raw.named_parameters() if parameter.requires_grad and id(parameter) not in used_param_ids]
        if uncovered or not groups:
            raise RuntimeError(f"Incomplete optimizer parameter coverage: {uncovered}")
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _forward_batch(model, batch, args, probability=1.0, validation=False):
    raw = unwrap_model(model)
    gt_norm = masks_to_boxes(batch["mask"], normalized=True)
    h, w = raw.medsam.prompt_encoder.input_image_size
    boxes = gt_norm * gt_norm.new_tensor([w, h, w, h])
    # boxes are used solely for box/prompt supervision, never the decoder path.
    return model(images=batch["image"], input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                 labels=batch["labels"], boxes=boxes, steer_texts=batch["steer_text"], stage=args.stage,
                 assistant_mask=batch["assistant_mask"], reasoning_mask=batch["reasoning_mask"],
                 compute_lm_loss=True, decode_masks=True, prompt_source="auto")


def _loss(out, batch, args):
    grounding = is_grounding_stage(args.stage)
    return compute_llm_medsam_loss(
        out["loss_lm"], out["pred_masks"], batch["mask"], args.lambda_lm, args.lambda_mask,
        spatial_logits=out.get("spatial_logits"), pred_box_norm=out.get("pred_box_norm"),
        gt_box_norm=masks_to_boxes(batch["mask"], normalized=True) if grounding else None,
        pred_box_embeddings=out.get("pred_box_embeddings"), gt_box_embeddings=out.get("gt_box_embeddings"),
        lambda_attn=args.lambda_attn, lambda_box=args.lambda_box, lambda_prompt=args.lambda_prompt,
    )


@torch.no_grad()
def validate(model, loader, device, args, distributed, local_rank):
    from tqdm import tqdm
    from medspar.generation import generate_reasoning, predict_from_generated
    from evaluation.eval_stage2_medspar import load_scoring_masks
    raw = unwrap_model(model)
    raw.eval()
    names = ["dice", "iou", "recall", "spatial_prior_dice", "spatial_prior_iou", "box_iou", "box_giou",
             "positive_point_hit_rate", "negative_point_correct_rate"]
    sums = torch.zeros(len(names) + 2, device=device, dtype=torch.float64)
    diagnostic_rows = []
    for batch in tqdm(loader, desc="val free generation", leave=False, disable=local_rank != 0):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, args.precision):
            generated = generate_reasoning(raw, batch["image"], batch["input_ids"], batch["attention_mask"],
                                           batch["question"], max_new_tokens=args.max_new_tokens,
                                           repetition_penalty=1.0, no_repeat_ngram_size=0)
            prediction = predict_from_generated(raw, batch["image"], batch["input_ids"], batch["attention_mask"],
                                                batch["question"], generated, stage=args.stage)
        # GT annotation IO is after inference; invalid SEG rows still count.
        gt = load_scoring_masks(batch["mask_path"], args.image_size, device)
        indices = prediction["valid_indices"]
        if indices.numel():
            out = prediction["model_outputs"]
            scored = native_metrics(out["pred_masks"], [batch["mask_path"][i] for i in indices.tolist()])
            metrics = {name: torch.tensor([row[name] for row in scored], device=device, dtype=torch.float64)
                       for name in ("dice", "iou", "recall")}
            metrics.update(localization_metrics(out, gt[indices]))
            sums[:len(names)] += torch.stack([metrics[name].double().sum() for name in names])
        from experiment_reporting import annotation_counts
        valid_lookup={idx:k for k,idx in enumerate(indices.tolist())}
        for i in range(batch['image'].shape[0]):
            valid=i in valid_lookup
            score=scored[valid_lookup[i]] if valid else dict(dice=0.,iou=0.,recall=0.,**annotation_counts(batch['mask_path'][i]))
            diagnostic_rows.append(dict(dataset=batch['dataset'][i],segmentation_failure=not valid,**score))
        sums[-2] += indices.numel()
        sums[-1] += batch["image"].shape[0]
    if distributed:
        dist.all_reduce(sums)
    if sums[-1] == 0:
        raise ValueError("Validation has no samples")
    from experiment_reporting import gather_validation_rows, diagnostic_summary
    diagnostic_rows = gather_validation_rows(diagnostic_rows, distributed)
    result = {"val_auto_"+name: float(sums[i]/sums[-1]) for i,name in enumerate(names)}
    result.update(val_num_samples=int(sums[-1]), val_success_rate=float(sums[-2]/sums[-1]),
                  val_valid_only_dice=float(sums[0]/sums[-2]) if sums[-2] else None)
    result["_diagnostic_rows"] = diagnostic_rows
    result.update({"val_"+k:v for k,v in diagnostic_summary(diagnostic_rows).items() if k in ("near_zero_rate","low_dice_rate")})
    return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    worker = torch.utils.data.get_worker_info()
    dataset = worker.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if hasattr(dataset, "rng"):
        dataset.rng.seed(seed)


def main():
    args = parse_args()
    args.protocol_version = PROTOCOL_VERSION
    from tqdm import tqdm
    distributed, rank, world_size, local_rank = setup_distributed(args)
    set_seed(args.seed, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    if rank == 0 and not args.resume and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {out_dir}")
    grounding = is_grounding_stage(args.stage)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "config.json", vars(args))
        save_json(out_dir / "train_config.json", vars(args))
        print(f"{args.stage}: {get_stage_config(args.stage).description}; validation={args.validation_mode}")
    model = build_model_from_args(args)
    configure_training_model(model, args)
    model.to(device)
    optimizer = build_optimizer(model, args)
    coverage = optimizer_coverage(model, optimizer)
    scaler = make_grad_scaler(args.precision, device)
    start_epoch, best_metric = 0, -1.0
    history = []
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer)
        if checkpoint.get("extra", {}).get("initialization_only"):
            raise ValueError("Verification-only checkpoints cannot resume formal training")
        saved_stage = checkpoint.get("extra", {}).get("stage")
        if saved_stage != args.stage:
            raise ValueError("Resume requires the same unified experiment")
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
        from checkpoint_io import validate_resume_config
        validate_resume_config(checkpoint['training_config'], vars(args))
        if "scaler" in checkpoint.get("extra", {}):
            scaler.load_state_dict(checkpoint["extra"]["scaler"])
        history_path = Path(args.resume).parent / "history.json"
        if history_path.exists():
            import json
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
            history = [row for row in history if row["epoch"] < start_epoch]
        del checkpoint
    if start_epoch >= args.epochs:
        raise ValueError("No epochs remain: increase --epochs to continue this experiment.")
    print_precision_policy(args, local_rank=rank)
    print_parameter_statistics(model, local_rank=rank)
    if args.print_trainable_names:
        print_trainable_parameter_names(model, max_lines=1000, local_rank=rank)
    if rank == 0:
        details = parameter_statistics(model)
        details["optimizer_coverage"] = coverage
        details["checkpoint_load"] = getattr(model, "checkpoint_load_report", None)
        details.update(config=vars(args), optimizer_groups=[{
            "name": group["name"], "lr": group["lr"], "params": sum(p.numel() for p in group["params"])
        } for group in optimizer.param_groups])
        save_json(out_dir / "training_details.json", details)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=args.ddp_find_unused_parameters)
    common = dict(data_root=args.data_root, datasets=args.datasets, image_size=args.image_size,
                  seg_token=args.seg_token, seed=args.seed, question_only=grounding)
    train_set = LLMReasonSegDataset(**common, split=args.train_split, is_train=True,
                                   train_box_mode="tight" if grounding else args.train_box_mode,
                                   bbox_shift=0 if grounding else args.bbox_shift,
                                   enlarge_ratio=args.enlarge_ratio)
    val_set = LLMReasonSegDataset(**common, split=args.val_split, is_train=False,
                                 load_supervision=False, load_reference_answer=False,
                                 eval_box_mode="tight" if grounding else args.eval_box_mode,
                                 enlarge_ratio=args.enlarge_ratio, skip_empty_mask=False)
    collator = LLMReasonSegCollator(unwrap_model(model).tokenizer, max_length=args.max_length,
                                   system_prompt=args.system_prompt, seg_token=args.seg_token)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    # Unlike DistributedSampler(shuffle=False), this does not duplicate samples.
    val_subset = Subset(val_set, list(range(rank, len(val_set), world_size))) if distributed else val_set
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda", collate_fn=collator, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=train_sampler is None, sampler=train_sampler,
                              drop_last=False, generator=torch.Generator().manual_seed(args.seed + rank), **loader_options)
    val_options = dict(loader_options)
    val_options["batch_size"] = args.val_batch_size or args.batch_size
    val_options["persistent_workers"] = args.num_workers > 0
    val_options["collate_fn"] = LLMGenerationCollator(unwrap_model(model).tokenizer, args.max_length, args.system_prompt)
    val_loader = DataLoader(val_subset, shuffle=False, **val_options)
    if rank == 0:
        print(f"Validation: {world_size} rank(s), batch_per_gpu={val_options['batch_size']}, "
              f"samples={len(val_set)}, disjoint shards without padding")
    loss_names = ["loss_total", "loss_mask", "loss_attn", "loss_box", "loss_prompt", "loss_lm"]
    budget = training_budget(train_loader, args.epochs, args.batch_size, world_size)
    args.training_budget = budget
    if rank == 0:
        save_json(out_dir / 'training_budget.json', budget)
        print('Training budget:', budget)
    reporter = TrainingReporter(out_dir, unwrap_model(model), optimizer, args, rank, world_size, device, history)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        keep_frozen_modules_eval(unwrap_model(model))
        probability = 1.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        reporter.start_epoch(epoch)
        sums = torch.zeros(len(loss_names) + 1, device=device, dtype=torch.float64)
        iterator = tqdm(train_loader, desc=f"{args.stage} {epoch+1}/{args.epochs}", disable=rank != 0)
        for batch_index, batch in enumerate(iterator):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with reporter.profiler.capture(batch["image"].shape[0], image_shape=list(batch["image"].shape),
                                           text_shape=list(batch["input_ids"].shape), precision=args.precision):
                with autocast_context(device, args.precision):
                    out = _forward_batch(model, batch, args, probability)
                loss_dict = _loss(out, batch, args)
                loss = loss_dict["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss_dict}")
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            n = batch["image"].shape[0]
            sums[:-1] += torch.stack([loss_dict[k].double() for k in loss_names]) * n
            sums[-1] += n
            iterator.set_postfix(loss=float(sums[0] / sums[-1]))
            del out, loss_dict, loss
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        if args.smoke_batches:
            if rank == 0:
                save_json(out_dir / "smoke_summary.json", {"status": "PASS", "config": vars(args),
                    "runtime_sanity": unwrap_model(model).runtime_sanity, "optimizer_coverage": coverage,
                    "note": "Real-data optimizer steps only; no formal checkpoint or evaluation claim."})
            cleanup_distributed()
            return
        # Validation needs no optimizer gradients from the last training batch.
        optimizer.zero_grad(set_to_none=True)
        reporter.finish_train(int(sums[-1]))
        if distributed:
            dist.all_reduce(sums)
        log = {"epoch": epoch, **{"train_"+name: float(sums[i] / sums[-1].clamp_min(1))
                                 for i, name in enumerate(loss_names)},
               "auto_prompt_probability": probability, "world_size": world_size,
               "validation_mode": args.validation_mode}
        metrics = {}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            metrics = validate(model, val_loader, device, args, distributed, rank)
            if metrics["val_num_samples"] != len(val_set):
                raise RuntimeError("Validation shard sample count differs from full validation set")
            rows = metrics.pop("_diagnostic_rows")
            if rank == 0:
                from experiment_reporting import save_diagnostic_outputs
                save_diagnostic_outputs(out_dir / "validation" / f"epoch_{epoch+1:04d}", rows)
            log.update(metrics)
        reporter.finish_validation(bool(metrics))
        log["epoch_time_sec"] = time.perf_counter() - started
        log["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.
        log["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.
        metric = metrics.get(args.best_metric_name, -1.0)
        improved = bool(metrics) and metric > best_metric
        if improved:
            best_metric = metric
        if rank == 0:
            history.append(log)
            print(log)
            extra = {"stage": args.stage, "config": vars(args), "metrics": metrics,
                     "best_metric_name": args.best_metric_name, "scaler": scaler.state_dict(),
                     "reporting": {"parameters": reporter.parameters}}
            if improved:
                save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, best_metric, extra)
            save_checkpoint(out_dir / "last.pth", model, optimizer, epoch, best_metric, extra)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(out_dir / f"epoch_{epoch + 1:04d}.pth", model, optimizer, epoch, best_metric, extra)
        if distributed:
            dist.barrier()
        reporter.finish_epoch(log)
        if rank == 0:
            save_json(out_dir / "history.json", {"history": history, "best_metric": best_metric,
                                                 "best_metric_name": args.best_metric_name})
    cleanup_distributed()


if __name__ == "__main__":
    main()
