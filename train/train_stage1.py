#!/usr/bin/env python3
"""Train Steer-MedSAM.

Single GPU:
    python train_steer_medsam.py --gpu_ids 0 --num_gpus 1 \
        --data_root /data/ReaSeg/data --datasets Skin BraTS Poly \
        --medsam_checkpoint /data/ReaSeg/pretrained/medsam_vit_b.pth

Multiple GPUs, recommended:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_steer_medsam.py \
        --gpu_ids 0,1 --num_gpus 2 \
        --data_root /data/ReaSeg/data --datasets Skin BraTS Poly \
        --medsam_checkpoint /data/ReaSeg/pretrained/medsam_vit_b.pth
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from TC_MVE.dataset import TextMedSAMDataset, text_medsam_collate
from TC_MVE.losses import binary_dice_iou_from_logits, compute_total_loss
from TC_MVE.model import build_steer_medsam, build_stage0_medsam
from TC_MVE.train_utils import (
    load_visual_checkpoint,
    move_batch_to_device,
    print_parameter_statistics,
    print_trainable_parameters,
    save_checkpoint,
    save_json,
)
from experiment_protocol import native_scores, native_metrics, training_budget
from experiment_reporting import TrainingReporter
from experiment_protocol import PROTOCOL_VERSION
from torch.utils.data import Subset
from experiment_protocol import seed_worker


def parse_args(stage0=False):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/data/ReaSeg/data")
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset folders, e.g. Skin BraTS Poly")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--medsam_checkpoint", type=str, required=True)
    p.add_argument("--model_type", type=str, default="vit_b")
    p.add_argument("--text_encoder", type=str, default="/data/ReaSeg/pretrained/bert-base-uncased")
    p.add_argument("--text_max_length", type=int, default=64)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--cross_attn_layers", nargs="+", type=int, default=[2, 5, 6, 11], help="0-indexed layers: [2,5,6,11] = layers 3,6,7,12")
    p.add_argument("--cross_attn_heads", type=int, default=12)
    p.add_argument("--vision_dim", type=int, default=768)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--prompt_p_context", type=float, default=0.8)
    p.add_argument("--train_box_mode", type=str, default="mixed", choices=["mixed", "tight", "enlarged", "full"])
    p.add_argument("--bbox_shift", type=int, default=20)
    p.add_argument("--enlarge_ratio", type=float, default=1.5)
    p.add_argument("--lambda_patch", type=float, default=0.2)
    p.add_argument("--no_patch_head", action="store_true")
    p.add_argument("--smoke_batches", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2, help="Per-process batch size. Global batch = batch_size * num_gpus in DDP.")
    p.add_argument("--val_batch_size", type=int, default=2, help="Per-GPU validation batch.")
    p.add_argument("--profile_compute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_connector", type=float, default=1e-4)
    p.add_argument("--lr_gca", type=float, default=1e-4)
    p.add_argument("--lr_mask_decoder", type=float, default=5e-5)
    p.add_argument("--lr_patch_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./work_dir/stage0_medsam" if stage0 else "./work_dir/stage1_TC-MVE")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--val_every", type=int, default=1)

    # GPU / distributed controls.
    p.add_argument("--gpu_ids", type=str, default="", help="Visible GPU ids, e.g. '0' or '0,1,2'. The shell script exports CUDA_VISIBLE_DEVICES with this value.")
    p.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use. For >1, launch with torchrun or use train_steer_medsam.sh.")
    p.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection. Normally not needed because the unsupervised IoU head is frozen.",
    )
    args = p.parse_args()
    args.training_stage = "stage0" if stage0 else "stage1"
    if stage0:
        args.no_patch_head = True
        args.lambda_patch = 0.0
    if min(args.batch_size, args.val_batch_size, args.epochs, args.val_every) < 1 or args.num_workers < 0:
        p.error("Batch sizes/epochs/val_every must be positive and num_workers nonnegative")
    return args


def maybe_set_cuda_visible_devices(args):
    if args.gpu_ids:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_ids)


def setup_distributed(args):
    """Initialize DDP when launched by torchrun."""
    maybe_set_cuda_visible_devices(args)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if args.num_gpus > 1:
            raise RuntimeError(
                "--num_gpus > 1 requires torchrun/DDP. Use: "
                f"CUDA_VISIBLE_DEVICES={args.gpu_ids or '0,1'} torchrun --standalone --nproc_per_node={args.num_gpus} train_steer_medsam.py ..."
            )
    return distributed, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, args):
    groups = []
    if model.text_encoder is not None:
        groups.append({"params": [p for p in model.text_encoder.connector.parameters() if p.requires_grad], "lr": args.lr_connector, "name": "connector"})

    gca_params = []
    for name, p in model.named_parameters():
        if "gated_cross_attn" in name and p.requires_grad:
            gca_params.append(p)
    groups.append({"params": gca_params, "lr": args.lr_gca, "name": "gca"})

    groups.append({"params": [p for p in model.medsam.mask_decoder.parameters() if p.requires_grad], "lr": args.lr_mask_decoder, "name": "mask_decoder"})

    if model.patch_head is not None:
        groups.append({"params": [p for p in model.patch_head.parameters() if p.requires_grad], "lr": args.lr_patch_head, "name": "patch_head"})

    groups = [g for g in groups if len(g["params"]) > 0]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def amp_dtype(args):
    if args.amp == "bf16":
        return torch.bfloat16
    if args.amp == "fp16":
        return torch.float16
    return None


def distributed_mean(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


@torch.no_grad()
def validate(model, loader, device, args, distributed: bool = False, local_rank: int = 0):
    model = model.module if hasattr(model, "module") else model
    model.eval()
    dtype = amp_dtype(args)
    total_loss = torch.tensor(0.0, device=device)
    total_dice = torch.tensor(0.0, device=device)
    total_iou = torch.tensor(0.0, device=device)
    total_recall = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    diagnostic_rows = []

    iterator = tqdm(loader, desc="val", leave=False, disable=(local_rank != 0))
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None and device.type == "cuda")):
            out = model(batch["image"], batch["box"], batch["text"])
            loss_dict = compute_total_loss(
                out["pred_masks"],
                batch["mask"],
                out["patch_logits"],
                lambda_patch=args.lambda_patch if not args.no_patch_head else 0.0,
            )
        scored = native_metrics(out["pred_masks"], batch["mask_path"])
        diagnostic_rows.extend(dict(dataset=batch["dataset"][i], **r) for i,r in enumerate(scored))
        dice, iou, recall = [torch.tensor([row[name] for row in scored], device=device)
                             for name in ("dice", "iou", "recall")]
        n = torch.tensor(float(batch["image"].shape[0]), device=device)
        total_loss += loss_dict["loss"].detach() * n
        total_dice += dice.detach().sum()
        total_iou += iou.detach().sum()
        total_recall += recall.sum()
        total_count += n

    if distributed:
        for t in [total_loss, total_dice, total_iou, total_recall, total_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    from experiment_reporting import gather_validation_rows
    diagnostic_rows = gather_validation_rows(diagnostic_rows, distributed)
    denom = total_count.clamp_min(1.0)
    return {
        "_diagnostic_rows": diagnostic_rows,
        "val_loss": float((total_loss / denom).detach().cpu()),
        "val_dice": float((total_dice / denom).detach().cpu()),
        "val_iou": float((total_iou / denom).detach().cpu()),
        "val_recall": float((total_recall / denom).detach().cpu()),
        "val_num_samples": int(total_count),
    }


def main(stage0=False):
    args = parse_args(stage0)
    args.protocol_version = PROTOCOL_VERSION
    distributed, rank, world_size, local_rank = setup_distributed(args)
    set_seed(args.seed, rank)

    out_dir = Path(args.output_dir)
    if rank == 0 and not args.resume and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {out_dir}")
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "config.json", vars(args))
        save_json(out_dir / "train_config.json", vars(args))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")
    if is_main_process(rank):
        print(f"Using device: {device}")
        print(f"Distributed: {distributed} | rank={rank} | world_size={world_size} | local_rank={local_rank}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    train_ds = TextMedSAMDataset(
        data_root=args.data_root,
        datasets=args.datasets,
        split=args.train_split,
        image_size=args.image_size,
        is_train=True,
        prompt_p_context=args.prompt_p_context,
        train_box_mode=args.train_box_mode,
        bbox_shift=args.bbox_shift,
        enlarge_ratio=args.enlarge_ratio,
        seed=args.seed,
    )
    val_ds = TextMedSAMDataset(
        data_root=args.data_root,
        datasets=args.datasets,
        split=args.val_split,
        image_size=args.image_size,
        is_train=False,
        eval_prompt_type="context",
        eval_box_mode="tight",
        enlarge_ratio=args.enlarge_ratio,
        seed=args.seed + 1,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    val_ds = Subset(val_ds, list(range(rank, len(val_ds), world_size)))
    val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + rank),
        pin_memory=True,
        collate_fn=text_medsam_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=text_medsam_collate,
    )

    builder = build_stage0_medsam if stage0 else build_steer_medsam
    raw_model = builder(
        checkpoint=args.medsam_checkpoint,
        model_type=args.model_type,
        text_encoder_name=args.text_encoder,
        cross_attn_layers=args.cross_attn_layers,
        vision_dim=args.vision_dim,
        cross_attn_heads=args.cross_attn_heads,
        text_max_length=args.text_max_length,
        use_patch_head=(not args.no_patch_head),
        local_files_only=args.local_files_only,
    ).to(device)
    raw_model.freeze_base_and_enable_trainable_parts()

    print_parameter_statistics(raw_model, local_rank=local_rank)
    # print_trainable_parameters(raw_model, max_lines=1000, local_rank=local_rank)

    optimizer = build_optimizer(raw_model, args)
    from medspar.train_utils import optimizer_coverage
    coverage = optimizer_coverage(raw_model, optimizer)
    if rank == 0:
        save_json(out_dir / "training_details.json", {"optimizer_coverage": coverage, "config": vars(args)})
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16" and device.type == "cuda"))

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        ckpt = load_visual_checkpoint(args.resume, raw_model, optimizer=optimizer, map_location=device)
        from checkpoint_io import validate_resume_config
        validate_resume_config(ckpt['extra'], vars(args))
        if "scaler" in ckpt.get("extra", {}): scaler.load_state_dict(ckpt["extra"]["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_metric", -1.0))

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    dtype = amp_dtype(args)
    history = []
    if args.resume:
        history_path = Path(args.resume).parent / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
            history = [row for row in history if row["epoch"] < start_epoch]
    reporter = TrainingReporter(out_dir, raw_model, optimizer, args, rank, world_size, device, history)
    if rank == 0:
        save_json(out_dir / "training_budget.json", training_budget(train_loader, args.epochs, args.batch_size, world_size))
        print(f"Validation: {world_size} rank(s), batch_per_gpu={args.val_batch_size}, disjoint shards")
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        reporter.start_epoch(epoch)
        processed_samples = 0
        running = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", disable=(local_rank != 0))
        for batch_index, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with reporter.profiler.capture(batch["image"].shape[0], image_shape=list(batch["image"].shape), precision=args.amp):
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None and device.type == "cuda")):
                    out = model(batch["image"], batch["box"], batch["text"])
                    loss_dict = compute_total_loss(
                        out["pred_masks"],
                        batch["mask"],
                        out["patch_logits"],
                        lambda_patch=args.lambda_patch if not args.no_patch_head else 0.0,
                    )
                    loss = loss_dict["loss"]

                if args.amp == "fp16" and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_([p for p in raw_model.parameters() if p.requires_grad], args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_([p for p in raw_model.parameters() if p.requires_grad], args.grad_clip)
                    optimizer.step()

            running.append(float(loss.detach().cpu()))
            processed_samples += batch["image"].shape[0]
            if local_rank == 0:
                pbar.set_postfix(loss=np.mean(running[-20:]))
            del out, loss_dict, loss
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                print("PASS: real-data forward/backward/optimizer smoke completed; no checkpoint saved", flush=True)
                cleanup_distributed()
                return

        optimizer.zero_grad(set_to_none=True)
        reporter.finish_train(processed_samples)
        train_loss = float(np.mean(running)) if running else 0.0
        train_loss = distributed_mean(train_loss, device, distributed)
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "time_sec": float(time.time() - t0),
            "world_size": world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
        }

        validation_performed = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        improved = False
        if validation_performed:
            val_log = validate(model, val_loader, device, args, distributed=distributed, local_rank=local_rank)
            rows = val_log.pop("_diagnostic_rows")
            from experiment_reporting import diagnostic_summary, save_diagnostic_outputs
            val_log.update({"val_"+k:v for k,v in diagnostic_summary(rows).items() if k in ("near_zero_rate","low_dice_rate")})
            if rank == 0:
                save_diagnostic_outputs(out_dir / "validation" / f"epoch_{epoch+1:04d}", rows)
            log.update(val_log)
            if is_main_process(rank):
                print(log)
            if val_log["val_dice"] > best_dice:
                best_dice = val_log["val_dice"]
                improved = True
        else:
            if is_main_process(rank):
                print(log)

        reporter.finish_validation(validation_performed)
        if is_main_process(rank):
            history.append(log)
            extra = {**vars(args), "scaler": scaler.state_dict(), "reporting": {"parameters": reporter.parameters}}
            if improved:
                save_checkpoint(out_dir / "best.pth", raw_model, optimizer, epoch, best_metric=best_dice, extra=extra)
                print(f"Saved best checkpoint: dice={best_dice:.4f}")
            save_checkpoint(out_dir / "last.pth", raw_model, optimizer, epoch, best_metric=best_dice, extra=extra)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(out_dir / f"epoch_{epoch+1:04d}.pth", raw_model, optimizer, epoch, best_metric=best_dice, extra=extra)

        if distributed:
            dist.barrier()
        reporter.finish_epoch(log)
        if rank == 0:
            save_json(out_dir / "history.json", {"history": history, "best_dice": best_dice,
                                                 "best_metric": best_dice, "best_metric_name": "val_dice"})

    cleanup_distributed()


if __name__ == "__main__":
    main()
