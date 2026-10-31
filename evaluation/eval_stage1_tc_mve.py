#!/usr/bin/env python3
"""Evaluate Steer-MedSAM with unified summaries and efficiency metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from TC_MVE.dataset import TextMedSAMDataset, text_medsam_collate
from TC_MVE.losses import binary_dice_iou_from_logits
from TC_MVE.model import build_steer_medsam, build_stage0_medsam
from TC_MVE.train_utils import load_visual_checkpoint, move_batch_to_device, save_json
from experiment_protocol import native_metrics, PROTOCOL_VERSION
from evaluation.eval_stage2_medspar import shard_indices, merge_rank_rows
from experiment_reporting import ComputeProfiler, memory_snapshot, parameter_report, training_metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/data/ReaSeg/data")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--medsam_checkpoint", type=str, required=True, help="Original MedSAM checkpoint")
    p.add_argument("--steer_checkpoint", type=str, required=True, help="Trained Steer-MedSAM checkpoint")
    p.add_argument("--model_type", type=str, default="vit_b")
    p.add_argument("--text_encoder", type=str, default="/data/ReaSeg/pretrained/bert-base-uncased")
    p.add_argument("--text_max_length", type=int, default=64)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--cross_attn_layers", nargs="+", type=int, default=[2, 5, 6, 11])
    p.add_argument("--cross_attn_heads", type=int, default=12)
    p.add_argument("--vision_dim", type=int, default=768)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--prompt_type", type=str, default="context", choices=["simple", "context", "location"])
    p.add_argument("--wrong_text", action="store_true")
    p.add_argument("--wrong_target", type=str, default="")
    p.add_argument("--box_mode", type=str, default="tight", choices=["tight", "enlarged", "full"])
    p.add_argument("--enlarge_ratio", type=float, default=1.5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--precision", "--amp", dest="precision", type=str, default="fp32", choices=["fp32", "none", "fp16", "bf16"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--output_dir", type=str, default="./work_dirs/eval_steer_medsam")
    p.add_argument("--measure_gflops", action="store_true", help="Compatibility alias for --profile_compute; reports partial supported-op FLOPs.")
    p.add_argument("--profile_compute", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.measure_gflops:
        args.profile_compute = True
    if args.batch_size < 1 or args.num_workers < 0 or not 0 <= args.threshold <= 1:
        p.error("Invalid batch size, workers or probability threshold")
    return args


def amp_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def get_cpu_rss_mb() -> Optional[float]:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return None


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio_percent": float(trainable / total * 100.0) if total > 0 else 0.0,
    }


def file_size_mb(path: str | os.PathLike) -> Optional[float]:
    try:
        return float(Path(path).stat().st_size / (1024 ** 2))
    except Exception:
        return None


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(vals: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def save_overall_summary_csv(path: Path, summary: Dict[str, Any], per_dataset: Dict[str, Dict[str, Any]]) -> None:
    fieldnames = ["scope", "dataset", "num_samples", "mean_dice", "std_dice", "mean_iou", "std_iou", "mean_recall", "std_recall"]
    rows = [{
        "scope": "global",
        "dataset": "ALL",
        "num_samples": summary.get("num_samples", 0),
        "mean_dice": summary.get("mean_dice", 0.0),
        "std_dice": summary.get("std_dice", 0.0),
        "mean_iou": summary.get("mean_iou", 0.0),
        "std_iou": summary.get("std_iou", 0.0),
        "mean_recall": summary.get("mean_recall"), "std_recall": summary.get("std_recall"),
    }]
    for ds, m in per_dataset.items():
        rows.append({
            "scope": "dataset",
            "dataset": ds,
            "num_samples": m.get("num_samples", 0),
            "mean_dice": m.get("mean_dice", 0.0),
            "std_dice": m.get("std_dice", 0.0),
            "mean_iou": m.get("mean_iou", 0.0),
            "std_iou": m.get("std_iou", 0.0),
            "mean_recall": m.get("mean_recall"), "std_recall": m.get("std_recall"),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def save_efficiency_csv(path: Path, metrics: Dict[str, Any]) -> None:
    flat = flatten_dict(metrics)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def summarize_rows(rows):
    result = dict(num_samples=len(rows), num_segmentation_failures=0,
                  num_successful_predictions=len(rows), success_rate=1.0 if rows else 0.0,
                  failure_rate=0.0)
    for metric in ("dice", "iou", "recall"):
        values = [row[metric] for row in rows]
        result["mean_" + metric] = float(np.mean(values)) if values else None
        result["std_" + metric] = float(np.std(values)) if values else None
        result["valid_only_mean_" + metric] = result["mean_" + metric]
    return result


def main(stage0=False):
    launch_started = time.perf_counter()
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Evaluation datasets must not contain duplicates")
    if args.precision == "none":
        args.precision = "fp32"
    metadata = torch.load(args.steer_checkpoint, map_location='cpu', weights_only=True)
    expected_stage = 'stage0' if stage0 else 'stage1'
    if metadata.get('pipeline_version') != 'medspar_visual_v1' or metadata.get('training_stage') != expected_stage:
        raise ValueError('Expected a new-project matching visual-stage checkpoint')
    saved = metadata['extra']
    for key in ('model_type', 'text_max_length', 'cross_attn_layers', 'cross_attn_heads', 'vision_dim', 'image_size'):
        setattr(args, key, saved[key])
    args.no_patch_head = saved['no_patch_head']
    del metadata
    out_dir = Path(args.output_dir)
    output_error = [None]
    if rank == 0 and out_dir.exists() and any(out_dir.iterdir()):
        output_error[0] = f"Output is not empty: {out_dir}"
    if world_size > 1:
        dist.broadcast_object_list(output_error, src=0)
    if output_error[0] is not None:
        raise FileExistsError(output_error[0])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    saved_config = {**vars(args), "world_size": world_size, "batch_size_per_gpu": args.batch_size,
                    "protocol_version": PROTOCOL_VERSION}
    if rank == 0:
        save_json(out_dir / "config.json", saved_config)
        save_json(out_dir / "eval_config.json", saved_config)  # Compatibility alias.
    builder = build_stage0_medsam if stage0 else build_steer_medsam
    model = builder(
        checkpoint=args.medsam_checkpoint, model_type=args.model_type,
        text_encoder_name=args.text_encoder, cross_attn_layers=args.cross_attn_layers,
        vision_dim=args.vision_dim, cross_attn_heads=args.cross_attn_heads,
        text_max_length=args.text_max_length, use_patch_head=not args.no_patch_head,
        local_files_only=args.local_files_only,
    ).to(device)
    checkpoint = load_visual_checkpoint(args.steer_checkpoint, model, optimizer=None, map_location="cpu")
    training = training_metadata(args.steer_checkpoint, checkpoint=checkpoint) if rank == 0 else None
    del checkpoint
    model.eval()
    dtype = amp_dtype(args.precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profiler = ComputeProfiler(args.profile_compute and rank == 0, device, "inference")
    profile_pending = bool(args.profile_compute and rank == 0)
    rows, batch_times = [], []
    expected_samples = timed_samples = profiled_samples = 0
    scoring_time = 0.0
    wall_start = time.perf_counter()
    with torch.no_grad():
        for dataset_name in args.datasets:
            dataset = TextMedSAMDataset(
                data_root=args.data_root, datasets=[dataset_name], split=args.split,
                image_size=args.image_size, is_train=False, eval_prompt_type=args.prompt_type,
                wrong_text=args.wrong_text, wrong_target=args.wrong_target or None,
                eval_box_mode=args.box_mode, enlarge_ratio=args.enlarge_ratio, skip_empty_mask=False,
            )
            expected_samples += len(dataset)
            indices = shard_indices(len(dataset), rank, world_size)
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                collate_fn=text_medsam_collate, drop_last=False)
            offset = 0
            print(f"[rank {rank}/{world_size}] {dataset_name}: {len(indices)}/{len(dataset)} samples", flush=True)
            for batch in tqdm(loader, desc=f"rank{rank} {dataset_name}", position=rank):
                batch = move_batch_to_device(batch, device)
                bsz = int(batch["image"].shape[0])
                positions = indices[offset:offset + bsz]
                sync_cuda(device)
                started = time.perf_counter()
                profiled = profile_pending
                with profiler.capture(bsz, image_size=args.image_size, precision=args.precision):
                    with torch.autocast(device_type="cuda", dtype=dtype,
                                        enabled=dtype is not None and device.type == "cuda"):
                        out = model(batch["image"], batch["box"], batch["text"])
                    sync_cuda(device)
                elapsed = time.perf_counter() - started
                profile_pending = False
                if profiled:
                    profiled_samples += bsz
                else:
                    batch_times.append(elapsed)
                    timed_samples += bsz
                scoring_started = time.perf_counter()
                scores = native_metrics(out["pred_masks"], batch["mask_path"], args.threshold)
                for i, score in enumerate(scores):
                    row = dict(dataset=batch["dataset"][i], evaluation_index=positions[i],
                               record_index=int(batch["record_index"][i]),
                               image_path=batch["image_path"][i], mask_path=batch["mask_path"][i],
                               target=batch["target"][i], text=batch["text"][i], question=batch["text"][i],
                               prompt_source="gt_derived_box", evaluation_kind="stage0_box_prompted" if stage0 else "stage1_box_prompted",
                               segmentation_failure=False, failure_reason="", generated_text=None, seg_count=None,
                               empty_prediction=score["pred_foreground_pixels"] == 0,
                               prediction_path=None, **score)
                    if args.save_preds:
                        pred_dir = out_dir / "preds" / dataset_name
                        pred_dir.mkdir(parents=True, exist_ok=True)
                        prediction_path = pred_dir / f"{positions[i]:07d}_{Path(batch['image_path'][i]).stem}.png"
                        pred = out["pred_masks"][i, 0].sigmoid().gt(args.threshold).byte().cpu().numpy() * 255
                        Image.fromarray(pred).save(prediction_path)
                        row["prediction_path"] = str(prediction_path)
                    rows.append(row)
                offset += bsz
                sync_cuda(device)
                scoring_time += time.perf_counter() - scoring_started
    sync_cuda(device)
    wall_time = time.perf_counter() - wall_start
    packet = dict(rows=rows, expected_samples=expected_samples, wall_time_sec=wall_time,
                  batch_times=batch_times, timed_samples=timed_samples, profiled_samples=profiled_samples,
                  scoring_time=scoring_time, memory={"rank": rank, **memory_snapshot(device), "cpu_rss_mb": get_cpu_rss_mb()})
    if world_size > 1:
        packets = [None] * world_size if rank == 0 else None
        dist.gather_object(packet, object_gather_list=packets, dst=0)
    else:
        packets = [packet]
    if rank == 0:
        rows = merge_rank_rows(packets, args.datasets)
        batch_times = [t for packet in packets for t in packet["batch_times"]]
        timed_samples = sum(p["timed_samples"] for p in packets)
        profiled_samples = sum(p["profiled_samples"] for p in packets)
        scoring_time = sum(p["scoring_time"] for p in packets)
        wall_time = max(p["wall_time_sec"] for p in packets)
        memories = [p["memory"] for p in packets]
        def maximum_memory(key):
            values = [m[key] for m in memories if m.get(key) is not None]
            return max(values) if values else None
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["dataset"]].append(row)
        per_dataset = {name: summarize_rows(values) for name, values in grouped.items()}
        summary = summarize_rows(rows)
        from experiment_reporting import annotation_counts, diagnostic_summary
        for row in rows:
            if row.get("gt_foreground_pixels") is None:
                row.update(annotation_counts(row["mask_path"]))
        summary.update({k:v for k,v in diagnostic_summary(rows).items() if k not in summary})
        summary.update(stage="stage0" if stage0 else "stage1", box_mode=args.box_mode, prompt_type=args.prompt_type,
                       wrong_text=args.wrong_text, precision=args.precision, world_size=world_size,
                       evaluation_kind="stage0_box_prompted" if stage0 else "stage1_box_prompted", inference_inputs=["image", "question", "gt_derived_box"],
                       checkpoint=args.steer_checkpoint, reference_answer_used=False,
                       empty_mask_policy="Valid prediction: both empty Dice/IoU 1; otherwise existing epsilon formula. Empty GT Recall 1.",
                       per_dataset=per_dataset)
        save_json(out_dir / "summary.json", summary)
        save_json(out_dir / "per_dataset_summary.json", per_dataset)
        save_overall_summary_csv(out_dir / "overall_summary.csv", summary, per_dataset)
        with (out_dir / "per_sample_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset", "dice", "iou", "recall"])
            writer.writeheader()
            writer.writerows(rows)
        prediction_fields = ("dataset", "record_index", "evaluation_index", "image_path", "mask_path", "question", "generated_text",
                             "seg_count", "segmentation_failure", "failure_reason", "prediction_path")
        save_json(out_dir / "predictions.json", [{key: row[key] for key in prediction_fields} for row in rows])
        parameters = parameter_report(model)
        training_parameters = training["parameters"]
        duration = sum(batch_times)
        efficiency = dict(
            stage="stage0" if stage0 else "stage1", task="box_prompted_segmentation", device=str(device), world_size=world_size,
            batch_size_per_gpu=args.batch_size, precision=args.precision, num_samples=len(rows),
            model=dict(total_params=parameters["total_params"],
                       training_trainable_params=training_parameters.get("trainable_params") if training_parameters else None,
                       training_parameter_source=training["source"]),
            checkpoint_size_mb=file_size_mb(args.steer_checkpoint),
            memory=dict(per_rank=memories, max_peak_allocated_gb=maximum_memory("peak_allocated_gb"),
                        max_peak_reserved_gb=maximum_memory("peak_reserved_gb")),
            speed=dict(wall_time_sec=wall_time, samples_per_sec=len(rows) / wall_time if wall_time else 0.,
                       inference_time_sec_sum_rank_batches=duration, inference_time_sec_max_rank_sum=max(sum(p["batch_times"]) for p in packets),
                       batch_latency_mean_ms=float(np.mean(batch_times) * 1000) if batch_times else None,
                       batch_latency_p95_ms=percentile([x * 1000 for x in batch_times], 95) if batch_times else None,
                       amortized_sample_latency_ms=1000 * duration / timed_samples if timed_samples else None,
                       latency_definition="Forward only, excluding transfer/scoring and profiling batch; amortized latency is not independent single-sample latency.",
                       timed_samples=timed_samples, profiled_samples=profiled_samples,
                       wall_time_definition="Maximum rank evaluation loop including input/GT-box loading, transfer, scoring, output and profiling; excludes model loading and final report saves.",
                       scoring_and_saving_time_sec_sum_ranks=scoring_time),
            training=training["efficiency"], elapsed_since_launch_sec=time.perf_counter() - launch_started)
        save_json(out_dir / "efficiency.json", efficiency)
        save_json(out_dir / "efficiency_metrics.json", efficiency)  # Compatibility alias.
        save_efficiency_csv(out_dir / "efficiency_metrics.csv", efficiency)
        save_json(out_dir / "compute_profile.json", profiler.report())
        from experiment_reporting import save_diagnostic_outputs
        save_diagnostic_outputs(out_dir, rows)
        print(json.dumps({key: value for key, value in summary.items() if key != "per_dataset"}, indent=2))
        print(f"Saved {len(rows)} results to {out_dir}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
