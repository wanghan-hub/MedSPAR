#!/usr/bin/env python3
"""Official image/question automatic-prompt segmentation and GT-box diagnostic.

The automatic path generates reasoning, repeats the LLM forward on the generated
tokens, and supplies only predicted geometry to SAM. Ground-truth masks are
opened by the scorer only after prediction. Missing/invalid SEG is a failure
with Dice/IoU zero and remains in the reported denominator.
"""

from __future__ import annotations

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
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from evaluation.generation_helpers import build_parser, generate_batch
from medspar.collator import LLMGenerationCollator
from medspar.dataset import LLMReasonSegDataset, _read_mask, _resize_mask_np
from medspar.generation import predict_from_generated
from medspar.train_utils import restore_evaluation_config
from medspar.losses import binary_dice_iou_from_logits, localization_metrics, masks_to_boxes
from medspar.medsam_bridge import encode_geometric_prompts, decode_masks_from_embeddings
from medspar.model import build_medspar
from medspar.train_utils import autocast_context, load_checkpoint, move_batch_to_device, save_json
from TC_MVE.box_ops import choose_eval_box
from experiment_protocol import native_metrics, PROTOCOL_VERSION
from experiment_reporting import ComputeProfiler, memory_snapshot, parameter_report, training_metadata


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
    fieldnames = ["scope", "dataset", "num_samples", "mean_dice", "std_dice", "mean_iou", "std_iou",
                  "mean_recall", "std_recall"]
    rows = [{
        "scope": "global",
        "dataset": "ALL",
        "num_samples": summary.get("num_samples", 0),
        "mean_dice": summary.get("mean_dice", 0.0),
        "std_dice": summary.get("std_dice", 0.0),
        "mean_iou": summary.get("mean_iou", 0.0),
        "std_iou": summary.get("std_iou", 0.0),
        "mean_recall": summary.get("mean_recall"),
        "std_recall": summary.get("std_recall"),
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
            "mean_recall": m.get("mean_recall"),
            "std_recall": m.get("std_recall"),
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


def verify_seg_token_ready(model, seg_token: str, llm_name_or_path: str) -> Dict[str, Any]:
    """Verify that [SEG] is a permanent single token and model vocab is resized."""
    tokenizer = model.tokenizer
    ids = tokenizer.encode(seg_token, add_special_tokens=False)
    token_id = tokenizer.convert_tokens_to_ids(seg_token)

    if len(ids) != 1:
        raise RuntimeError(
            f"{seg_token} must be a single token, but tokenizer.encode returned {ids}. "
            "Please use the permanent SEG-token LLM directory."
        )

    if int(ids[0]) != int(token_id):
        raise RuntimeError(
            f"{seg_token} tokenizer mismatch: encode={ids}, convert_tokens_to_ids={token_id}."
        )

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and int(token_id) == int(unk_id):
        raise RuntimeError(
            f"{seg_token} maps to unk_token_id={unk_id}. "
            "Please add [SEG] as an additional_special_token."
        )

    if hasattr(model, "seg_token_id") and int(model.seg_token_id) != int(token_id):
        raise RuntimeError(
            f"model.seg_token_id={int(model.seg_token_id)} does not match tokenizer id={int(token_id)}."
        )

    input_emb = model.llm.get_input_embeddings()
    if input_emb is None or int(input_emb.weight.shape[0]) != len(tokenizer):
        rows = None if input_emb is None else int(input_emb.weight.shape[0])
        raise RuntimeError(
            f"Tokenizer length ({len(tokenizer)}) does not match LLM input embedding rows ({rows}). "
            "Please use the permanent SEG-token model directory with resized embeddings."
        )

    output_emb = model.llm.get_output_embeddings()
    output_rows = None
    if output_emb is not None and hasattr(output_emb, "weight"):
        output_rows = int(output_emb.weight.shape[0])
        if output_rows != len(tokenizer):
            raise RuntimeError(
                f"Tokenizer length ({len(tokenizer)}) does not match LM head rows ({output_rows})."
            )

    info_path = Path(llm_name_or_path) / "seg_token_info.json"
    return {
        "seg_token": seg_token,
        "seg_token_id": int(token_id),
        "seg_encode": [int(x) for x in ids],
        "tokenizer_length": int(len(tokenizer)),
        "input_embedding_rows": int(input_emb.weight.shape[0]),
        "lm_head_rows": output_rows,
        "seg_token_info_json_exists": bool(info_path.exists()),
        "llm_name_or_path": str(llm_name_or_path),
    }




def parse_args():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir="./output/eval_auto_prompt_seg")
    parser.add_argument("--prompt_source", choices=["auto"], default="auto",
                        help="Formal unified evaluation accepts predicted prompts only.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualize_samples", type=int, default=8, help="Visualizations per dataset.")
    parser.add_argument("--save_preds", action="store_true")
    import argparse
    parser.add_argument("--profile_compute", action=argparse.BooleanOptionalAction, default=True,
                        help="Profile supported FLOPs on rank 0's first inference batch; excludes scoring.")
    args = parser.parse_args()
    if args.rescore_csv:
        parser.error("--rescore_csv belongs to the text generation evaluator.")
    return args


def load_scoring_masks(mask_paths: List[str], image_size: int, device) -> torch.Tensor:
    """Scorer-only IO, deliberately outside image/question inference helpers."""
    masks = []
    for path in mask_paths:
        if not path:
            raise ValueError("Segmentation scoring requires a mask path for every sample.")
        masks.append(torch.from_numpy(_resize_mask_np(_read_mask(Path(path)), image_size).copy())[None].float())
    return torch.stack(masks).to(device)


@torch.no_grad()
def predict_gt_box_diagnostic(model, images, questions, gt_masks, args):
    """Pure Steer-MedSAM + GT box upper-bound diagnostic, no reference answer."""
    boxes = []
    for mask in gt_masks:
        arr = mask[0].detach().cpu().numpy()
        if arr.any():
            boxes.append(choose_eval_box(arr, args.image_size, args.image_size,
                                        mode=args.eval_box_mode, enlarge_ratio=args.enlarge_ratio))
        else:
            boxes.append((0.0, 0.0, 0.0, 0.0))
    boxes = images.new_tensor(boxes, dtype=torch.float32)
    prompt_h, prompt_w = model.medsam.prompt_encoder.input_image_size
    boxes = boxes * boxes.new_tensor([prompt_w / images.shape[-1], prompt_h / images.shape[-2],
                                      prompt_w / images.shape[-1], prompt_h / images.shape[-2]])
    embeddings, _ = model.encode_image(images, steer_texts=list(questions))
    sparse, dense = encode_geometric_prompts(model.medsam.prompt_encoder, boxes=boxes)
    low_res, iou = decode_masks_from_embeddings(
        model.medsam.prompt_encoder, model.medsam.mask_decoder, embeddings, sparse, dense,
    )
    return {"pred_masks": F.interpolate(low_res, size=images.shape[-2:], mode="bilinear", align_corners=False),
            "low_res_masks": low_res, "iou_predictions": iou, "diagnostic_gt_box": boxes}


def _draw_panel(image: np.ndarray, title: str, size=320) -> Image.Image:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    tile = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize((size, size))
    panel = Image.new("RGB", (size, size + 26), "white")
    panel.paste(tile, (0, 26))
    ImageDraw.Draw(panel).text((8, 7), title, fill="black")
    return panel


def save_visualization(path: Path, image, gt_mask, prediction, output, index, prompt_hw, threshold,
                       points_enabled=True, coarse_mask_enabled=True):
    """Six annotated panels: image, GT, prior, geometry, coarse mask, final mask."""
    rgb = image.detach().float().cpu().permute(1, 2, 0).numpy()
    gt = gt_mask.detach().float().cpu().squeeze().numpy()
    pred = prediction.detach().float().sigmoid().cpu().squeeze().numpy() if prediction is not None else np.zeros_like(gt)
    height, width = gt.shape
    prior = coarse = np.zeros_like(gt)
    if output is not None and "spatial_logits" in output:
        prior = F.interpolate(output["spatial_logits"][index:index + 1].float(), (height, width),
                              mode="bilinear", align_corners=False).sigmoid()[0, 0].cpu().numpy()
        coarse = F.interpolate(output["pred_coarse_mask_logits"][index:index + 1].float(), (height, width),
                               mode="bilinear", align_corners=False).sigmoid()[0, 0].cpu().numpy()
    heat = np.stack([prior, np.zeros_like(prior), 1 - prior], axis=-1)
    geometry = _draw_panel(rgb, ("Points + boxes" if points_enabled else "Boxes (points disabled)")
                           + ": GT green / pred red")
    draw = ImageDraw.Draw(geometry)
    gt_indices = np.argwhere(gt > 0)
    if len(gt_indices):
        y1, x1 = gt_indices.min(0)
        y2, x2 = gt_indices.max(0) + 1
        draw.rectangle((x1 * 320 / width, y1 * 320 / height + 26,
                        x2 * 320 / width, y2 * 320 / height + 26), outline="lime", width=2)
    if output is not None and "pred_box_xyxy" in output:
        ph, pw = prompt_hw
        box = output["pred_box_xyxy"][index].detach().float().cpu().tolist()
        draw.rectangle((box[0] * 320 / pw, box[1] * 320 / ph + 26,
                        box[2] * 320 / pw, box[3] * 320 / ph + 26), outline="red", width=2)
        if points_enabled:
            coords = output["point_coords"][index].detach().float().cpu().tolist()
            labels = output["point_labels"][index].detach().cpu().tolist()
            for (x, y), label in zip(coords, labels):
                x, y = x * 320 / pw, y * 320 / ph + 26
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="lime" if label == 1 else "cyan")
    panels = [_draw_panel(rgb, "Input image"), _draw_panel(gt, "Ground truth (scoring only)"),
              _draw_panel(0.55 * rgb + 0.45 * heat, "Spatial Prior Map"), geometry,
              _draw_panel(coarse, "Predicted coarse mask" if coarse_mask_enabled else "Coarse mask (NOT used)"),
              _draw_panel((pred > threshold).astype(float), "Final mask" if prediction is not None else "FAILED: no valid prediction")]
    canvas = Image.new("RGB", (960, 692), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 3) * 320, (i // 3) * 346))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


METRIC_KEYS = ("dice", "iou", "spatial_prior_dice", "spatial_prior_iou", "box_iou", "box_giou",
               "positive_point_hit_rate", "negative_point_correct_rate")


def summarize_rows(rows):
    summary = {"num_samples": len(rows),
               "num_segmentation_failures": sum(bool(r["segmentation_failure"]) for r in rows),
               "failure_rate": float(np.mean([r["segmentation_failure"] for r in rows])) if rows else 0.0}
    summary["success_rate"] = 1.0 - summary["failure_rate"] if rows else 0.0
    for key in (*METRIC_KEYS, "recall"):
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        summary["mean_" + key] = float(np.mean(values)) if values else None
        if key in {"dice", "iou", "recall"}:
            summary["std_" + key] = float(np.std(values)) if values else None
    valid = [r for r in rows if not r["segmentation_failure"]]
    summary["num_successful_predictions"] = len(valid)
    summary["valid_only_mean_dice"] = float(np.mean([r["dice"] for r in valid])) if valid else None
    summary["valid_only_mean_iou"] = float(np.mean([r["iou"] for r in valid])) if valid else None
    summary["valid_only_mean_recall"] = float(np.mean([r["recall"] for r in valid])) if valid else None
    return summary


def shard_indices(length, rank, world_size):
    """Every evaluation item belongs to exactly one rank, without padding."""
    if length < 0 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("Invalid evaluation shard")
    return list(range(rank, length, world_size))


def merge_rank_rows(packets, datasets):
    expected = packets[0]["expected_samples"]
    if any(packet["expected_samples"] != expected for packet in packets):
        raise RuntimeError("Evaluation ranks disagree on dataset length")
    rows = [row for packet in packets for row in packet["rows"]]
    keys = [(row["dataset"], row["evaluation_index"]) for row in rows]
    if len(rows) != expected or len(set(keys)) != len(rows):
        raise RuntimeError(f"Evaluation coverage error: rows={len(rows)}, unique={len(set(keys))}, expected={expected}")
    order = {name: i for i, name in enumerate(datasets)}
    return sorted(rows, key=lambda row: (order[row["dataset"]], row["evaluation_index"]))


def main():
    launch_started = time.perf_counter()
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA evaluation requested but unavailable")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    args = restore_evaluation_config(args)
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Evaluation datasets must not contain duplicate names")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    from tqdm import tqdm
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
    if rank == 0:
        saved_config = {**vars(args), "world_size": world_size, "batch_size_per_gpu": args.batch_size}
        save_json(out_dir / "config.json", saved_config)
        save_json(out_dir / "eval_config.json", saved_config)  # Compatibility alias.
    print(f"[rank {rank}/{world_size}] Loading model on {device}; per-GPU batch={args.batch_size}", flush=True)
    model = build_medspar(
        vision_type=args.vision_type, medsam_checkpoint=args.medsam_checkpoint,
        llm_name_or_path=args.llm_name_or_path, model_type=args.model_type,
        stage0_checkpoint=args.stage0_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        steer_text_encoder=args.steer_text_encoder, steer_cross_attn_layers=args.steer_cross_attn_layers,
        steer_text_max_length=args.steer_text_max_length, vision_dim=args.vision_dim,
        visual_grid=args.visual_grid, llm_torch_dtype=args.llm_torch_dtype,
        local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code,
        use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_target_modules=args.lora_target_modules,
        seg_token=args.seg_token, seg_loss_weight=args.seg_loss_weight, stage=args.stage, experiment_id=args.experiment_id,
        num_reasoning_queries=args.num_reasoning_queries, num_pos_points=args.num_pos_points,
        num_neg_points=args.num_neg_points, grounder_layers=args.grounder_layers,
        grounder_dim=args.grounder_dim,
    )
    model.model_config = dict(args.restored_model_config)
    checkpoint = load_checkpoint(args.checkpoint, model, map_location="cpu")
    training = training_metadata(args.checkpoint, checkpoint=checkpoint) if rank == 0 else None
    del checkpoint
    model.model_config = dict(args.restored_model_config)
    model.to(device).eval()
    policy = model.ablation_config
    seg_check = verify_seg_token_ready(model, args.seg_token, args.llm_name_or_path)
    if rank == 0:
        save_json(out_dir / "seg_token_check.json", seg_check)
    prompt_hw = tuple(model.medsam.prompt_encoder.input_image_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profiler = ComputeProfiler(bool(args.profile_compute and rank == 0), device, "inference")
    profile_pending = bool(args.profile_compute and rank == 0)
    rows, batch_times = [], []
    total_generated_tokens = expected_samples = timed_samples = profiled_samples = 0
    scoring_time = 0.0
    if world_size > 1:
        dist.barrier()
    wall_start = time.perf_counter()
    with torch.no_grad():
        for dataset_name in args.datasets:
            dataset = LLMReasonSegDataset(
                args.data_root, [dataset_name], split=args.split, image_size=args.image_size,
                is_train=False, seg_token=args.seg_token,
                load_supervision=False, load_reference_answer=False, question_only=True,
            )
            length = min(args.max_samples, len(dataset)) if args.max_samples > 0 else len(dataset)
            expected_samples += length
            indices = shard_indices(length, rank, world_size)
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                collate_fn=LLMGenerationCollator(
                                    model.tokenizer, args.max_length, args.system_prompt))
            local_offset = 0
            print(f"[rank {rank}] {dataset_name}: {len(indices)}/{length} samples", flush=True)
            for batch in tqdm(loader, desc=f"rank{rank} {dataset_name}", position=rank):
                batch = move_batch_to_device(batch, device)
                bsz = batch["image"].shape[0]
                positions = indices[local_offset:local_offset + bsz]
                local_offset += bsz
                sync_cuda(device)
                started = time.perf_counter()
                profiled = profile_pending
                with profiler.capture(bsz, image_size=args.image_size,
                                      max_new_tokens=args.max_new_tokens, experiment_id=args.experiment_id):
                    generated = generate_batch(model, batch, device, args)
                    with autocast_context(device, args.precision):
                        result = predict_from_generated(
                            model, batch["image"], batch["input_ids"], batch["attention_mask"],
                            batch["question"], generated, stage=args.stage,
                        )
                    sync_cuda(device)
                elapsed = time.perf_counter() - started
                profile_pending = False
                if profiled:
                    profiled_samples += bsz
                    profiler.result["input"]["generated_token_counts"] = generated["generated_token_counts"]
                    profiler.result["input"]["prompt_token_counts"] = batch["attention_mask"].long().sum(-1).tolist()
                else:
                    batch_times.append(elapsed)
                    timed_samples += bsz
                output = result["model_outputs"]
                valid_indices = result["valid_indices"].tolist()
                total_generated_tokens += sum(generated["generated_token_counts"])
                # Annotation IO and metrics are excluded from model inference timing.
                scoring_started = time.perf_counter()
                gt_masks = load_scoring_masks(batch["mask_path"], args.image_size, device)
                local_index = {global_i: local_i for local_i, global_i in enumerate(valid_indices)}
                localization, scores = None, []
                if output is not None:
                    scores = native_metrics(output["pred_masks"], [batch["mask_path"][i] for i in valid_indices], args.threshold)
                    localization = localization_metrics(output, gt_masks[valid_indices], input_image_size=prompt_hw)
                gt_boxes = masks_to_boxes(gt_masks, normalized=True)
                for i in range(bsz):
                    valid = i in local_index
                    k = local_index.get(i, 0)
                    score = scores[k] if valid else {}
                    row = {
                        "dataset": batch["dataset"][i], "record_index": batch["record_index"][i],
                        "evaluation_index": positions[i],
                        "image_path": batch["image_path"][i], "mask_path": batch["mask_path"][i],
                        "question": batch["question"][i], "prompt_source": args.prompt_source,
                        "experiment_id": args.experiment_id, "vision_backbone": args.vision_type,
                        "points_enabled": policy.point_prompts,
                        "coarse_mask_enabled": policy.coarse_mask_prompt,
                        "semantic_prompt_enabled": policy.semantic_prompt,
                        "evaluation_kind": "official_automatic",
                        "generated_text": generated["texts"][i],
                        "generated_tokens": int(generated["generated_token_counts"][i]),
                        "seg_count": int(generated["seg_counts"][i]),
                        "segmentation_failure": not valid, "failure_reason": result["failure_reasons"][i],
                        "dice": float(score["dice"]) if valid else 0.0,
                        "iou": float(score["iou"]) if valid else 0.0,
                        "recall": float(score["recall"]) if valid else 0.0,
                        "tp": score.get("tp"), "fp": score.get("fp"), "fn": score.get("fn"),
                        "gt_foreground_pixels": score.get("gt_foreground_pixels"),
                        "pred_foreground_pixels": score.get("pred_foreground_pixels"),
                        "native_height": score.get("native_height"), "native_width": score.get("native_width"),
                        "empty_prediction": score["pred_foreground_pixels"] == 0 if valid else None,
                        "gt_box": json.dumps(gt_boxes[i].cpu().tolist()),
                        "pred_box": None, "box_coordinate_system": "normalized_xyxy",
                        "num_positive_points": 0, "num_negative_points": 0,
                        "prediction_path": None,
                    }
                    for metric in METRIC_KEYS[2:]:
                        row[metric] = float(localization[metric][k]) if localization is not None and valid else 0.0
                    if localization is not None and valid:
                        row["pred_box"] = json.dumps(output["pred_box_norm"][k].float().cpu().tolist())
                        row["num_positive_points"] = int(output["point_labels"][k].eq(1).sum()) if policy.point_prompts else 0
                        row["num_negative_points"] = int(output["point_labels"][k].eq(0).sum()) if policy.point_prompts else 0
                    row["positive_hit_rate"] = row["positive_point_hit_rate"]
                    row["negative_correct_rate"] = row["negative_point_correct_rate"]
                    stem = f"{positions[i]:07d}_{Path(batch['image_path'][i]).stem}"
                    # Limit across the entire dataset, not per rank.
                    if positions[i] < args.visualize_samples:
                        save_visualization(out_dir / "visualizations" / dataset_name / (stem + ".png"),
                                           batch["image"][i], gt_masks[i], output["pred_masks"][k] if valid else None,
                                           output if valid else None, k, prompt_hw, args.threshold,
                                           points_enabled=row["points_enabled"], coarse_mask_enabled=row["coarse_mask_enabled"])
                    if args.save_preds and valid:
                        pred_dir = out_dir / "preds" / dataset_name
                        pred_dir.mkdir(parents=True, exist_ok=True)
                        pred = output["pred_masks"][k, 0].sigmoid().gt(args.threshold).byte().cpu().numpy() * 255
                        prediction_path = pred_dir / (stem + ".png")
                        Image.fromarray(pred).save(prediction_path)
                        row["prediction_path"] = str(prediction_path)
                    from experiment_reporting import annotation_counts
                    if row.get("gt_foreground_pixels") is None:
                        row.update(annotation_counts(row["mask_path"]))
                    rows.append(row)
                sync_cuda(device)
                scoring_time += time.perf_counter() - scoring_started
    sync_cuda(device)
    packet = {"rank": rank, "rows": rows, "expected_samples": expected_samples,
              "wall_time_sec": time.perf_counter() - wall_start,
              "batch_times": batch_times, "timed_samples": timed_samples,
              "profiled_samples": profiled_samples, "generated_tokens": total_generated_tokens,
              "scoring_and_saving_time_sec": scoring_time,
              "memory": {"rank": rank, **memory_snapshot(device), "cpu_rss_mb": get_cpu_rss_mb()}}
    if world_size > 1:
        packets = [None] * world_size if rank == 0 else None
        dist.gather_object(packet, object_gather_list=packets, dst=0)
    else:
        packets = [packet]
    if rank == 0:
        rows = merge_rank_rows(packets, args.datasets)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["dataset"]].append(row)
        per_dataset = {name: summarize_rows(grouped[name]) for name in args.datasets}
        summary = summarize_rows(rows)
        from experiment_reporting import annotation_counts, diagnostic_summary
        for row in rows:
            if row.get("gt_foreground_pixels") is None:
                row.update(annotation_counts(row["mask_path"]))
        summary.update({k:v for k,v in diagnostic_summary(rows).items() if k not in summary})
        summary.update(stage=args.stage, prompt_source=args.prompt_source,
                       experiment_id=args.experiment_id, vision_backbone=args.vision_type, checkpoint=args.checkpoint,
                       evaluation_kind="official_automatic", inference_inputs=["image", "question"],
                       reference_answer_used=False, world_size=world_size,
                       invalid_generation_policy="failure; Dice/IoU/Recall zero; included in all-sample denominator",
                       empty_mask_policy="Valid prediction: both empty Dice/IoU 1; otherwise existing epsilon formula. Empty GT Recall 1. Invalid generation always scores 0.",
                       localization_failure_policy="zero; diagnostic-only localization fields are unavailable",
                       per_dataset=per_dataset)
        save_json(out_dir / "summary.json", summary)
        save_json(out_dir / "per_dataset_summary.json", per_dataset)
        save_overall_summary_csv(out_dir / "overall_summary.csv", summary, per_dataset)
        with open(out_dir / "per_sample_metrics.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset", "dice", "iou", "recall"])
            writer.writeheader()
            writer.writerows(rows)
        prediction_fields = ("dataset", "record_index", "evaluation_index", "image_path", "mask_path", "question",
                             "generated_text", "generated_tokens", "seg_count", "segmentation_failure", "failure_reason",
                             "pred_box", "prediction_path")
        save_json(out_dir / "predictions.json", [{key: row[key] for key in prediction_fields} for row in rows])
        batch_times = [value for item in packets for value in item["batch_times"]]
        duration = sum(batch_times)
        timed_samples = sum(item["timed_samples"] for item in packets)
        wall_time = max(item["wall_time_sec"] for item in packets)
        memories = [item["memory"] for item in packets]
        def maximum_memory(key):
            values = [item[key] for item in memories if item.get(key) is not None]
            return max(values) if values else None
        parameters = parameter_report(model)
        training_parameters = training.get("parameters")
        efficiency = {
            "stage": "stage2", "task": "image_question_auto_prompt_segmentation", "device": str(device),
            "world_size": world_size, "batch_size_per_gpu": args.batch_size,
            "precision": args.precision, "num_samples": len(rows),
            "model": {"total_params": parameters["total_params"],
                      "training_trainable_params": training_parameters.get("trainable_params") if training_parameters else None,
                      "training_parameter_source": training.get("source")},
            "checkpoint_size_mb": file_size_mb(args.checkpoint),
            "memory": {"per_rank": memories, "max_peak_allocated_gb": maximum_memory("peak_allocated_gb"),
                       "max_peak_reserved_gb": maximum_memory("peak_reserved_gb")},
            "speed": {"wall_time_sec": wall_time,
                      "samples_per_sec": len(rows) / wall_time if wall_time else 0.0,
                      "inference_time_sec_sum_rank_batches": duration,
                      "inference_time_sec_max_rank_sum": max(sum(item["batch_times"]) for item in packets),
                      "batch_latency_mean_ms": float(np.mean(batch_times) * 1000) if batch_times else None,
                      "batch_latency_p95_ms": percentile([value * 1000 for value in batch_times], 95) if batch_times else None,
                      "amortized_sample_latency_ms": duration * 1000 / timed_samples if timed_samples else None,
                      "latency_definition": "Batch latency is generation plus second forward/mask decoding; amortized latency is not independent single-sample latency. Profiling batch excluded from latency statistics.",
                      "timed_samples": timed_samples,
                      "profiled_samples": sum(item["profiled_samples"] for item in packets),
                      "wall_time_definition": "Maximum rank evaluation-loop time, including loading, scoring, visualization, prediction saves and profiling; excludes model loading and final result aggregation.",
                      "scoring_and_saving_time_sec_sum_ranks": sum(item["scoring_and_saving_time_sec"] for item in packets)},
            "generation": {"generated_tokens": sum(item["generated_tokens"] for item in packets),
                           "end_to_end_tokens_per_sec": sum(item["generated_tokens"] for item in packets) / wall_time if wall_time else 0.0},
            "training": training.get("efficiency"),
            "elapsed_since_launch_sec": time.perf_counter() - launch_started,
        }
        save_json(out_dir / "efficiency.json", efficiency)
        save_json(out_dir / "efficiency_metrics.json", efficiency)  # Compatibility alias.
        save_efficiency_csv(out_dir / "efficiency_metrics.csv", efficiency)
        save_json(out_dir / "compute_profile.json", profiler.report())
        from experiment_reporting import save_diagnostic_outputs
        save_diagnostic_outputs(out_dir, rows)
        print(json.dumps({key: value for key, value in summary.items() if key != "per_dataset"}, indent=2))
        print(f"Saved {len(rows)} unique results to {out_dir}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
