"""Training utilities for Steer-MedSAM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from experiment_protocol import load_matching_state, validate_protocol


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when using DistributedDataParallel/DataParallel."""
    return model.module if hasattr(model, "module") else model


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _get_stats(module: Optional[torch.nn.Module]) -> tuple[int, int, float]:
    if module is None:
        return 0, 0, 0.0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    return total, trainable, ratio


def _iter_gca_modules(image_encoder: torch.nn.Module):
    for m in image_encoder.modules():
        if m.__class__.__name__ in {"GatedTextCrossAttention", "GatedCrossAttention"}:
            yield m


def print_parameter_statistics(model: torch.nn.Module, local_rank: int = 0) -> None:
    """Print component-level total/trainable parameter statistics.

    This is tailored to SteerMedSAM. It also works when the model is wrapped by
    DistributedDataParallel or DataParallel.
    """
    if local_rank != 0:
        return

    model = unwrap_model(model)
    medsam = getattr(model, "medsam", model)

    image_encoder = getattr(medsam, "image_encoder", None)
    prompt_encoder = getattr(medsam, "prompt_encoder", None)
    mask_decoder = getattr(medsam, "mask_decoder", None)
    iou_head = getattr(mask_decoder, "iou_prediction_head", None) if mask_decoder is not None else None
    text_encoder = getattr(model, "text_encoder", None)
    patch_head = getattr(model, "patch_head", None)

    text_backbone = getattr(text_encoder, "text_model", None) if text_encoder is not None else None
    text_connector = getattr(text_encoder, "connector", None) if text_encoder is not None else None

    # Count only the inserted gated cross-attention modules inside the image encoder.
    if image_encoder is not None:
        gca_modules = torch.nn.ModuleList(list(_iter_gca_modules(image_encoder)))
    else:
        gca_modules = None

    rows = [
        ("Steer-MedSAM (整体模型)", model),
        ("  ├─ TC-MVE / MedSAM Image Encoder", image_encoder),
        ("  │   └─ Gated Cross-Attn", gca_modules),
        ("  ├─ MedSAM Prompt Encoder", prompt_encoder),
        ("  ├─ MedSAM Mask Decoder", mask_decoder),
        ("  │   └─ IoU Prediction Head", iou_head),
        ("  ├─ Text Encoder (backbone)", text_backbone),
        ("  ├─ Text-to-Vision MLP", text_connector),
        ("  └─ Aux Patch Head", patch_head),
    ]

    print("\n" + "=" * 98)
    print(f"{'组件 (Component)':<32} | {'参数总量 (Total)':<18} | {'训练参数量 (Trainable)':<20} | {'训练占比 (%)':<12}")
    print("-" * 98)
    for name, module in rows:
        total, trainable, ratio = _get_stats(module)
        print(f"{name:<32} | {total:<18,} | {trainable:<20,} | {ratio:>10.2f}%")
    print("=" * 98)
    print("期望看到：Image Encoder整体大多冻结，但 Gated Cross-Attn 可训练；Prompt Encoder 和 Text Encoder backbone 为 0 trainable；Mask Decoder 大部分可训练，但 IoU Prediction Head 为 0 trainable；Text-to-Vision MLP、Aux Patch Head 可训练。\n")


def print_trainable_parameters(model: torch.nn.Module, max_lines: int = 300, local_rank: int = 0) -> None:
    if local_rank != 0:
        return
    model = unwrap_model(model)
    stats = count_parameters(model)
    print(f"Total params:     {stats['total']:,}")
    print(f"Trainable params: {stats['trainable']:,}")
    print(f"Frozen params:    {stats['frozen']:,}")
    print("\nTrainable parameter names:")
    n = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {tuple(p.shape)}")
            n += 1
            if n >= max_lines:
                print(f"  ... truncated after {max_lines} lines")
                break


def save_checkpoint(path, model, optimizer=None, epoch=0, best_metric=0., extra=None):
    from checkpoint_io import optimizer_signature
    raw=unwrap_model(model)
    state=dict(pipeline_version='medspar_visual_v1', training_stage=raw.training_stage,
        model=raw.state_dict(), epoch=int(epoch), best_metric=float(best_metric), extra=extra or {})
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), optimizer_signature=optimizer_signature(raw, optimizer))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,path)


def load_visual_checkpoint(path, model, optimizer=None, map_location='cpu'):
    from checkpoint_io import strict_state_load, restore_optimizer
    ckpt=torch.load(str(path),map_location=map_location,weights_only=True)
    raw=unwrap_model(model)
    if ckpt.get('pipeline_version')!='medspar_visual_v1' or ckpt.get('training_stage')!=raw.training_stage:
        raise RuntimeError('Only matching new-project Stage0/Stage1 checkpoints are supported')
    validate_protocol(ckpt['extra'])
    raw.checkpoint_load_report=strict_state_load(raw,ckpt['model'],path)
    if optimizer is not None: restore_optimizer(raw,optimizer,ckpt)
    return ckpt


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
