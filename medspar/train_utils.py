"""Training helpers for LLM-MedSAM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from experiment_protocol import validate_protocol


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def amp_dtype(precision: str):
    precision = (precision or "fp32").lower()
    if precision in {"fp16", "float16"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


def autocast_context(device: torch.device, precision: str):
    dtype = amp_dtype(precision)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None and device.type == "cuda"))


def make_grad_scaler(precision: str, device: torch.device):
    enabled = precision.lower() in {"fp16", "float16"} and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def prepare_trainable_precision(model):
    """Keep optimizer master parameters in fp32; leave frozen LLM storage alone.

    Run after freezing and checkpoint initialization, before creating optimizer
    or DDP. In particular, GradScaler cannot unscale fp16 embedding gradients.
    Parameter identities (including tied embeddings) are preserved.
    """
    converted = []
    for name, parameter in unwrap_model(model).named_parameters():
        if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
            if parameter.grad is not None:
                raise RuntimeError("Prepare trainable precision before the first backward")
            parameter.data = parameter.data.float()
            converted.append(name)
    if converted:
        print(f"FP32 optimizer parameters: converted {len(converted)} trainable tensors; "
              f"examples={converted[:8]}")
    return converted


def print_precision_policy(args, local_rank: int = 0) -> None:
    if local_rank != 0:
        return
    print("\n" + "=" * 90)
    print("Precision policy")
    print("- Original checkpoint precision: usually fp32 unless checkpoint/model loader casts weights.")
    print(f"- LLM load dtype (--llm_torch_dtype): {args.llm_torch_dtype}")
    print(f"- Forward autocast (--precision): {args.precision}")
    print("- Images/boxes/masks are loaded as fp32.")
    print("- Visual prefix embeddings are cast to the LLM token embedding dtype before concatenation.")
    print("- [SEG] sparse prompt is cast to image_embeddings dtype before SAM mask decoder.")
    print("- Losses are computed in fp32 by casting logits/masks to float().")
    print("- GradScaler is enabled only for fp16, not bf16/fp32.")
    print("=" * 90 + "\n")


def _stats(module: Optional[torch.nn.Module]):
    if module is None:
        return 0, 0, 0.0
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train, train / total * 100 if total else 0.0


def print_parameter_statistics(model: torch.nn.Module, local_rank: int = 0) -> None:
    if local_rank != 0:
        return
    m = unwrap_model(model)
    rows = [
        ("MedSPAR", m),
        ("  ├─ MedSAM Image Encoder", getattr(getattr(m, "medsam", None), "image_encoder", None)),
        ("  ├─ MedSAM Prompt Encoder", getattr(getattr(m, "medsam", None), "prompt_encoder", None)),
        ("  ├─ MedSAM Mask Decoder", getattr(getattr(m, "medsam", None), "mask_decoder", None)),
        ("  ├─ LLM", getattr(m, "llm", None)),
        ("  ├─ Visual Token Compressor", getattr(m, "visual_compressor", None)),
        ("  ├─ Visual Projector", getattr(m, "visual_projector", None)),
        
    ]
    rows.extend((name, getattr(m, name, None)) for name in (
        "reasoning_query_pooler", "seg_anchor_projector", "semantic_spatial_grounder",
        "automatic_geometric_prompt_generator", "semantic_prompt_projector", "prompt_fusion_gate",
    ))
    print("\n" + "=" * 100)
    print(f"{'Component':<34} | {'Total':<18} | {'Trainable':<18} | {'Trainable %':<12}")
    print("-" * 100)
    for name, module in rows:
        total, train, ratio = _stats(module)
        print(f"{name:<34} | {total:<18,} | {train:<18,} | {ratio:>10.2f}%")
    print("=" * 100 + "\n")


def print_trainable_parameter_names(model: torch.nn.Module, max_lines: int = 300, local_rank: int = 0) -> None:
    if local_rank != 0:
        return
    m = unwrap_model(model)
    print("Trainable parameter names:")
    n = 0
    for name, p in m.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {tuple(p.shape)}")
            n += 1
            if n >= max_lines:
                print(f"  ... truncated after {max_lines} lines")
                break


def save_checkpoint(path, model, optimizer=None, epoch=0, best_metric=0., extra=None):
    from .experiments import UNIFIED_VERSION, ExperimentConfig
    from checkpoint_io import optimizer_signature
    raw=unwrap_model(model)
    state=dict(pipeline_version=UNIFIED_VERSION, experiment_id=raw.experiment_id,
        effective_flags=ExperimentConfig(raw.experiment_id).flags(), vision_backbone=raw.vision_backbone,
        model_config=raw.model_config, training_config=(extra or {}).get('config',{}),
        model=raw.state_dict(), epoch=int(epoch),best_metric=float(best_metric),extra=extra or {},
        seg_token_metadata=dict(seg_token=raw.seg_token,seg_token_id=int(raw.seg_token_id),vocab_size=len(raw.tokenizer)))
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(),optimizer_signature=optimizer_signature(raw,optimizer))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,path)





def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    from .experiments import UNIFIED_VERSION, ExperimentConfig
    from checkpoint_io import strict_state_load, restore_optimizer
    raw=unwrap_model(model)
    ckpt=torch.load(str(path),map_location=map_location,weights_only=True)
    validate_protocol(ckpt.get('training_config',{}))
    if ckpt.get('pipeline_version')!=UNIFIED_VERSION or ckpt.get('experiment_id')!=raw.experiment_id:
        raise RuntimeError('Checkpoint must belong to this new-project experiment')
    if ckpt.get('effective_flags')!=ExperimentConfig(raw.experiment_id).flags() or ckpt.get('model_config')!=raw.model_config:
        raise RuntimeError('Checkpoint architecture/tokenization/experiment flags differ')
    metadata=dict(seg_token=raw.seg_token,seg_token_id=int(raw.seg_token_id),vocab_size=len(raw.tokenizer))
    if ckpt.get('seg_token_metadata')!=metadata: raise RuntimeError('SEG token metadata mismatch')
    raw.checkpoint_load_report=strict_state_load(raw,ckpt['model'],path)
    current=raw.state_dict()
    for label,module in [('SEG embedding',raw.llm.get_input_embeddings()),('SEG LM head',raw.llm.get_output_embeddings())]:
        keys=[n+'.weight' for n,m in raw.named_modules() if m is module]
        if not keys or keys[0] not in ckpt['model']: raise RuntimeError(f'Missing {label}')
        key=keys[0]
        expected=ckpt['model'][key][raw.seg_token_id].to(module.weight.device,module.weight.dtype)
        if not torch.equal(module.weight[raw.seg_token_id],expected): raise RuntimeError(f'{label} restoration failed')
        print(f'  VERIFIED {label}: {key}[{raw.seg_token_id}]')
    if optimizer is not None: restore_optimizer(raw,optimizer,ckpt)
    return ckpt


def optimizer_coverage(model, optimizer, require_complete=True):
    """Audit by parameter identity, accounting for shared/tied weights once."""
    named = {id(p): (name, p) for name, p in unwrap_model(model).named_parameters()}
    included = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [id(p) for p in included]
    seen = set(ids)
    missing = [name for key, (name, p) in named.items() if p.requires_grad and key not in seen]
    frozen = [named[id(p)][0] for p in included if id(p) in named and not p.requires_grad]
    foreign = [key for key in seen if key not in named]
    duplicates = len(ids) - len(seen)
    report = {"trainable_not_in_optimizer": missing, "frozen_in_optimizer": frozen,
              "foreign_parameter_count": len(foreign), "duplicate_parameter_count": duplicates,
              "optimizer_parameter_tensors": len(ids),
              "optimizer_parameter_elements": sum(p.numel() for p in included),
              "groups": [{"name": g.get("name", str(i)), "lr": g["lr"],
                          "parameter_tensors": len(g["params"]),
                          "parameter_elements": sum(p.numel() for p in g["params"])}
                         for i, g in enumerate(optimizer.param_groups)]}
    if require_complete and (missing or frozen or foreign or duplicates):
        raise RuntimeError(f"Optimizer parameter coverage failed: {report}")
    return report


def parameter_statistics(model):
    """Module totals overlap where the existing wrapper shares SAM parameters."""
    m = unwrap_model(model)
    modules = {
        "LLM": m.llm, "Vision Wrapper": m.vision,
        ("Original MedSAM Image Encoder" if m.vision_backbone == "medsam" else "Steer-MedSAM Image Encoder"): m.medsam.image_encoder,
        "MedSAM Prompt Encoder": m.medsam.prompt_encoder,
        "MedSAM Mask Decoder": m.medsam.mask_decoder,
        "Visual Token Compressor": m.visual_compressor, "Visual Projector": m.visual_projector,
        "Reasoning Query Pooler": m.reasoning_query_pooler,
        "SEG Anchor Projector": m.seg_anchor_projector,
        "Semantic-Spatial Grounder": m.semantic_spatial_grounder,
        "Automatic Geometric Prompt Generator": m.automatic_geometric_prompt_generator,
        "Semantic Prompt Projector": m.semantic_prompt_projector,
        "Prompt Fusion Gate": m.prompt_fusion_gate,
    }
    result = {}
    for name, module in modules.items():
        total, trainable, ratio = _stats(module)
        result[name] = dict(total_params=total, trainable_params=trainable, trainable_ratio_percent=ratio)
    total, trainable, ratio = _stats(m)
    return {"modules": result, "total_unique_params": total, "trainable_unique_params": trainable,
            "trainable_ratio_percent": ratio,
            "trainable_parameter_names": [n for n, p in m.named_parameters() if p.requires_grad],
            "note": "Module rows overlap through shared SAM references; unique totals count each parameter once."}


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def restore_evaluation_config(args):
    from .experiments import UNIFIED_VERSION, MODEL_CONFIG_FIELDS, ExperimentConfig
    checkpoint=torch.load(str(args.checkpoint),map_location='cpu',weights_only=True)
    if checkpoint.get('pipeline_version')!=UNIFIED_VERSION:
        raise RuntimeError('Only new MedSPAR experiment checkpoints are accepted')
    config=checkpoint.get('model_config',{})
    mode=checkpoint.get('experiment_id')
    policy=ExperimentConfig(mode)
    if checkpoint.get('effective_flags')!=policy.flags(): raise RuntimeError('Checkpoint flags differ')
    missing=set(MODEL_CONFIG_FIELDS)-set(config)
    if missing or config.get('experiment_id')!=mode or config.get('vision_type')!=policy.vision_backbone:
        raise RuntimeError(f'Incomplete/inconsistent model configuration: {sorted(missing)}')
    explicit=getattr(args,'experiment_id',None)
    if explicit is not None and explicit!=mode: raise ValueError('Requested experiment differs from checkpoint')
    for k in MODEL_CONFIG_FIELDS: setattr(args,k,config[k])
    saved=checkpoint['training_config']
    validate_protocol(saved)
    args.max_new_tokens=saved['max_new_tokens']
    args.steer_text_encoder=getattr(args,'steer_text_encoder',None) or saved['steer_text_encoder']
    args.vision_backbone=policy.vision_backbone
    args.stage1_checkpoint=args.stage0_checkpoint=''
    args.restored_model_config=dict(config)
    print(f'Restored MedSPAR experiment={mode}; flags={policy.flags()}; checkpoint={args.checkpoint}')
    return args
