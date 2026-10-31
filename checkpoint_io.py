"""Strict new-project checkpoint operations; no historical adapters."""
import torch

def strict_state_load(model, state, path):
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected)-set(state)), sorted(set(state)-set(expected))
    shapes = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    print(f'Checkpoint: {path}; matched={len(set(expected)&set(state))}; missing={len(missing)} {missing}; unexpected={len(unexpected)} {unexpected}; shape_mismatch={shapes}', flush=True)
    if missing or unexpected or shapes: raise RuntimeError('Checkpoint is incomplete or incompatible')
    model.load_state_dict(state, strict=True)
    groups = ('llm.', 'visual_compressor.', 'visual_projector.', 'seg_anchor_projector.',
              'reasoning_query_pooler.', 'semantic_spatial_grounder.', 'automatic_geometric_prompt_generator.',
              'semantic_prompt_projector.', 'prompt_fusion_gate.', 'medsam.image_encoder.',
              'medsam.prompt_encoder.', 'medsam.mask_decoder.', 'text_encoder.')
    for prefix in groups:
        count=sum(k.startswith(prefix) for k in state)
        if count: print(f'  restored {prefix}: {count} tensors', flush=True)
    return dict(path=str(path), missing_keys=missing, unexpected_keys=unexpected, shape_mismatch=shapes)

def optimizer_signature(model, optimizer):
    names={id(p):n for n,p in model.named_parameters()}
    return [dict(name=g.get('name'), parameters=[(names[id(p)],list(p.shape)) for p in g['params']],
                 lr=g['lr'],weight_decay=g.get('weight_decay')) for g in optimizer.param_groups]

def restore_optimizer(model, optimizer, checkpoint):
    if 'optimizer' not in checkpoint or checkpoint.get('optimizer_signature') != optimizer_signature(model, optimizer):
        raise RuntimeError('Resume optimizer groups, ordered parameter names/shapes or hyperparameters differ')
    optimizer.load_state_dict(checkpoint['optimizer'])


def validate_resume_config(saved, current):
    mutable = {'resume','output_dir','epochs','save_every','val_every','print_trainable_names',
               'num_workers','gpu_ids','medsam_checkpoint','llm_name_or_path','text_encoder',
               'steer_text_encoder','stage0_checkpoint','stage1_checkpoint','data_root',
               'profile_compute','scaler','reporting','training_budget'}
    mismatches = {k: (saved.get(k), v) for k,v in current.items()
                  if k not in mutable and saved.get(k) != v}
    if mismatches:
        raise ValueError(f'Resume configuration differs: {mismatches}')
