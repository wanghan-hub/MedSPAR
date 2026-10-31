"""Reproducible input, supervision and scoring contract, copied into each project.

Each project owns this module; no sibling checkout is required at runtime.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

PROTOCOL_VERSION = 'question-square-native-v1'
SYSTEM_PROMPT = 'You are a helpful medical image segmentation assistant.'


def question_text(sample):
    value = sample.get('input')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing nonempty input: {sample.get('image_path', sample.get('image'))}")
    if '[SEG]' in value:
        raise ValueError('User input must not contain the segmentation control token')
    return value.strip()


def answer_text(sample, seg_token='[SEG]'):
    value = sample.get('output')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing nonempty output: {sample.get('image_path', sample.get('image'))}")
    value = value.strip()
    if seg_token not in value:
        value += ' ' + seg_token
    if value.count(seg_token) != 1:
        raise ValueError('One target mask requires exactly one segmentation token')
    return value


def format_prompt(question, system_prompt=SYSTEM_PROMPT):
    return (f'### System:\n{system_prompt}\n\n### User:\n<image>\n'
            f'{question.strip()}\n\n### Assistant:\n')


def conversation_ids(tokenizer, question, answer=None, max_length=1024, system_prompt=SYSTEM_PROMPT):
    prompt = list(tokenizer.encode(format_prompt(question, system_prompt), add_special_tokens=True))
    seg = tokenizer.convert_tokens_to_ids('[SEG]')
    if seg in prompt:
        raise ValueError('Prompt contains [SEG]')
    output = []
    if answer is not None:
        output = list(tokenizer.encode(answer.strip(), add_special_tokens=False))
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError('The common tokenizer must define eos_token_id')
        while output and output[-1] == eos:
            output.pop()
        output.append(eos)
        if output.count(seg) != 1:
            raise ValueError('The supervised answer must encode exactly one [SEG]')
    ids = prompt + output
    if len(ids) > max_length:
        raise ValueError(f'Text length {len(ids)} exceeds max_length={max_length}; no truncation')
    return ids, len(prompt)


def check_context(llm, length):
    config = getattr(llm.config, 'text_config', llm.config)
    limit = getattr(config, 'max_position_embeddings', None)
    if limit and length > limit:
        raise ValueError(f'Multimodal context {length} exceeds model limit {limit}')


def read_square_image(path, image_size=1024):
    with Image.open(path) as source:
        image = source.convert('RGB')
        original_size = (image.height, image.width)
        array = np.asarray(image.resize((image_size, image_size), Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0, original_size


def read_mask(path, image_size=None):
    with Image.open(path) as source:
        array = (np.asarray(source.convert('L')) > 0).astype(np.uint8)
    if image_size is not None:
        array = np.asarray(Image.fromarray(array).resize((image_size, image_size), Image.Resampling.NEAREST)).copy()
    return torch.from_numpy(array).float()[None]


def stage1_box(mask, training=False, mode='tight', shift=20, enlarge_ratio=1.5, rng=None):
    import random
    rng = rng or random
    array = np.asarray(mask).squeeze()
    height, width = array.shape
    ys, xs = np.where(array > 0)
    if not len(xs):
        return np.array([0, 0, width - 1, height - 1], dtype=np.float32)
    if mode == 'mixed':
        value = rng.random()
        mode = 'tight' if value < .7 else 'enlarged' if value < .9 else 'full'
    if mode == 'full':
        return np.array([0, 0, width - 1, height - 1], dtype=np.float32)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    if mode == 'enlarged':
        center = (box[:2] + box[2:]) / 2
        half = (box[2:] - box[:2] + 1) * enlarge_ratio / 2
        box = np.concatenate([center - half, center + half])
    elif mode == 'tight':
        if training and shift:
            box += np.array([-rng.randint(0, shift), -rng.randint(0, shift), rng.randint(0, shift), rng.randint(0, shift)])
    else:
        raise ValueError(f'Unknown box mode: {mode}')
    return np.clip(box, [0, 0, 0, 0], [width-1, height-1, width-1, height-1]).astype(np.float32)


def seed_worker(worker_id):
    import random
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    dataset = torch.utils.data.get_worker_info().dataset
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if hasattr(dataset, 'rng'):
        dataset.rng.seed(seed)


def common_mask_loss(logits, targets, bce_weight=2.0, dice_weight=0.5):
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f'Mask shape mismatch: {logits.shape}, {targets.shape}')
    logits, targets = logits.float(), targets.float()
    p, y = logits.sigmoid().flatten(1), targets.flatten(1)
    numerator = 2 * (p / 1000 * y).sum(1)
    denominator = (p / 1000).sum(1) + (y / 1000).sum(1)
    dice = (1 - (numerator + 1e-6) / (denominator + 1e-6)).sum() / (len(logits) + 1e-8)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none').flatten(1).mean(1)
    bce = bce.sum() / (len(logits) + 1e-8)
    return bce_weight * bce + dice_weight * dice


def native_prediction(logits, mask_path, threshold=0.5):
    target = read_mask(mask_path).to(logits.device)
    prediction = F.interpolate(logits.reshape(1, 1, *logits.shape[-2:]).float(), target.shape[-2:],
                               mode='bilinear', align_corners=False)[0].sigmoid().gt(threshold)
    return prediction, target.bool()


def native_scores(logits, mask_paths, threshold=0.5):
    if len(logits) != len(mask_paths):
        raise ValueError('Prediction and GT path counts disagree')
    dice, iou = [], []
    for mask, path in zip(logits, mask_paths):
        pred, target = native_prediction(mask, path, threshold)
        inter = (pred & target).sum().float()
        total = pred.sum().float() + target.sum().float()
        dice.append((2 * inter + 1e-6) / (total + 1e-6))
        iou.append((inter + 1e-6) / (total - inter + 1e-6))
    return torch.stack(dice), torch.stack(iou)


def native_metrics(logits, mask_paths, threshold=0.5):
    """Native-grid metrics/counts; preserves the historical Dice/IoU formula.

    Recall is TP / GT foreground. For a valid prediction and empty GT it is 1
    (no positive target to miss); invalid generation is scored as 0 by caller.
    """
    if len(logits) != len(mask_paths):
        raise ValueError('Prediction and GT path counts disagree')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be a probability in [0,1]')
    rows = []
    for mask, path in zip(logits, mask_paths):
        pred, target = native_prediction(mask, path, threshold)
        counts = torch.stack(((pred & target).sum(), pred.sum(), target.sum())).cpu().tolist()
        tp, predicted, foreground = (int(value) for value in counts)
        # Calculate in float32, exactly as native_scores (including epsilon).
        inter = mask.new_tensor(float(tp), dtype=torch.float32)
        total = mask.new_tensor(float(predicted + foreground), dtype=torch.float32)
        rows.append(dict(dice=float((2 * inter + 1e-6) / (total + 1e-6)),
                         iou=float((inter + 1e-6) / (total - inter + 1e-6)),
                         recall=tp / foreground if foreground else 1.0,
                         tp=tp, fp=predicted - tp, fn=foreground - tp,
                         gt_foreground_pixels=foreground, pred_foreground_pixels=predicted,
                         native_height=int(target.shape[-2]), native_width=int(target.shape[-1])))
    return rows


def validate_protocol(config):
    if config.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Checkpoint uses another/legacy experiment protocol. Start a new unified run; '
                         'do not silently resume legacy preprocessing or teacher-forced selection.')


def prepare_trainable_precision(model):
    """Use FP32 master parameters/Adam states; autocast controls forward precision."""
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point():
            parameter.data = parameter.data.float()


def training_budget(loader, epochs, batch_size, world_size, accumulation=1):
    import math
    return dict(epochs=epochs, batches_per_epoch=len(loader),
                optimizer_steps_per_epoch=math.ceil(len(loader) / accumulation),
                planned_optimizer_steps=epochs * math.ceil(len(loader) / accumulation),
                nominal_global_batch=batch_size * world_size * accumulation,
                accumulation_steps=accumulation, scheduler='constant')


def load_matching_state(model, state, path):
    """Strip only unambiguous outer wrappers; require every model tensor to match."""
    expected = set(model.state_dict())
    candidates = [dict(state)]
    for _ in range(4):
        current = candidates[-1]
        prefix = next((p for p in ('module.', 'model.', 'base_model.model.', 'base_model.')
                       if current and all(k.startswith(p) for k in current)), None)
        if prefix is None:
            break
        candidates.append({k[len(prefix):]: v for k, v in current.items()})
    matched = min(candidates, key=lambda s: len(expected ^ set(s)))
    missing, unexpected = sorted(expected - set(matched)), sorted(set(matched) - expected)
    print(f'Checkpoint: {path}\nmissing_keys={len(missing)} {missing}\nunexpected_keys={len(unexpected)} {unexpected}')
    if missing or unexpected:
        raise RuntimeError('Checkpoint keys do not match this model; no partial load was performed')
    model.load_state_dict(matched, strict=True)
