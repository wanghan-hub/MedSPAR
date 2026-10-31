"""Image/question generation and a second forward for automatic segmentation.

Neither public inference helper accepts a mask, bounding box, reference answer,
or annotation-derived vision text. SEG is never inserted into generated output.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Sequence

import torch

from .collator import pad_token_id
from experiment_protocol import check_context


@contextmanager
def _evaluation_mode(model):
    previous = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in previous:
            module.training = training


def generated_token_mask(ids: torch.Tensor, eos_ids: Sequence[int], pad_id: int) -> torch.Tensor:
    """Include the first EOS, exclude everything after it, including padding."""
    valid = torch.ones_like(ids, dtype=torch.bool)
    for i in range(ids.shape[0]):
        end = ids.shape[1]
        for j, value in enumerate(ids[i].tolist()):
            if int(value) in eos_ids:
                end = j + 1
                break
            if int(value) == pad_id:
                end = j
                break
        valid[i, end:] = False
    return valid


def _autocast_state(device):
    enabled = torch.is_autocast_enabled(device.type)
    return enabled, torch.get_autocast_dtype(device.type) if enabled else None


@torch.no_grad()
def generate_reasoning(
    model, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    questions: Sequence[str], *, max_new_tokens: int = 256, do_sample: bool = False,
    temperature: float = 1.0, top_p: float = 0.9, num_beams: int = 1,
    repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
    stop_on_seg: bool = True, cache_visual_features: bool = True,
) -> Dict[str, Any]:
    """Generate once from image + question, retaining original continuation IDs."""
    if input_ids.shape != attention_mask.shape or images.shape[0] != input_ids.shape[0]:
        raise ValueError("Image, question-token, and attention-mask batch dimensions disagree.")
    if len(questions) != images.shape[0] or max_new_tokens < 1:
        raise ValueError("Each image needs one question and max_new_tokens must be positive.")
    tokenizer = model.tokenizer
    pad_id = pad_token_id(tokenizer)
    seg_id = int(model.seg_token_id)
    with _evaluation_mode(model):
        image_embeddings, patch_tokens, spatial_hw = model._encode_for_grounding(images, list(questions))
        visual_embeds = model.visual_projector(model.visual_compressor(patch_tokens))
        embedding = model.llm.get_input_embeddings()
        visual_embeds = visual_embeds.to(dtype=embedding.weight.dtype, device=input_ids.device)
        text_lengths = attention_mask.long().sum(-1)
        nv = visual_embeds.shape[1]
        width = int(text_lengths.max().item()) + nv
        full_ids = input_ids.new_full((images.shape[0], width), pad_id)
        full_mask = attention_mask.new_zeros((images.shape[0], width))
        inputs_embeds = embedding(full_ids)
        # Left-pad the complete multimodal prefix so each row ends at Assistant:.
        # Appending visual tokens in front of padded text leaves internal holes.
        for i, length in enumerate(text_lengths.tolist()):
            start = width - nv - length
            valid_text = input_ids[i, attention_mask[i].bool()]
            inputs_embeds[i, start:start + nv] = visual_embeds[i]
            inputs_embeds[i, start + nv:] = embedding(valid_text)
            full_ids[i, start + nv:] = valid_text
            full_mask[i, start:] = 1
        check_context(model.llm, width + max_new_tokens)
        eos_ids = []
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        if stop_on_seg:
            eos_ids.append(seg_id)
        eos_ids = list(dict.fromkeys(eos_ids))
        kwargs = dict(
            max_new_tokens=int(max_new_tokens), do_sample=bool(do_sample),
            num_beams=int(num_beams), repetition_penalty=float(repetition_penalty),
            pad_token_id=pad_id, use_cache=True,
            return_dict_in_generate=False, output_scores=False,
        )
        if eos_ids:
            kwargs["eos_token_id"] = eos_ids
        if do_sample:
            kwargs.update(temperature=float(temperature), top_p=float(top_p))
        if no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        sequences = model.llm.generate(
            input_ids=full_ids, inputs_embeds=inputs_embeds,
            attention_mask=full_mask, **kwargs,
        )
        if (sequences.ndim != 2 or sequences.shape[0] != images.shape[0]
                or sequences.shape[1] < width
                or not torch.equal(sequences[:, :width], full_ids)):
            raise RuntimeError(
                "LLM generate() did not return the supplied token prefix. "
                "This backbone needs explicit generation adaptation; output is not silently re-generated."
            )
        generated_ids = sequences[:, width:]
        generated_mask = generated_token_mask(generated_ids, eos_ids, pad_id)
        seg_counts = (generated_ids.eq(seg_id) & generated_mask).sum(-1)
        texts = []
        for ids, valid in zip(generated_ids, generated_mask):
            values = ids[valid].tolist()
            # Keep SEG even though it is a special token; drop tokenizer wrappers.
            wrappers = set(getattr(tokenizer, "all_special_ids", [])) - {seg_id}
            values = [v for v in values if v not in wrappers]
            texts.append(tokenizer.decode(values, skip_special_tokens=False).strip())
        result = {
            "generated_ids": generated_ids,
            "generated_attention_mask": generated_mask,
            "seg_counts": seg_counts,
            "texts": texts,
            "generated_token_counts": generated_mask.sum(-1).tolist(),
            "generation_mode": "image_question_free_generation",
        }
        if cache_visual_features:
            # Local to this generation result; never stored on the model or on disk.
            result["_visual_cache"] = {
                "features": (image_embeddings, patch_tokens, spatial_hw),
                "images": images, "questions": tuple(questions), "model": model,
                "image_version": None if torch.is_inference(images) else images._version,
                "autocast_state": _autocast_state(images.device),
            }
        return result


def build_generated_forward_inputs(
    tokenizer, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
    generated_ids: torch.Tensor, generated_mask: torch.Tensor, seg_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Concatenate exact prompt/generated tokens and locate assistant reasoning."""
    sequences, prompt_lengths = [], []
    for pids, pmask, gids, gmask in zip(prompt_ids, prompt_mask, generated_ids, generated_mask):
        pids, gids = pids[pmask.bool()], gids[gmask.bool()]
        sequences.append(torch.cat([pids, gids]))
        prompt_lengths.append(pids.numel())
    width = max(x.numel() for x in sequences)
    ids = prompt_ids.new_full((len(sequences), width), pad_token_id(tokenizer))
    attention = prompt_mask.new_zeros((len(sequences), width))
    assistant = torch.zeros_like(ids, dtype=torch.bool)
    for i, (sequence, plen) in enumerate(zip(sequences, prompt_lengths)):
        ids[i, :sequence.numel()] = sequence
        attention[i, :sequence.numel()] = 1
        assistant[i, plen:sequence.numel()] = True
    seg_locations = assistant & ids.eq(int(seg_token_id))
    positions = torch.arange(width, device=ids.device).expand_as(ids)
    first_seg = positions.masked_fill(~seg_locations, width).min(-1).values
    reasoning = assistant & positions.lt(first_seg.unsqueeze(1))
    for special_id in getattr(tokenizer, "all_special_ids", []):
        reasoning &= ids.ne(int(special_id))
    return {"input_ids": ids, "attention_mask": attention,
            "assistant_mask": assistant, "reasoning_mask": reasoning}


@torch.no_grad()
def predict_from_generated(
    model, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    questions: Sequence[str], generation: Dict[str, Any], *, stage: str = "unified",
) -> Dict[str, Any]:
    """Second LLM forward and automatic prompts, only for valid generated answers."""
    prepared = build_generated_forward_inputs(
        model.tokenizer, input_ids, attention_mask,
        generation["generated_ids"], generation["generated_attention_mask"], model.seg_token_id,
    )
    valid = generation["seg_counts"].eq(1)
    indices = torch.where(valid)[0]
    cached_features = None
    cache = generation.get("_visual_cache")
    if cache is not None:
        version = None if torch.is_inference(images) else images._version
        if (cache["model"] is not model or cache["images"] is not images
                or cache["questions"] != tuple(questions) or cache["image_version"] != version
                or cache["autocast_state"] != _autocast_state(images.device)):
            raise ValueError("Generated visual cache model/image/question batch or autocast policy differs")
        embeddings, tokens, spatial_hw = cache["features"]
        if embeddings.shape[0] != images.shape[0] or tokens.shape[0] != images.shape[0]:
            raise ValueError("Generated visual cache does not cover the complete original batch")
        if indices.numel():
            cached_features = (embeddings, tokens, spatial_hw) if indices.numel() == images.shape[0] else (
                embeddings.index_select(0, indices), tokens.index_select(0, indices), spatial_hw)
    outputs = None
    if indices.numel():
        subset = {key: value[indices] for key, value in prepared.items()}
        with _evaluation_mode(model):
            outputs = model(
                images=images[indices], **subset, labels=None, boxes=None,
                steer_texts=[questions[i] for i in indices.tolist()], stage=stage,
                prompt_source="auto", compute_lm_loss=False,
                cached_visual_features=cached_features,
            )
    reasons = []
    for i in range(images.shape[0]):
        count = int(generation["seg_counts"][i])
        reasons.append("" if bool(valid[i]) else
                       "missing_seg" if count == 0 else
                       "multiple_seg")
    return {"valid_indices": indices, "valid_generation": valid,
            "failure_reasons": reasons, "model_outputs": outputs}
