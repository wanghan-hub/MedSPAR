"""Prompt construction utilities for text-steered MedSAM.

Current project decision:
- Training does NOT use the long `output` field.
- Training uses 80% prompt_context and 20% prompt_simple.
- prompt_location is not used in the training pipeline.
"""
# steer_medsam/prompt_builder.py

from __future__ import annotations

import random
from typing import Any, Dict, Optional


def _norm_text(x: Any) -> str:
    """Normalize json text fields."""
    if x is None:
        return ""
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def _clean_phrase(x: Any) -> str:
    """Return a clean natural-language phrase."""
    if x is None:
        return ""
    return str(x).strip().replace("_", " ").replace("-", " ")


# Dataset/body_part/target-aware phrase mapping.
# Key format: (dataset, body_part, target)
# Use lowercase normalized strings.
TARGET_PHRASE_MAP = {
    # Original datasets
    ("skin", "skin", "skin lesion"): "skin lesion",
    ("skin", "skin", "lesion"): "skin lesion",
    ("poly", "colon", "polyp"): "polyp",
    ("poly", "colorectum", "polyp"): "polyp",
    ("brats", "brain", "whole tumor"): "whole tumor",
    ("brats", "brain", "tumor core"): "tumor core",
    ("brats", "brain", "enhancing tumor"): "enhancing tumor",
    ("brats", "brain", "tumor"): "brain tumor",

    # Added datasets
    ("breast", "breast", "tumor"): "breast tumor",
    ("breast", "breast", "lesion"): "breast lesion",

    ("kidney", "kidney", "kidney"): "kidney",
    ("kidney", "kidney", "tumor"): "kidney tumor",
    ("kidney", "kidney", "lesion"): "kidney lesion",

    ("liver", "liver", "liver"): "liver",
    ("liver", "liver", "tumor"): "liver tumor",
    ("liver", "liver", "cancer"): "liver tumor",
    ("liver", "liver", "lesion"): "liver lesion",

    ("lung", "lung", "tumor"): "lung tumor",
    ("lung", "lung", "nodule"): "lung nodule",
    ("lung", "lung", "lesion"): "lung lesion",

    ("pancreas", "pancreas", "pancreas"): "pancreas",
    ("pancreas", "pancreas", "tumor"): "pancreatic tumor",
    ("pancreas", "pancreas", "cancer"): "pancreatic tumor",
    ("pancreas", "pancreas", "lesion"): "pancreatic lesion",

    ("prostate", "prostate", "prostate"): "prostate",
    ("prostate", "prostate", "tumor"): "prostate tumor",
    ("prostate", "prostate", "cancer"): "prostate tumor",
    ("prostate", "prostate", "lesion"): "prostate lesion",

    ("thyroid", "thyroid", "nodule"): "thyroid nodule",
    ("thyroid", "thyroid", "tumor"): "thyroid tumor",
    ("thyroid", "thyroid", "lesion"): "thyroid lesion",
}


# Fallback mapping when dataset is not reliable but body_part + target are available.
BODY_TARGET_PHRASE_MAP = {
    ("skin", "lesion"): "skin lesion",
    ("skin", "skin lesion"): "skin lesion",

    ("colon", "polyp"): "polyp",
    ("colorectum", "polyp"): "polyp",

    ("brain", "tumor"): "brain tumor",
    ("brain", "whole tumor"): "whole tumor",
    ("brain", "tumor core"): "tumor core",
    ("brain", "enhancing tumor"): "enhancing tumor",

    ("breast", "tumor"): "breast tumor",
    ("breast", "lesion"): "breast lesion",

    ("kidney", "kidney"): "kidney",
    ("kidney", "tumor"): "kidney tumor",
    ("kidney", "lesion"): "kidney lesion",

    ("liver", "liver"): "liver",
    ("liver", "tumor"): "liver tumor",
    ("liver", "cancer"): "liver tumor",
    ("liver", "lesion"): "liver lesion",

    ("lung", "tumor"): "lung tumor",
    ("lung", "nodule"): "lung nodule",
    ("lung", "lesion"): "lung lesion",

    ("pancreas", "pancreas"): "pancreas",
    ("pancreas", "tumor"): "pancreatic tumor",
    ("pancreas", "cancer"): "pancreatic tumor",
    ("pancreas", "lesion"): "pancreatic lesion",

    ("prostate", "prostate"): "prostate",
    ("prostate", "tumor"): "prostate tumor",
    ("prostate", "cancer"): "prostate tumor",
    ("prostate", "lesion"): "prostate lesion",

    ("thyroid", "nodule"): "thyroid nodule",
    ("thyroid", "tumor"): "thyroid tumor",
    ("thyroid", "lesion"): "thyroid lesion",
}


def get_target_phrase(sample: Dict[str, Any], dataset_name: Optional[str] = None) -> str:
    """
    Return a natural-language target phrase for prompt construction.

    Priority:
    1. sample["target_phrase"], if it exists and is non-empty.
    2. dataset + body_part + target mapping.
    3. body_part + target mapping.
    4. fallback target.
    5. "target".
    """
    explicit = _clean_phrase(sample.get("target_phrase", ""))
    if explicit:
        return explicit

    dataset = _norm_text(dataset_name or sample.get("dataset", ""))
    body_part = _norm_text(sample.get("body_part", ""))
    target = _norm_text(sample.get("target", ""))

    # Some dataset names may come as KiTS23, ISIC, etc.
    # Normalize them to the canonical project dataset names when possible.
    dataset_alias = {
        "kits23": "kidney",
        "kits": "kidney",
        "kidney": "kidney",
        "brats": "brats",
        "skin": "skin",
        "poly": "poly",
        "polyp": "poly",
        "breast": "breast",
        "liver": "liver",
        "lung": "lung",
        "pancreas": "pancreas",
        "prostate": "prostate",
        "thyroid": "thyroid",
    }
    dataset = dataset_alias.get(dataset, dataset)

    key_full = (dataset, body_part, target)
    if key_full in TARGET_PHRASE_MAP:
        return TARGET_PHRASE_MAP[key_full]

    key_body = (body_part, target)
    if key_body in BODY_TARGET_PHRASE_MAP:
        return BODY_TARGET_PHRASE_MAP[key_body]

    # If target is too generic but body_part exists, make it organ-specific.
    if target in {"tumor", "lesion", "cancer", "nodule"} and body_part:
        if body_part == "pancreas" and target in {"tumor", "cancer"}:
            return "pancreatic tumor"
        return f"{body_part} {target}"

    if target:
        return _clean_phrase(target)

    return "target"


def get_modality_phrase(sample: Dict[str, Any]) -> str:
    modality = _clean_phrase(sample.get("modality", ""))
    return modality if modality else "medical"


def get_body_part_phrase(sample: Dict[str, Any]) -> str:
    body_part = _clean_phrase(sample.get("body_part", ""))
    return body_part if body_part else "anatomical region"


def build_prompt_simple(sample: Dict[str, Any], dataset_name: Optional[str] = None) -> str:
    target_phrase = get_target_phrase(sample, dataset_name=dataset_name)
    return f"Segment the {target_phrase}."


def build_prompt_context(sample: Dict[str, Any], dataset_name: Optional[str] = None) -> str:
    target_phrase = get_target_phrase(sample, dataset_name=dataset_name)
    modality = get_modality_phrase(sample)
    body_part = get_body_part_phrase(sample)
    return f"Segment the {target_phrase} in this {modality} image of the {body_part}."


def choose_train_prompt(
    sample: Dict[str, Any],
    dataset_name: Optional[str] = None,
    p_context: float = 0.8,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Training prompt policy:
    - 80% prompt_context
    - 20% prompt_simple
    """
    rng = rng or random
    if rng.random() < p_context:
        return build_prompt_context(sample, dataset_name=dataset_name)
    return build_prompt_simple(sample, dataset_name=dataset_name)


def build_eval_prompt(
    sample: Dict[str, Any],
    dataset_name: Optional[str] = None,
    prompt_type: str = "context",
) -> str:
    if prompt_type == "simple":
        return build_prompt_simple(sample, dataset_name=dataset_name)
    if prompt_type == "context":
        return build_prompt_context(sample, dataset_name=dataset_name)
    raise ValueError(f"Unknown prompt_type: {prompt_type}. Expected 'simple' or 'context'.")

def build_wrong_prompt(
    sample: Dict[str, Any],
    wrong_target: Optional[str] = None,
    prompt_type: str = "context",
) -> str:
    """Build an intentionally wrong prompt for evaluation only.

    This is used to test whether the model actually uses text. It does not change
    the image or box. If wrong_target is not provided, a small rule-based target
    swap is used.
    """
    target = get_target(sample).lower()
    modality = get_modality(sample)
    body_part = get_body_part(sample)

    if wrong_target is None:
        candidates = [
            "skin lesion",
            "polyp",
            "brain tumor",
            "lung nodule",
            "breast lesion",
            "kidney tumor",
            "liver tumor",
            "cell nucleus",
        ]
        wrong_target = next((x for x in candidates if x.lower() != target), "different target")

    fake_sample = dict(sample)
    fake_sample["target"] = wrong_target
    fake_sample["modality"] = modality
    fake_sample["body_part"] = body_part
    return build_eval_prompt(fake_sample, prompt_type=prompt_type)