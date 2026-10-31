"""Dataset for LLM + MedSAM reasoning segmentation.

This loader reads the unified JSON format:

    image
    mask
    input
    output
    body_part
    modality
    target
    target_position

The output field is used for Stage 2 LLM training and should contain [SEG].
Both language and vision are conditioned only on JSON input. Annotation
metadata is returned solely for supervision/reporting. Set load_supervision=False
and load_reference_answer=False for image/question inference with no mask IO.
Metadata-derived model input is rejected.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from TC_MVE.box_ops import choose_eval_box, choose_train_box
from TC_MVE.prompt_builder import (
    build_eval_prompt,
    get_target_phrase,
)
from .seg_token import DEFAULT_SEG_TOKEN
from experiment_protocol import question_text, answer_text


def _load_json_records(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["data", "samples", "annotations", "items"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(f"Unsupported JSON format in {json_path}")


def _resolve_path(split_dir: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else split_dir / path


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _resize_image_np(img: np.ndarray, image_size: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(pil)


def _resize_mask_np(mask: np.ndarray, image_size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    pil = pil.resize((image_size, image_size), resample=Image.NEAREST)
    arr = np.asarray(pil)
    return (arr > 0).astype(np.uint8)


class LLMReasonSegDataset(Dataset):
    """Read user JSONs and return image/mask/box plus conversation text.

    This dataset is used for Stage 2 LLM + MedSAM training/evaluation.

    Returned text fields:
        question:
            User instruction from JSON["input"] or JSON["question"]. A metadata
            fallback exists only with explicit historical question_only=False.

        answer:
            Assistant response. Prefer JSON["output"], fallback to generated answer.
            Must contain exactly one [SEG] after ensure_single_seg_token().

        steer_text:
            The same user question for the Steer-MedSAM vision branch in P1-P4.
    """

    def __init__(
        self,
        data_root: str | os.PathLike,
        datasets: Sequence[str] | str,
        split: str = "train",
        image_size: int = 1024,
        is_train: bool = True,
        train_box_mode: str = "mixed",
        eval_box_mode: str = "tight",
        bbox_shift: int = 20,
        enlarge_ratio: float = 1.5,
        seg_token: str = DEFAULT_SEG_TOKEN,
        append_seg_if_missing: bool = True,
        skip_empty_mask: bool = False,
        seed: int = 2026,
        load_supervision: bool = True,
        load_reference_answer: bool = True,
        question_only: bool = True,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)

        if isinstance(datasets, str):
            datasets = [x.strip() for x in datasets.split(",") if x.strip()]
        self.datasets = list(datasets)

        self.split = split
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.train_box_mode = train_box_mode
        self.eval_box_mode = eval_box_mode
        self.bbox_shift = int(bbox_shift)
        self.enlarge_ratio = float(enlarge_ratio)
        self.seg_token = seg_token
        self.append_seg_if_missing = bool(append_seg_if_missing)
        self.skip_empty_mask = bool(skip_empty_mask)
        self.rng = random.Random(seed)
        self.load_supervision = bool(load_supervision)
        self.load_reference_answer = bool(load_reference_answer)
        self.question_only = bool(question_only)
        if not self.question_only:
            raise ValueError("Metadata-derived text is not supported by the unified protocol")

        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

        if not self.samples:
            raise RuntimeError(
                f"No samples found: root={data_root}, "
                f"datasets={datasets}, split={split}"
            )

    def _load_samples(self) -> None:
        for ds_name in self.datasets:
            split_dir = self.data_root / ds_name / self.split
            json_path = split_dir / f"{self.split}.json"

            if not json_path.exists():
                candidates = sorted(split_dir.glob("*.json"))
                if not candidates:
                    raise FileNotFoundError(f"No json found in {split_dir}")
                json_path = candidates[0]

            records = _load_json_records(json_path)

            for i, rec in enumerate(records):
                if "image" not in rec or (self.load_supervision and "mask" not in rec):
                    raise ValueError(f"Missing required image/mask field in {json_path}, record {i}")

                item = dict(rec)

                item["dataset"] = item.get("dataset", ds_name)
                item["_dataset_name"] = ds_name

                item["split_dir"] = str(split_dir)
                item["json_path"] = str(json_path)
                item["record_index"] = i
                item["image_path"] = str(_resolve_path(split_dir, item["image"]))
                item["mask_path"] = str(_resolve_path(split_dir, item["mask"])) if item.get("mask") else ""

                self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def _dataset_name(self, sample: Dict[str, Any]) -> str:
        return sample.get("_dataset_name", sample.get("dataset", ""))

    def _build_question(self, sample):
        return question_text(sample)

    def _build_steer_text(self, sample):
        return self._build_question(sample)

    def _build_answer(self, sample):
        return answer_text(sample, self.seg_token)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.load_supervision:
            sample = self.samples[index]
            image = _resize_image_np(_read_rgb(Path(sample["image_path"])), self.image_size)
            item = {
                "image": torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0,
                "question": self._build_question(sample),
                "steer_text": self._build_steer_text(sample),
                "dataset": sample.get("dataset", self._dataset_name(sample)),
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "record_index": int(sample.get("record_index", index)),
            }
            if self.load_reference_answer:
                item.update({
                    "answer": self._build_answer(sample),
                    "target": sample.get("target", ""),
                    "target_phrase": get_target_phrase(sample, dataset_name=self._dataset_name(sample)),
                    "body_part": sample.get("body_part", ""),
                    "modality": sample.get("modality", ""),
                })
            return item
        allow_resample = self.is_train and self.skip_empty_mask
        # Only explicitly excluded empty training masks may be resampled.
        # Corrupt files and interface errors must surface with their traceback.
        for _ in range(10):
            sample = self.samples[index]
            image_path = Path(sample["image_path"])
            mask_path = Path(sample["mask_path"])
            image = _read_rgb(image_path)
            mask = _read_mask(mask_path)
            if self.skip_empty_mask and mask.sum() == 0:
                if not allow_resample:
                    raise ValueError(f"Empty mask: {mask_path}")
                print(f"Training empty-mask resample: {mask_path}", flush=True)
                index = self.rng.randrange(len(self.samples))
                continue
            break
        else:
            raise RuntimeError("Failed to fetch a valid non-empty mask after 10 attempts.")

        image = _resize_image_np(image, self.image_size)
        mask = _resize_mask_np(mask, self.image_size)

        if self.is_train:
            box = choose_train_box(
                mask,
                width=self.image_size,
                height=self.image_size,
                mode=self.train_box_mode,
                bbox_shift=self.bbox_shift,
                enlarge_ratio=self.enlarge_ratio,
                rng=self.rng,
            )
        else:
            box = choose_eval_box(
                mask,
                width=self.image_size,
                height=self.image_size,
                mode=self.eval_box_mode,
                bbox_shift=0,
                enlarge_ratio=self.enlarge_ratio,
            )

        image_t = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        box_t = torch.as_tensor(box, dtype=torch.float32)

        dataset_name = self._dataset_name(sample)
        target_phrase = get_target_phrase(sample, dataset_name=dataset_name)

        return {
            "image": image_t,
            "mask": mask_t,
            "box": box_t,
            "question": self._build_question(sample),
            "answer": self._build_answer(sample) if self.load_reference_answer else "",
            "steer_text": self._build_steer_text(sample),
            "dataset": sample.get("dataset", dataset_name),
            "dataset_name": dataset_name,
            "target": sample.get("target", ""),
            "target_phrase": target_phrase,
            "modality": sample.get("modality", ""),
            "body_part": sample.get("body_part", ""),
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
            "record_index": int(sample.get("record_index", index)),
        }
