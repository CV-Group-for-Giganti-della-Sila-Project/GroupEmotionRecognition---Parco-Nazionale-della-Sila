#!/usr/bin/env python3
"""
QLoRA fine-tuning script for MiniCPM-V on FERPlus-style emotion data.

Expected input record format, either in a JSON array file or JSONL file:

{
    "image": "/workspace/datasets/images/train/happiness/img.png",
    "output": "{\"primary_emotion\": \"happiness\"}"
}

The script:
1. Loads the dataset.
2. Extracts the primary emotion from each sample.
3. Keeps a stratified fraction of each emotion class, defaulting to 1/3.
4. Converts each FERPlus single-person sample into a MiniCPM-V chat example.
5. Fine-tunes MiniCPM-V with 4-bit QLoRA for a single epoch by default.

Example:

python finetune_minicpmv_emotion_qlora.py \
  --train_json /workspace/datasets/ferplus_train.json \
  --output_dir ./minicpmv-ferplus-emotion-qlora

Useful packages on an AWS g4dn.xlarge / Tesla T4:

pip install -U "transformers>=4.57.0" accelerate peft bitsandbytes pillow

Notes:
- Tesla T4 does not support bfloat16, so this script uses fp16.
- FERPlus is mostly single-face emotion data. It can teach emotion recognition
  and output formatting, but true multi-person group reasoning is best learned
  from an additional dataset containing group images and per-person labels.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    # MiniCPM-V 4.6 is exposed through this auto class in current Transformers.
    from transformers import AutoModelForImageTextToText

    MODEL_AUTO_CLASS = AutoModelForImageTextToText
except ImportError:
    # Older MiniCPM-V checkpoints used trust_remote_code with AutoModel.
    # Keeping this fallback makes the script usable with older environments.
    from transformers import AutoModel

    MODEL_AUTO_CLASS = AutoModel

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-4.6"
DEFAULT_EMOTIONS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

TORCH_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    """Collect CLI arguments in one place so the training behavior is explicit."""

    parser = argparse.ArgumentParser(
        description="Fine-tune MiniCPM-V with QLoRA on a FERPlus-style emotion dataset."
    )

    # Dataset and output paths.
    parser.add_argument("--train_json", required=True, help="Path to the FERPlus train JSON/JSONL file.")
    parser.add_argument(
        "--eval_json",
        default=None,
        help="Optional validation JSON/JSONL file. If omitted, a small stratified split is taken from train.",
    )
    parser.add_argument(
        "--output_dir",
        default="./minicpmv-ferplus-emotion-qlora",
        help="Directory where the LoRA adapter and processor will be saved.",
    )
    parser.add_argument(
        "--training_summary_json",
        default=None,
        help="Where to save the final training summary JSON. Defaults to <output_dir>/training_summary.json.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory to resume from, e.g. ./output/checkpoint-1200.",
    )

    # Model loading. The default is intentionally configurable because MiniCPM-V
    # variants differ a lot in memory needs.
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id, e.g. openbmb/MiniCPM-V-4_5 or another MiniCPM-V checkpoint.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model/processor code from the checkpoint repository.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="Attention backend. Use sdpa on T4 unless flash-attn is installed and verified.",
    )
    parser.add_argument(
        "--model_torch_dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Dtype for non-quantized model modules. float32 is slower but can help diagnose NaN gradients.",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        default="float16",
        choices=("float16", "float32"),
        help="bitsandbytes 4-bit matmul compute dtype. float32 can be more stable than float16 on difficult runs.",
    )

    # Data reduction and validation split. sample_fraction=1/3 implements your
    # requirement: keep one third of each emotion subset.
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0 / 3.0,
        help="Fraction sampled independently from each emotion class.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="Validation ratio carved out of the sampled training set when --eval_json is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class-wise sampling, train/eval splitting, and Trainer reproducibility.",
    )
    parser.add_argument(
        "--emotion_labels",
        default=",".join(DEFAULT_EMOTIONS),
        help="Comma-separated allowed emotion labels used in prompts. Samples with other labels are skipped.",
    )
    parser.add_argument(
        "--target_schema",
        choices=("group", "primary"),
        default="group",
        help=(
            "group: train the model to output subjects plus group_emotion. "
            "primary: train the model to reproduce {'primary_emotion': label}."
        ),
    )

    # MiniCPM-V image processor controls. Smaller max_slice_nums is friendlier
    # to a 15 GB T4, especially with batch size 1.
    parser.add_argument(
        "--downsample_mode",
        default="16x",
        help="MiniCPM-V image-token downsampling mode. 16x is lighter; 4x keeps more visual detail.",
    )
    parser.add_argument(
        "--max_slice_nums",
        type=int,
        default=4,
        help="Maximum image slices. Lower values reduce memory at the cost of visual detail.",
    )
    parser.add_argument(
        "--image_tensor_dtype",
        default="float32",
        choices=("auto", "float16", "float32"),
        help="Dtype for processor image tensors. float32 is safer; auto preserves the processor output dtype.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum token length for a training conversation.",
    )

    # One epoch is the default to preserve compute. Batch size 1 plus gradient
    # accumulation is usually the safest starting point on a T4.
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override epoch-based training with a fixed number of optimizer steps. Useful for stability smoke tests.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Micro-batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Eval micro-batch size.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Accumulate small batches to simulate a larger effective batch size.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="LoRA learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup fraction.")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. This is an extra guardrail against unstable updates.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp16 mixed precision. Use --no-fp16 for a slower but more stable diagnostic run.",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Trainer logging interval.")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint save interval.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Eval interval when validation data exists.")

    # Loss-divergence guardrails. These defaults are conservative for QLoRA:
    # they do not react to ordinary noisy batches, but they do intervene if the
    # logged loss becomes non-finite or jumps to a clearly dangerous level.
    parser.add_argument(
        "--stop_on_nan_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training if the training loss or eval_loss becomes NaN/Inf.",
    )
    parser.add_argument(
        "--stop_on_nan_grad_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop when logged grad_norm is NaN/Inf. Default is false because fp16 AMP can report "
            "non-finite grad_norm during early scale calibration even when loss is finite."
        ),
    )
    parser.add_argument(
        "--max_consecutive_nan_grad_norm",
        type=int,
        default=3,
        help="When --stop_on_nan_grad_norm is enabled, stop after this many consecutive non-finite grad_norm logs.",
    )
    parser.add_argument(
        "--debug_non_finite_grads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the first trainable parameter names whose gradients become NaN/Inf.",
    )
    parser.add_argument(
        "--max_non_finite_grad_names",
        type=int,
        default=8,
        help="Maximum number of non-finite gradient parameter names to record when debugging.",
    )
    parser.add_argument(
        "--loss_guard_min_logs",
        type=int,
        default=3,
        help="Number of finite logged train-loss values to observe before checking for loss explosion.",
    )
    parser.add_argument(
        "--loss_explosion_factor",
        type=float,
        default=3.0,
        help="Reduce LR when logged train loss is this many times worse than the best logged train loss.",
    )
    parser.add_argument(
        "--loss_explosion_abs_threshold",
        type=float,
        default=10.0,
        help="Minimum absolute logged loss required before treating a spike as an explosion.",
    )
    parser.add_argument(
        "--lr_reduction_factor",
        type=float,
        default=0.5,
        help="Multiply the current LR and scheduler base LRs by this factor after a loss explosion.",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
        help="Do not reduce LR below this value.",
    )
    parser.add_argument(
        "--lr_reduction_cooldown_steps",
        type=int,
        default=50,
        help="Minimum optimizer steps between automatic LR reductions.",
    )
    parser.add_argument(
        "--max_lr_reductions",
        type=int,
        default=3,
        help="Stop training if loss keeps exploding after this many LR reductions. Use 0 for unlimited reductions.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Stop after this many eval rounds without eval_loss improvement. Requires validation data.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum eval_loss improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload the best eval_loss checkpoint at the end when validation checkpoints exist.",
    )

    # QLoRA settings. The target module names match the Qwen-style language
    # backbone used by recent MiniCPM-V checkpoints.
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help=(
            "Comma-separated module suffixes to adapt, or 'auto' to infer common linear module names. "
            "The default targets the language model and leaves the vision encoder frozen."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to reduce VRAM usage.",
    )

    # Dataset workers. Keeping this small avoids CPU RAM spikes on modest EC2
    # instances while still allowing PIL image loading to overlap a bit.
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="DataLoader worker count.")

    return parser.parse_args()


def normalize_emotion(value: str) -> str:
    """Normalize emotion names to make grouping robust across JSON variants."""

    return value.strip().lower().replace(" ", "_")


def validate_guardrail_args(args: argparse.Namespace) -> None:
    """Fail early if guardrail hyperparameters cannot behave as intended."""

    if not 0 < args.lr_reduction_factor < 1:
        raise ValueError("--lr_reduction_factor must be in the interval (0, 1).")
    if args.loss_explosion_factor <= 1:
        raise ValueError("--loss_explosion_factor must be greater than 1.")
    if args.loss_explosion_abs_threshold <= 0:
        raise ValueError("--loss_explosion_abs_threshold must be positive.")
    if args.min_learning_rate < 0:
        raise ValueError("--min_learning_rate cannot be negative.")
    if args.max_lr_reductions < 0:
        raise ValueError("--max_lr_reductions cannot be negative. Use 0 for unlimited reductions.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience cannot be negative.")
    if args.max_consecutive_nan_grad_norm <= 0:
        raise ValueError("--max_consecutive_nan_grad_norm must be positive.")
    if args.max_non_finite_grad_names <= 0:
        raise ValueError("--max_non_finite_grad_names must be positive.")
    if args.max_steps < -1 or args.max_steps == 0:
        raise ValueError("--max_steps must be -1 or a positive integer.")
    if args.eval_steps <= 0:
        raise ValueError("--eval_steps must be positive.")
    if args.save_steps <= 0:
        raise ValueError("--save_steps must be positive.")


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load either a JSON array file or a JSONL file.

    Many dataset preparation scripts write one JSON object per line for large
    datasets, while smaller examples are often stored as one JSON array. This
    helper accepts both without changing the training code.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty.")

    if text[0] == "[":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list of records.")
        return records

    if text[0] == "{":
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return loaded
            if isinstance(loaded, dict) and "image" in loaded:
                return [loaded]
        except json.JSONDecodeError:
            # Fall through to JSONL parsing below.
            pass

    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
        records.append(item)

    return records


def parse_output_payload(raw_output: Any) -> Dict[str, Any]:
    """
    Convert the dataset's output field into a dictionary.

    Your example stores JSON inside a string:
    "{\"primary_emotion\": \"happiness\"}".
    Some pipelines may already store it as a dict, so both formats are accepted.
    """

    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse output as JSON: {raw_output}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed output is not a JSON object: {raw_output}")
        return parsed
    raise ValueError(f"Unsupported output field type: {type(raw_output).__name__}")


def extract_emotion(record: Dict[str, Any]) -> str:
    """
    Extract the label used for stratified sampling and target generation.

    Primary source: record['output']['primary_emotion'].
    Fallback: the parent directory name from the image path, which matches the
    common /train/<emotion>/image.png dataset organization.
    """

    output = parse_output_payload(record.get("output", {}))
    emotion = output.get("primary_emotion") or output.get("emotion") or output.get("label")

    if emotion is None:
        image_path = record.get("image")
        if not image_path:
            raise ValueError(f"Record has no usable emotion label and no image path: {record}")
        emotion = Path(str(image_path)).parent.name

    return normalize_emotion(str(emotion))


def validate_and_annotate_records(
    records: Sequence[Dict[str, Any]],
    allowed_emotions: Sequence[str],
    require_images: bool = True,
) -> List[Dict[str, Any]]:
    """
    Attach a normalized '_emotion' field and filter unsupported labels.

    FERPlus can contain labels such as unknown/non-face depending on preprocessing.
    The default prompt emotion list includes the eight standard FERPlus emotions;
    unsupported labels are skipped so the model is not trained to emit labels
    that are absent from the prompt.
    """

    allowed = {normalize_emotion(label) for label in allowed_emotions if label.strip()}
    annotated: List[Dict[str, Any]] = []
    skipped_missing_image = 0
    skipped_label = Counter()

    for record in records:
        if "image" not in record:
            skipped_missing_image += 1
            continue

        image_path = Path(str(record["image"]))
        if require_images and not image_path.exists():
            skipped_missing_image += 1
            continue

        emotion = extract_emotion(record)
        if allowed and emotion not in allowed:
            skipped_label[emotion] += 1
            continue

        annotated_record = dict(record)
        annotated_record["_emotion"] = emotion
        annotated.append(annotated_record)

    if skipped_missing_image:
        print(f"Skipped {skipped_missing_image} records with missing image paths.")
    if skipped_label:
        print(f"Skipped labels outside --emotion_labels: {dict(skipped_label)}")
    if not annotated:
        raise ValueError("No usable records remain after validation/filtering.")

    return annotated


def stratified_sample(
    records: Sequence[Dict[str, Any]],
    fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Keep the requested fraction from each emotion class independently.

    This implements "use 1/3 of each subset of emotion": every class is sampled
    separately, so frequent classes do not consume the entire reduced dataset.
    """

    if not 0 < fraction <= 1:
        raise ValueError("--sample_fraction must be in the interval (0, 1].")

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["_emotion"]].append(record)

    sampled: List[Dict[str, Any]] = []
    for emotion, group in sorted(groups.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        keep_count = max(1, math.floor(len(shuffled) * fraction))
        sampled.extend(shuffled[:keep_count])
        print(f"Class {emotion}: kept {keep_count}/{len(shuffled)} samples.")

    rng.shuffle(sampled)
    return sampled


def stratified_train_eval_split(
    records: Sequence[Dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split sampled records into train/eval while preserving emotion coverage.

    When a class has only one sampled example, it stays in train. This avoids an
    empty class in training after aggressive 1/3 sampling.
    """

    if eval_ratio <= 0:
        return list(records), []
    if eval_ratio >= 1:
        raise ValueError("--eval_ratio must be less than 1.")

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["_emotion"]].append(record)

    train_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        eval_count = max(1, math.floor(len(shuffled) * eval_ratio)) if len(shuffled) > 1 else 0
        eval_records.extend(shuffled[:eval_count])
        train_records.extend(shuffled[eval_count:])

    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return train_records, eval_records


def build_target_json(emotion: str, schema: str) -> str:
    """
    Build the supervised assistant answer.

    The group schema uses a single subject because FERPlus images normally
    contain one face. At inference time on group images, the prompt asks the
    model to extend the same schema to multiple subjects and to choose the
    majority emotion as group_emotion.
    """

    if schema == "primary":
        payload = {"primary_emotion": emotion}
    elif schema == "group":
        payload = {
            "subjects": [{"id": 1, "emotion": emotion}],
            "group_emotion": emotion,
        }
    else:
        raise ValueError(f"Unsupported target schema: {schema}")

    # Compact JSON reduces target tokens and makes exact-format learning easier.
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def build_user_prompt(emotion_labels: Sequence[str], schema: str) -> str:
    """
    Build the instruction shown with every image.

    The prompt is intentionally strict: we want the model to emit machine-parseable
    JSON instead of prose, because downstream IoT/analytics pipelines are much
    easier to build when the response schema is stable.
    """

    labels = ", ".join(emotion_labels)

    if schema == "primary":
        return (
            "Analyze the visible face in the image. "
            f"Choose exactly one emotion from this list: {labels}. "
            'Return only valid JSON in this schema: {"primary_emotion":"<emotion>"}'
        )

    return (
        "Analyze the image and identify each visible person separately. "
        f"For every subject, choose exactly one emotion from this list: {labels}. "
        "Set group_emotion to the most representative emotion; when several subjects are present, "
        "use the majority emotion, and if there is a tie choose the most central or salient subject. "
        'Return only valid JSON in this schema: {"subjects":[{"id":1,"emotion":"<emotion>"}],"group_emotion":"<emotion>"}'
    )


def make_conversation(
    image: Image.Image,
    prompt: str,
    answer: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create a MiniCPM-V/Hugging Face multimodal chat conversation.

    The image is placed in the user turn. The assistant turn is present for
    supervised training examples and omitted for prompt-only examples used to
    compute which tokens should be masked from the loss.
    """

    conversation: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if answer is not None:
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )

    return conversation


class EmotionConversationDataset(Dataset):
    """Small PyTorch Dataset wrapper around validated emotion records."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        prompt: str,
        target_schema: str,
    ) -> None:
        self.records = list(records)
        self.prompt = prompt
        self.target_schema = target_schema

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Load images lazily so the dataset does not keep all images in RAM.

        Converting to RGB avoids mode-specific surprises such as grayscale or
        palette images reaching the vision processor.
        """

        record = self.records[index]
        image = Image.open(record["image"]).convert("RGB")
        emotion = record["_emotion"]
        answer = build_target_json(emotion, self.target_schema)

        return {
            "image": image,
            "prompt": self.prompt,
            "answer": answer,
            "emotion": emotion,
        }


@dataclass
class MiniCPMVDataCollator:
    """
    Convert PIL images and text conversations into tensors accepted by MiniCPM-V.

    The label mask is the key supervised fine-tuning detail:
    - User/image/prompt tokens are set to -100 and do not contribute to loss.
    - Assistant JSON answer tokens keep their ids and are optimized.
    - Padding tokens are set to -100.
    """

    processor: Any
    max_length: int
    downsample_mode: str
    max_slice_nums: int
    image_tensor_dtype: str = "float32"
    _template_attempt_index: Optional[int] = None

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        full_conversations = [
            make_conversation(example["image"], example["prompt"], example["answer"])
            for example in examples
        ]
        prompt_conversations = [
            make_conversation(example["image"], example["prompt"], answer=None)
            for example in examples
        ]

        full_inputs = self._apply_chat_template(
            full_conversations,
            add_generation_prompt=False,
        )
        prompt_inputs = self._apply_chat_template(
            prompt_conversations,
            add_generation_prompt=True,
        )

        labels = full_inputs["input_ids"].clone()
        attention_mask = full_inputs.get("attention_mask")

        # Mask the non-answer prefix for every row. prompt_inputs includes the
        # assistant generation prefix, so the first unmasked token corresponds
        # to the first token of the JSON answer.
        prompt_attention = prompt_inputs.get("attention_mask")
        for row_index in range(labels.size(0)):
            if prompt_attention is not None:
                prompt_length = int(prompt_attention[row_index].sum().item())
            else:
                prompt_length = int(prompt_inputs["input_ids"][row_index].ne(self.pad_token_id).sum().item())
            labels[row_index, :prompt_length] = -100

        # Mask all padding from the supervised loss.
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        else:
            labels[full_inputs["input_ids"] == self.pad_token_id] = -100

        full_inputs["labels"] = labels

        # Keep image tensors in a stable dtype. The first versions of this
        # script cast every floating tensor to fp16; on T4 that can make the
        # first backward pass produce NaN gradients for some MiniCPM-V batches.
        for key, value in list(full_inputs.items()):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                full_inputs[key] = self._cast_float_input(key, value)

        return dict(full_inputs)

    def _cast_float_input(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Cast only image-like floating tensors according to --image_tensor_dtype."""

        if self.image_tensor_dtype == "auto":
            return value
        if "pixel" not in key.lower() and "image" not in key.lower():
            return value
        return value.to(dtype=TORCH_DTYPE_BY_NAME[self.image_tensor_dtype])

    @property
    def pad_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            return int(tokenizer.pad_token_id)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        return 0

    def _apply_chat_template(
        self,
        conversations: Sequence[List[Dict[str, Any]]],
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        """
        Call processor.apply_chat_template with compatibility fallbacks.

        Recent Transformers versions require processor-call options to live in
        processor_kwargs. Keep only chat-template controls at the top level so
        Transformers does not warn on every batch.
        """

        base_kwargs = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        processor_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "downsample_mode": self.downsample_mode,
            "max_slice_nums": self.max_slice_nums,
        }
        attempts = [
            {
                **base_kwargs,
                "processor_kwargs": processor_kwargs,
            },
        ]

        if self._template_attempt_index is not None:
            return self.processor.apply_chat_template(
                list(conversations),
                **attempts[self._template_attempt_index],
            )

        last_error: Optional[TypeError] = None
        for attempt_index, kwargs in enumerate(attempts):
            try:
                result = self.processor.apply_chat_template(list(conversations), **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            self._template_attempt_index = attempt_index
            return result

        assert last_error is not None
        raise last_error


class GuardrailTrainingStop(RuntimeError):
    """Raised when a training guardrail needs to abort the run immediately."""


class LossDivergenceGuardrailCallback(TrainerCallback):
    """
    Stop unstable runs and reduce LR when the loss clearly explodes.

    Transformers callbacks can see logged metrics and the optimizer/scheduler.
    This callback uses that hook to:
    - stop on non-finite train/eval losses,
    - reduce LR when logged train loss spikes far above the best observed loss,
    - record all guardrail actions for the final JSON report.
    """

    def __init__(
        self,
        stop_on_nan_loss: bool,
        stop_on_nan_grad_norm: bool,
        max_consecutive_nan_grad_norm: int,
        debug_non_finite_grads: bool,
        max_non_finite_grad_names: int,
        loss_guard_min_logs: int,
        loss_explosion_factor: float,
        loss_explosion_abs_threshold: float,
        lr_reduction_factor: float,
        min_learning_rate: float,
        lr_reduction_cooldown_steps: int,
        max_lr_reductions: int,
    ) -> None:
        self.stop_on_nan_loss = stop_on_nan_loss
        self.stop_on_nan_grad_norm = stop_on_nan_grad_norm
        self.max_consecutive_nan_grad_norm = max_consecutive_nan_grad_norm
        self.debug_non_finite_grads = debug_non_finite_grads
        self.max_non_finite_grad_names = max_non_finite_grad_names
        self.loss_guard_min_logs = max(0, loss_guard_min_logs)
        self.loss_explosion_factor = loss_explosion_factor
        self.loss_explosion_abs_threshold = loss_explosion_abs_threshold
        self.lr_reduction_factor = lr_reduction_factor
        self.min_learning_rate = min_learning_rate
        self.lr_reduction_cooldown_steps = max(0, lr_reduction_cooldown_steps)
        self.max_lr_reductions = max_lr_reductions

        self.best_logged_loss: Optional[float] = None
        self.latest_logged_loss: Optional[float] = None
        self.latest_eval_loss: Optional[float] = None
        self.latest_learning_rate: Optional[float] = None
        self.latest_grad_norm: Optional[float] = None
        self.consecutive_nan_grad_norm_logs = 0
        self.non_finite_grad_examples: List[Dict[str, Any]] = []
        self.logged_train_loss_count = 0
        self.last_lr_reduction_step = -10**12
        self.lr_reduction_events: List[Dict[str, Any]] = []
        self.stop_reason: Optional[str] = None
        self.fatal_stop = False

    def observe_training_step_loss(self, loss: Any, state: Any) -> None:
        """
        Inspect the raw per-step training loss.

        This catches NaN/Inf losses immediately instead of waiting until the next
        logging interval.
        """

        loss_value = self._to_float(loss)
        if loss_value is None:
            return
        if self.stop_on_nan_loss and not math.isfinite(loss_value):
            step = getattr(state, "global_step", None)
            self.fatal_stop = True
            self.stop_reason = f"non_finite_training_loss_at_step_{step}"
            raise GuardrailTrainingStop(
                f"Training stopped because loss became non-finite at step {step}: {loss_value}"
            )

    def observe_non_finite_parameter_grads(self, model: torch.nn.Module, state: Any) -> None:
        """Record trainable parameter names whose gradients contain NaN/Inf."""

        if not self.debug_non_finite_grads:
            return

        examples: List[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            grad = parameter.grad
            if torch.is_floating_point(grad) and not torch.isfinite(grad).all():
                examples.append(name)
                if len(examples) >= self.max_non_finite_grad_names:
                    break

        if examples:
            event = {
                "step": getattr(state, "global_step", None),
                "parameter_names": examples,
            }
            if not self.non_finite_grad_examples or self.non_finite_grad_examples[-1] != event:
                self.non_finite_grad_examples.append(event)
                print(f"Non-finite gradient examples at step {event['step']}: {examples}")

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """React to logged training/evaluation losses."""

        if not logs:
            return control

        if "learning_rate" in logs:
            self.latest_learning_rate = self._to_float(logs["learning_rate"])

        if "grad_norm" in logs:
            grad_norm = self._to_float(logs["grad_norm"])
            self.latest_grad_norm = grad_norm
            if grad_norm is None or not math.isfinite(grad_norm):
                self.consecutive_nan_grad_norm_logs += 1
                print(
                    f"Guardrail warning: grad_norm is non-finite at step {state.global_step}: {grad_norm} "
                    f"({self.consecutive_nan_grad_norm_logs}/{self.max_consecutive_nan_grad_norm})"
                )
                if self.stop_on_nan_grad_norm and self.consecutive_nan_grad_norm_logs >= self.max_consecutive_nan_grad_norm:
                    self.fatal_stop = True
                    self.stop_reason = f"non_finite_grad_norm_at_step_{state.global_step}"
                    control.should_training_stop = True
                    raise GuardrailTrainingStop(
                        f"Training stopped because grad_norm stayed non-finite for "
                        f"{self.consecutive_nan_grad_norm_logs} logged step(s); latest step {state.global_step}: {grad_norm}"
                    )
            else:
                self.consecutive_nan_grad_norm_logs = 0

        if "eval_loss" in logs:
            eval_loss = self._to_float(logs["eval_loss"])
            self.latest_eval_loss = eval_loss
            if self.stop_on_nan_loss and (eval_loss is None or not math.isfinite(eval_loss)):
                self.fatal_stop = True
                self.stop_reason = f"non_finite_eval_loss_at_step_{state.global_step}"
                control.should_training_stop = True
                print(f"Guardrail stop: eval_loss is non-finite at step {state.global_step}: {eval_loss}")
                raise GuardrailTrainingStop(
                    f"Training stopped because eval_loss became non-finite at step {state.global_step}: {eval_loss}"
                )

        if "loss" not in logs:
            return control

        loss = self._to_float(logs["loss"])
        self.latest_logged_loss = loss
        if loss is None:
            return control

        if self.stop_on_nan_loss and not math.isfinite(loss):
            self.fatal_stop = True
            self.stop_reason = f"non_finite_logged_loss_at_step_{state.global_step}"
            control.should_training_stop = True
            print(f"Guardrail stop: logged train loss is non-finite at step {state.global_step}: {loss}")
            raise GuardrailTrainingStop(
                f"Training stopped because logged train loss became non-finite at step {state.global_step}: {loss}"
            )

        self.logged_train_loss_count += 1
        if self.best_logged_loss is None or loss < self.best_logged_loss:
            self.best_logged_loss = loss
            return control

        if not self._is_loss_explosion(loss):
            return control

        if not self._can_reduce_lr(state.global_step):
            return control

        if self.max_lr_reductions > 0 and len(self.lr_reduction_events) >= self.max_lr_reductions:
            self.stop_reason = f"max_lr_reductions_reached_at_step_{state.global_step}"
            control.should_training_stop = True
            print(
                "Guardrail stop: loss is still exploding after "
                f"{self.max_lr_reductions} LR reductions."
            )
            return control

        reduced = self._reduce_learning_rate(
            optimizer=kwargs.get("optimizer"),
            lr_scheduler=kwargs.get("lr_scheduler"),
            step=state.global_step,
            loss=loss,
        )
        if reduced:
            print(
                f"Guardrail action: loss spike detected at step {state.global_step} "
                f"(loss={loss:.6f}, best={self.best_logged_loss:.6f}); reduced learning rate."
            )
        else:
            self.stop_reason = self.stop_reason or f"loss_explosion_but_lr_could_not_be_reduced_at_step_{state.global_step}"
            control.should_training_stop = True
            print("Guardrail stop: loss exploded, but the learning rate could not be reduced further.")

        return control

    def _is_loss_explosion(self, loss: float) -> bool:
        """Return True when a logged loss is far above the best stable value."""

        if self.best_logged_loss is None:
            return False
        if self.logged_train_loss_count < self.loss_guard_min_logs:
            return False
        if self.best_logged_loss <= 0:
            return loss >= self.loss_explosion_abs_threshold

        relative_threshold = self.best_logged_loss * self.loss_explosion_factor
        absolute_threshold = self.loss_explosion_abs_threshold
        explosion_threshold = max(relative_threshold, absolute_threshold)
        return loss >= explosion_threshold

    def _can_reduce_lr(self, step: int) -> bool:
        """Enforce a cooldown so one bad log cannot trigger repeated reductions."""

        return (step - self.last_lr_reduction_step) >= self.lr_reduction_cooldown_steps

    def _reduce_learning_rate(
        self,
        optimizer: Any,
        lr_scheduler: Any,
        step: int,
        loss: float,
    ) -> bool:
        """Reduce optimizer and scheduler learning rates in-place."""

        if optimizer is None:
            self.stop_reason = f"loss_explosion_but_optimizer_unavailable_at_step_{step}"
            return False

        old_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
        new_lrs = [
            max(old_lr * self.lr_reduction_factor, self.min_learning_rate)
            for old_lr in old_lrs
        ]

        if old_lrs == new_lrs:
            return False

        for group, new_lr in zip(optimizer.param_groups, new_lrs):
            group["lr"] = new_lr

        # LR schedulers compute future rates from base_lrs. Reducing those base
        # values prevents the next scheduler step from undoing this guardrail.
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [
                max(float(base_lr) * self.lr_reduction_factor, self.min_learning_rate)
                for base_lr in lr_scheduler.base_lrs
            ]
        if lr_scheduler is not None and hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = new_lrs

        self.last_lr_reduction_step = step
        self.latest_learning_rate = new_lrs[0] if new_lrs else None
        self.lr_reduction_events.append(
            {
                "step": step,
                "loss": loss,
                "best_logged_loss": self.best_logged_loss,
                "old_learning_rates": old_lrs,
                "new_learning_rates": new_lrs,
                "lr_reduction_factor": self.lr_reduction_factor,
            }
        )
        return True

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Best-effort conversion for Python numbers and scalar tensors."""

        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class GuardedTrainer(Trainer):
    """Trainer variant that can immediately stop on non-finite raw loss."""

    def __init__(self, *args: Any, loss_guardrail_callback: Optional[LossDivergenceGuardrailCallback] = None, **kwargs: Any) -> None:
        self.loss_guardrail_callback = loss_guardrail_callback
        super().__init__(*args, **kwargs)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        loss = super().training_step(model, inputs, *args, **kwargs)
        if self.loss_guardrail_callback is not None:
            self.loss_guardrail_callback.observe_training_step_loss(loss, self.state)
            self.loss_guardrail_callback.observe_non_finite_parameter_grads(model, self.state)
        return loss


def resolve_lora_targets(model: torch.nn.Module, target_arg: str) -> List[str]:
    """
    Resolve LoRA target modules.

    The default explicit list is usually correct for recent MiniCPM-V language
    backbones. The 'auto' path is a convenience fallback that scans linear layer
    suffixes and prefers attention/MLP projection names while avoiding obvious
    vision and output heads.
    """

    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    excluded_fragments = (
        "vision",
        "visual",
        "image",
        "pixel",
        "embed",
        "lm_head",
        "output",
    )

    requested = [name.strip() for name in target_arg.split(",") if name.strip()]
    explicit_mode = target_arg.strip().lower() != "auto"
    requested_suffixes = set(requested) if explicit_mode else preferred

    discovered: List[str] = []
    for module_name, module in model.named_modules():
        type_name = type(module).__name__.lower()
        if "linear" not in type_name:
            continue
        if any(fragment in module_name.lower() for fragment in excluded_fragments):
            continue
        suffix = module_name.split(".")[-1]
        if suffix in requested_suffixes:
            # Returning full module names keeps PEFT from also matching vision
            # modules that happen to share q_proj/v_proj-style suffixes.
            discovered.append(module_name)

    if discovered:
        return sorted(discovered)

    if explicit_mode:
        print(
            "Warning: could not resolve explicit LoRA suffixes to text modules; "
            "passing the requested values directly to PEFT."
        )
        return requested

    raise ValueError(
        "Could not infer LoRA target modules automatically. "
        "Pass --lora_target_modules explicitly after inspecting model.named_modules()."
    )


def load_model_and_processor(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any]:
    """
    Load MiniCPM-V in 4-bit NF4 and attach LoRA adapters.

    QLoRA keeps the base model quantized and trains only small low-rank adapter
    matrices, which is the practical path for a 15 GB Tesla T4.
    """

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        # Decoder-only models need a pad token for batch padding. Reusing EOS is
        # standard for supervised fine-tuning when the tokenizer lacks PAD.
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model_torch_dtype = TORCH_DTYPE_BY_NAME[args.model_torch_dtype]
    bnb_compute_dtype = TORCH_DTYPE_BY_NAME[args.bnb_4bit_compute_dtype]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bnb_compute_dtype,
    )

    model = MODEL_AUTO_CLASS.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=model_torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    # Prepare quantized weights for adapter training. Newer PEFT accepts
    # gradient_checkpointing_kwargs; older PEFT does not, so we keep a fallback.
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    lora_targets = resolve_lora_targets(model, args.lora_target_modules)
    if len(lora_targets) > 20:
        preview = ", ".join(lora_targets[:8])
        print(f"LoRA target modules: {len(lora_targets)} text linear layers. First matches: {preview}, ...")
    else:
        print(f"LoRA target modules: {lora_targets}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def print_dataset_summary(name: str, records: Sequence[Dict[str, Any]]) -> None:
    """Print concise class counts so sampling mistakes are visible immediately."""

    counts = Counter(record["_emotion"] for record in records)
    print(f"{name}: {len(records)} samples")
    for emotion, count in sorted(counts.items()):
        print(f"  {emotion}: {count}")


def build_datasets(args: argparse.Namespace) -> Tuple[EmotionConversationDataset, Optional[EmotionConversationDataset], List[str]]:
    """Load, filter, sample, split, and wrap records in PyTorch Dataset objects."""

    emotion_labels = [normalize_emotion(label) for label in args.emotion_labels.split(",") if label.strip()]
    if not emotion_labels:
        raise ValueError("--emotion_labels cannot be empty.")

    raw_train = read_json_or_jsonl(args.train_json)
    train_records = validate_and_annotate_records(raw_train, emotion_labels)
    sampled_train = stratified_sample(train_records, args.sample_fraction, args.seed)

    if args.eval_json:
        raw_eval = read_json_or_jsonl(args.eval_json)
        eval_records = validate_and_annotate_records(raw_eval, emotion_labels)
        # The validation file is not reduced unless you explicitly pass a reduced
        # file. This keeps evaluation more informative.
        final_train_records = sampled_train
        final_eval_records = eval_records
    else:
        final_train_records, final_eval_records = stratified_train_eval_split(
            sampled_train,
            args.eval_ratio,
            args.seed,
        )

    prompt = build_user_prompt(emotion_labels, args.target_schema)
    train_dataset = EmotionConversationDataset(final_train_records, prompt, args.target_schema)
    eval_dataset = (
        EmotionConversationDataset(final_eval_records, prompt, args.target_schema)
        if final_eval_records
        else None
    )

    print_dataset_summary("Train", final_train_records)
    if final_eval_records:
        print_dataset_summary("Eval", final_eval_records)
    else:
        print("Eval: disabled because no validation records are available.")

    return train_dataset, eval_dataset, emotion_labels


def make_json_safe(value: Any) -> Any:
    """Convert common non-JSON Python values into JSON-serializable objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    return str(value)


def get_optimizer_learning_rates(optimizer: Any) -> List[float]:
    """Read current LR values from optimizer parameter groups."""

    if optimizer is None:
        return []
    return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]


def estimate_total_update_steps(args: argparse.Namespace, train_dataset: Dataset) -> int:
    """
    Estimate optimizer update steps for this run.

    The estimate is used only to keep evaluation/checkpoint intervals from being
    larger than the whole run, which would make early stopping and best-model
    reloading impossible on short one-epoch jobs.
    """

    if args.max_steps > 0:
        return args.max_steps

    micro_batches_per_epoch = math.ceil(len(train_dataset) / max(1, args.per_device_train_batch_size))
    updates_per_epoch = math.ceil(micro_batches_per_epoch / max(1, args.gradient_accumulation_steps))
    return max(1, math.ceil(updates_per_epoch * args.num_train_epochs))


def resolve_eval_and_save_steps(
    args: argparse.Namespace,
    train_dataset: Dataset,
    load_best_model_at_end: bool,
) -> Tuple[int, int, int]:
    """
    Choose effective eval/save intervals that are valid for Trainer.

    When load_best_model_at_end=True with step-based evaluation, Transformers
    requires save_steps to align with eval_steps. Matching them is the simplest
    robust choice. The eval interval is also capped to the estimated run length
    so one-epoch runs still produce an eval_loss and a best checkpoint.
    """

    estimated_total_steps = estimate_total_update_steps(args, train_dataset)
    effective_eval_steps = max(1, min(args.eval_steps, estimated_total_steps))
    effective_save_steps = effective_eval_steps if load_best_model_at_end else args.save_steps
    return effective_eval_steps, effective_save_steps, estimated_total_steps


def reload_best_checkpoint_if_available(
    trainer: Trainer,
    best_checkpoint: Optional[str],
    should_reload: bool,
) -> Dict[str, Any]:
    """
    Reload the best checkpoint and return truthful status metadata.

    Trainer usually reloads the best model automatically after a normal
    train() return when load_best_model_at_end=True. This helper still attempts
    an explicit reload so guardrail-interrupted runs get the best checkpoint too,
    but it records whether that actually succeeded.
    """

    result = {
        "requested": should_reload,
        "checkpoint": best_checkpoint,
        "succeeded": False,
        "skipped_reason": None,
        "error": None,
    }

    if not should_reload:
        result["skipped_reason"] = "disabled_or_no_eval_dataset"
        return result
    if not best_checkpoint:
        result["skipped_reason"] = "no_best_checkpoint_recorded"
        return result
    if not Path(best_checkpoint).exists():
        result["skipped_reason"] = "best_checkpoint_path_missing"
        return result

    load_method = getattr(trainer, "_load_best_model", None)
    if load_method is None:
        result["skipped_reason"] = "trainer_has_no_load_best_model_method"
        return result

    try:
        print(f"Reloading best checkpoint before final save: {best_checkpoint}")
        load_method()
    except Exception as exc:  # noqa: BLE001 - report reload failure without hiding the training summary.
        result["error"] = repr(exc)
        return result

    result["succeeded"] = True
    return result


def build_training_summary(
    args: argparse.Namespace,
    trainer: Trainer,
    train_metrics: Dict[str, Any],
    emotion_labels: Sequence[str],
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    guardrail_callback: LossDivergenceGuardrailCallback,
    has_eval: bool,
    train_interrupted: bool,
    effective_eval_steps: Optional[int],
    effective_save_steps: int,
    estimated_total_update_steps: int,
    best_checkpoint_reload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the final JSON object requested by the user.

    The hyperparameters section includes both the original settings and the
    final optimizer LR, which may differ after automatic LR reductions or after
    the scheduler reaches the end of training.
    """

    final_lrs = get_optimizer_learning_rates(getattr(trainer, "optimizer", None))
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)

    hyperparameters = {
        "model_id": args.model_id,
        "target_schema": args.target_schema,
        "emotion_labels": list(emotion_labels),
        "sample_fraction": args.sample_fraction,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "initial_learning_rate": args.learning_rate,
        "final_optimizer_learning_rates": final_lrs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": args.max_grad_norm,
        "optim": "paged_adamw_8bit",
        "fp16": args.fp16,
        "bf16": False,
        "model_torch_dtype": args.model_torch_dtype,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "gradient_checkpointing": args.gradient_checkpointing,
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
        "image_tensor_dtype": args.image_tensor_dtype,
        "max_length": args.max_length,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "requested_eval_steps": args.eval_steps,
        "effective_eval_steps": effective_eval_steps,
        "requested_save_steps": args.save_steps,
        "effective_save_steps": effective_save_steps,
        "estimated_total_update_steps": estimated_total_update_steps,
    }

    if guardrail_callback.stop_reason:
        training_status = "stopped_by_guardrail"
    elif train_interrupted:
        training_status = "interrupted"
    else:
        training_status = "completed"

    guardrails = {
        "stop_on_nan_loss": args.stop_on_nan_loss,
        "stop_on_nan_grad_norm": args.stop_on_nan_grad_norm,
        "max_consecutive_nan_grad_norm": args.max_consecutive_nan_grad_norm,
        "debug_non_finite_grads": args.debug_non_finite_grads,
        "max_non_finite_grad_names": args.max_non_finite_grad_names,
        "loss_guard_min_logs": args.loss_guard_min_logs,
        "loss_explosion_factor": args.loss_explosion_factor,
        "loss_explosion_abs_threshold": args.loss_explosion_abs_threshold,
        "lr_reduction_factor": args.lr_reduction_factor,
        "min_learning_rate": args.min_learning_rate,
        "lr_reduction_cooldown_steps": args.lr_reduction_cooldown_steps,
        "max_lr_reductions": args.max_lr_reductions,
        "early_stopping_enabled": has_eval and args.early_stopping_patience > 0 and args.load_best_model_at_end,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "load_best_model_at_end": has_eval and args.load_best_model_at_end,
        "best_checkpoint_reload": best_checkpoint_reload,
        "stop_reason": guardrail_callback.stop_reason,
        "train_interrupted_by_guardrail": train_interrupted,
        "best_logged_train_loss": guardrail_callback.best_logged_loss,
        "latest_logged_train_loss": guardrail_callback.latest_logged_loss,
        "latest_eval_loss": guardrail_callback.latest_eval_loss,
        "latest_learning_rate_seen_in_logs": guardrail_callback.latest_learning_rate,
        "latest_grad_norm": guardrail_callback.latest_grad_norm,
        "consecutive_nan_grad_norm_logs": guardrail_callback.consecutive_nan_grad_norm_logs,
        "non_finite_grad_examples": guardrail_callback.non_finite_grad_examples,
        "lr_reduction_events": guardrail_callback.lr_reduction_events,
    }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": args.output_dir,
        "training_finished": training_status == "completed",
        "training_status": training_status,
        "hyperparameters_used": hyperparameters,
        "guardrails": guardrails,
        "trainer_state": {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": best_checkpoint,
            "log_history": trainer.state.log_history,
        },
        "train_metrics": train_metrics,
    }


def write_training_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    """Persist the final training JSON report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    """Main training orchestration."""

    args = parse_args()
    validate_guardrail_args(args)
    set_seed(args.seed)

    # Keeps CUDA allocation behavior a little less fragmented on long runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_dataset, eval_dataset, emotion_labels = build_datasets(args)
    model, processor = load_model_and_processor(args)

    collator = MiniCPMVDataCollator(
        processor=processor,
        max_length=args.max_length,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        image_tensor_dtype=args.image_tensor_dtype,
    )

    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    load_best_model_at_end = has_eval and args.load_best_model_at_end
    effective_eval_steps, effective_save_steps, estimated_total_update_steps = resolve_eval_and_save_steps(
        args=args,
        train_dataset=train_dataset,
        load_best_model_at_end=load_best_model_at_end,
    )
    if has_eval and effective_eval_steps != args.eval_steps:
        print(
            f"Adjusted eval_steps from {args.eval_steps} to {effective_eval_steps} "
            "so this run produces at least one eval_loss."
        )
    if load_best_model_at_end and effective_save_steps != args.save_steps:
        print(
            f"Adjusted save_steps from {args.save_steps} to {effective_save_steps} "
            "so best-checkpoint saving aligns with evaluation."
        )

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": effective_save_steps,
        "save_total_limit": 2,
        "eval_steps": effective_eval_steps if has_eval else None,
        "fp16": args.fp16,
        "bf16": False,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
    }

    # Transformers renamed evaluation_strategy to eval_strategy. Supporting
    # both keeps the script usable across the versions people commonly install.
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_arg_names:
        training_kwargs["eval_strategy"] = "steps" if has_eval else "no"
    else:
        training_kwargs["evaluation_strategy"] = "steps" if has_eval else "no"
    if "save_strategy" in training_arg_names:
        training_kwargs["save_strategy"] = "steps"
    if "load_best_model_at_end" in training_arg_names:
        training_kwargs["load_best_model_at_end"] = load_best_model_at_end
    if load_best_model_at_end:
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False
    if "logging_nan_inf_filter" in training_arg_names:
        # If this stays True, Transformers may hide NaN/Inf values in logs,
        # which defeats the explicit stop-on-NaN guardrail.
        training_kwargs["logging_nan_inf_filter"] = False
    if "logging_first_step" in training_arg_names:
        training_kwargs["logging_first_step"] = True

    training_args = TrainingArguments(**training_kwargs)

    guardrail_callback = LossDivergenceGuardrailCallback(
        stop_on_nan_loss=args.stop_on_nan_loss,
        stop_on_nan_grad_norm=args.stop_on_nan_grad_norm,
        max_consecutive_nan_grad_norm=args.max_consecutive_nan_grad_norm,
        debug_non_finite_grads=args.debug_non_finite_grads,
        max_non_finite_grad_names=args.max_non_finite_grad_names,
        loss_guard_min_logs=args.loss_guard_min_logs,
        loss_explosion_factor=args.loss_explosion_factor,
        loss_explosion_abs_threshold=args.loss_explosion_abs_threshold,
        lr_reduction_factor=args.lr_reduction_factor,
        min_learning_rate=args.min_learning_rate,
        lr_reduction_cooldown_steps=args.lr_reduction_cooldown_steps,
        max_lr_reductions=args.max_lr_reductions,
    )

    callbacks: List[TrainerCallback] = [guardrail_callback]
    if has_eval and args.early_stopping_patience > 0 and load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    elif has_eval and args.early_stopping_patience > 0:
        print("Early stopping disabled because --load_best_model_at_end is false.")
    elif not has_eval:
        print("Early stopping disabled because no validation records are available.")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": callbacks,
    }

    # New Transformers versions prefer processing_class; older ones still use
    # tokenizer. The processor is saved manually after training either way.
    trainer_arg_names = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_arg_names:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_arg_names:
        trainer_kwargs["tokenizer"] = getattr(processor, "tokenizer", None)

    trainer = GuardedTrainer(**trainer_kwargs, loss_guardrail_callback=guardrail_callback)

    train_metrics: Dict[str, Any] = {}
    train_interrupted = False
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        train_metrics = train_result.metrics
    except GuardrailTrainingStop as exc:
        train_interrupted = True
        train_metrics = {"guardrail_stop": str(exc)}
        print(f"Training stopped by guardrail: {exc}")

    best_checkpoint = trainer.state.best_model_checkpoint
    best_checkpoint_reload = reload_best_checkpoint_if_available(
        trainer=trainer,
        best_checkpoint=best_checkpoint,
        should_reload=load_best_model_at_end,
    )
    if best_checkpoint_reload["error"]:
        print(f"Best-checkpoint reload failed: {best_checkpoint_reload['error']}")
    elif best_checkpoint_reload["skipped_reason"]:
        print(f"Best-checkpoint reload skipped: {best_checkpoint_reload['skipped_reason']}")

    # For QLoRA, this saves the adapter weights, not a full merged model. If the
    # run stopped on a non-finite loss before any good checkpoint existed, avoid
    # overwriting the output directory with unstable weights.
    should_save_final_model = not guardrail_callback.fatal_stop or bool(best_checkpoint_reload["succeeded"])
    if should_save_final_model:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Saved QLoRA adapter and processor to: {args.output_dir}")
    else:
        print(
            "Skipped final adapter save because training stopped on non-finite loss "
            "and no best checkpoint was successfully reloaded."
        )

    summary = build_training_summary(
        args=args,
        trainer=trainer,
        train_metrics=train_metrics,
        emotion_labels=emotion_labels,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        guardrail_callback=guardrail_callback,
        has_eval=has_eval,
        train_interrupted=train_interrupted,
        effective_eval_steps=effective_eval_steps if has_eval else None,
        effective_save_steps=effective_save_steps,
        estimated_total_update_steps=estimated_total_update_steps,
        best_checkpoint_reload=best_checkpoint_reload,
    )
    summary["final_model_saved"] = should_save_final_model
    summary["summary_json_path"] = args.training_summary_json or str(Path(args.output_dir) / "training_summary.json")

    if trainer.is_world_process_zero():
        write_training_summary(summary, summary["summary_json_path"])
        print("Final training summary JSON:")
        print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
