#!/usr/bin/env python3
"""
Evaluate a fine-tuned MiniCPM-V QLoRA adapter on the FERPlus test set.

Expected FERPlus test record format, either JSON array or JSONL:

{
    "image": "/workspace/datasets/images/test/happiness/img.png",
    "output": "{\"primary_emotion\": \"happiness\"}"
}

The script:
1. Loads the FERPlus test set from --test_json.
2. Loads the base MiniCPM-V model plus the trained QLoRA adapter.
3. Generates one emotion prediction for every test image.
4. Computes common classification metrics.
5. Saves the metrics into a JSON file for later inspection.

Example:

python evaluate_minicpmv_emotion.py \
  --test_json /workspace/datasets/ferplus_test.json \
  --adapter_dir ./minicpmv-ferplus-emotion-qlora \
  --metrics_json ./ferplus_test_metrics.json \
  --predictions_jsonl ./ferplus_test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    # Recent MiniCPM-V checkpoints are available through this auto class.
    from transformers import AutoModelForImageTextToText

    MODEL_AUTO_CLASS = AutoModelForImageTextToText
except ImportError:
    # Older MiniCPM-V environments may still need the remote-code AutoModel path.
    from transformers import AutoModel

    MODEL_AUTO_CLASS = AutoModel


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


def parse_args() -> argparse.Namespace:
    """Define all evaluation options explicitly from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate a MiniCPM-V QLoRA emotion adapter on FERPlus test data."
    )

    # The test set path is required because metrics are only meaningful against
    # a held-out labeled split.
    parser.add_argument(
        "--test_json",
        required=True,
        help="Path to the FERPlus test JSON/JSONL file.",
    )
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="Directory containing the trained QLoRA adapter saved by the training script.",
    )
    parser.add_argument(
        "--metrics_json",
        default="./ferplus_test_metrics.json",
        help="Where to save the evaluation metrics JSON file.",
    )
    parser.add_argument(
        "--predictions_jsonl",
        default=None,
        help="Optional path for per-sample predictions. Useful for error analysis.",
    )
    parser.add_argument(
        "--checkpoint_jsonl",
        default=None,
        help=(
            "Prediction checkpoint JSONL. Defaults to <predictions_stem>.checkpoint.jsonl "
            "when --predictions_jsonl is set, otherwise <metrics_stem>.checkpoint.jsonl."
        ),
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Write partial metrics every N newly evaluated samples. Predictions are checkpointed every sample.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by skipping samples already present in --checkpoint_jsonl. Use --no-resume_from_checkpoint to start fresh.",
    )
    parser.add_argument(
        "--partial_metrics_json",
        default=None,
        help="Where to save partial metrics during evaluation. Defaults to <metrics_stem>.partial.json.",
    )

    # The base model is usually read from adapter_config.json. The argument is
    # still available in case the adapter config is missing or you want to force
    # a specific compatible checkpoint.
    parser.add_argument(
        "--model_id",
        default=None,
        help=f"Base MiniCPM-V model id. Defaults to the adapter config, then {DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom code from the MiniCPM-V checkpoint repository.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="Attention backend. sdpa is the safest default for a Tesla T4.",
    )

    # These prompt/schema options should match the fine-tuning run. If you used
    # the default training command, keep target_schema=group.
    parser.add_argument(
        "--target_schema",
        choices=("group", "primary"),
        default="group",
        help="Schema expected from the model; should match training.",
    )
    parser.add_argument(
        "--emotion_labels",
        default=",".join(DEFAULT_EMOTIONS),
        help="Comma-separated emotion labels allowed during evaluation.",
    )

    # Generation settings. Greedy decoding is used by default so metrics are
    # deterministic and repeatable across runs.
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
        help="Maximum number of generated tokens per test image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 uses greedy decoding; values above 0 enable sampling.",
    )
    parser.add_argument(
        "--downsample_mode",
        default="16x",
        help="MiniCPM-V image-token downsampling mode. Match the training script when possible.",
    )
    parser.add_argument(
        "--max_slice_nums",
        type=int,
        default=4,
        help="Maximum image slices. Lower values reduce VRAM usage.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum prompt token length.",
    )

    # Optional limit is useful for a fast smoke test before running the full
    # FERPlus test split.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of test samples to evaluate.",
    )

    return parser.parse_args()


def normalize_emotion(value: str) -> str:
    """Normalize labels so JSON, folders, and model text compare reliably."""

    return value.strip().lower().replace(" ", "_")


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load either a JSON array file or a JSONL file.

    Dataset conversion scripts often differ on this detail, so accepting both
    formats makes evaluation less fragile.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty.")

    if text[0] == "[":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return records

    if text[0] == "{":
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return loaded
            if isinstance(loaded, dict) and "image" in loaded:
                return [loaded]
        except json.JSONDecodeError:
            # If the full file is not one JSON object, parse it as JSONL below.
            pass

    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
        records.append(record)

    return records


def parse_output_payload(raw_output: Any) -> Dict[str, Any]:
    """
    Parse the ground-truth output field from the dataset.

    Your FERPlus records store a JSON object inside a string, but this function
    also accepts already-parsed dictionaries.
    """

    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed output is not a JSON object: {raw_output}")
        return parsed
    raise ValueError(f"Unsupported output type: {type(raw_output).__name__}")


def extract_true_emotion(record: Dict[str, Any]) -> str:
    """
    Extract the ground-truth emotion from a FERPlus record.

    The preferred source is output.primary_emotion. If that is absent, the code
    falls back to the image parent folder, e.g. /test/happiness/img.png.
    """

    output = parse_output_payload(record.get("output", {}))
    emotion = output.get("primary_emotion") or output.get("group_emotion") or output.get("emotion") or output.get("label")

    if emotion is None:
        image_path = record.get("image")
        if not image_path:
            raise ValueError(f"Cannot find a label in this record: {record}")
        emotion = Path(str(image_path)).parent.name

    return normalize_emotion(str(emotion))


def validate_test_records(
    records: Sequence[Dict[str, Any]],
    allowed_emotions: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Keep only test records with an existing image and an allowed emotion label.

    Unsupported FERPlus labels, such as unknown/non-face if present in your
    preprocessing, are skipped because they are not valid prediction classes.
    """

    allowed = {normalize_emotion(label) for label in allowed_emotions}
    validated: List[Dict[str, Any]] = []
    skipped_missing_image = 0
    skipped_label = Counter()

    for record in records:
        image_path = Path(str(record.get("image", "")))
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        true_emotion = extract_true_emotion(record)
        if true_emotion not in allowed:
            skipped_label[true_emotion] += 1
            continue

        item = dict(record)
        item["_true_emotion"] = true_emotion
        validated.append(item)

    if skipped_missing_image:
        print(f"Skipped {skipped_missing_image} test records with missing image files.")
    if skipped_label:
        print(f"Skipped labels outside --emotion_labels: {dict(skipped_label)}")
    if not validated:
        raise ValueError("No valid test records remain after filtering.")

    return validated


def build_user_prompt(emotion_labels: Sequence[str], schema: str) -> str:
    """
    Build the same instruction style used during fine-tuning.

    Keeping the prompt consistent reduces evaluation noise: we want to measure
    the adapter, not a prompt mismatch.
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


def make_conversation(image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    """Create the multimodal chat message consumed by the MiniCPM-V processor."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def apply_chat_template_for_generation(
    processor: Any,
    conversation: List[Dict[str, Any]],
    downsample_mode: str,
    max_slice_nums: int,
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenize one multimodal prompt with compatibility fallbacks.

    Recent Transformers versions require processor-call options to live in
    processor_kwargs. Keep only chat-template controls at the top level so
    Transformers does not warn on every sample.
    """

    base_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    processor_kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": max_length,
        "downsample_mode": downsample_mode,
        "max_slice_nums": max_slice_nums,
    }
    attempts = [
        {
            **base_kwargs,
            "processor_kwargs": processor_kwargs,
        },
    ]

    last_error: Optional[TypeError] = None
    for kwargs in attempts:
        try:
            return processor.apply_chat_template([conversation], **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def move_inputs_to_model_device(inputs: Dict[str, Any], model: torch.nn.Module) -> Dict[str, Any]:
    """
    Move tensor inputs to the first model device.

    device_map='auto' can shard large models, but for a single T4 this usually
    places the model on cuda:0. Non-tensor values are kept unchanged.
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def decode_generated_text(
    processor: Any,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> str:
    """
    Decode only the newly generated answer tokens, not the original prompt.

    Some model.generate outputs include the prompt followed by the answer. By
    slicing off input_ids length, we keep just the assistant response.
    """

    prompt_length = input_ids.shape[-1]
    answer_ids = generated_ids[:, prompt_length:]
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def load_model_and_processor(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any, str]:
    """
    Load the base MiniCPM-V model in 4-bit mode and attach the QLoRA adapter.

    This mirrors the memory-saving setup used for training, which is important
    on the 15 GB Tesla T4 in an AWS g4dn.xlarge instance.
    """

    adapter_dir = Path(args.adapter_dir)
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_id = args.model_id or peft_config.base_model_name_or_path or DEFAULT_MODEL_ID

    try:
        # The training script saves the processor into the adapter directory.
        # Loading it from there preserves tokenizer/template details from the
        # fine-tuning run.
        processor = AutoProcessor.from_pretrained(
            adapter_dir,
            trust_remote_code=args.trust_remote_code,
        )
    except OSError:
        # If only adapter weights were copied, fall back to the base model's
        # processor. This is usually equivalent, but the adapter copy is better
        # when available.
        processor = AutoProcessor.from_pretrained(
            base_model_id,
            trust_remote_code=args.trust_remote_code,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = MODEL_AUTO_CLASS.from_pretrained(
        base_model_id,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, processor, str(base_model_id)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse the model response as JSON.

    The ideal response is already pure JSON. The fallback extracts the first
    {...} block in case the model adds accidental surrounding text.
    """

    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def majority_vote(labels: Sequence[str]) -> Optional[str]:
    """
    Return the most frequent emotion in a subject list.

    This is a fallback for group-schema outputs missing group_emotion but
    containing valid per-subject emotions.
    """

    if not labels:
        return None
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def parse_predicted_emotion(
    raw_text: str,
    allowed_emotions: Sequence[str],
    target_schema: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Convert model text into a single emotion label for metric computation.

    For FERPlus, the test image has one ground-truth label. If the model was
    trained with the group schema, group_emotion is compared against that label.
    """

    allowed = {normalize_emotion(label) for label in allowed_emotions}
    parsed = extract_json_object(raw_text)
    parse_info: Dict[str, Any] = {
        "parsed_json": parsed,
        "parse_status": "json_ok" if parsed is not None else "json_failed",
    }

    prediction: Optional[str] = None
    if parsed is not None:
        if target_schema == "primary":
            prediction = parsed.get("primary_emotion") or parsed.get("emotion") or parsed.get("label")
        else:
            prediction = parsed.get("group_emotion") or parsed.get("primary_emotion")
            if prediction is None and isinstance(parsed.get("subjects"), list):
                subject_emotions = []
                for subject in parsed["subjects"]:
                    if isinstance(subject, dict) and subject.get("emotion") is not None:
                        subject_emotions.append(normalize_emotion(str(subject["emotion"])))
                prediction = majority_vote(subject_emotions)

    if prediction is not None:
        normalized_prediction = normalize_emotion(str(prediction))
        if normalized_prediction in allowed:
            parse_info["parse_status"] = "valid_label"
            return normalized_prediction, parse_info
        parse_info["parse_status"] = "label_not_allowed"
        parse_info["raw_label"] = normalized_prediction

    # Last-resort fallback: look for one of the known emotion words in the raw
    # text. This keeps metrics computable while still recording that the output
    # was not valid JSON.
    raw_lower = normalize_emotion(raw_text)
    found_labels = [label for label in allowed if re.search(rf"\b{re.escape(label)}\b", raw_lower)]
    if len(found_labels) == 1:
        parse_info["parse_status"] = "text_label_fallback"
        return found_labels[0], parse_info

    return None, parse_info


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    processor: Any,
    image_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    """Run MiniCPM-V generation for one test image and return raw text."""

    image = Image.open(image_path).convert("RGB")
    conversation = make_conversation(image, prompt)
    inputs = apply_chat_template_for_generation(
        processor=processor,
        conversation=conversation,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        max_length=args.max_length,
    )
    inputs = move_inputs_to_model_device(inputs, model)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature if args.temperature > 0.0 else None,
        "pad_token_id": getattr(getattr(processor, "tokenizer", None), "pad_token_id", None),
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    # Some MiniCPM-V versions expect downsample_mode during forward/generation.
    try:
        generated_ids = model.generate(
            **inputs,
            downsample_mode=args.downsample_mode,
            **generation_kwargs,
        )
    except TypeError:
        generated_ids = model.generate(**inputs, **generation_kwargs)

    return decode_generated_text(processor, inputs["input_ids"], generated_ids)


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid ZeroDivisionError and match sklearn's zero_division=0 behavior."""

    return numerator / denominator if denominator else 0.0


def compute_classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[Optional[str]],
    emotion_labels: Sequence[str],
) -> Dict[str, Any]:
    """
    Compute common single-label classification metrics.

    Metrics included:
    - accuracy
    - balanced accuracy
    - macro precision/recall/F1
    - weighted precision/recall/F1
    - micro precision/recall/F1
    - per-class precision/recall/F1/support
    - confusion matrix

    Invalid/unparsed predictions are counted as incorrect and tracked under
    prediction_status, but they are not added as an extra class.
    """

    labels = [normalize_emotion(label) for label in emotion_labels]
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = [[0 for _ in labels] for _ in labels]

    total = len(true_labels)
    correct = 0
    invalid_predictions = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_label = normalize_emotion(true_label)
        predicted_label = normalize_emotion(predicted_label) if predicted_label else None

        if predicted_label is None or predicted_label not in label_to_index:
            invalid_predictions += 1
            continue

        if true_label == predicted_label:
            correct += 1

        if true_label in label_to_index:
            confusion[label_to_index[true_label]][label_to_index[predicted_label]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    macro_precision_values: List[float] = []
    macro_recall_values: List[float] = []
    macro_f1_values: List[float] = []
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0
    total_support = 0

    for index, label in enumerate(labels):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(labels)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(labels)) if col != index)
        support = sum(confusion[index])

        # Invalid predictions for this true class are false negatives. They do
        # not appear in the confusion matrix columns, so add them to fn.
        invalid_for_class = sum(
            1
            for true_label, predicted_label in zip(true_labels, predicted_labels)
            if normalize_emotion(true_label) == label and (predicted_label is None or normalize_emotion(predicted_label) not in label_to_index)
        )
        fn += invalid_for_class
        support += invalid_for_class

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        macro_precision_values.append(precision)
        macro_recall_values.append(recall)
        macro_f1_values.append(f1)
        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support
        total_support += support

    accuracy = safe_divide(correct, total)
    macro_precision = safe_divide(sum(macro_precision_values), len(labels))
    macro_recall = safe_divide(sum(macro_recall_values), len(labels))
    macro_f1 = safe_divide(sum(macro_f1_values), len(labels))
    weighted_precision = safe_divide(weighted_precision_sum, total_support)
    weighted_recall = safe_divide(weighted_recall_sum, total_support)
    weighted_f1 = safe_divide(weighted_f1_sum, total_support)

    # In ordinary single-label multiclass classification, micro precision,
    # recall, and F1 collapse to accuracy. Generative models can also emit
    # invalid labels, so we count those as false negatives without inventing a
    # fake prediction class.
    true_positive_total = correct
    false_positive_total = sum(
        confusion[row][col]
        for row in range(len(labels))
        for col in range(len(labels))
        if row != col
    )
    false_negative_total = false_positive_total + invalid_predictions
    micro_precision = safe_divide(true_positive_total, true_positive_total + false_positive_total)
    micro_recall = safe_divide(true_positive_total, true_positive_total + false_negative_total)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "overall": {
            "num_samples": total,
            "num_correct": correct,
            "num_incorrect": total - correct,
            "num_invalid_predictions": invalid_predictions,
            "invalid_prediction_rate": safe_divide(invalid_predictions, total),
            "accuracy": accuracy,
            "balanced_accuracy": macro_recall,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion,
            "rows": "true_labels",
            "columns": "predicted_labels",
            "note": "Invalid or unparsed predictions are counted as errors but are not represented as a prediction column.",
        },
    }


def save_predictions_jsonl(path: str | Path, predictions: Sequence[Dict[str, Any]]) -> None:
    """Write one JSON object per evaluated sample for later error analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def add_suffix_before_extension(path: str | Path, suffix: str) -> Path:
    """
    Insert a suffix before the final file extension.

    Example: val_predictions.jsonl + .checkpoint.jsonl becomes
    val_predictions.checkpoint.jsonl instead of val_predictions.jsonl.checkpoint.
    """

    path = Path(path)
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def legacy_checkpoint_path(args: argparse.Namespace) -> Path:
    """Return the old checkpoint default used by earlier versions of this script."""

    if args.predictions_jsonl:
        return Path(f"{args.predictions_jsonl}.checkpoint")
    return Path(f"{args.metrics_json}.checkpoint.jsonl")


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    """Choose the checkpoint JSONL path used for resume."""

    if args.checkpoint_jsonl:
        return Path(args.checkpoint_jsonl)

    if args.predictions_jsonl:
        checkpoint_path = add_suffix_before_extension(args.predictions_jsonl, ".checkpoint.jsonl")
    else:
        checkpoint_path = add_suffix_before_extension(args.metrics_json, ".checkpoint.jsonl")

    old_path = legacy_checkpoint_path(args)
    if (
        args.resume_from_checkpoint
        and old_path != checkpoint_path
        and old_path.exists()
        and old_path.stat().st_size > 0
        and not checkpoint_path.exists()
    ):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_path, checkpoint_path)
        print(f"Migrated legacy checkpoint from {old_path} to {checkpoint_path}")

    return checkpoint_path


def resolve_partial_metrics_path(args: argparse.Namespace) -> Path:
    """Choose where periodic partial metrics should be written."""

    if args.partial_metrics_json:
        return Path(args.partial_metrics_json)
    return add_suffix_before_extension(args.metrics_json, ".partial.json")


def load_prediction_checkpoint(path: str | Path) -> Dict[int, Dict[str, Any]]:
    """
    Load previously completed predictions from a JSONL checkpoint.

    If the last line is partially written because a process was interrupted at
    exactly the wrong moment, that line is skipped instead of failing the resume.
    """

    path = Path(path)
    if not path.exists():
        return {}

    loaded: Dict[int, Dict[str, Any]] = {}
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                print(f"Warning: skipped invalid checkpoint line {line_number} in {path}.")
                continue
            if not isinstance(item, dict) or "index" not in item:
                skipped += 1
                print(f"Warning: skipped malformed checkpoint line {line_number} in {path}.")
                continue
            loaded[int(item["index"])] = item

    print(f"Loaded {len(loaded)} completed predictions from checkpoint: {path}")
    if skipped:
        print(f"Skipped {skipped} malformed checkpoint lines.")
    return loaded


def align_checkpoint_with_records(
    checkpoint_items: Dict[int, Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Keep only checkpoint entries that still match the current test set.

    This prevents accidentally resuming with predictions from a different
    --test_json, --limit, or dataset ordering.
    """

    aligned: Dict[int, Dict[str, Any]] = {}
    skipped = 0

    for index, item in checkpoint_items.items():
        if index < 0 or index >= len(test_records):
            skipped += 1
            continue

        current_image = str(test_records[index].get("image", ""))
        checkpoint_image = str(item.get("image", ""))
        if checkpoint_image != current_image:
            skipped += 1
            continue

        normalized_item = dict(item)
        normalized_item["true_emotion"] = test_records[index]["_true_emotion"]
        normalized_item["is_correct"] = normalized_item.get("predicted_emotion") == normalized_item["true_emotion"]
        aligned[index] = normalized_item

    if skipped:
        print(f"Skipped {skipped} checkpoint entries that do not match the current test set.")
    return aligned


def prepare_checkpoint_file(path: str | Path, resume: bool) -> None:
    """Create the checkpoint file, truncating it only for a fresh non-resume run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        path.touch(exist_ok=True)
        return

    with path.open("w", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def append_prediction_checkpoint(path: str | Path, prediction: Dict[str, Any]) -> None:
    """Append one completed prediction and force it to disk immediately."""

    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_metrics_report(
    predictions: Sequence[Dict[str, Any]],
    emotion_labels: Sequence[str],
    args: argparse.Namespace,
    base_model_id: str,
    status: str,
    total_samples: int,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """Build a metrics JSON report from completed predictions."""

    true_labels = [item["true_emotion"] for item in predictions]
    predicted_labels = [item.get("predicted_emotion") for item in predictions]
    parse_status_counts = Counter(item.get("parse_status", "unknown") for item in predictions)

    metrics = compute_classification_metrics(true_labels, predicted_labels, emotion_labels)
    metrics["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "test_json": str(args.test_json),
        "adapter_dir": str(args.adapter_dir),
        "base_model_id": base_model_id,
        "target_schema": args.target_schema,
        "emotion_labels": list(emotion_labels),
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "limit": args.limit,
        "total_samples": total_samples,
        "completed_samples": len(predictions),
        "remaining_samples": max(0, total_samples - len(predictions)),
        "checkpoint_jsonl": str(checkpoint_path),
    }
    metrics["prediction_parsing"] = dict(parse_status_counts)
    return metrics


def save_metrics_json(path: str | Path, metrics: Dict[str, Any]) -> None:
    """Write metrics JSON, creating the parent directory if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    """Run end-to-end evaluation and save the metrics report."""

    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint_every must be a positive integer.")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    emotion_labels = [normalize_emotion(label) for label in args.emotion_labels.split(",") if label.strip()]
    if not emotion_labels:
        raise ValueError("--emotion_labels cannot be empty.")

    raw_records = read_json_or_jsonl(args.test_json)
    test_records = validate_test_records(raw_records, emotion_labels)
    if args.limit is not None:
        test_records = test_records[: args.limit]

    print(f"Loaded {len(test_records)} valid FERPlus test samples.")
    model, processor, base_model_id = load_model_and_processor(args)
    prompt = build_user_prompt(emotion_labels, args.target_schema)

    checkpoint_path = resolve_checkpoint_path(args)
    partial_metrics_path = resolve_partial_metrics_path(args)
    loaded_checkpoint = (
        load_prediction_checkpoint(checkpoint_path)
        if args.resume_from_checkpoint
        else {}
    )
    completed_by_index = align_checkpoint_with_records(loaded_checkpoint, test_records)
    predictions_for_file: List[Dict[str, Any]] = [
        completed_by_index[index]
        for index in sorted(completed_by_index)
        if 0 <= index < len(test_records)
    ]

    if predictions_for_file:
        print(f"Resuming evaluation: {len(predictions_for_file)}/{len(test_records)} samples already completed.")
    else:
        if not args.resume_from_checkpoint and checkpoint_path.exists():
            print(f"Starting fresh evaluation; overwriting old checkpoint: {checkpoint_path}")
        print(f"Writing evaluation checkpoint to: {checkpoint_path}")

    prepare_checkpoint_file(checkpoint_path, resume=args.resume_from_checkpoint)

    newly_completed = 0
    interrupted = False

    try:
        for index, record in enumerate(test_records):
            if index in completed_by_index:
                continue

            true_label = record["_true_emotion"]
            raw_prediction = predict_one(
                model=model,
                processor=processor,
                image_path=record["image"],
                prompt=prompt,
                args=args,
            )
            predicted_label, parse_info = parse_predicted_emotion(
                raw_text=raw_prediction,
                allowed_emotions=emotion_labels,
                target_schema=args.target_schema,
            )

            prediction_item = {
                "index": index,
                "image": record["image"],
                "true_emotion": true_label,
                "predicted_emotion": predicted_label,
                "is_correct": predicted_label == true_label,
                "parse_status": parse_info["parse_status"],
                "raw_model_output": raw_prediction,
                "parsed_model_output": parse_info.get("parsed_json"),
            }

            append_prediction_checkpoint(checkpoint_path, prediction_item)
            completed_by_index[index] = prediction_item
            predictions_for_file.append(prediction_item)
            newly_completed += 1

            completed_count = len(completed_by_index)
            if (
                newly_completed % args.checkpoint_every == 0
                or completed_count == len(test_records)
            ):
                partial_metrics = build_metrics_report(
                    predictions=sorted(predictions_for_file, key=lambda item: item["index"]),
                    emotion_labels=emotion_labels,
                    args=args,
                    base_model_id=base_model_id,
                    status="partial" if completed_count < len(test_records) else "completed",
                    total_samples=len(test_records),
                    checkpoint_path=checkpoint_path,
                )
                save_metrics_json(partial_metrics_path, partial_metrics)
                print(
                    f"Evaluated {completed_count}/{len(test_records)} samples. "
                    f"Checkpoint: {checkpoint_path}. Partial metrics: {partial_metrics_path}"
                )

    except KeyboardInterrupt:
        interrupted = True
        print("Evaluation interrupted. Saving partial metrics before exiting.")

    predictions_for_file = sorted(predictions_for_file, key=lambda item: item["index"])
    status = "interrupted" if interrupted else ("completed" if len(predictions_for_file) == len(test_records) else "partial")
    metrics = build_metrics_report(
        predictions=predictions_for_file,
        emotion_labels=emotion_labels,
        args=args,
        base_model_id=base_model_id,
        status=status,
        total_samples=len(test_records),
        checkpoint_path=checkpoint_path,
    )

    metrics_path = Path(args.metrics_json)
    save_metrics_json(metrics_path, metrics)
    print(f"Saved metrics JSON to: {metrics_path}")

    if args.predictions_jsonl:
        save_predictions_jsonl(args.predictions_jsonl, predictions_for_file)
        print(f"Saved per-sample predictions to: {args.predictions_jsonl}")

    if interrupted:
        print(f"Resume later with the same command. Completed samples are stored in: {checkpoint_path}")


if __name__ == "__main__":
    main()
