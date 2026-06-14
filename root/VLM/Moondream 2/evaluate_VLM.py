
"""
Evaluate a fine-tuned Moondream model on the FERPlus test set.

Expected test record format, either JSON array or JSONL:

{
    "image": "/workspace/datasets/images/test/happiness/img.png",
    "output": "{\"primary_emotion\": \"happiness\"}"
}
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
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "vikhyatk/moondream2"
DEFAULT_REVISION = "2024-04-02"
DEFAULT_LOCAL_MODEL_DIR = "moondream2-base"
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
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Moondream emotion model.")

    parser.add_argument("--test_json", required=True, help="Path to the FERPlus test JSON/JSONL file.")
    parser.add_argument(
        "--adapter_dir",
        "--model_dir",
        dest="model_dir",
        required=True,
        help="Directory containing the fine-tuned Moondream model saved by main.py.",
    )
    parser.add_argument("--metrics_json", default="./ferplus_test_metrics.json")
    parser.add_argument("--predictions_jsonl", default=None)
    parser.add_argument("--checkpoint_jsonl", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial_metrics_json", default=None)

    parser.add_argument("--base_model_dir", default=DEFAULT_LOCAL_MODEL_DIR, help="Fallback base Moondream directory.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Fallback HF id if local loading needs base code.")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--torch_dtype", default="auto", choices=("auto", "float16", "float32", "bfloat16"))

    parser.add_argument("--target_schema", choices=("group", "primary"), default="group")
    parser.add_argument("--emotion_labels", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)


    parser.add_argument("--downsample_mode", default=None)
    parser.add_argument("--max_slice_nums", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--attn_implementation", default=None)

    return parser.parse_args()


def normalize_emotion(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty.")

    if text[0] == "[":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return records

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
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed output is not a JSON object: {raw_output}")
        return parsed
    raise ValueError(f"Unsupported output type: {type(raw_output).__name__}")


def extract_true_emotion(record: Dict[str, Any]) -> str:
    output = parse_output_payload(record.get("output", {}))
    emotion = output.get("primary_emotion") or output.get("group_emotion") or output.get("emotion") or output.get("label")
    if emotion is None:
        image_path = record.get("image")
        if not image_path:
            raise ValueError(f"Cannot find a label in this record: {record}")
        emotion = Path(str(image_path)).parent.name
    return normalize_emotion(str(emotion))


def validate_test_records(records: Sequence[Dict[str, Any]], allowed_emotions: Sequence[str]) -> List[Dict[str, Any]]:
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
        "Set group_emotion to the most representative emotion. "
        'Return only valid JSON in this schema: {"subjects":[{"id":1,"emotion":"<emotion>"}],"group_emotion":"<emotion>"}'
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_moondream(args: argparse.Namespace) -> Tuple[Any, Any, str]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    model_dir = Path(args.model_dir)
    base_model_dir = Path(args.base_model_dir)
    if not base_model_dir.is_absolute():
        base_model_dir = Path(__file__).resolve().parent / base_model_dir

    common_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **common_kwargs).to(device)
        model_source = str(model_dir)
    except Exception as exc:
        if not base_model_dir.exists():
            raise RuntimeError(
                f"Could not load fine-tuned model directly from {model_dir}, and base model dir "
                f"{base_model_dir} does not exist. Keep ./moondream2-base next to this script "
                "or pass --base_model_dir."
            ) from exc
        print(f"Direct model load from {model_dir} failed; loading base code from {base_model_dir} and weights from model dir.")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(base_model_dir, **common_kwargs).to(device)
        weight_file = model_dir / "pytorch_model.bin"
        safetensors_file = model_dir / "model.safetensors"
        if safetensors_file.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(safetensors_file), device=str(device))
        elif weight_file.exists():
            state_dict = torch.load(weight_file, map_location=device)
        else:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_dir}")
        model.load_state_dict(state_dict, strict=False)
        model_source = f"{model_dir} via base code {base_model_dir}"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    model.eval()
    return model, tokenizer, model_source


@torch.inference_mode()
def predict_one(model: Any, tokenizer: Any, image_path: str, prompt: str, args: argparse.Namespace) -> str:
    image = Image.open(image_path).convert("RGB")

    if hasattr(model, "encode_image") and hasattr(model, "answer_question"):
        encoded_image = model.encode_image(image)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0.0,
            "temperature": args.temperature if args.temperature > 0.0 else None,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        try:
            return model.answer_question(encoded_image, prompt, tokenizer, **generation_kwargs).strip()
        except TypeError:
            return model.answer_question(encoded_image, prompt, tokenizer).strip()

    raise RuntimeError("Loaded Moondream model does not expose encode_image/answer_question.")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
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
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def parse_predicted_emotion(
    raw_text: str,
    allowed_emotions: Sequence[str],
    target_schema: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
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
                subject_emotions = [
                    normalize_emotion(str(subject["emotion"]))
                    for subject in parsed["subjects"]
                    if isinstance(subject, dict) and subject.get("emotion") is not None
                ]
                prediction = majority_vote(subject_emotions)

    if prediction is not None:
        normalized_prediction = normalize_emotion(str(prediction))
        if normalized_prediction in allowed:
            parse_info["parse_status"] = "valid_label"
            return normalized_prediction, parse_info
        parse_info["parse_status"] = "label_not_allowed"
        parse_info["raw_label"] = normalized_prediction

    raw_lower = normalize_emotion(raw_text)
    found_labels = [label for label in allowed if re.search(rf"\b{re.escape(label)}\b", raw_lower)]
    if len(found_labels) == 1:
        parse_info["parse_status"] = "text_label_fallback"
        return found_labels[0], parse_info

    return None, parse_info


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[Optional[str]],
    emotion_labels: Sequence[str],
) -> Dict[str, Any]:
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
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
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
    false_positive_total = sum(confusion[row][col] for row in range(len(labels)) for col in range(len(labels)) if row != col)
    false_negative_total = false_positive_total + invalid_predictions
    micro_precision = safe_divide(correct, correct + false_positive_total)
    micro_recall = safe_divide(correct, correct + false_negative_total)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def add_suffix_before_extension(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def legacy_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.predictions_jsonl:
        return Path(f"{args.predictions_jsonl}.checkpoint")
    return Path(f"{args.metrics_json}.checkpoint.jsonl")


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_jsonl:
        return Path(args.checkpoint_jsonl)
    if args.predictions_jsonl:
        checkpoint_path = add_suffix_before_extension(args.predictions_jsonl, ".checkpoint.jsonl")
    else:
        checkpoint_path = add_suffix_before_extension(args.metrics_json, ".checkpoint.jsonl")

    old_path = legacy_checkpoint_path(args)
    if args.resume_from_checkpoint and old_path != checkpoint_path and old_path.exists() and old_path.stat().st_size > 0 and not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_path, checkpoint_path)
        print(f"Migrated legacy checkpoint from {old_path} to {checkpoint_path}")
    return checkpoint_path


def resolve_partial_metrics_path(args: argparse.Namespace) -> Path:
    if args.partial_metrics_json:
        return Path(args.partial_metrics_json)
    return add_suffix_before_extension(args.metrics_json, ".partial.json")


def load_prediction_checkpoint(path: str | Path) -> Dict[int, Dict[str, Any]]:
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        path.touch(exist_ok=True)
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def append_prediction_checkpoint(path: str | Path, prediction: Dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_metrics_report(
    predictions: Sequence[Dict[str, Any]],
    emotion_labels: Sequence[str],
    args: argparse.Namespace,
    model_source: str,
    status: str,
    total_samples: int,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    true_labels = [item["true_emotion"] for item in predictions]
    predicted_labels = [item.get("predicted_emotion") for item in predictions]
    parse_status_counts = Counter(item.get("parse_status", "unknown") for item in predictions)
    metrics = compute_classification_metrics(true_labels, predicted_labels, emotion_labels)
    metrics["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "test_json": str(args.test_json),
        "model_dir": str(args.model_dir),
        "model_source": model_source,
        "target_schema": args.target_schema,
        "emotion_labels": list(emotion_labels),
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
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
    model, tokenizer, model_source = load_moondream(args)
    prompt = build_user_prompt(emotion_labels, args.target_schema)

    checkpoint_path = resolve_checkpoint_path(args)
    partial_metrics_path = resolve_partial_metrics_path(args)
    loaded_checkpoint = load_prediction_checkpoint(checkpoint_path) if args.resume_from_checkpoint else {}
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
            raw_prediction = predict_one(model, tokenizer, record["image"], prompt, args)
            predicted_label, parse_info = parse_predicted_emotion(raw_prediction, emotion_labels, args.target_schema)
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
            if newly_completed % args.checkpoint_every == 0 or completed_count == len(test_records):
                partial_metrics = build_metrics_report(
                    predictions=sorted(predictions_for_file, key=lambda item: item["index"]),
                    emotion_labels=emotion_labels,
                    args=args,
                    model_source=model_source,
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
        model_source=model_source,
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
