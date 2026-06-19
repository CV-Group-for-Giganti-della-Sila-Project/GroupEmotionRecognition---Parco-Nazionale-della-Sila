#!/usr/bin/env python3
"""Predict emotions for every image in a folder with a trained Moondream model."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Moondream emotion prediction on a folder of images.")
    parser.add_argument("--image_dir", required=True, help="Folder containing images to analyze.")
    parser.add_argument(
        "--model_dir",
        "--adapter_dir",
        dest="model_dir",
        required=True,
        help="Fine-tuned Moondream model/checkpoint directory to use.",
    )
    parser.add_argument("--output_jsonl", default="./folder_predictions.jsonl", help="Where to save JSONL predictions.")
    parser.add_argument("--output_csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--base_model_dir", default=DEFAULT_LOCAL_MODEL_DIR, help="Fallback base Moondream directory.")
    parser.add_argument("--target_schema", choices=("primary", "group"), default="primary")
    parser.add_argument("--emotion_labels", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--torch_dtype", default="auto", choices=("auto", "float16", "float32", "bfloat16"))
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_emotion(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


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


def load_moondream(args: argparse.Namespace) -> Tuple[Any, Any, str, torch.device]:
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
                f"Could not load {model_dir} directly, and fallback base model dir {base_model_dir} does not exist."
            ) from exc
        print(f"Direct model load from {model_dir} failed; loading base code from {base_model_dir} and weights from model dir.")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(base_model_dir, **common_kwargs).to(device)
        state_dict = load_state_dict_from_model_dir(model_dir, device)
        model.load_state_dict(state_dict, strict=False)
        model_source = f"{model_dir} via base code {base_model_dir}"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    model.eval()
    return model, tokenizer, model_source, device


def load_state_dict_from_model_dir(model_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    safetensors_file = model_dir / "model.safetensors"
    weight_file = model_dir / "pytorch_model.bin"
    if safetensors_file.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_file), device=str(device))
    if weight_file.exists():
        return torch.load(weight_file, map_location=device)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_dir}")


def iter_image_paths(image_dir: str | Path, recursive: bool, limit: Optional[int]) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    globber = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(path for path in globber if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No supported images found in {root}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")
    return paths


@torch.inference_mode()
def predict_raw(model: Any, tokenizer: Any, image_path: Path, prompt: str, args: argparse.Namespace) -> str:
    image = Image.open(image_path).convert("RGB")
    if not hasattr(model, "encode_image") or not hasattr(model, "answer_question"):
        raise RuntimeError("Loaded model does not expose Moondream encode_image/answer_question methods.")
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


def parse_emotion(raw_text: str, allowed_emotions: Sequence[str], target_schema: str) -> Tuple[Optional[str], Dict[str, Any]]:
    allowed = {normalize_emotion(label) for label in allowed_emotions}
    parsed = extract_json_object(raw_text)
    info: Dict[str, Any] = {
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
        normalized = normalize_emotion(str(prediction))
        if normalized in allowed:
            info["parse_status"] = "valid_label"
            return normalized, info
        info["parse_status"] = "label_not_allowed"
        info["raw_label"] = normalized

    raw_lower = normalize_emotion(raw_text)
    found_labels = [label for label in allowed if re.search(rf"\b{re.escape(label)}\b", raw_lower)]
    if len(found_labels) == 1:
        info["parse_status"] = "text_label_fallback"
        return found_labels[0], info
    return None, info


def write_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "predicted_emotion", "parse_status", "raw_model_output"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    emotion_labels = [normalize_emotion(label) for label in args.emotion_labels.split(",") if label.strip()]
    if not emotion_labels:
        raise ValueError("--emotion_labels cannot be empty.")

    image_paths = iter_image_paths(args.image_dir, args.recursive, args.limit)
    print(f"Found {len(image_paths)} image(s).")
    model, tokenizer, model_source, _device = load_moondream(args)
    print(f"Using model: {model_source}")
    prompt = build_user_prompt(emotion_labels, args.target_schema)

    rows: List[Dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        raw_output = predict_raw(model, tokenizer, image_path, prompt, args)
        predicted_emotion, info = parse_emotion(raw_output, emotion_labels, args.target_schema)
        row = {
            "image": str(image_path),
            "predicted_emotion": predicted_emotion,
            "parse_status": info["parse_status"],
            "raw_model_output": raw_output,
            "parsed_model_output": info.get("parsed_json"),
        }
        rows.append(row)
        print(f"[{index}/{len(image_paths)}] {image_path.name}: {predicted_emotion or 'INVALID'}")

    write_jsonl(args.output_jsonl, rows)
    print(f"Saved JSONL predictions to: {args.output_jsonl}")
    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"Saved CSV predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
