#!/usr/bin/env python3
"""
Fine-tune Moondream locally on FERPlus-style emotion data.

Expected input record format, either JSON array or JSONL:

{
    "image": "/workspace/datasets/images/train/happiness/img.png",
    "output": "{\"primary_emotion\": \"happiness\"}"
}

This script is intentionally Moondream-specific. It uses the Hugging Face
Moondream2 fine-tuning surface that exposes `vision_encoder` and `text_model`,
freezes the vision encoder, and trains the language model from image tokens plus
question/answer tokens.

Example:
python main.py \
  --train_json /workspace/datasets/ferplus_train.json \
  --output_dir ./moondream-ferplus-emotion

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

try:
    from bitsandbytes.optim import Adam8bit
except ImportError:
    Adam8bit = None


DEFAULT_MODEL_ID = "vikhyatk/moondream2"
DEFAULT_REVISION = "2024-04-02"
DEFAULT_LOCAL_MODEL_DIR = "moondream2-base"
ANSWER_EOS = "<|endoftext|>"
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


@dataclass
class RunStats:
    train_loss: List[float]
    eval_loss: List[float]
    best_eval_loss: Optional[float]
    best_checkpoint: Optional[str]
    global_step: int
    optimizer_steps: int
    stopped_reason: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Moondream2 locally on emotion-recognition records.")

    parser.add_argument("--train_json", required=True, help="Path to train JSON/JSONL.")
    parser.add_argument("--eval_json", default=None, help="Optional eval JSON/JSONL.")
    parser.add_argument("--output_dir", default="./moondream-ferplus-emotion", help="Where to save the fine-tuned model.")
    parser.add_argument("--training_summary_json", default=None, help="Defaults to <output_dir>/training_summary.json.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Local path or Hugging Face id for Moondream.")
    parser.add_argument(
        "--local_model_dir",
        default=DEFAULT_LOCAL_MODEL_DIR,
        help=(
            "Where to download the default base Moondream model, relative to main.py unless absolute. "
            "Only used when --model_id is left as vikhyatk/moondream2."
        ),
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=(
            "Moondream revision. The default is pinned to the revision whose HF wrapper exposes "
            "vision_encoder/text_model for local fine-tuning. Use --revision main only if your "
            "local checkpoint still exposes that API."
        ),
    )
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="none", choices=("none", "eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--torch_dtype", default="auto", choices=("auto", "float16", "float32", "bfloat16"))
    parser.add_argument("--use_8bit_optimizer", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sample_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emotion_labels", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--target_schema", choices=("group", "primary"), default="group")
    parser.add_argument("--max_length", type=int, default=1536)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable text-model gradient checkpointing to reduce VRAM usage.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--stop_on_nan_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

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
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
        records.append(item)
    return records


def parse_output_payload(raw_output: Any) -> Dict[str, Any]:
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed output is not an object: {raw_output}")
        return parsed
    return {}


def extract_emotion(record: Dict[str, Any]) -> str:
    output = parse_output_payload(record.get("output", {}))
    emotion = output.get("primary_emotion") or output.get("emotion") or output.get("label")
    if emotion is None:
        image_path = record.get("image")
        if not image_path:
            raise ValueError(f"Record has no emotion label and no image path: {record}")
        emotion = Path(str(image_path)).parent.name
    return normalize_emotion(str(emotion))


def validate_and_annotate_records(records: Sequence[Dict[str, Any]], allowed_emotions: Sequence[str]) -> List[Dict[str, Any]]:
    allowed = {normalize_emotion(label) for label in allowed_emotions if label.strip()}
    annotated: List[Dict[str, Any]] = []
    skipped_missing_image = 0
    skipped_label = Counter()

    for record in records:
        image = record.get("image")
        if not image or not Path(str(image)).exists():
            skipped_missing_image += 1
            continue
        emotion = extract_emotion(record)
        if allowed and emotion not in allowed:
            skipped_label[emotion] += 1
            continue
        item = dict(record)
        item["_emotion"] = emotion
        annotated.append(item)

    if skipped_missing_image:
        print(f"Skipped {skipped_missing_image} records with missing image paths.")
    if skipped_label:
        print(f"Skipped labels outside --emotion_labels: {dict(skipped_label)}")
    if not annotated:
        raise ValueError("No usable records remain after validation/filtering.")
    return annotated


def stratified_sample(records: Sequence[Dict[str, Any]], fraction: float, seed: int) -> List[Dict[str, Any]]:
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
    if schema == "primary":
        payload = {"primary_emotion": emotion}
    else:
        payload = {"subjects": [{"id": 1, "emotion": emotion}], "group_emotion": emotion}
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


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


class EmotionQADataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], prompt: str, target_schema: str) -> None:
        self.records = list(records)
        self.prompt = prompt
        self.target_schema = target_schema

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        emotion = record["_emotion"]
        return {
            "image": Image.open(record["image"]).convert("RGB"),
            "question": self.prompt,
            "answer": build_target_json(emotion, self.target_schema),
            "emotion": emotion,
        }


def print_dataset_summary(name: str, records: Sequence[Dict[str, Any]]) -> None:
    counts = Counter(record["_emotion"] for record in records)
    print(f"{name}: {len(records)} samples")
    for emotion, count in sorted(counts.items()):
        print(f"  {emotion}: {count}")


def build_datasets(args: argparse.Namespace) -> Tuple[EmotionQADataset, Optional[EmotionQADataset], List[str]]:
    emotion_labels = [normalize_emotion(label) for label in args.emotion_labels.split(",") if label.strip()]
    if not emotion_labels:
        raise ValueError("--emotion_labels cannot be empty.")

    raw_train = read_json_or_jsonl(args.train_json)
    train_records = validate_and_annotate_records(raw_train, emotion_labels)
    sampled_train = stratified_sample(train_records, args.sample_fraction, args.seed)

    if args.eval_json:
        raw_eval = read_json_or_jsonl(args.eval_json)
        final_train_records = sampled_train
        final_eval_records = validate_and_annotate_records(raw_eval, emotion_labels)
    else:
        final_train_records, final_eval_records = stratified_train_eval_split(sampled_train, args.eval_ratio, args.seed)

    prompt = build_user_prompt(emotion_labels, args.target_schema)
    print_dataset_summary("Train", final_train_records)
    if final_eval_records:
        print_dataset_summary("Eval", final_eval_records)
    else:
        print("Eval: disabled because no validation records are available.")

    train_dataset = EmotionQADataset(final_train_records, prompt, args.target_schema)
    eval_dataset = EmotionQADataset(final_eval_records, prompt, args.target_schema) if final_eval_records else None
    return train_dataset, eval_dataset, emotion_labels


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


def resolve_model_source(args: argparse.Namespace) -> Tuple[str, bool]:
    """
    Return the path/id to pass to from_pretrained.

    By default, keep the base Moondream files beside this script instead of in
    the global Hugging Face cache. If --model_id is changed, respect it exactly.
    """

    if args.model_id != DEFAULT_MODEL_ID:
        model_path = Path(args.model_id)
        return args.model_id, model_path.exists()

    local_dir = Path(args.local_model_dir)
    if not local_dir.is_absolute():
        local_dir = Path(__file__).resolve().parent / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ensuring base Moondream is available in: {local_dir}")
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(local_dir),
    )
    return str(local_dir), True


def load_moondream(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Any]:
    dtype = resolve_dtype(args.torch_dtype, device)
    model_source, is_local_model = resolve_model_source(args)

    pretrained_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if not is_local_model:
        pretrained_kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **pretrained_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or ANSWER_EOS
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if not is_local_model:
        model_kwargs["revision"] = args.revision
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs).to(device)

    if not hasattr(model, "vision_encoder") or not hasattr(model, "text_model"):
        raise RuntimeError(
            "This Moondream checkpoint does not expose the fine-tuning API used here "
            "(`vision_encoder` and `text_model`). Keep the default --revision 2024-04-02, "
            "or pass a local Moondream2 checkpoint with that API."
        )

    model.vision_encoder.eval()
    for parameter in model.vision_encoder.parameters():
        parameter.requires_grad = False

    if hasattr(model.text_model, "config"):
        model.text_model.config.use_cache = False
    if hasattr(model.text_model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        try:
            model.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.text_model.gradient_checkpointing_enable()

    model.text_model.train()
    return model, tokenizer


@dataclass
class MoondreamCollator:
    tokenizer: Any
    vision_encoder: Any
    max_length: int
    image_tokens: int

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = [sample["image"] for sample in batch]

        token_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        for sample in batch:
            tokens = [self.tokenizer.bos_token_id]
            labels = [-100] * (self.image_tokens + 1)

            question_ids = self.tokenizer(
                f"\n\nQuestion: {sample['question']}\n\nAnswer:",
                add_special_tokens=False,
            ).input_ids
            tokens.extend(question_ids)
            labels.extend([-100] * len(question_ids))

            answer_ids = self.tokenizer(
                f" {sample['answer']}{ANSWER_EOS}",
                add_special_tokens=False,
            ).input_ids
            tokens.extend(answer_ids)
            labels.extend(answer_ids)

            max_token_count = max(1, self.max_length - self.image_tokens)
            token_rows.append(tokens[:max_token_count])
            label_rows.append(labels[: self.max_length])

        max_tokens = max(len(row) for row in token_rows)
        max_labels = max(len(row) for row in label_rows)

        token_tensor_rows: List[torch.Tensor] = []
        label_tensor_rows: List[torch.Tensor] = []
        attention_rows: List[torch.Tensor] = []
        for tokens, labels in zip(token_rows, label_rows):
            token_pad = max_tokens - len(tokens)
            label_pad = max_labels - len(labels)
            token_tensor_rows.append(torch.tensor(tokens + [self.tokenizer.pad_token_id] * token_pad, dtype=torch.long))
            label_tensor_rows.append(torch.tensor(labels + [-100] * label_pad, dtype=torch.long))
            attention_rows.append(torch.tensor([1] * len(labels) + [0] * label_pad, dtype=torch.bool))

        return {
            "images": images,
            "tokens": torch.stack(token_tensor_rows),
            "labels": torch.stack(label_tensor_rows),
            "attention_mask": torch.stack(attention_rows),
        }


def infer_image_token_count(model: Any, dataset: Dataset, device: torch.device) -> int:
    sample = dataset[0]
    with torch.no_grad():
        image_embeddings = model.vision_encoder([sample["image"]])
    return int(image_embeddings.shape[1])


def compute_loss(model: Any, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    images = batch["images"]
    tokens = batch["tokens"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        image_embeddings = model.vision_encoder(images)
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.to(device)

    token_embeddings = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((token_embeddings[:, 0:1, :], image_embeddings, token_embeddings[:, 1:, :]), dim=1)

    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attention_mask,
    )
    return outputs.loss


def make_optimizer(model: Any, args: argparse.Namespace) -> torch.optim.Optimizer:
    trainable_parameters = [parameter for parameter in model.text_model.parameters() if parameter.requires_grad]
    if args.use_8bit_optimizer and Adam8bit is not None:
        return Adam8bit(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.use_8bit_optimizer and Adam8bit is None:
        print(
            "bitsandbytes is not installed; falling back to torch AdamW. "
            "On a 15 GB GPU this may OOM because AdamW keeps large optimizer states. "
            "Install bitsandbytes on CUDA Linux for the intended 8-bit optimizer path."
        )
    return torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def estimate_optimizer_steps(args: argparse.Namespace, dataset: Dataset) -> int:
    if args.max_steps > 0:
        return args.max_steps
    micro_batches = math.ceil(len(dataset) / max(1, args.per_device_train_batch_size))
    steps_per_epoch = math.ceil(micro_batches / max(1, args.gradient_accumulation_steps))
    return max(1, math.ceil(steps_per_epoch * args.num_train_epochs))


def evaluate(model: Any, dataloader: DataLoader, device: torch.device) -> float:
    model.text_model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            loss = compute_loss(model, batch, device)
            losses.append(float(loss.detach().cpu().item()))
    model.text_model.train()
    return float(sum(losses) / max(1, len(losses)))


def prune_checkpoints(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    for checkpoint in checkpoints[:-limit]:
        for child in sorted(checkpoint.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        checkpoint.rmdir()


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, step: int, limit: int) -> str:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    prune_checkpoints(output_dir, limit)
    return str(checkpoint_dir)


def train(
    model: Any,
    tokenizer: Any,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> RunStats:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = make_optimizer(model, args)
    total_steps = estimate_optimizer_steps(args, train_loader.dataset)
    warmup_steps = max(0, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train_losses: List[float] = []
    eval_losses: List[float] = []
    best_eval_loss: Optional[float] = None
    best_checkpoint: Optional[str] = None
    stale_eval_count = 0
    global_step = 0
    optimizer_steps = 0
    stopped_reason: Optional[str] = None

    optimizer.zero_grad(set_to_none=True)
    epochs = max(1, math.ceil(args.num_train_epochs))
    progress = tqdm(total=total_steps, desc="Train")

    for epoch in range(epochs):
        if args.max_steps > 0 and optimizer_steps >= args.max_steps:
            break
        for batch in train_loader:
            global_step += 1
            loss = compute_loss(model, batch, device) / max(1, args.gradient_accumulation_steps)
            raw_loss = float(loss.detach().cpu().item() * max(1, args.gradient_accumulation_steps))

            if args.stop_on_nan_loss and not math.isfinite(raw_loss):
                stopped_reason = f"non_finite_training_loss_at_micro_step_{global_step}"
                print(f"Stopping: {stopped_reason}")
                return RunStats(train_losses, eval_losses, best_eval_loss, best_checkpoint, global_step, optimizer_steps, stopped_reason)

            loss.backward()
            train_losses.append(raw_loss)

            if global_step % args.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.text_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            progress.update(1)

            if optimizer_steps % args.logging_steps == 0:
                progress.set_postfix({"loss": f"{raw_loss:.4f}", "lr": optimizer.param_groups[0]["lr"]})

            if optimizer_steps % args.save_steps == 0:
                checkpoint = save_checkpoint(model, tokenizer, output_dir, optimizer_steps, args.save_total_limit)
                print(f"Saved checkpoint: {checkpoint}")

            if eval_loader is not None and optimizer_steps % args.eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, device)
                eval_losses.append(eval_loss)
                print(f"Eval loss at step {optimizer_steps}: {eval_loss:.6f}")
                if args.stop_on_nan_loss and not math.isfinite(eval_loss):
                    stopped_reason = f"non_finite_eval_loss_at_step_{optimizer_steps}"
                    break
                if best_eval_loss is None or eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    stale_eval_count = 0
                    best_checkpoint = save_checkpoint(model, tokenizer, output_dir, optimizer_steps, args.save_total_limit)
                    print(f"New best checkpoint: {best_checkpoint}")
                else:
                    stale_eval_count += 1
                    if args.early_stopping_patience > 0 and stale_eval_count >= args.early_stopping_patience:
                        stopped_reason = f"early_stopping_at_step_{optimizer_steps}"
                        break

            if args.max_steps > 0 and optimizer_steps >= args.max_steps:
                break
            if optimizer_steps >= total_steps:
                break

        if stopped_reason or optimizer_steps >= total_steps:
            break

    progress.close()
    return RunStats(train_losses, eval_losses, best_eval_loss, best_checkpoint, global_step, optimizer_steps, stopped_reason)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def write_summary(
    args: argparse.Namespace,
    stats: RunStats,
    output_path: Path,
    emotion_labels: Sequence[str],
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    device: torch.device,
) -> None:
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": args.output_dir,
        "training_finished": stats.stopped_reason is None,
        "training_status": "completed" if stats.stopped_reason is None else "stopped",
        "stopped_reason": stats.stopped_reason,
        "hyperparameters_used": {
            "model_id": args.model_id,
            "revision": args.revision,
            "local_model_dir": args.local_model_dir,
            "target_schema": args.target_schema,
            "emotion_labels": list(emotion_labels),
            "sample_fraction": args.sample_fraction,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_length": args.max_length,
            "device": str(device),
            "torch_dtype": args.torch_dtype,
            "use_8bit_optimizer": args.use_8bit_optimizer and Adam8bit is not None,
            "seed": args.seed,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        },
        "run_stats": asdict(stats),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print("Final training summary JSON:")
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=True))


def main() -> None:
    args = parse_args()
    if args.max_steps < -1 or args.max_steps == 0:
        raise ValueError("--max_steps must be -1 or a positive integer.")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset, eval_dataset, emotion_labels = build_datasets(args)
    model, tokenizer = load_moondream(args, device)
    image_tokens = infer_image_token_count(model, train_dataset, device)
    print(f"Moondream image token count: {image_tokens}")

    collator = MoondreamCollator(tokenizer, model.vision_encoder, args.max_length, image_tokens)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.dataloader_num_workers,
        )
        if eval_dataset is not None
        else None
    )

    stats = train(model, tokenizer, train_loader, eval_loader, args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned Moondream model to: {output_dir}")

    summary_path = Path(args.training_summary_json or output_dir / "training_summary.json")
    write_summary(args, stats, summary_path, emotion_labels, train_dataset, eval_dataset, device)


if __name__ == "__main__":
    main()
