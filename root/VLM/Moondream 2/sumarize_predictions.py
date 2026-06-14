#!/usr/bin/env python3
"""Print readable accuracy metrics from an evaluation predictions JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize emotion prediction accuracy.")
    parser.add_argument("predictions_jsonl", help="Path to test_predictions.jsonl.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    return parser.parse_args()


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            predictions.append(item)
    if not predictions:
        raise ValueError(f"No predictions found in {path}.")
    return predictions


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def summarize(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for item in predictions if item.get("is_correct") is True)
    invalid = sum(1 for item in predictions if item.get("predicted_emotion") in (None, ""))
    parse_status = Counter(str(item.get("parse_status", "unknown")) for item in predictions)

    per_class_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in predictions:
        true_label = str(item.get("true_emotion", "unknown"))
        predicted_label = item.get("predicted_emotion")
        predicted_label = str(predicted_label) if predicted_label else "INVALID"
        per_class_counts[true_label]["total"] += 1
        per_class_counts[true_label]["correct"] += int(item.get("is_correct") is True)
        confusion[true_label][predicted_label] += 1

    per_class = {
        label: {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for label, counts in sorted(per_class_counts.items())
    }

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "invalid_predictions": invalid,
        "invalid_prediction_rate": invalid / total,
        "parse_status": dict(parse_status),
        "per_class": per_class,
        "confusion": {label: dict(counter) for label, counter in sorted(confusion.items())},
    }


def print_text(summary: Dict[str, Any]) -> None:
    print("Overall")
    print(f"  Samples: {summary['total']}")
    print(f"  Correct: {summary['correct']}")
    print(f"  Incorrect: {summary['incorrect']}")
    print(f"  Accuracy: {percent(summary['accuracy'])}")
    print(f"  Invalid predictions: {summary['invalid_predictions']} ({percent(summary['invalid_prediction_rate'])})")

    print("\nParse Status")
    for status, count in sorted(summary["parse_status"].items()):
        print(f"  {status}: {count}")

    print("\nPer-Class Accuracy")
    for label, item in summary["per_class"].items():
        print(f"  {label}: {percent(item['accuracy'])} ({item['correct']}/{item['total']})")


def main() -> None:
    args = parse_args()
    summary = summarize(load_predictions(args.predictions_jsonl))
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    else:
        print_text(summary)


if __name__ == "__main__":
    main()
