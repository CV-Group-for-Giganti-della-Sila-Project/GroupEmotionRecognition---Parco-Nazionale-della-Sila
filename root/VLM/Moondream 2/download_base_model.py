#!/usr/bin/env python3
"""Download the pinned Moondream 2 base model used by the local scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "vikhyatk/moondream2"
DEFAULT_REVISION = "2024-04-02"
DEFAULT_LOCAL_MODEL_DIR = "moondream2-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Moondream 2 base files locally.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model repo to download.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Model revision to download.")
    parser.add_argument(
        "--local_model_dir",
        default=DEFAULT_LOCAL_MODEL_DIR,
        help="Output folder. Relative paths are resolved beside this script.",
    )
    parser.add_argument(
        "--force_download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-download even if files already exist in the Hugging Face cache.",
    )
    return parser.parse_args()


def resolve_local_dir(path: str) -> Path:
    local_dir = Path(path).expanduser()
    if not local_dir.is_absolute():
        local_dir = Path(__file__).resolve().parent / local_dir
    return local_dir


def main() -> None:
    args = parse_args()
    local_dir = resolve_local_dir(args.local_model_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id}@{args.revision} to: {local_dir}")
    downloaded_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(local_dir),
        force_download=args.force_download,
    )
    print(f"Base model ready in: {downloaded_path}")


if __name__ == "__main__":
    main()
