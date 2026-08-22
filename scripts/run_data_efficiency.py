from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.curriculum import load_stage_model
from src.data import load_wtq
from src.data_efficiency import (
    condition_name,
    deterministic_subset_indices,
    subset_fingerprint,
    write_data_efficiency_summary,
)
from src.model import build_model, load_tokenizer
from src.train import train_model
from src.utils import model_dtype, select_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune identical WTQ subsets from base or curriculum initialization"
    )
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs/serialized_table_lora.yaml"),
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--official-cache-dir", required=True)
    parser.add_argument(
        "--initialization", choices=("base", "curriculum"), required=True
    )
    parser.add_argument("--curriculum-checkpoint", default=None)
    parser.add_argument("--wtq-fraction", type=float, required=True)
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    return parser.parse_args()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _validate_or_write_metadata(path: Path, metadata: dict[str, object]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparison_keys = set(metadata) - {"git_commit"}
        if any(existing.get(key) != metadata[key] for key in comparison_keys):
            raise ValueError(
                f"Existing run metadata does not match this request: {path}. "
                "Use a new output root for changed settings."
            )
        return
    write_json(metadata, path)


def main() -> None:
    args = parse_args()
    curriculum_checkpoint: Path | None = None
    if args.initialization == "curriculum":
        if not args.curriculum_checkpoint:
            raise ValueError(
                "--curriculum-checkpoint is required for curriculum initialization"
            )
        curriculum_checkpoint = Path(args.curriculum_checkpoint)
        if not curriculum_checkpoint.is_file():
            raise FileNotFoundError(curriculum_checkpoint)

    output_root = Path(args.output_root)
    run_name = condition_name(args.wtq_fraction, args.initialization)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.base_config)
    if config.experiment_type != "serialized" or not config.lora.enabled:
        raise ValueError("Data-efficiency runs require the serialized LoRA config")
    if args.base_model:
        config.model.name = args.base_model
    config.training.output_dir = str(run_dir)
    config.training.mirror_output_dir = None
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.checkpoint_every_steps = args.checkpoint_every_steps
    config.training.early_stopping_patience = args.early_stopping_patience
    config.training.seed = args.training_seed
    config.training.max_validation_examples = args.max_validation_examples
    config.evaluation.official_cache_dir = args.official_cache_dir

    set_seed(args.training_seed)
    dataset = load_wtq(config.data.dataset, config.data.revision)
    indices = deterministic_subset_indices(
        len(dataset["train"]), args.wtq_fraction, args.subset_seed
    )
    train_subset = dataset["train"].select(indices)
    config.training.max_train_examples = len(train_subset)
    config.validate()

    fingerprint = subset_fingerprint(indices)
    metadata: dict[str, object] = {
        "git_commit": _git_commit(),
        "initialization": args.initialization,
        "curriculum_checkpoint": (
            str(curriculum_checkpoint) if curriculum_checkpoint else None
        ),
        "wtq_fraction": args.wtq_fraction,
        "wtq_percentage": int(args.wtq_fraction * 100),
        "training_examples": len(train_subset),
        "dataset_size": len(dataset["train"]),
        "subset_seed": args.subset_seed,
        "subset_fingerprint": fingerprint,
        "training_seed": args.training_seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "base_model": config.model.name,
    }
    _validate_or_write_metadata(run_dir / "run_metadata.json", metadata)
    write_json(indices, run_dir / "wtq_train_indices.json")

    print("=" * 88, flush=True)
    print(f"DATA EFFICIENCY: {run_name}", flush=True)
    print(f"INITIALIZATION: {args.initialization}", flush=True)
    print(
        f"WTQ TRAIN: {len(train_subset)}/{len(dataset['train'])} "
        f"({args.wtq_fraction:.0%}) | fingerprint={fingerprint[:12]}",
        flush=True,
    )
    print(f"OUTPUT: {run_dir}", flush=True)
    print("=" * 88, flush=True)

    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    tokenizer = load_tokenizer(config)
    model = build_model(config, tokenizer, device, dtype)
    if curriculum_checkpoint is not None:
        load_stage_model(curriculum_checkpoint, model)
        print(f"Loaded curriculum initialization: {curriculum_checkpoint}", flush=True)

    history = train_model(
        model,
        tokenizer,
        train_subset,
        dataset["validation"],
        config,
        device,
    )
    write_data_efficiency_summary(output_root)
    print(
        f"FINISHED: {run_name} | status={history['status']} | "
        f"best={history.get('best_metric')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
