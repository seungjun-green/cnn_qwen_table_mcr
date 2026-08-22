from __future__ import annotations

import argparse
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
from src.curriculum import CurriculumRunConfig, CurriculumRunner
from src.data import load_wtq
from src.model import build_model, load_tokenizer
from src.utils import model_dtype, select_device, set_seed
from src.wtq_evaluation import (
    ensure_official_tagged_data,
    load_official_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential synthetic table-reasoning curriculum SFT"
    )
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs/serialized_table_lora.yaml"),
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-cache-dir", default=None)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--epochs-per-level",
        nargs="+",
        type=int,
        default=[3],
        help=(
            "One shared epoch count, or one count per selected level "
            "(for example: --levels 1 2 3 4 --epochs-per-level 3 6 3 3)"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-examples-per-level", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument(
        "--normalize-synthetic-format",
        action="store_true",
        help="Render synthetic Markdown tables with the same serializer as WTQ",
    )
    parser.add_argument("--run-mixed-phase", action="store_true")
    parser.add_argument("--synthetic-ratio", type=int, default=25)
    parser.add_argument("--wtq-ratio", type=int, default=75)
    parser.add_argument("--mixed-epochs", type=int, default=1)
    parser.add_argument("--mixed-total-examples", type=int, default=None)
    parser.add_argument("--run-final-test", action="store_true")
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


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.base_config)
    if args.base_model:
        experiment_config.model.name = args.base_model
    if experiment_config.experiment_type != "serialized":
        raise ValueError("Curriculum SFT currently requires a serialized base config")
    if not experiment_config.lora.enabled:
        raise ValueError("Curriculum SFT requires LoRA to preserve the base model")
    experiment_config.training.output_dir = args.output_dir
    experiment_config.training.max_validation_examples = args.max_validation_examples
    run_config = CurriculumRunConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        epochs_per_level=(
            args.epochs_per_level[0]
            if len(args.epochs_per_level) == 1
            else tuple(args.epochs_per_level)
        ),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_sequence_length=args.max_sequence_length,
        max_answer_tokens=experiment_config.data.max_answer_tokens,
        max_examples_per_level=args.max_examples_per_level,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every_steps=args.log_every_steps,
        seed=args.seed,
        normalize_synthetic_format=args.normalize_synthetic_format,
        run_mixed_phase=args.run_mixed_phase,
        synthetic_to_wtq_ratio=(args.synthetic_ratio, args.wtq_ratio),
        mixed_epochs=args.mixed_epochs,
        mixed_total_examples=args.mixed_total_examples,
        run_final_test=args.run_final_test,
    )
    run_config.validate()
    set_seed(args.seed)
    device = select_device()
    dtype = model_dtype(device, experiment_config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    print(f"Git commit: {_git_commit()}", flush=True)
    print(
        "Curriculum: "
        f"levels={list(run_config.levels)} | "
        f"epochs={list(run_config.level_epoch_schedule())} | "
        "synthetic_format="
        f"{'wtq_serialized' if run_config.normalize_synthetic_format else 'original'}",
        flush=True,
    )
    tokenizer = load_tokenizer(experiment_config)
    dataset = load_wtq(
        experiment_config.data.dataset, experiment_config.data.revision
    )
    print(
        "WTQ sizes: "
        f"train={len(dataset['train'])}, validation={len(dataset['validation'])}, "
        f"test={len(dataset['test'])}",
        flush=True,
    )
    model = build_model(experiment_config, tokenizer, device, dtype)
    official_cache = (
        Path(args.official_cache_dir)
        if args.official_cache_dir
        else Path(args.output_dir) / "diagnostics" / "wtq_official_1.0.2"
    )
    tagged_data = ensure_official_tagged_data(official_cache)
    official_targets = load_official_targets(tagged_data)
    print(f"Loaded {len(official_targets)} official WTQ targets", flush=True)
    runner = CurriculumRunner(
        model=model,
        tokenizer=tokenizer,
        wtq_dataset=dataset,
        experiment_config=experiment_config,
        run_config=run_config,
        device=device,
        official_targets=official_targets,
        experiment_metadata={
            "git_commit": _git_commit(),
            "base_model": experiment_config.model.name,
            "precision": str(dtype),
            "device": str(device),
            "adaptation": "LoRA",
        },
    )
    runner.run()


if __name__ == "__main__":
    main()
