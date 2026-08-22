from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.curriculum import CurriculumRunConfig, CurriculumRunner, load_stage_model
from src.data import load_wtq
from src.model import build_model, load_tokenizer
from src.train import train_model
from src.utils import model_dtype, select_device, set_seed, write_json
from src.wtq_evaluation import ensure_official_tagged_data, load_official_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic-pretrain LoRA+GNN residual, then fine-tune both on WTQ"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/gnn_residual_relational_early.yaml"),
    )
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--official-cache-dir", required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--synthetic-epochs",
        nargs="+",
        type=int,
        default=[3],
        help="One shared value or one epoch count per synthetic level",
    )
    parser.add_argument("--synthetic-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--wtq-epochs", type=int, default=10)
    parser.add_argument("--wtq-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--wtq-early-stopping-patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples-per-level", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
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


def _epoch_schedule(values: list[int]) -> int | tuple[int, ...]:
    return values[0] if len(values) == 1 else tuple(values)


def _trainable_groups(model: torch.nn.Module) -> dict[str, int]:
    groups = {"lora": 0, "gnn_residual": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"] += parameter.numel()
        elif not name.startswith("language_model."):
            groups["gnn_residual"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    if groups["lora"] == 0 or groups["gnn_residual"] == 0:
        raise RuntimeError(
            "Expected both LoRA and GNN residual parameters to be trainable; "
            f"received {groups}"
        )
    if groups["other"]:
        raise RuntimeError(f"Unexpected trainable Qwen backbone parameters: {groups}")
    return groups


def _best_wtq_epoch(history: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        record
        for record in history.get("epochs", [])
        if "denotation_accuracy" in record.get("validation", {})
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda record: float(record["validation"]["denotation_accuracy"]),
    )
    return {
        "epoch": int(best["epoch"]),
        "denotation_accuracy": float(
            best["validation"]["denotation_accuracy"]
        ),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    synthetic_output = output_root / "stage_1_synthetic"
    wtq_output = output_root / "stage_2_wtq"
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if config.experiment_type != "serialized_gnn_residual":
        raise ValueError("Two-stage pretraining requires serialized_gnn_residual")
    if not config.lora.enabled or not config.model.freeze_backbone:
        raise ValueError("The experiment requires frozen Qwen with trainable LoRA")
    config.training.output_dir = str(synthetic_output)
    config.training.max_validation_examples = args.max_validation_examples
    config.validate()

    set_seed(args.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    print(f"Git commit: {_git_commit()}", flush=True)
    tokenizer = load_tokenizer(config)
    dataset = load_wtq(config.data.dataset, config.data.revision)
    model = build_model(config, tokenizer, device, dtype)
    trainable_groups = _trainable_groups(model)
    print(
        "Trainable parameters: "
        f"LoRA={trainable_groups['lora']:,} | "
        f"GNN residual={trainable_groups['gnn_residual']:,} | "
        "Qwen backbone=0",
        flush=True,
    )

    tagged_data = ensure_official_tagged_data(Path(args.official_cache_dir))
    official_targets = load_official_targets(tagged_data)
    print(f"Loaded {len(official_targets)} official WTQ targets", flush=True)

    synthetic_epochs = _epoch_schedule(args.synthetic_epochs)
    curriculum_config = CurriculumRunConfig(
        data_root=args.data_root,
        output_dir=str(synthetic_output),
        levels=tuple(args.levels),
        epochs_per_level=synthetic_epochs,
        learning_rate=args.synthetic_learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_sequence_length=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        max_examples_per_level=args.max_examples_per_level,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every_steps=args.checkpoint_every_steps,
        seed=args.seed,
        normalize_synthetic_format=True,
    )
    curriculum_config.validate()
    print("\n" + "=" * 88, flush=True)
    print("STAGE 1/2: SYNTHETIC PRETRAINING OF LoRA + GNN RESIDUAL", flush=True)
    print(
        f"LEVELS: {list(curriculum_config.levels)} | "
        f"EPOCHS: {list(curriculum_config.level_epoch_schedule())}",
        flush=True,
    )
    print(f"OUTPUT: {synthetic_output}", flush=True)
    print("=" * 88, flush=True)
    curriculum_runner = CurriculumRunner(
        model=model,
        tokenizer=tokenizer,
        wtq_dataset=dataset,
        experiment_config=config,
        run_config=curriculum_config,
        device=device,
        official_targets=official_targets,
        experiment_metadata={
            "git_commit": _git_commit(),
            "base_model": config.model.name,
            "precision": str(dtype),
            "device": str(device),
            "adaptation": "LoRA + GNN residual",
        },
    )
    curriculum_state = curriculum_runner.run()
    final_level = args.levels[-1]
    synthetic_checkpoint = (
        synthetic_output / "checkpoints" / f"level_{final_level}" / "checkpoint.pt"
    )
    if not synthetic_checkpoint.is_file():
        raise FileNotFoundError(synthetic_checkpoint)
    load_stage_model(synthetic_checkpoint, model)
    synthetic_gate = float(torch.tanh(model.residual_gate).detach().float())

    print("\n" + "=" * 88, flush=True)
    print("STAGE 2/2: WTQ FINE-TUNING OF LoRA + GNN RESIDUAL", flush=True)
    print(f"INITIALIZATION: {synthetic_checkpoint}", flush=True)
    print(f"OUTPUT: {wtq_output}", flush=True)
    print("=" * 88, flush=True)
    wtq_config = copy.deepcopy(config)
    wtq_config.training.output_dir = str(wtq_output)
    wtq_config.training.mirror_output_dir = None
    wtq_config.training.epochs = args.wtq_epochs
    wtq_config.training.learning_rate = args.wtq_learning_rate
    wtq_config.training.batch_size = args.batch_size
    wtq_config.training.gradient_accumulation_steps = (
        args.gradient_accumulation_steps
    )
    wtq_config.training.checkpoint_every_steps = args.checkpoint_every_steps
    wtq_config.training.early_stopping_patience = (
        args.wtq_early_stopping_patience
    )
    wtq_config.training.seed = args.seed
    wtq_config.training.max_train_examples = args.max_train_examples
    wtq_config.training.max_validation_examples = args.max_validation_examples
    wtq_config.evaluation.official_cache_dir = args.official_cache_dir
    wtq_config.validate()
    wtq_history = train_model(
        model,
        tokenizer,
        dataset["train"],
        dataset["validation"],
        wtq_config,
        device,
    )
    wtq_gate = float(torch.tanh(model.residual_gate).detach().float())
    summary = {
        "git_commit": _git_commit(),
        "trainable_parameter_groups": trainable_groups,
        "stage_1": {
            "levels": list(curriculum_config.levels),
            "epochs_per_level": list(curriculum_config.level_epoch_schedule()),
            "final_checkpoint": str(synthetic_checkpoint),
            "final_residual_gate": synthetic_gate,
            "validation_results": curriculum_state["results"],
        },
        "stage_2": {
            "status": wtq_history["status"],
            "best": _best_wtq_epoch(wtq_history),
            "best_checkpoint": str(wtq_output / "checkpoint_best.pt"),
            "last_checkpoint": str(wtq_output / "checkpoint_last.pt"),
            "final_residual_gate": wtq_gate,
        },
    }
    write_json(summary, output_root / "two_stage_summary.json")
    print("\n" + "=" * 88, flush=True)
    print(f"TWO-STAGE TRAINING FINISHED: {summary['stage_2']['best']}", flush=True)
    print(f"SUMMARY: {output_root / 'two_stage_summary.json'}", flush=True)
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
