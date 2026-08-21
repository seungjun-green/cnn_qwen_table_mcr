from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import load_config
from src.data import load_wtq
from src.diagnostics import build_table_shuffled_examples, table_dependence_metrics
from src.evaluate import evaluate_model
from src.model import build_model, load_tokenizer, load_trainable_checkpoint
from src.utils import model_dtype, select_device, set_seed, write_json

DEFAULT_CONFIGS = [
    "configs/baseline.yaml",
    "configs/pooling_max.yaml",
    "configs/pooling_attention.yaml",
    "configs/cell_dim_128.yaml",
    "configs/cell_dim_512.yaml",
    "configs/grid_16x8.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose table dependence in saved CNN checkpoints"
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--mirror-root",
        default="/content/drive/MyDrive/cnn_qwen_table_mcr/outputs",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--sample-predictions", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["trained", "no_thinking"],
        default=["trained", "no_thinking"],
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _checkpoint_path(run_dir: Path, preference: str) -> Path | None:
    preferred = run_dir / f"checkpoint_{preference}.pt"
    fallback = run_dir / "checkpoint_last.pt"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    return None


def _print_samples(
    run_name: str,
    mode: str,
    correct_records: list[dict[str, Any]],
    shuffled_records: list[dict[str, Any]],
    count: int,
) -> None:
    print(f"\n[{run_name} | {mode}] sample predictions")
    for index, (correct, shuffled) in enumerate(zip(correct_records, shuffled_records)):
        if index >= count:
            break
        print(f"  Q: {correct['question']}")
        print(f"  Gold: {correct['gold_answers']}")
        print(f"  Correct table:  {correct['prediction']!r}")
        print(f"  Shuffled table: {shuffled['prediction']!r}")


def main() -> None:
    args = parse_args()
    if args.max_examples < 2:
        raise ValueError("--max-examples must be at least 2")
    mirror_root = Path(args.mirror_root)
    output_dir = Path(args.output_dir or mirror_root / "diagnostics/table_dependence")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [load_config(path) for path in args.configs]
    first_config = configs[0]
    set_seed(first_config.training.seed)
    device = select_device()
    dtype = model_dtype(device, first_config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}")

    dataset = load_wtq(first_config.data.dataset, first_config.data.revision)[
        args.split
    ]
    dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    shuffled_dataset = build_table_shuffled_examples(dataset)
    summary: dict[str, Any] = {
        "split": args.split,
        "max_examples": len(dataset),
        "checkpoint_preference": args.checkpoint,
        "modes": args.modes,
        "runs": {},
    }

    for position, (config_path, config) in enumerate(
        zip(args.configs, configs), start=1
    ):
        run_name = Path(config.training.output_dir).name
        run_dir = mirror_root / run_name
        checkpoint_path = _checkpoint_path(run_dir, args.checkpoint)
        separator = "=" * 88
        print(f"\n{separator}")
        print(f"DIAGNOSTIC {position}/{len(configs)}: {run_name}")
        print(f"CONFIG:     {Path(config_path).resolve()}")
        print(f"CHECKPOINT: {checkpoint_path or 'NOT FOUND'}")
        print(separator, flush=True)
        if checkpoint_path is None:
            summary["runs"][run_name] = {"status": "checkpoint_not_found"}
            write_json(summary, output_dir / "summary.json")
            continue

        if args.max_new_tokens is not None:
            config.generation.max_new_tokens = args.max_new_tokens
        tokenizer = load_tokenizer(config)
        model = build_model(config, tokenizer, device, dtype)
        load_trainable_checkpoint(model, checkpoint_path)
        run_summary: dict[str, Any] = {
            "status": "completed",
            "checkpoint": str(checkpoint_path),
            "modes": {},
        }

        for mode in args.modes:
            enable_thinking = None if mode == "trained" else False
            mode_dir = output_dir / run_name / mode
            correct_metrics, correct_records = evaluate_model(
                model,
                tokenizer,
                dataset,
                config,
                device,
                predictions_path=mode_dir / "correct_tables/predictions.json",
                description=f"[{run_name} | {mode}] correct tables",
                enable_thinking=enable_thinking,
            )
            shuffled_metrics, shuffled_records = evaluate_model(
                model,
                tokenizer,
                shuffled_dataset,
                config,
                device,
                predictions_path=mode_dir / "shuffled_tables/predictions.json",
                description=f"[{run_name} | {mode}] shuffled tables",
                enable_thinking=enable_thinking,
            )
            dependence = table_dependence_metrics(correct_records, shuffled_records)
            dependence["correct_metrics"] = correct_metrics
            dependence["shuffled_metrics"] = shuffled_metrics
            run_summary["modes"][mode] = dependence
            write_json(dependence, mode_dir / "comparison.json")
            _print_samples(
                run_name,
                mode,
                correct_records,
                shuffled_records,
                args.sample_predictions,
            )
            print(
                f"[{run_name} | {mode}] correct EM="
                f"{dependence['correct_table_exact_match']:.4f}; shuffled EM="
                f"{dependence['shuffled_table_exact_match']:.4f}; drop="
                f"{dependence['exact_match_drop']:.4f}; prediction change="
                f"{dependence['prediction_change_rate']:.4f}"
            )

        summary["runs"][run_name] = run_summary
        write_json(summary, output_dir / "summary.json")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("TABLE-DEPENDENCE SUMMARY")
    print("=" * 88)
    for run_name, run_result in summary["runs"].items():
        if run_result.get("status") != "completed":
            print(f"{run_name:24s} {run_result.get('status', 'unknown')}")
            continue
        for mode, metrics in run_result["modes"].items():
            print(
                f"{run_name:24s} {mode:12s} "
                f"correct={metrics['correct_table_exact_match']:.4f} "
                f"shuffled={metrics['shuffled_table_exact_match']:.4f} "
                f"drop={metrics['exact_match_drop']:.4f} "
                f"changed={metrics['prediction_change_rate']:.4f}"
            )
    print(f"\nDiagnostic summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
