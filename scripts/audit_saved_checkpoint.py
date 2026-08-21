from __future__ import annotations

import argparse
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
from src.utils import model_dtype, normalize_answer, select_device, set_seed, write_json
from src.wtq_evaluation import (
    OfficialTarget,
    ensure_official_tagged_data,
    load_official_targets,
    score_prediction,
    split_prediction_items,
    truncation_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a saved WTQ checkpoint without retraining"
    )
    parser.add_argument("--config", default="configs/continuous_prefix_lora.yaml")
    parser.add_argument(
        "--run-dir",
        default=(
            "/content/drive/MyDrive/cnn_qwen_table_mcr/outputs/"
            "continuous_prefix_lora"
        ),
    )
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--official-cache-dir", default=None)
    parser.add_argument("--official-data-dir", default=None)
    return parser.parse_args()


def _annotate_records(
    records: list[dict[str, Any]], targets: dict[str, OfficialTarget]
) -> None:
    for record in records:
        example_id = str(record["id"])
        if example_id not in targets:
            raise KeyError(f"Official WTQ target not found for example {example_id!r}")
        target = targets[example_id]
        record["predicted_items"] = split_prediction_items(record["prediction"])
        record["official_gold_answers"] = list(target.original_strings)
        record["official_answer_count"] = len(target.values)
        record["official_denotation_correct"] = score_prediction(
            record["prediction"], target
        )


def _official_dependence_metrics(
    correct_records: list[dict[str, Any]],
    shuffled_records: list[dict[str, Any]],
) -> dict[str, float | int]:
    total = len(correct_records)
    if total != len(shuffled_records):
        raise ValueError("Correct and shuffled evaluations must have equal lengths")
    correct = sum(record["official_denotation_correct"] for record in correct_records)
    shuffled = sum(
        record["official_denotation_correct"] for record in shuffled_records
    )
    changed = sum(
        normalize_answer(left["prediction"]) != normalize_answer(right["prediction"])
        for left, right in zip(correct_records, shuffled_records)
    )
    correct_to_wrong = sum(
        left["official_denotation_correct"]
        and not right["official_denotation_correct"]
        for left, right in zip(correct_records, shuffled_records)
    )
    denominator = max(total, 1)
    return {
        "number_evaluated": total,
        "correct_table_denotation_accuracy": correct / denominator,
        "shuffled_table_denotation_accuracy": shuffled / denominator,
        "denotation_accuracy_drop": (correct - shuffled) / denominator,
        "prediction_change_rate": changed / denominator,
        "correct_to_wrong_count": correct_to_wrong,
    }


def _sample_pair(
    correct: dict[str, Any], shuffled: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": correct["id"],
        "question": correct["question"],
        "gold_answers": correct["official_gold_answers"],
        "correct_table_prediction": correct["prediction"],
        "shuffled_table_prediction": shuffled["prediction"],
        "legacy_any_answer_correct": correct["correct"],
        "official_denotation_correct": correct["official_denotation_correct"],
    }


def _collect_samples(
    correct_records: list[dict[str, Any]],
    shuffled_records: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[dict[str, Any]]] = {
        "representative": [],
        "multi_answer": [],
        "legacy_correct_but_denotation_wrong": [],
        "prediction_changed_after_shuffle": [],
        "correct_table_helped": [],
    }
    for correct, shuffled in zip(correct_records, shuffled_records):
        sample = _sample_pair(correct, shuffled)
        conditions = {
            "representative": True,
            "multi_answer": correct["official_answer_count"] > 1,
            "legacy_correct_but_denotation_wrong": bool(correct["correct"])
            and not bool(correct["official_denotation_correct"]),
            "prediction_changed_after_shuffle": normalize_answer(
                correct["prediction"]
            )
            != normalize_answer(shuffled["prediction"]),
            "correct_table_helped": bool(correct["official_denotation_correct"])
            and not bool(shuffled["official_denotation_correct"]),
        }
        for category, include in conditions.items():
            if include and len(categories[category]) < limit:
                categories[category].append(sample)
    return categories


def _print_samples(samples: dict[str, list[dict[str, Any]]]) -> None:
    for category, records in samples.items():
        print(f"\nSAMPLES: {category} ({len(records)})")
        for record in records:
            print(f"  [{record['id']}] {record['question']}")
            print(f"    gold:     {record['gold_answers']}")
            print(f"    correct:  {record['correct_table_prediction']!r}")
            print(f"    shuffled: {record['shuffled_table_prediction']!r}")


def main() -> None:
    args = parse_args()
    if args.max_examples < 2:
        raise ValueError("--max-examples must be at least 2")
    config = load_config(args.config)
    if args.max_new_tokens is not None:
        config.generation.max_new_tokens = args.max_new_tokens
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / f"checkpoint_{args.checkpoint}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Confirm Drive is mounted and training saved it."
        )
    output_dir = Path(args.output_dir or run_dir / "diagnostics/checkpoint_audit")
    official_cache = Path(
        args.official_cache_dir
        or run_dir.parent / "diagnostics" / "wtq_official_1.0.2"
    )
    tagged_data_dir = (
        Path(args.official_data_dir)
        if args.official_data_dir
        else ensure_official_tagged_data(official_cache)
    )
    print(f"Reading official WTQ targets from {tagged_data_dir}", flush=True)
    targets = load_official_targets(tagged_data_dir)
    print(f"Loaded {len(targets)} official targets", flush=True)

    set_seed(config.training.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    separator = "=" * 88
    print(f"\n{separator}")
    print("SAVED CHECKPOINT AUDIT (NO TRAINING)")
    print(f"CONFIG:     {Path(args.config).resolve()}")
    print(f"CHECKPOINT: {checkpoint_path}")
    print(f"SPLIT:      {args.split}")
    print(f"DEVICE:     {device} | DTYPE: {dtype}")
    print(separator, flush=True)

    all_examples = load_wtq(config.data.dataset, config.data.revision)[args.split]
    generated_examples = all_examples.select(
        range(min(args.max_examples, len(all_examples)))
    )
    shuffled_examples = build_table_shuffled_examples(generated_examples)

    tokenizer = load_tokenizer(config)
    model = build_model(config, tokenizer, device, dtype)
    checkpoint = load_trainable_checkpoint(model, checkpoint_path)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_best_metric = checkpoint.get("best_metric")
    print(
        f"Loaded checkpoint epoch={checkpoint_epoch}; "
        f"saved best legacy EM={checkpoint_best_metric}",
        flush=True,
    )

    correct_metrics, correct_records = evaluate_model(
        model,
        tokenizer,
        generated_examples,
        config,
        device,
        predictions_path=output_dir / "correct_tables/predictions.json",
        description="[checkpoint audit] correct tables",
    )
    shuffled_metrics, shuffled_records = evaluate_model(
        model,
        tokenizer,
        shuffled_examples,
        config,
        device,
        predictions_path=output_dir / "shuffled_tables/predictions.json",
        description="[checkpoint audit] shuffled tables",
    )
    _annotate_records(correct_records, targets)
    _annotate_records(shuffled_records, targets)
    write_json(correct_records, output_dir / "correct_tables/predictions.json")
    write_json(shuffled_records, output_dir / "shuffled_tables/predictions.json")

    legacy = table_dependence_metrics(correct_records, shuffled_records)
    legacy["correct_metrics"] = correct_metrics
    legacy["shuffled_metrics"] = shuffled_metrics
    official = _official_dependence_metrics(correct_records, shuffled_records)
    coverage = truncation_coverage(
        (dict(all_examples[index]) for index in range(len(all_examples))),
        targets,
        config.data.max_rows,
        config.data.max_cols,
        sample_limit=args.sample_count,
    )
    samples = _collect_samples(correct_records, shuffled_records, args.sample_count)
    summary = {
        "status": "completed",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_saved_best_legacy_em": checkpoint_best_metric,
        "split": args.split,
        "generation_examples": len(generated_examples),
        "coverage_examples": len(all_examples),
        "prediction_item_separator_policy": "pipe, semicolon, newline, or tab",
        "legacy_any_answer_metrics": legacy,
        "official_denotation_metrics": official,
        "truncation_coverage": coverage,
        "samples": samples,
    }
    write_json(summary, output_dir / "audit_summary.json")
    _print_samples(samples)

    print(f"\n{separator}")
    print("AUDIT SUMMARY")
    print(separator)
    print(
        "Legacy any-answer EM:  "
        f"correct={legacy['correct_table_exact_match']:.4f} | "
        f"shuffled={legacy['shuffled_table_exact_match']:.4f} | "
        f"drop={legacy['exact_match_drop']:.4f}"
    )
    print(
        "Official denotation:   "
        f"correct={official['correct_table_denotation_accuracy']:.4f} | "
        f"shuffled={official['shuffled_table_denotation_accuracy']:.4f} | "
        f"drop={official['denotation_accuracy_drop']:.4f}"
    )
    print(
        "Prediction changes:    "
        f"{official['prediction_change_rate']:.4f} after table shuffling"
    )
    print(
        "Multi-answer examples: "
        f"{coverage['multi_answer_rate']:.4f} "
        f"({coverage['multi_answer_count']}/{coverage['number_evaluated']})"
    )
    print(
        "Truncation removed a directly present complete answer: "
        f"{coverage['truncation_removed_rate_overall']:.4f} "
        f"({coverage['truncation_removed_direct_answer_count']}/"
        f"{coverage['number_evaluated']})"
    )
    print(f"\nSaved audit to: {output_dir / 'audit_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
