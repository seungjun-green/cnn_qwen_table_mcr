from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_wtq
from src.evaluate import evaluate_model
from src.model import build_model, load_tokenizer, load_trainable_checkpoint
from src.utils import mirror_directory, model_dtype, select_device, set_seed
from src.wtq_evaluation import ensure_official_tagged_data, load_official_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Table-CNN MRC checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--mirror-output-dir",
        default=None,
        help="Also copy evaluation artifacts to this directory",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mirror_output_dir:
        config.training.mirror_output_dir = args.mirror_output_dir
    set_seed(config.training.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    tokenizer = load_tokenizer(config)
    dataset = load_wtq(config.data.dataset, config.data.revision)[args.split]
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    model = build_model(config, tokenizer, device, dtype)
    load_trainable_checkpoint(model, args.checkpoint)
    output_dir = Path(args.output_dir or config.training.output_dir) / args.split
    official_targets = None
    if config.evaluation.primary_metric == "denotation_accuracy":
        if config.evaluation.official_data_dir:
            tagged_data_dir = Path(config.evaluation.official_data_dir)
        else:
            cache_dir = Path(
                config.evaluation.official_cache_dir
                or output_dir.parent / "diagnostics" / "wtq_official_1.0.2"
            )
            tagged_data_dir = ensure_official_tagged_data(cache_dir)
        official_targets = load_official_targets(tagged_data_dir)
    run_name = Path(config.training.output_dir).name or "experiment"
    metrics, _ = evaluate_model(
        model,
        tokenizer,
        dataset,
        config,
        device,
        predictions_path=output_dir / "predictions.json",
        description=f"[{run_name}] {args.split} evaluation",
        official_targets=official_targets,
    )
    if config.training.mirror_output_dir:
        mirror_directory(
            output_dir,
            Path(config.training.mirror_output_dir) / args.split,
        )
    print(metrics)


if __name__ == "__main__":
    main()
