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
from src.model import build_model, load_tokenizer
from src.train import train_model
from src.utils import model_dtype, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Table-CNN MRC experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mirror-output-dir",
        default=None,
        help="Also copy run artifacts here at checkpoints and epoch boundaries",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-from",
        default=None,
        help="Resume from a specific full-state checkpoint",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing local and mirrored checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mirror_output_dir:
        config.training.mirror_output_dir = args.mirror_output_dir
    if args.resume_from:
        config.training.resume_from_checkpoint = args.resume_from
        config.training.auto_resume = False
    elif args.no_resume:
        config.training.resume_from_checkpoint = None
        config.training.auto_resume = False
    run_name = Path(config.training.output_dir).name or Path(args.config).stem
    separator = "=" * 88
    print(f"\n{separator}", flush=True)
    print(f"EXPERIMENT: {run_name}", flush=True)
    print(f"CONFIG:     {Path(args.config).resolve()}", flush=True)
    print(
        f"MODEL:      {config.model.name} | TYPE: {config.experiment_type} | "
        f"EPOCHS: {config.training.epochs}",
        flush=True,
    )
    if config.lora.enabled:
        print(
            f"LORA:       rank={config.lora.rank} | alpha={config.lora.alpha} | "
            f"dropout={config.lora.dropout} | "
            f"targets={','.join(config.lora.target_modules)}",
            flush=True,
        )
    print(f"OUTPUT:     {config.training.output_dir}", flush=True)
    if config.training.mirror_output_dir:
        print(f"DRIVE:      {config.training.mirror_output_dir}", flush=True)
    print(f"{separator}\n", flush=True)
    set_seed(config.training.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}")
    tokenizer = load_tokenizer(config)
    dataset = load_wtq(config.data.dataset, config.data.revision)
    model = build_model(config, tokenizer, device, dtype)
    history = train_model(
        model, tokenizer, dataset["train"], dataset["validation"], config, device
    )
    best_metric = history.get("best_exact_match")
    best_text = "n/a" if best_metric is None else f"{best_metric:.4f}"
    print(f"\n{separator}", flush=True)
    print(f"EXPERIMENT FINISHED: {run_name}", flush=True)
    print(f"STATUS: {history.get('status', 'unknown')} | BEST EXACT MATCH: {best_text}")
    print(f"{separator}\n", flush=True)


if __name__ == "__main__":
    main()
