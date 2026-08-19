from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        help="Also copy run artifacts to this directory after every epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mirror_output_dir:
        config.training.mirror_output_dir = args.mirror_output_dir
    set_seed(config.training.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}")
    tokenizer = load_tokenizer(config)
    dataset = load_wtq(config.data.dataset, config.data.revision)
    model = build_model(config, tokenizer, device, dtype)
    train_model(
        model, tokenizer, dataset["train"], dataset["validation"], config, device
    )


if __name__ == "__main__":
    main()
