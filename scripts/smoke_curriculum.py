from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.curriculum import (
    CurriculumRunConfig,
    CurriculumRunner,
    load_curriculum_checkpoint,
)
from src.data import load_wtq
from src.model import build_model, load_tokenizer
from src.synthetic_curriculum import (
    load_all_synthetic_levels,
    synthetic_level_summary,
)
from src.utils import model_dtype, select_device, set_seed
from src.wtq_evaluation import ensure_official_tagged_data, load_official_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the curriculum pipeline")
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs/serialized_table_lora.yaml"),
    )
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--official-cache-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = load_all_synthetic_levels(args.data_root)
    for level, records in levels.items():
        summary = synthetic_level_summary(records)
        print(
            f"Level {level}: rows={summary['rows']} | "
            f"tasks={summary['task_type_distribution']}"
        )
    config = load_config(args.base_config)
    config.training.max_validation_examples = 1
    set_seed(config.training.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    tokenizer = load_tokenizer(config)
    dataset = load_wtq(config.data.dataset, config.data.revision)
    model = build_model(config, tokenizer, device, dtype)
    tagged_data = ensure_official_tagged_data(args.official_cache_dir)
    official_targets = load_official_targets(tagged_data)
    with tempfile.TemporaryDirectory(prefix="table_curriculum_smoke_") as temporary:
        config.training.output_dir = temporary
        run_config = CurriculumRunConfig(
            data_root=args.data_root,
            output_dir=temporary,
            levels=(1,),
            epochs_per_level=1,
            learning_rate=5.0e-5,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_sequence_length=2048,
            max_answer_tokens=config.data.max_answer_tokens,
            max_examples_per_level=1,
            checkpoint_every_steps=1,
            log_every_steps=1,
            seed=config.training.seed,
        )
        runner = CurriculumRunner(
            model,
            tokenizer,
            dataset,
            config,
            run_config,
            device,
            official_targets,
            {"smoke_test": True},
        )
        state = runner.run()
        if len(state["results"]) != 2:
            raise AssertionError("Smoke test did not produce base and Level-1 results")
        if not runner.root_checkpoint.is_file():
            raise AssertionError("Smoke test did not save checkpoint_last.pt")
        restored = load_curriculum_checkpoint(
            runner.root_checkpoint,
            runner.model,
            runner.optimizer,
            runner.signature,
            device,
        )
        if restored["phase"] != "curriculum_complete":
            raise AssertionError("Reloaded curriculum checkpoint has invalid state")
    print("Curriculum smoke test passed.")


if __name__ == "__main__":
    main()
