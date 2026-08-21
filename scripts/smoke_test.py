from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import MRCBatchCollator, load_wtq, normalize_table
from src.model import (
    ContinuousPrefixQwen,
    Structured2DQwen,
    TableCNNQwen,
    build_model,
    load_tokenizer,
)
from src.utils import gradient_norm, model_dtype, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-example end-to-end smoke test")
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def assert_gradient(name: str, module: torch.nn.Module) -> float:
    norm = gradient_norm(module)
    if not math.isfinite(norm) or norm <= 0:
        raise AssertionError(
            f"Expected a finite non-zero gradient for {name}, got {norm}"
        )
    print(f"{name} gradient norm: {norm:.8f}")
    return norm


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    supported_types = {"cnn", "continuous_prefix", "serialized", "structured_2d"}
    if config.experiment_type not in supported_types:
        raise ValueError(
            "Unsupported experiment type for the smoke test"
        )
    set_seed(config.training.seed)
    device = select_device()
    dtype = model_dtype(device, config.training.bf16)
    print(f"Device: {device}; dtype: {dtype}")
    tokenizer = load_tokenizer(config)
    example = load_wtq(config.data.dataset, config.data.revision)[args.split][
        args.index
    ]
    table = normalize_table(example["table"])
    answers = example.get("answers", example.get("answer", []))
    print(f"Question: {example['question']}")
    print(f"Gold answers: {answers}")
    print(f"Original table shape (including header): {table.shape}")

    model = build_model(config, tokenizer, device, dtype)
    collator = MRCBatchCollator(
        tokenizer,
        experiment_type=config.experiment_type,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        max_question_tokens=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        training=True,
        answer_mode=config.data.answer_mode,
        answer_separator=config.data.answer_separator,
        table_selection=config.data.table_selection,
        selection_neighbor_radius=config.data.selection_neighbor_radius,
    )
    batch = collator([example])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    model.train()
    if isinstance(model, (TableCNNQwen, ContinuousPrefixQwen, Structured2DQwen)):
        memory, memory_mask, shapes = model.encode_tables(batch["tables"])
        if "cell_grid" in shapes:
            print(f"After cell MLP: {shapes['cell_grid']}")
            print(f"After 2D CNN: {shapes['cnn_output']}")
            print(f"Flattened CNN memory: {shapes['flattened_cnn']}")
            print(f"Projected table memory: {shapes['table_memory']}")
        if "table_tokens" in shapes:
            print(f"Structured table tokens: {shapes['table_tokens']}")
        if "table_prefix" in shapes:
            print(f"Table prefix: {shapes['table_prefix']}")
        print(f"Valid table positions: {int(memory_mask.sum())}")
        del memory

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        tables=batch["tables"],
        use_cache=False,
    )
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss}")
    print(f"Loss: {float(loss.detach().float()):.8f}")
    loss.backward()
    if isinstance(model, (TableCNNQwen, ContinuousPrefixQwen)):
        assert_gradient("cell encoder", model.cell_encoder)
        assert_gradient("CNN", model.table_cnn)
        assert_gradient("projector", model.projector)
    if isinstance(model, Structured2DQwen):
        assert_gradient("row embeddings", model.row_embeddings)
        assert_gradient("column embeddings", model.column_embeddings)
        assert_gradient("cell type embeddings", model.cell_type_embeddings)
    if config.lora.enabled:
        assert_gradient("LoRA adapters", model.language_model)
    if isinstance(model, TableCNNQwen):
        assert_gradient("cross-attention", model.cross_attention)

    model.eval()
    eval_collator = MRCBatchCollator(
        tokenizer,
        experiment_type=config.experiment_type,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        max_question_tokens=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        training=False,
        answer_mode=config.data.answer_mode,
        answer_separator=config.data.answer_separator,
        table_selection=config.data.table_selection,
        selection_neighbor_radius=config.data.selection_neighbor_radius,
    )
    generation_batch = eval_collator([example])
    generation_input = generation_batch["input_ids"].to(device)
    generation_mask = generation_batch["attention_mask"].to(device)
    generated = model.generate(
        input_ids=generation_input,
        attention_mask=generation_mask,
        tables=generation_batch["tables"],
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(
        generated[0, generation_input.shape[1] :], skip_special_tokens=True
    ).strip()
    print(f"Generated answer (accuracy is not checked): {answer!r}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
