from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import MRCBatchCollator
from .utils import exact_match, write_json
from .wtq_evaluation import OfficialTarget, score_prediction


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    config: ExperimentConfig,
    device: torch.device,
    predictions_path: str | Path | None = None,
    description: str = "Evaluating",
    enable_thinking: bool | None = None,
    official_targets: dict[str, OfficialTarget] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collator = MRCBatchCollator(
        tokenizer=tokenizer,
        experiment_type=config.experiment_type,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        max_question_tokens=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        training=False,
        enable_thinking=enable_thinking,
        answer_mode=config.data.answer_mode,
        answer_separator=config.data.answer_separator,
        table_selection=config.data.table_selection,
        selection_neighbor_radius=config.data.selection_neighbor_radius,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    model.eval()
    records: list[dict[str, Any]] = []
    correct = 0
    denotation_correct = 0
    plain_progress = os.environ.get("TABLE_MRC_PLAIN_PROGRESS") == "1"
    plain_log_every = max(
        int(os.environ.get("TABLE_MRC_PLAIN_EVAL_EVERY", "250")), 1
    )
    for batch_index, batch in enumerate(
        tqdm(loader, desc=description, disable=plain_progress), start=1
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generation_kwargs: dict[str, Any] = {}
        if config.experiment_type in {
            "serialized_cnn_residual",
            "serialized_gnn_residual",
        }:
            generation_kwargs["table_cell_indices"] = batch[
                "table_cell_indices"
            ].to(device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tables=batch["tables"],
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        new_tokens = generated[:, input_ids.shape[1] :]
        predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for example_id, question, prediction, golds, shape, selected_shape in zip(
            batch["example_ids"],
            batch["questions"],
            predictions,
            batch["gold_answers"],
            batch["original_shapes"],
            batch["selected_shapes"],
        ):
            prediction = prediction.strip()
            is_correct = exact_match(prediction, golds)
            correct += int(is_correct)
            is_denotation_correct: bool | None = None
            if official_targets is not None:
                target = official_targets.get(str(example_id))
                if target is None:
                    raise KeyError(
                        f"Official WTQ target not found for example {example_id!r}"
                    )
                is_denotation_correct = score_prediction(prediction, target)
                denotation_correct += int(is_denotation_correct)
            records.append(
                {
                    "id": example_id,
                    "question": question,
                    "prediction": prediction,
                    "gold_answers": golds,
                    "correct": is_correct,
                    "denotation_correct": is_denotation_correct,
                    "table_rows": shape[0],
                    "table_cols": shape[1],
                    "selected_table_rows": selected_shape[0],
                    "selected_table_cols": selected_shape[1],
                }
            )
        if plain_progress and (
            batch_index % plain_log_every == 0 or batch_index == len(loader)
        ):
            print(
                f"{description}: {batch_index}/{len(loader)} batches evaluated",
                flush=True,
            )
    total = len(records)
    metrics = {
        "exact_match": correct / total if total else 0.0,
        "number_evaluated": total,
        "number_correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    if official_targets is not None:
        metrics["denotation_accuracy"] = denotation_correct / total if total else 0.0
        metrics["number_denotation_correct"] = denotation_correct
    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        write_json(records, predictions_path)
        write_json(metrics, predictions_path.with_name("metrics.json"))
    return metrics, records
