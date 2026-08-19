from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import MRCBatchCollator
from .utils import exact_match, write_json


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    config: ExperimentConfig,
    device: torch.device,
    predictions_path: str | Path | None = None,
    description: str = "Evaluating",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collator = MRCBatchCollator(
        tokenizer=tokenizer,
        experiment_type=config.experiment_type,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        max_question_tokens=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        training=False,
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
    for batch in tqdm(loader, desc=description):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tables=batch["tables"],
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, input_ids.shape[1] :]
        predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for question, prediction, golds, shape in zip(
            batch["questions"],
            predictions,
            batch["gold_answers"],
            batch["original_shapes"],
        ):
            prediction = prediction.strip()
            is_correct = exact_match(prediction, golds)
            correct += int(is_correct)
            records.append(
                {
                    "question": question,
                    "prediction": prediction,
                    "gold_answers": golds,
                    "correct": is_correct,
                    "table_rows": shape[0],
                    "table_cols": shape[1],
                }
            )
    total = len(records)
    metrics = {
        "exact_match": correct / total if total else 0.0,
        "number_evaluated": total,
        "number_correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        write_json(records, predictions_path)
        write_json(metrics, predictions_path.with_name("metrics.json"))
    return metrics, records
