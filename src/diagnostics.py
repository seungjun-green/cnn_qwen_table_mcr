from __future__ import annotations

import json
from typing import Any

from .utils import normalize_answer


def build_table_shuffled_examples(dataset: Any) -> list[dict[str, Any]]:
    """Keep each question/gold answer but replace its table with a different one."""
    examples = [dict(dataset[index]) for index in range(len(dataset))]
    if len(examples) < 2:
        raise ValueError("Table-shuffling diagnostics require at least two examples")
    signatures = [
        json.dumps(example["table"], sort_keys=True, ensure_ascii=False)
        for example in examples
    ]
    shuffled: list[dict[str, Any]] = []
    initial_offset = max(1, len(examples) // 2)
    for index, example in enumerate(examples):
        replacement_index = (index + initial_offset) % len(examples)
        for _ in range(len(examples)):
            if signatures[replacement_index] != signatures[index]:
                break
            replacement_index = (replacement_index + 1) % len(examples)
        if signatures[replacement_index] == signatures[index]:
            raise ValueError("All selected validation examples use the same table")
        shuffled_example = dict(example)
        shuffled_example["table"] = examples[replacement_index]["table"]
        shuffled.append(shuffled_example)
    return shuffled


def table_dependence_metrics(
    correct_records: list[dict[str, Any]],
    shuffled_records: list[dict[str, Any]],
) -> dict[str, float | int]:
    if len(correct_records) != len(shuffled_records):
        raise ValueError("Correct and shuffled evaluations must have equal lengths")
    total = len(correct_records)
    correct_count = sum(bool(record["correct"]) for record in correct_records)
    shuffled_count = sum(bool(record["correct"]) for record in shuffled_records)
    changed = sum(
        normalize_answer(correct["prediction"])
        != normalize_answer(shuffled["prediction"])
        for correct, shuffled in zip(correct_records, shuffled_records)
    )
    correct_to_wrong = sum(
        bool(correct["correct"]) and not bool(shuffled["correct"])
        for correct, shuffled in zip(correct_records, shuffled_records)
    )
    denominator = max(total, 1)
    correct_em = correct_count / denominator
    shuffled_em = shuffled_count / denominator
    return {
        "number_evaluated": total,
        "correct_table_exact_match": correct_em,
        "shuffled_table_exact_match": shuffled_em,
        "exact_match_drop": correct_em - shuffled_em,
        "prediction_change_rate": changed / denominator,
        "correct_to_wrong_count": correct_to_wrong,
    }
