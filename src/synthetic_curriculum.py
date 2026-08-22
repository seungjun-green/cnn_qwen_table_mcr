from __future__ import annotations

import csv
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .data import (
    SERIALIZED_SYSTEM_PROMPT,
    extract_answers,
    normalize_table,
    select_table_for_question,
    serialize_answers,
    serialize_table,
    truncate_table,
)

SYNTHETIC_FILENAMES = tuple(
    f"dataset_level_{level}.csv" for level in range(1, 6)
)
MARKDOWN_SEPARATOR = re.compile(r"^:?-{3,}:?$")


class SFTRecordDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def normalize_synthetic_prompt_to_wtq(prompt: str) -> str:
    """Convert a curriculum Markdown table prompt to the WTQ serializer format."""
    table_marker = "Table:\n"
    question_marker = "\n\nQuestion:"
    table_start = prompt.find(table_marker)
    question_start = prompt.find(question_marker, table_start + len(table_marker))
    if table_start < 0 or question_start < 0:
        raise ValueError("Synthetic prompt must contain Table and Question sections")

    markdown_table = prompt[
        table_start + len(table_marker) : question_start
    ].strip()
    question = prompt[question_start + len(question_marker) :].strip()
    if not markdown_table or not question:
        raise ValueError("Synthetic prompt contains an empty table or question")

    serialized_rows: list[str] = []
    for line in markdown_table.splitlines():
        line = line.strip()
        if not line:
            continue
        if not (line.startswith("|") and line.endswith("|")):
            raise ValueError(f"Expected a Markdown table row, received: {line!r}")
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if cells and all(MARKDOWN_SEPARATOR.fullmatch(cell) for cell in cells):
            continue
        serialized_rows.append(" | ".join(cells))

    if len(serialized_rows) < 2:
        raise ValueError("Synthetic prompt table must contain a header and data row")
    serialized_table = "\n".join(serialized_rows)
    return f"Table:\n{serialized_table}\n\nQuestion: {question}"


def load_synthetic_level(
    data_root: str | Path,
    level: int,
    normalize_wtq_format: bool = False,
) -> list[dict[str, str]]:
    if level not in range(1, 6):
        raise ValueError("Synthetic curriculum level must be between 1 and 5")
    path = Path(data_root) / f"dataset_level_{level}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing synthetic curriculum file: {path}. Commit all five "
            "dataset_level_*.csv files before running Colab."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"Synthetic curriculum file is empty: {path}")
    missing_columns = {"prompt", "answer"} - set(records[0])
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    for row_index, record in enumerate(records, start=2):
        for field in ("prompt", "answer"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"{path}:{row_index} has an empty {field!r} field")
        if normalize_wtq_format:
            try:
                record["prompt"] = normalize_synthetic_prompt_to_wtq(record["prompt"])
            except ValueError as error:
                raise ValueError(f"{path}:{row_index}: {error}") from error
        record["source"] = "synthetic"
        record["curriculum_level"] = str(level)
        record["prompt_format"] = (
            "wtq_serialized" if normalize_wtq_format else "synthetic_markdown"
        )
    return records


def load_all_synthetic_levels(
    data_root: str | Path,
    normalize_wtq_format: bool = False,
) -> dict[int, list[dict[str, str]]]:
    return {
        level: load_synthetic_level(data_root, level, normalize_wtq_format)
        for level in range(1, 6)
    }


def synthetic_level_summary(records: Sequence[dict[str, str]]) -> dict[str, Any]:
    task_types = Counter(
        str(record.get("task_type") or record.get("operation") or "unknown")
        for record in records
    )
    return {
        "rows": len(records),
        "task_type_distribution": dict(sorted(task_types.items())),
        "first_example": {
            "id": records[0].get("id", ""),
            "prompt": records[0]["prompt"],
            "answer": records[0]["answer"],
        },
    }


def _chat_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": SERIALIZED_SYSTEM_PROMPT},
        {"role": "user", "content": str(prompt)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    text = (
        f"System: {SERIALIZED_SYSTEM_PROMPT}\n"
        f"User: {prompt}\nAssistant:"
    )
    return list(tokenizer.encode(text, add_special_tokens=True))


class SyntheticSFTCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_sequence_length: int,
        max_answer_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_answer_tokens = max_answer_tokens

    def __call__(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        for record in records:
            prompt_ids = _chat_prompt_ids(self.tokenizer, str(record["prompt"]))
            answer_ids = list(
                self.tokenizer.encode(
                    str(record["answer"]), add_special_tokens=False
                )
            )[: self.max_answer_tokens]
            prompt_budget = self.max_sequence_length - len(answer_ids) - 1
            if prompt_budget < 1:
                raise ValueError(
                    "max_sequence_length is too small for the configured answer length"
                )
            prompt_ids = prompt_ids[-prompt_budget:]
            sequences.append(prompt_ids + answer_ids + [eos_id])
            labels.append([-100] * len(prompt_ids) + answer_ids + [eos_id])

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id
        maximum = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), maximum), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        label_tensor = torch.full_like(input_ids, -100)
        for index, (sequence, label) in enumerate(zip(sequences, labels)):
            length = len(sequence)
            input_ids[index, :length] = torch.tensor(sequence)
            attention_mask[index, :length] = 1
            label_tensor[index, :length] = torch.tensor(label)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "ids": [str(record.get("id", "")) for record in records],
            "sources": [str(record.get("source", "synthetic")) for record in records],
        }


def wtq_training_records(dataset: Any, data_config: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for example in dataset:
        question = str(example["question"])
        table = normalize_table(example["table"])
        if data_config.table_selection == "question_relevance":
            table = select_table_for_question(
                table,
                question,
                data_config.max_rows,
                data_config.max_cols,
                data_config.selection_neighbor_radius,
            )
        else:
            table = truncate_table(
                table, data_config.max_rows, data_config.max_cols
            )
        answers = extract_answers(example)
        records.append(
            {
                "id": str(example.get("id", "")),
                "source": "wtq",
                "prompt": (
                    f"Table:\n{serialize_table(table, data_config.max_rows, data_config.max_cols)}"
                    f"\n\nQuestion: {question}"
                ),
                "answer": serialize_answers(
                    answers,
                    data_config.answer_mode,
                    data_config.answer_separator,
                ),
            }
        )
    return records


def build_mixed_dataset(
    synthetic_records: Sequence[dict[str, Any]],
    wtq_records: Sequence[dict[str, Any]],
    ratio: tuple[int, int],
    seed: int,
    total_examples: int | None = None,
) -> SFTRecordDataset:
    synthetic_weight, wtq_weight = ratio
    if synthetic_weight < 0 or wtq_weight < 0 or synthetic_weight + wtq_weight <= 0:
        raise ValueError("Synthetic:WTQ ratio must contain non-negative weights")
    if not synthetic_records or not wtq_records:
        raise ValueError("Both synthetic and WTQ records are required for mixing")
    total = total_examples or len(wtq_records)
    synthetic_count = round(total * synthetic_weight / (synthetic_weight + wtq_weight))
    wtq_count = total - synthetic_count
    generator = random.Random(seed)
    mixed = [generator.choice(synthetic_records) for _ in range(synthetic_count)]
    mixed.extend(generator.choice(wtq_records) for _ in range(wtq_count))
    generator.shuffle(mixed)
    return SFTRecordDataset(mixed)
