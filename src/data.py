from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

SYSTEM_PROMPT = (
    "Answer the user's question using the table representation provided separately. "
    "Return only the final answer."
)
SERIALIZED_SYSTEM_PROMPT = (
    "Answer the user's question using the provided table. Return only the final answer."
)
CONTINUOUS_PREFIX_SYSTEM_PROMPT = (
    "A learned representation of the table precedes this conversation. "
    "Answer the user's question using that table. Return only the final answer."
)


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: list[list[str]]

    @property
    def shape(self) -> tuple[int, int]:
        return 1 + len(self.rows), len(self.header)


def normalize_table(raw: Any) -> Table:
    if isinstance(raw, Table):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Expected table dict, got {type(raw).__name__}")
    header = raw.get("header", raw.get("column_names", []))
    rows = raw.get("rows", raw.get("data", []))
    if hasattr(header, "tolist"):
        header = header.tolist()
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    header = [str(value) for value in header]
    normalized_rows = [[str(value) for value in row] for row in rows]
    if not header and normalized_rows:
        header = [f"column_{index}" for index in range(len(normalized_rows[0]))]
    width = len(header)
    normalized_rows = [(row + [""] * width)[:width] for row in normalized_rows]
    return Table(header=header, rows=normalized_rows)


def truncate_table(table: Table, max_rows: int, max_cols: int) -> Table:
    """Keep the header plus leading rows, within a total max_rows grid height."""
    columns = min(len(table.header), max_cols)
    data_rows = max(max_rows - 1, 0)
    return Table(
        header=table.header[:columns],
        rows=[row[:columns] for row in table.rows[:data_rows]],
    )


def serialize_table(raw_table: Any, max_rows: int, max_cols: int) -> str:
    table = truncate_table(normalize_table(raw_table), max_rows, max_cols)
    lines = [" | ".join(table.header)]
    lines.extend(" | ".join(row) for row in table.rows)
    return "\n".join(lines)


def load_wtq(dataset_name: str, revision: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, revision=revision)


def _prompt_ids(
    tokenizer: Any,
    question: str,
    system_prompt: str,
    enable_thinking: bool | None = None,
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(question)},
    ]
    if getattr(tokenizer, "chat_template", None):
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        ids = tokenizer.apply_chat_template(messages, **template_kwargs)
        return list(ids)
    prompt = f"System: {system_prompt}\nUser: {question}\nAssistant:"
    return list(tokenizer.encode(prompt, add_special_tokens=True))


def build_prompt_ids(
    tokenizer: Any,
    question: str,
    table: Any | None = None,
    experiment_type: str = "cnn",
    max_rows: int = 32,
    max_cols: int = 8,
    enable_thinking: bool | None = None,
) -> list[int]:
    if experiment_type == "serialized":
        text_table = serialize_table(table, max_rows, max_cols)
        question = f"Table:\n{text_table}\n\nQuestion: {question}"
        return _prompt_ids(
            tokenizer,
            question,
            SERIALIZED_SYSTEM_PROMPT,
            enable_thinking=enable_thinking,
        )
    if experiment_type == "continuous_prefix":
        return _prompt_ids(
            tokenizer,
            question,
            CONTINUOUS_PREFIX_SYSTEM_PROMPT,
            enable_thinking=enable_thinking,
        )
    return _prompt_ids(
        tokenizer,
        question,
        SYSTEM_PROMPT,
        enable_thinking=enable_thinking,
    )


def extract_answers(example: dict[str, Any]) -> list[str]:
    answers = example.get("answers", example.get("answer", []))
    if isinstance(answers, str):
        return [answers]
    return [str(answer) for answer in answers]


class MRCBatchCollator:
    def __init__(
        self,
        tokenizer: Any,
        experiment_type: str,
        max_rows: int,
        max_cols: int,
        max_question_tokens: int,
        max_answer_tokens: int,
        training: bool,
        enable_thinking: bool | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.experiment_type = experiment_type
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens
        self.training = training
        self.enable_thinking = (
            False
            if experiment_type == "continuous_prefix" and enable_thinking is None
            else enable_thinking
        )

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        tables: list[Table] = []
        questions: list[str] = []
        example_ids: list[str] = []
        gold_answers: list[list[str]] = []
        original_shapes: list[tuple[int, int]] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id")

        for example in examples:
            question = str(example["question"])
            table = normalize_table(example["table"])
            answers = extract_answers(example)
            if self.training and not answers:
                raise ValueError("Training example has no answer")
            prompt = build_prompt_ids(
                self.tokenizer,
                question,
                table,
                self.experiment_type,
                self.max_rows,
                self.max_cols,
                self.enable_thinking,
            )[-self.max_question_tokens :]
            if self.training:
                answer_ids = list(
                    self.tokenizer.encode(answers[0], add_special_tokens=False)
                )[: self.max_answer_tokens]
                sequence = prompt + answer_ids + [eos_id]
                label = [-100] * len(prompt) + answer_ids + [eos_id]
            else:
                sequence = prompt
                label = [-100] * len(prompt)
            sequences.append(sequence)
            labels.append(label)
            tables.append(truncate_table(table, self.max_rows, self.max_cols))
            questions.append(question)
            example_ids.append(str(example.get("id", "")))
            gold_answers.append(answers)
            original_shapes.append(table.shape)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        label_tensor = torch.full((len(sequences), max_length), -100, dtype=torch.long)
        for index, (sequence, label) in enumerate(zip(sequences, labels)):
            length = len(sequence)
            start = 0 if self.training else max_length - length
            stop = start + length
            input_ids[index, start:stop] = torch.tensor(sequence)
            attention_mask[index, start:stop] = 1
            label_tensor[index, start:stop] = torch.tensor(label)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "tables": tables,
            "questions": questions,
            "example_ids": example_ids,
            "gold_answers": gold_answers,
            "original_shapes": original_shapes,
        }
