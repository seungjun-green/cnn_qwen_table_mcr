from __future__ import annotations

import re
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
STRUCTURED_2D_SYSTEM_PROMPT = (
    "A tokenized table with learned row, column, and cell-type structure precedes "
    "this conversation. Answer the user's question using that table. Return only "
    "the final answer; separate multiple answer items with |."
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


def _terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
    return {
        token
        for token in re.findall(r"[\w]+", str(text).casefold(), flags=re.UNICODE)
        if token and token not in stopwords
    }


def _is_question_mention(
    value: str, question_text: str, question_terms: set[str]
) -> bool:
    normalized = str(value).strip().casefold()
    if not normalized:
        return False
    if len(normalized) < 3 and normalized not in question_terms:
        return False
    return normalized in question_text


def select_table_for_question(
    table: Table,
    question: str,
    max_rows: int,
    max_cols: int,
    neighbor_radius: int = 1,
) -> Table:
    """Select relevant columns and rows, retaining neighbors for relational QA."""
    if max_rows < 1 or max_cols < 1:
        raise ValueError("max_rows and max_cols must be positive")
    question_terms = _terms(question)
    question_text = str(question).casefold()

    column_scores: list[tuple[float, int]] = []
    for column_index, header in enumerate(table.header):
        header_overlap = len(_terms(header) & question_terms)
        value_overlap = 0
        exact_mentions = 0
        for row in table.rows:
            if column_index >= len(row):
                continue
            value = row[column_index]
            value_overlap += len(_terms(value) & question_terms)
            if _is_question_mention(value, question_text, question_terms):
                exact_mentions += 1
        score = 8.0 * header_overlap + 2.0 * exact_mentions + value_overlap
        column_scores.append((score, column_index))
    ranked_columns = sorted(column_scores, key=lambda item: (-item[0], item[1]))
    selected_columns = sorted(
        index for _, index in ranked_columns[: min(max_cols, len(table.header))]
    )

    row_capacity = max(max_rows - 1, 0)
    row_scores: list[tuple[float, int]] = []
    for row_index, row in enumerate(table.rows):
        row_text = " ".join(row)
        overlap = len(_terms(row_text) & question_terms)
        exact_mentions = sum(
            1
            for value in row
            if _is_question_mention(value, question_text, question_terms)
        )
        row_scores.append((3.0 * exact_mentions + overlap, row_index))
    ranked_rows = sorted(row_scores, key=lambda item: (-item[0], item[1]))

    selected_row_set: set[int] = set()
    if row_capacity:
        for _, anchor in ranked_rows:
            offsets = [0]
            for distance in range(1, neighbor_radius + 1):
                offsets.extend([distance, -distance])
            for offset in offsets:
                candidate = anchor + offset
                if 0 <= candidate < len(table.rows):
                    selected_row_set.add(candidate)
                    if len(selected_row_set) >= row_capacity:
                        break
            if len(selected_row_set) >= row_capacity:
                break
        if len(selected_row_set) < row_capacity:
            for row_index in range(len(table.rows)):
                selected_row_set.add(row_index)
                if len(selected_row_set) >= row_capacity:
                    break
    selected_rows = sorted(selected_row_set)
    return Table(
        header=[table.header[index] for index in selected_columns],
        rows=[
            [table.rows[row_index][column_index] for column_index in selected_columns]
            for row_index in selected_rows
        ],
    )


def serialize_table(raw_table: Any, max_rows: int, max_cols: int) -> str:
    table = truncate_table(normalize_table(raw_table), max_rows, max_cols)
    lines = [" | ".join(table.header)]
    lines.extend(" | ".join(row) for row in table.rows)
    return "\n".join(lines)


def _serialize_table_with_cell_spans(
    raw_table: Any, max_rows: int, max_cols: int
) -> tuple[str, list[tuple[int, int, int]]]:
    """Serialize a table and retain character spans for each non-empty cell."""
    table = truncate_table(normalize_table(raw_table), max_rows, max_cols)
    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    grid = [table.header, *table.rows]
    for row_index, row in enumerate(grid):
        if row_index:
            pieces.append("\n")
            offset += 1
        for column_index, value in enumerate(row):
            if column_index:
                pieces.append(" | ")
                offset += 3
            value = str(value)
            start = offset
            pieces.append(value)
            offset += len(value)
            if value:
                spans.append(
                    (start, offset, row_index * max_cols + column_index)
                )
    return "".join(pieces), spans


def _render_chat_prompt(
    tokenizer: Any,
    question: str,
    system_prompt: str,
    enable_thinking: bool | None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(question)},
    ]
    if getattr(tokenizer, "chat_template", None):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return str(tokenizer.apply_chat_template(messages, **template_kwargs))
    return f"System: {system_prompt}\nUser: {question}\nAssistant:"


def _tokenize_with_offsets(
    tokenizer: Any, text: str
) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(
        offsets[0][0], list
    ):
        offsets = offsets[0]
    return list(input_ids), [tuple(pair) for pair in offsets]


def build_serialized_prompt_with_cell_alignment(
    tokenizer: Any,
    question: str,
    table: Any,
    max_rows: int,
    max_cols: int,
    enable_thinking: bool | None = False,
) -> tuple[list[int], list[int]]:
    """Build the normal serialized prompt plus a flat cell index per token."""
    table_text, local_spans = _serialize_table_with_cell_spans(
        table, max_rows, max_cols
    )
    user_content = f"Table:\n{table_text}\n\nQuestion: {question}"
    rendered = _render_chat_prompt(
        tokenizer,
        user_content,
        SERIALIZED_SYSTEM_PROMPT,
        enable_thinking,
    )
    encoded = _tokenize_with_offsets(tokenizer, rendered)
    if encoded is None:
        # Slow/custom tokenizers without offset mappings can still run the
        # serialized model, but cannot safely align repeated cell values.
        raise TypeError(
            "serialized_cnn_residual requires a fast tokenizer with "
            "return_offsets_mapping support"
        )
    input_ids, offsets = encoded
    content_start = rendered.find(user_content)
    if content_start < 0:
        raise ValueError("Could not locate the serialized user content in the prompt")
    table_start = content_start + len("Table:\n")
    spans = [
        (table_start + start, table_start + stop, cell_index)
        for start, stop, cell_index in local_spans
    ]
    alignment = [-1] * len(input_ids)
    span_cursor = 0
    for token_index, (token_start, token_stop) in enumerate(offsets):
        if token_stop <= token_start:
            continue
        while span_cursor < len(spans) and spans[span_cursor][1] <= token_start:
            span_cursor += 1
        candidate = span_cursor
        best_overlap = 0
        best_cell = -1
        while candidate < len(spans) and spans[candidate][0] < token_stop:
            start, stop, cell_index = spans[candidate]
            overlap = min(token_stop, stop) - max(token_start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cell = cell_index
            candidate += 1
        alignment[token_index] = best_cell
    if not any(cell_index >= 0 for cell_index in alignment):
        raise ValueError("No serialized table tokens could be aligned to table cells")
    return input_ids, alignment


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
    if experiment_type in {"serialized", "serialized_cnn_residual"}:
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
    if experiment_type == "structured_2d":
        return _prompt_ids(
            tokenizer,
            question,
            STRUCTURED_2D_SYSTEM_PROMPT,
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


def serialize_answers(
    answers: Sequence[str], mode: str = "first", separator: str = " | "
) -> str:
    if not answers:
        return ""
    if mode == "first":
        return str(answers[0])
    if mode == "all":
        return separator.join(str(answer) for answer in answers)
    raise ValueError(f"Unsupported answer mode: {mode}")


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
        answer_mode: str = "first",
        answer_separator: str = " | ",
        table_selection: str = "leading",
        selection_neighbor_radius: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.experiment_type = experiment_type
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens
        self.training = training
        self.answer_mode = answer_mode
        self.answer_separator = answer_separator
        self.table_selection = table_selection
        self.selection_neighbor_radius = selection_neighbor_radius
        self.enable_thinking = (
            False
            if experiment_type
            in {
                "continuous_prefix",
                "serialized",
                "structured_2d",
                "serialized_cnn_residual",
            }
            and enable_thinking is None
            else enable_thinking
        )

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        cell_alignments: list[list[int]] = []
        tables: list[Table] = []
        questions: list[str] = []
        example_ids: list[str] = []
        gold_answers: list[list[str]] = []
        original_shapes: list[tuple[int, int]] = []
        selected_shapes: list[tuple[int, int]] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id")

        for example in examples:
            question = str(example["question"])
            original_table = normalize_table(example["table"])
            if self.table_selection == "question_relevance":
                table = select_table_for_question(
                    original_table,
                    question,
                    self.max_rows,
                    self.max_cols,
                    self.selection_neighbor_radius,
                )
            else:
                table = truncate_table(original_table, self.max_rows, self.max_cols)
            answers = extract_answers(example)
            if self.training and not answers:
                raise ValueError("Training example has no answer")
            if self.experiment_type == "serialized_cnn_residual":
                prompt, prompt_alignment = build_serialized_prompt_with_cell_alignment(
                    self.tokenizer,
                    question,
                    table,
                    self.max_rows,
                    self.max_cols,
                    self.enable_thinking,
                )
                prompt = prompt[-self.max_question_tokens :]
                prompt_alignment = prompt_alignment[-self.max_question_tokens :]
            else:
                prompt = build_prompt_ids(
                    self.tokenizer,
                    question,
                    table,
                    self.experiment_type,
                    self.max_rows,
                    self.max_cols,
                    self.enable_thinking,
                )[-self.max_question_tokens :]
                prompt_alignment = [-1] * len(prompt)
            if self.training:
                answer_text = serialize_answers(
                    answers, self.answer_mode, self.answer_separator
                )
                answer_ids = list(
                    self.tokenizer.encode(answer_text, add_special_tokens=False)
                )[: self.max_answer_tokens]
                sequence = prompt + answer_ids + [eos_id]
                label = [-100] * len(prompt) + answer_ids + [eos_id]
                alignment = prompt_alignment + [-1] * (len(answer_ids) + 1)
            else:
                sequence = prompt
                label = [-100] * len(prompt)
                alignment = prompt_alignment
            sequences.append(sequence)
            labels.append(label)
            cell_alignments.append(alignment)
            tables.append(table)
            questions.append(question)
            example_ids.append(str(example.get("id", "")))
            gold_answers.append(answers)
            original_shapes.append(original_table.shape)
            selected_shapes.append(table.shape)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        label_tensor = torch.full((len(sequences), max_length), -100, dtype=torch.long)
        cell_index_tensor = torch.full(
            (len(sequences), max_length), -1, dtype=torch.long
        )
        for index, (sequence, label, alignment) in enumerate(
            zip(sequences, labels, cell_alignments)
        ):
            length = len(sequence)
            start = 0 if self.training else max_length - length
            stop = start + length
            input_ids[index, start:stop] = torch.tensor(sequence)
            attention_mask[index, start:stop] = 1
            label_tensor[index, start:stop] = torch.tensor(label)
            cell_index_tensor[index, start:stop] = torch.tensor(alignment)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "table_cell_indices": cell_index_tensor,
            "tables": tables,
            "questions": questions,
            "example_ids": example_ids,
            "gold_answers": gold_answers,
            "original_shapes": original_shapes,
            "selected_shapes": selected_shapes,
        }
