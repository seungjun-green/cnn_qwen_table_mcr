from __future__ import annotations

"""WikiTableQuestions denotation scoring and direct answer-coverage auditing.

The value normalization and matching rules are a Python 3 adaptation of the
WikiTableQuestions 1.0.2 evaluator:
https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py
"""

import csv
import math
import re
import shutil
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data import Table, normalize_table, truncate_table

WTQ_COMPACT_URL = (
    "https://github.com/ppasupat/WikiTableQuestions/releases/download/"
    "v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
)


def normalize_wtq_string(text: str) -> str:
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(text))
        if unicodedata.category(character) != "Mn"
    )
    value = re.sub(r"[‘’´`]", "'", value)
    value = re.sub(r"[“”]", '"', value)
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    while True:
        previous = value
        value = re.sub(
            r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", value.strip()
        )
        value = re.sub(r"(?<!^)( \([^)]*\))*$", "", value.strip())
        value = re.sub(r'^"([^"]*)"$', r"\1", value.strip())
        if value == previous:
            break
    if value.endswith("."):
        value = value[:-1]
    return re.sub(r"\s+", " ", value).lower().strip()


@dataclass(frozen=True)
class WTQValue:
    kind: str
    value: str | int | float | tuple[int, int, int]
    normalized: str = field(compare=False, hash=False)

    def matches(self, other: "WTQValue") -> bool:
        if self.normalized == other.normalized:
            return True
        if self.kind != other.kind:
            return False
        if self.kind == "number":
            return abs(float(self.value) - float(other.value)) < 1e-6
        return self.value == other.value


@dataclass(frozen=True)
class OfficialTarget:
    original_strings: tuple[str, ...]
    canonical_strings: tuple[str, ...]
    values: tuple[WTQValue, ...]


def _parse_number(text: str) -> int | float | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        try:
            amount = float(text)
        except (TypeError, ValueError):
            return None
        if math.isnan(amount) or math.isinf(amount):
            return None
        return amount


def _parse_date(text: str) -> tuple[int, int, int] | None:
    try:
        parts = str(text).lower().split("-")
        if len(parts) != 3:
            return None
        year = -1 if parts[0] in {"xx", "xxxx"} else int(parts[0])
        month = -1 if parts[1] == "xx" else int(parts[1])
        day = -1 if parts[2] == "xx" else int(parts[2])
        if year == month == day == -1:
            return None
        if month != -1 and not 1 <= month <= 12:
            return None
        if day != -1 and not 1 <= day <= 31:
            return None
        return year, month, day
    except (TypeError, ValueError):
        return None


def to_wtq_value(original: str, canonical: str | None = None) -> WTQValue:
    canonical = str(original) if canonical in {None, ""} else str(canonical)
    normalized = normalize_wtq_string(original)
    number = _parse_number(canonical)
    if number is not None:
        rounded = round(float(number))
        number = int(rounded) if abs(float(number) - rounded) < 1e-6 else float(number)
        return WTQValue("number", number, normalized)
    date = _parse_date(canonical)
    if date is not None:
        if date[1] == date[2] == -1:
            return WTQValue("number", date[0], normalized)
        return WTQValue("date", date, normalized)
    return WTQValue("string", normalized, normalized)


def to_wtq_values(
    originals: Sequence[str], canonicals: Sequence[str] | None = None
) -> tuple[WTQValue, ...]:
    if canonicals is not None and len(originals) != len(canonicals):
        raise ValueError("Original and canonical answer counts differ")
    converted = [
        to_wtq_value(original, None if canonicals is None else canonicals[index])
        for index, original in enumerate(originals)
    ]
    unique: list[WTQValue] = []
    for value in converted:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def check_denotation(
    target_values: Sequence[WTQValue], predicted_values: Sequence[WTQValue]
) -> bool:
    if len(target_values) != len(predicted_values):
        return False
    return all(
        any(target.matches(prediction) for prediction in predicted_values)
        for target in target_values
    )


def split_prediction_items(prediction: str) -> list[str]:
    """Convert model text to evaluator items without splitting commas in cells."""
    text = str(prediction).strip()
    if not text:
        return []
    items = [item.strip() for item in re.split(r"\s*(?:\||;|\n|\t)\s*", text)]
    return [item for item in items if item]


def score_prediction(prediction: str, target: OfficialTarget) -> bool:
    predicted_values = to_wtq_values(split_prediction_items(prediction))
    return check_denotation(target.values, predicted_values)


def _tsv_unescape(text: str) -> str:
    return text.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


def _tsv_unescape_list(text: str) -> tuple[str, ...]:
    return tuple(_tsv_unescape(item) for item in text.split("|"))


def load_official_targets(tagged_data_dir: str | Path) -> dict[str, OfficialTarget]:
    tagged_data_dir = Path(tagged_data_dir)
    files = sorted(tagged_data_dir.glob("*.tagged"))
    if not files:
        raise FileNotFoundError(f"No .tagged files found under {tagged_data_dir}")
    targets: dict[str, OfficialTarget] = {}
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                originals = _tsv_unescape_list(row["targetValue"])
                canonicals = _tsv_unescape_list(row["targetCanon"])
                targets[row["id"]] = OfficialTarget(
                    original_strings=originals,
                    canonical_strings=canonicals,
                    values=to_wtq_values(originals, canonicals),
                )
    return targets


def _safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in WTQ archive: {member.filename}")
        archive.extractall(destination)


def ensure_official_tagged_data(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    existing = sorted(cache_dir.glob("**/tagged/data")) if cache_dir.exists() else []
    for candidate in existing:
        if any(candidate.glob("*.tagged")):
            return candidate
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "WikiTableQuestions-1.0.2-compact.zip"
    if not archive_path.is_file():
        temporary = archive_path.with_suffix(".zip.tmp")
        print(f"Downloading official WikiTableQuestions metadata to {archive_path}")
        with urllib.request.urlopen(WTQ_COMPACT_URL) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        temporary.replace(archive_path)
    _safe_extract(archive_path, cache_dir)
    candidates = sorted(cache_dir.glob("**/tagged/data"))
    for candidate in candidates:
        if any(candidate.glob("*.tagged")):
            return candidate
    raise FileNotFoundError("The WTQ archive did not contain tagged/data/*.tagged")


def _table_values(table: Table) -> tuple[WTQValue, ...]:
    strings: list[str] = list(table.header)
    strings.extend(cell for row in table.rows for cell in row)
    return to_wtq_values(strings)


def target_is_directly_covered(target: OfficialTarget, table: Table) -> bool:
    cells = _table_values(table)
    return all(any(value.matches(cell) for cell in cells) for value in target.values)


def truncation_coverage(
    examples: Iterable[dict[str, Any]],
    targets: dict[str, OfficialTarget],
    max_rows: int,
    max_cols: int,
    sample_limit: int = 10,
) -> dict[str, Any]:
    total = full_covered = truncated_covered = removed = multi_answer = 0
    removed_samples: list[dict[str, Any]] = []
    for example in examples:
        example_id = str(example.get("id", ""))
        target = targets.get(example_id)
        if target is None:
            raise KeyError(f"Official target not found for example {example_id!r}")
        table = normalize_table(example["table"])
        shortened = truncate_table(table, max_rows, max_cols)
        full_has_answer = target_is_directly_covered(target, table)
        truncated_has_answer = target_is_directly_covered(target, shortened)
        total += 1
        full_covered += int(full_has_answer)
        truncated_covered += int(truncated_has_answer)
        multi_answer += int(len(target.values) > 1)
        was_removed = full_has_answer and not truncated_has_answer
        removed += int(was_removed)
        if was_removed and len(removed_samples) < sample_limit:
            removed_samples.append(
                {
                    "id": example_id,
                    "question": str(example["question"]),
                    "gold_answers": list(target.original_strings),
                    "original_rows": table.shape[0],
                    "original_cols": table.shape[1],
                    "kept_rows": shortened.shape[0],
                    "kept_cols": shortened.shape[1],
                }
            )
    denominator = max(total, 1)
    return {
        "number_evaluated": total,
        "multi_answer_count": multi_answer,
        "multi_answer_rate": multi_answer / denominator,
        "full_table_direct_coverage_count": full_covered,
        "full_table_direct_coverage_rate": full_covered / denominator,
        "truncated_table_direct_coverage_count": truncated_covered,
        "truncated_table_direct_coverage_rate": truncated_covered / denominator,
        "truncation_removed_direct_answer_count": removed,
        "truncation_removed_rate_overall": removed / denominator,
        "truncation_removed_rate_among_full_covered": removed
        / max(full_covered, 1),
        "removed_samples": removed_samples,
        "note": (
            "Direct coverage checks normalized string/number/date matches in table "
            "cells. Computed answers may correctly be absent from the full table."
        ),
    }
