from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .utils import write_json


def deterministic_subset_indices(
    dataset_size: int,
    fraction: float,
    seed: int,
) -> list[int]:
    """Return a reproducible subset; smaller fractions nest inside larger ones."""
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1]")
    count = max(1, round(dataset_size * fraction))
    order = list(range(dataset_size))
    random.Random(seed).shuffle(order)
    return sorted(order[:count])


def subset_fingerprint(indices: list[int]) -> str:
    encoded = ",".join(str(index) for index in indices).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def condition_name(fraction: float, initialization: str) -> str:
    percentage = fraction * 100
    if not percentage.is_integer():
        raise ValueError("WTQ fraction must represent a whole percentage")
    return f"wtq_{int(percentage):02d}pct_{initialization}"


def _best_epoch(history: dict[str, Any]) -> tuple[int | None, float | None]:
    metric_name = str(history.get("primary_metric", "denotation_accuracy"))
    candidates = [
        (int(record["epoch"]), float(record["validation"][metric_name]))
        for record in history.get("epochs", [])
        if metric_name in record.get("validation", {})
    ]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def write_data_efficiency_summary(output_root: str | Path) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(output_root.glob("wtq_*pct_*/run_metadata.json")):
        run_dir = metadata_path.parent
        history_path = run_dir / "history.json"
        if not history_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best_epoch, best_score = _best_epoch(history)
        rows.append(
            {
                "wtq_fraction": float(metadata["wtq_fraction"]),
                "wtq_percentage": int(metadata["wtq_percentage"]),
                "initialization": str(metadata["initialization"]),
                "training_examples": int(metadata["training_examples"]),
                "subset_fingerprint": str(metadata["subset_fingerprint"]),
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "status": str(history.get("status", "unknown")),
                "best_checkpoint": str(run_dir / "checkpoint_best.pt"),
            }
        )

    by_percentage: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_percentage.setdefault(row["wtq_percentage"], {})[
            row["initialization"]
        ] = row
    for conditions in by_percentage.values():
        base = conditions.get("base")
        curriculum = conditions.get("curriculum")
        delta = None
        if (
            base
            and curriculum
            and base["best_validation_score"] is not None
            and curriculum["best_validation_score"] is not None
        ):
            if base["subset_fingerprint"] != curriculum["subset_fingerprint"]:
                raise ValueError("Paired runs used different WTQ training subsets")
            delta = (
                curriculum["best_validation_score"]
                - base["best_validation_score"]
            )
        for row in conditions.values():
            row["curriculum_delta_vs_base"] = delta

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(rows, results_dir / "data_efficiency_results.json")
    if rows:
        with (results_dir / "data_efficiency_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
