from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def exact_match(prediction: str, gold_answers: Iterable[str]) -> bool:
    prediction = normalize_answer(prediction)
    return any(prediction == normalize_answer(gold) for gold in gold_answers)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_dtype(device: torch.device, use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if use_bf16 and device.type == "cuda" else torch.float32


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def mirror_directory(source: str | Path, destination: str | Path) -> None:
    """Atomically update a persistent mirror without removing existing files."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Mirror destination must be outside the source directory")
    if not source.is_dir():
        raise FileNotFoundError(f"Mirror source does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Refusing to mirror symbolic link: {item}")
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.name.endswith(".tmp"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            source_stat = item.stat()
            target_stat = target.stat()
            if (
                source_stat.st_size == target_stat.st_size
                and target_stat.st_mtime_ns == source_stat.st_mtime_ns
            ):
                continue
        temporary_target = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(item, temporary_target)
        temporary_target.replace(target)


def restore_directory_from_mirror(mirror: str | Path, destination: str | Path) -> bool:
    """Restore a run when Drive has a checkpoint newer than the local copy."""
    mirror = Path(mirror)
    destination = Path(destination)
    mirror_checkpoint = mirror / "checkpoint_last.pt"
    if not mirror_checkpoint.is_file():
        return False
    local_checkpoint = destination / "checkpoint_last.pt"
    if local_checkpoint.is_file():
        mirror_mtime = mirror_checkpoint.stat().st_mtime
        if local_checkpoint.stat().st_mtime > mirror_mtime:
            return False
    destination.mkdir(parents=True, exist_ok=True)
    mirror_directory(mirror, destination)
    return True


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item()) ** 2
    return total**0.5
