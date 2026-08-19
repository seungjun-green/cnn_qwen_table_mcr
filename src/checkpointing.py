from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig

CHECKPOINT_VERSION = 1


def architecture_signature(config: ExperimentConfig) -> str:
    config_dict = config.to_dict()
    relevant = {
        "experiment_type": config_dict["experiment_type"],
        "model": config_dict["model"],
        "data": config_dict["data"],
        "cell_encoder": config_dict["cell_encoder"],
        "cnn": config_dict["cnn"],
        "cross_attention": config_dict["cross_attention"],
        "generation": config_dict["generation"],
        "training": {
            key: config_dict["training"][key]
            for key in (
                "batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "seed",
                "max_train_examples",
                "max_validation_examples",
            )
        },
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing = expected - set(state)
    if missing:
        raise RuntimeError(f"Missing trainable checkpoint keys: {sorted(missing)}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {list(incompatible.unexpected_keys)}"
        )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_optimizer_state(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_optimizer_state(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_optimizer_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_state(item, device) for item in value)
    return value


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    *,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    history: dict[str, Any],
    best_metric: float | None,
    epochs_without_improvement: int,
    epoch_loss_sum: float = 0.0,
    epoch_batches_seen: int = 0,
    training_complete: bool = False,
    stop_reason: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture_signature": architecture_signature(config),
        "config": config.to_dict(),
        "model_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "history": history,
        "best_metric": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_batches_seen": epoch_batches_seen,
        "training_complete": training_complete,
        "stop_reason": stop_reason,
        "rng_state": capture_rng_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    expected = architecture_signature(config)
    if checkpoint.get("architecture_signature") != expected:
        raise ValueError(
            "Checkpoint architecture does not match the current config. Use the "
            "matching config or start a new run with --no-resume."
        )
    load_trainable_state_dict(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_optimizer_state(value, device)
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
