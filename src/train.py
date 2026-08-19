from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ExperimentConfig, save_config
from .data import MRCBatchCollator
from .evaluate import evaluate_model
from .model import save_trainable_checkpoint
from .utils import mirror_directory, trainable_parameter_count, write_json


def _limit(dataset: Any, maximum: int | None):
    if maximum is None:
        return dataset
    return dataset.select(range(min(maximum, len(dataset))))


def train_model(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    mirror_output_dir = config.training.mirror_output_dir
    if mirror_output_dir:
        mirror_directory(output_dir, mirror_output_dir)
    train_dataset = _limit(train_dataset, config.training.max_train_examples)
    validation_dataset = _limit(
        validation_dataset, config.training.max_validation_examples
    )
    collator = MRCBatchCollator(
        tokenizer=tokenizer,
        experiment_type=config.experiment_type,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        max_question_tokens=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        training=True,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError(
            "The selected model has no trainable parameters. Unfreeze the serialized "
            "baseline or use the CNN model."
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: dict[str, Any] = {
        "trainable_parameters": trainable_parameter_count(model),
        "epochs": [],
        "started_at_unix": time.time(),
    }
    accumulation = config.training.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(config.training.epochs):
        model.train()
        running_loss = 0.0
        optimizer_steps = 0
        progress = tqdm(loader, desc=f"Train epoch {epoch + 1}")
        for batch_index, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            use_autocast = config.training.bf16 and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    tables=batch["tables"],
                    use_cache=False,
                )
                raw_loss = outputs.loss
                loss = raw_loss / accumulation
            loss.backward()
            running_loss += float(raw_loss.detach().float().item())
            should_step = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(loader)
            )
            if should_step:
                remainder = len(loader) % accumulation
                if batch_index + 1 == len(loader) and remainder:
                    correction = accumulation / remainder
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(
                    parameters, config.training.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
            if (batch_index + 1) % config.training.log_every == 0:
                progress.set_postfix(loss=f"{running_loss / (batch_index + 1):.4f}")

        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "training_loss": running_loss / max(len(loader), 1),
            "optimizer_steps": optimizer_steps,
            "global_step": global_step,
        }
        should_evaluate = (
            (epoch + 1) % config.training.eval_every_epochs == 0
            or epoch + 1 == config.training.epochs
        )
        if should_evaluate:
            validation_dir = output_dir / f"validation_epoch_{epoch + 1}"
            metrics, _ = evaluate_model(
                model,
                tokenizer,
                validation_dataset,
                config,
                device,
                predictions_path=validation_dir / "predictions.json",
                description=f"Validation epoch {epoch + 1}",
            )
            epoch_record["validation"] = metrics
        history["epochs"].append(epoch_record)
        save_trainable_checkpoint(model, output_dir / "checkpoint_last.pt")
        write_json(history, output_dir / "history.json")
        if mirror_output_dir:
            mirror_directory(output_dir, mirror_output_dir)
    history["finished_at_unix"] = time.time()
    write_json(history, output_dir / "history.json")
    if mirror_output_dir:
        mirror_directory(output_dir, mirror_output_dir)
    return history
