from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import load_training_checkpoint, save_training_checkpoint
from .config import ExperimentConfig, save_config
from .data import MRCBatchCollator
from .evaluate import evaluate_model
from .utils import (
    mirror_directory,
    restore_directory_from_mirror,
    trainable_parameter_count,
    write_json,
)


def _limit(dataset: Any, maximum: int | None):
    if maximum is None:
        return dataset
    return dataset.select(range(min(maximum, len(dataset))))


def _training_loader(
    dataset: Any,
    collator: MRCBatchCollator,
    config: ExperimentConfig,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.training.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )


def train_model(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = Path(config.training.output_dir)
    run_name = output_dir.name or "experiment"
    mirror_output_dir = config.training.mirror_output_dir
    if config.training.auto_resume and mirror_output_dir:
        restored = restore_directory_from_mirror(mirror_output_dir, output_dir)
        if restored:
            print(f"[{run_name}] Restored run artifacts from {mirror_output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

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
        "loss_history": [],
        "resume_events": [],
        "status": "running",
        "started_at_unix": time.time(),
    }
    start_epoch = 0
    start_batch_index = 0
    global_step = 0
    best_metric: float | None = None
    epochs_without_improvement = 0
    resumed_epoch_loss = 0.0
    resumed_batches_seen = 0

    resume_path: Path | None = None
    if config.training.resume_from_checkpoint:
        resume_path = Path(config.training.resume_from_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif config.training.auto_resume:
        candidate = output_dir / "checkpoint_last.pt"
        if candidate.is_file():
            resume_path = candidate

    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path, model, optimizer, config, device
        )
        history = checkpoint["history"]
        history.setdefault("loss_history", [])
        history.setdefault("resume_events", [])
        history["status"] = "running"
        history["resume_events"].append(
            {
                "time_unix": time.time(),
                "checkpoint": str(resume_path),
                "epoch": checkpoint["epoch"],
                "next_batch_index": checkpoint["next_batch_index"],
                "global_step": checkpoint["global_step"],
            }
        )
        start_epoch = int(checkpoint["epoch"])
        start_batch_index = int(checkpoint["next_batch_index"])
        global_step = int(checkpoint["global_step"])
        best_metric = checkpoint["best_metric"]
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        resumed_epoch_loss = float(checkpoint.get("epoch_loss_sum", 0.0))
        resumed_batches_seen = int(checkpoint.get("epoch_batches_seen", 0))
        print(
            f"[{run_name}] Resuming epoch {start_epoch + 1} at batch "
            f"{start_batch_index}; "
            f"optimizer step {global_step}"
        )
        stopped_early = checkpoint.get("stop_reason") == "early_stopping"
        already_finished = start_epoch >= config.training.epochs
        if stopped_early or already_finished:
            history["status"] = "early_stopped" if stopped_early else "completed"
            print(
                f"[{run_name}] Run is already {history['status']}; "
                "no training is needed."
            )
            write_json(history, output_dir / "history.json")
            if mirror_output_dir:
                mirror_directory(output_dir, mirror_output_dir)
            return history

    if mirror_output_dir:
        mirror_directory(output_dir, mirror_output_dir)

    accumulation = config.training.gradient_accumulation_steps
    patience = config.training.early_stopping_patience
    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(start_epoch, config.training.epochs):
        loader = _training_loader(train_dataset, collator, config, epoch)
        if len(loader) == 0:
            raise ValueError("The training dataset is empty")
        resume_batch = start_batch_index if epoch == start_epoch else 0
        if resume_batch > len(loader):
            raise ValueError(
                f"Checkpoint batch index {resume_batch} exceeds epoch length {len(loader)}"
            )
        running_loss = resumed_epoch_loss if epoch == start_epoch else 0.0
        batches_seen = resumed_batches_seen if epoch == start_epoch else 0
        optimizer_steps = math.ceil(resume_batch / accumulation)
        step_loss_sum = 0.0
        microbatches_in_step = 0
        model.train()

        iterator = iter(loader)
        for _ in range(resume_batch):
            next(iterator)
        progress = tqdm(
            iterator,
            total=len(loader),
            initial=resume_batch,
            desc=f"[{run_name}] Train epoch {epoch + 1}/{config.training.epochs}",
        )
        for batch_index, batch in enumerate(progress, start=resume_batch):
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
            loss_value = float(raw_loss.detach().float().item())
            running_loss += loss_value
            batches_seen += 1
            step_loss_sum += loss_value
            microbatches_in_step += 1

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
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, config.training.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                step_loss = step_loss_sum / max(microbatches_in_step, 1)
                history["loss_history"].append(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "loss": step_loss,
                        "gradient_norm": float(gradient_norm),
                    }
                )
                step_loss_sum = 0.0
                microbatches_in_step = 0

                checkpoint_interval = config.training.checkpoint_every_steps
                if checkpoint_interval and global_step % checkpoint_interval == 0:
                    save_training_checkpoint(
                        output_dir / "checkpoint_last.pt",
                        model,
                        optimizer,
                        config,
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                        global_step=global_step,
                        history=history,
                        best_metric=best_metric,
                        epochs_without_improvement=epochs_without_improvement,
                        epoch_loss_sum=running_loss,
                        epoch_batches_seen=batches_seen,
                    )
                    write_json(history, output_dir / "history.json")
                    if mirror_output_dir:
                        mirror_directory(output_dir, mirror_output_dir)

            if (batch_index + 1) % config.training.log_every == 0 or should_step:
                progress.set_postfix(
                    loss=f"{running_loss / max(batches_seen, 1):.4f}",
                    step=global_step,
                )

        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "training_loss": running_loss / max(batches_seen, 1),
            "optimizer_steps": optimizer_steps,
            "global_step": global_step,
        }
        should_evaluate = (
            (epoch + 1) % config.training.eval_every_epochs == 0
            or epoch + 1 == config.training.epochs
        )
        improved = False
        if should_evaluate:
            validation_dir = output_dir / f"validation_epoch_{epoch + 1}"
            metrics, _ = evaluate_model(
                model,
                tokenizer,
                validation_dataset,
                config,
                device,
                predictions_path=validation_dir / "predictions.json",
                description=f"[{run_name}] Validation epoch {epoch + 1}",
            )
            epoch_record["validation"] = metrics
            metric = float(metrics["exact_match"])
            improved = best_metric is None or (
                metric > best_metric + config.training.early_stopping_min_delta
            )
            if improved:
                best_metric = metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epoch_record["best_exact_match"] = best_metric
            epoch_record["epochs_without_improvement"] = epochs_without_improvement
            improvement_text = "improved" if improved else "no improvement"
            patience_text = (
                "disabled"
                if patience is None
                else f"{epochs_without_improvement}/{patience}"
            )
            print(
                f"[{run_name}] Epoch {epoch + 1} validation Exact Match: "
                f"{metric:.4f} | best: {best_metric:.4f} | {improvement_text} | "
                f"patience: {patience_text}",
                flush=True,
            )
            stop_training = (
                not improved
                and patience is not None
                and epochs_without_improvement >= patience
            )

        history["epochs"].append(epoch_record)
        next_epoch = epoch + 1
        completed_all_epochs = next_epoch >= config.training.epochs
        stop_reason = "early_stopping" if stop_training else None
        training_complete = completed_all_epochs or stop_training
        history["status"] = (
            "early_stopped"
            if stop_training
            else "completed"
            if completed_all_epochs
            else "running"
        )
        if training_complete:
            history["finished_at_unix"] = time.time()
            history["best_exact_match"] = best_metric
        if improved:
            save_training_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                config,
                epoch=next_epoch,
                next_batch_index=0,
                global_step=global_step,
                history=history,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                training_complete=training_complete,
                stop_reason=stop_reason,
            )
        save_training_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            config,
            epoch=next_epoch,
            next_batch_index=0,
            global_step=global_step,
            history=history,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            training_complete=training_complete,
            stop_reason=stop_reason,
        )
        write_json(history, output_dir / "history.json")
        if mirror_output_dir:
            mirror_directory(output_dir, mirror_output_dir)
        if stop_training:
            print(
                f"[{run_name}] Early stopping after epoch {epoch + 1}: validation "
                f"Exact Match did not improve for {epochs_without_improvement} "
                "evaluation(s)."
            )
            break
        start_batch_index = 0
        resumed_epoch_loss = 0.0
        resumed_batches_seen = 0

    history["status"] = "early_stopped" if stop_training else "completed"
    history["finished_at_unix"] = time.time()
    history["best_exact_match"] = best_metric
    write_json(history, output_dir / "history.json")
    if mirror_output_dir:
        mirror_directory(output_dir, mirror_output_dir)
    return history
