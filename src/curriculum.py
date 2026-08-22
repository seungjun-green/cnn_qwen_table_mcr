from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import (
    capture_rng_state,
    load_trainable_state_dict,
    restore_rng_state,
    trainable_state_dict,
)
from .config import ExperimentConfig
from .evaluate import evaluate_model
from .synthetic_curriculum import (
    SFTRecordDataset,
    SyntheticSFTCollator,
    build_mixed_dataset,
    load_all_synthetic_levels,
    wtq_training_records,
)
from .utils import write_json

CURRICULUM_CHECKPOINT_VERSION = 1


@dataclass
class CurriculumRunConfig:
    data_root: str
    output_dir: str
    levels: tuple[int, ...] = (1, 2, 3, 4, 5)
    epochs_per_level: int | tuple[int, ...] = 3
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 2048
    max_answer_tokens: int = 128
    max_examples_per_level: int | None = None
    checkpoint_every_steps: int = 25
    log_every_steps: int = 25
    seed: int = 42
    run_mixed_phase: bool = False
    synthetic_to_wtq_ratio: tuple[int, int] = (25, 75)
    mixed_epochs: int = 1
    mixed_total_examples: int | None = None
    run_final_test: bool = False

    def validate(self) -> None:
        if not self.levels or any(level not in range(1, 6) for level in self.levels):
            raise ValueError("levels must contain values from 1 through 5")
        if tuple(sorted(set(self.levels))) != self.levels:
            raise ValueError("levels must be unique and ordered")
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_sequence_length",
            "max_answer_tokens",
            "log_every_steps",
            "mixed_epochs",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        schedule = self.level_epoch_schedule()
        if any(epochs < 1 for epochs in schedule):
            raise ValueError("epochs_per_level values must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps cannot be negative")

    def level_epoch_schedule(self) -> tuple[int, ...]:
        if isinstance(self.epochs_per_level, int):
            return (self.epochs_per_level,) * len(self.levels)
        schedule = tuple(int(value) for value in self.epochs_per_level)
        if len(schedule) != len(self.levels):
            raise ValueError(
                "epochs_per_level must contain one value per selected level; "
                f"received {len(schedule)} epoch values for {len(self.levels)} levels"
            )
        return schedule

    def epochs_for_level(self, level: int) -> int:
        try:
            position = self.levels.index(level)
        except ValueError as error:
            raise ValueError(f"Level {level} is not part of this curriculum") from error
        return self.level_epoch_schedule()[position]


def curriculum_signature(
    experiment_config: ExperimentConfig, run_config: CurriculumRunConfig
) -> str:
    run_values = asdict(run_config)
    for workflow_key in (
        "data_root",
        "output_dir",
        "run_mixed_phase",
        "run_final_test",
    ):
        run_values.pop(workflow_key, None)
    payload = {
        "experiment": experiment_config.to_dict(),
        "curriculum": run_values,
    }
    payload["experiment"]["training"]["output_dir"] = "<runtime-output>"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def save_curriculum_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    signature: str,
    experiment_config: ExperimentConfig,
    run_config: CurriculumRunConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CURRICULUM_CHECKPOINT_VERSION,
        "signature": signature,
        "model_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "state": state,
        "experiment_config": experiment_config.to_dict(),
        "run_config": asdict(run_config),
        "rng_state": capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_curriculum_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_signature: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != CURRICULUM_CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported curriculum checkpoint: {path}")
    if checkpoint.get("signature") != expected_signature:
        raise ValueError(
            "Curriculum checkpoint does not match the active model or training "
            "configuration. Use a new output directory for changed hyperparameters."
        )
    load_trainable_state_dict(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for optimizer_state in optimizer.state.values():
        for key, value in list(optimizer_state.items()):
            optimizer_state[key] = _move_optimizer_state(value, device)
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["state"]


def load_stage_model(path: str | Path, model: torch.nn.Module) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, checkpoint["model_state"])


class CurriculumRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        wtq_dataset: Any,
        experiment_config: ExperimentConfig,
        run_config: CurriculumRunConfig,
        device: torch.device,
        official_targets: dict[str, Any],
        experiment_metadata: dict[str, Any],
    ) -> None:
        run_config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.wtq_dataset = wtq_dataset
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.device = device
        self.official_targets = official_targets
        self.output_dir = Path(run_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.signature = curriculum_signature(experiment_config, run_config)
        self.parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not self.parameters:
            raise ValueError("Curriculum SFT requires trainable LoRA parameters")
        self.optimizer = self._new_optimizer()
        self.collator = SyntheticSFTCollator(
            tokenizer,
            run_config.max_sequence_length,
            run_config.max_answer_tokens,
        )
        self.synthetic_levels = load_all_synthetic_levels(run_config.data_root)
        if run_config.max_examples_per_level is not None:
            if run_config.max_examples_per_level < 1:
                raise ValueError("max_examples_per_level must be positive")
            self.synthetic_levels = {
                level: records[: run_config.max_examples_per_level]
                for level, records in self.synthetic_levels.items()
            }
        self.root_checkpoint = self.output_dir / "checkpoint_last.pt"
        self.state = self._initial_state()
        if self.root_checkpoint.is_file():
            self.state = load_curriculum_checkpoint(
                self.root_checkpoint,
                self.model,
                self.optimizer,
                self.signature,
                self.device,
            )
            print(
                "[curriculum] Resumed persistent state at "
                f"phase={self.state['phase']} level={self.state['next_level']}",
                flush=True,
            )
        metadata = {
            **experiment_metadata,
            "experiment_config": experiment_config.to_dict(),
            "curriculum_config": asdict(run_config),
            "curriculum_signature": self.signature,
        }
        write_json(metadata, self.results_dir / "experiment_config.json")

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters,
            lr=self.run_config.learning_rate,
            weight_decay=self.run_config.weight_decay,
        )

    def _initial_state(self) -> dict[str, Any]:
        return {
            "phase": "curriculum",
            "next_level": self.run_config.levels[0],
            "active_stage": None,
            "epoch": 0,
            "next_batch_index": 0,
            "global_step": 0,
            "stage_loss_sum": 0.0,
            "stage_batches_seen": 0,
            "stage_training_complete": False,
            "results": [],
            "mixed_result": None,
            "started_at_unix": time.time(),
        }

    def _save(self, path: str | Path | None = None) -> None:
        save_curriculum_checkpoint(
            path or self.root_checkpoint,
            self.model,
            self.optimizer,
            self.state,
            self.signature,
            self.experiment_config,
            self.run_config,
        )

    def _stage_checkpoint(self, stage: str) -> Path:
        return self.checkpoints_dir / stage / "checkpoint.pt"

    def _write_results(self) -> None:
        results = self.state["results"]
        write_json(results, self.results_dir / "curriculum_results.json")
        csv_path = self.results_dir / "curriculum_results.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        fieldnames = [
            "stage",
            "training_data_just_added",
            "wtq_validation_score",
            "delta_vs_base",
            "checkpoint",
        ]
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        temporary.replace(csv_path)
        if results:
            best = max(results, key=lambda item: item["wtq_validation_score"])
            base_score = float(results[0]["wtq_validation_score"])
            summary = {
                "best_stage": best["stage"],
                "best_validation_score": best["wtq_validation_score"],
                "absolute_improvement_over_base": (
                    float(best["wtq_validation_score"]) - base_score
                ),
                "best_checkpoint": best["checkpoint"],
            }
            write_json(summary, self.results_dir / "curriculum_summary.json")

    def _evaluation_config(self, maximum: int | None = None) -> ExperimentConfig:
        config = self.experiment_config
        config.training.max_validation_examples = maximum
        return config

    def _evaluate(self, split: str, stage: str) -> float:
        dataset = self.wtq_dataset[split]
        maximum = self.experiment_config.training.max_validation_examples
        if maximum is not None:
            dataset = dataset.select(range(min(maximum, len(dataset))))
        metrics, _ = evaluate_model(
            self.model,
            self.tokenizer,
            dataset,
            self._evaluation_config(maximum),
            self.device,
            predictions_path=(
                self.output_dir / "evaluations" / f"{stage}_{split}" / "predictions.json"
            ),
            description=f"[{stage}] WTQ {split}",
            official_targets=self.official_targets,
        )
        score = float(metrics["denotation_accuracy"])
        print(f"[{stage}] WTQ {split} denotation accuracy: {score:.4f}", flush=True)
        return score

    def _append_result(self, stage: str, training_data: str, score: float) -> None:
        if any(item["stage"] == stage for item in self.state["results"]):
            return
        base_score = (
            score
            if not self.state["results"]
            else float(self.state["results"][0]["wtq_validation_score"])
        )
        checkpoint = str(self._stage_checkpoint(stage))
        self.state["results"].append(
            {
                "stage": stage,
                "training_data_just_added": training_data,
                "wtq_validation_score": score,
                "delta_vs_base": score - base_score,
                "checkpoint": checkpoint,
            }
        )
        self._write_results()

    def _loader(self, dataset: SFTRecordDataset, epoch: int, stage_seed: int):
        generator = torch.Generator().manual_seed(
            self.run_config.seed + stage_seed * 1000 + epoch
        )
        return DataLoader(
            dataset,
            batch_size=self.run_config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=self.collator,
        )

    def _train_stage(
        self,
        stage: str,
        dataset: SFTRecordDataset,
        epochs: int,
        stage_seed: int,
    ) -> None:
        if self.state["active_stage"] != stage:
            self.state.update(
                {
                    "active_stage": stage,
                    "epoch": 0,
                    "next_batch_index": 0,
                    "stage_loss_sum": 0.0,
                    "stage_batches_seen": 0,
                    "stage_training_complete": False,
                }
            )
            self._save()
        if self.state["stage_training_complete"]:
            return

        accumulation = self.run_config.gradient_accumulation_steps
        plain_progress = os.environ.get("TABLE_MRC_PLAIN_PROGRESS") == "1"
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(int(self.state["epoch"]), epochs):
            self.model.train()
            loader = self._loader(dataset, epoch, stage_seed)
            resume_batch = (
                int(self.state["next_batch_index"])
                if epoch == int(self.state["epoch"])
                else 0
            )
            iterator = iter(loader)
            for _ in range(resume_batch):
                next(iterator)
            progress = tqdm(
                iterator,
                total=len(loader),
                initial=resume_batch,
                desc=f"[{stage}] SFT epoch {epoch + 1}/{epochs}",
                disable=plain_progress,
            )
            for batch_index, batch in enumerate(progress, start=resume_batch):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                use_autocast = (
                    self.experiment_config.training.bf16
                    and self.device.type == "cuda"
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    raw_loss = output.loss
                    loss = raw_loss / accumulation
                loss.backward()
                loss_value = float(raw_loss.detach().float().item())
                self.state["stage_loss_sum"] += loss_value
                self.state["stage_batches_seen"] += 1
                should_step = (batch_index + 1) % accumulation == 0 or (
                    batch_index + 1 == len(loader)
                )
                if should_step:
                    remainder = len(loader) % accumulation
                    if batch_index + 1 == len(loader) and remainder:
                        correction = accumulation / remainder
                        for parameter in self.parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters, self.experiment_config.training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state["global_step"] += 1
                    self.state["epoch"] = epoch
                    self.state["next_batch_index"] = batch_index + 1
                    global_step = int(self.state["global_step"])
                    interval = self.run_config.checkpoint_every_steps
                    if interval and global_step % interval == 0:
                        self._save()
                    if global_step % self.run_config.log_every_steps == 0:
                        average = self.state["stage_loss_sum"] / max(
                            self.state["stage_batches_seen"], 1
                        )
                        print(
                            f"[{stage}] epoch {epoch + 1}/{epochs} | batch "
                            f"{batch_index + 1}/{len(loader)} | loss={average:.4f} | "
                            f"step={global_step}",
                            flush=True,
                        )
                    if not plain_progress:
                        progress.set_postfix(loss=f"{loss_value:.4f}")
            self.state["epoch"] = epoch + 1
            self.state["next_batch_index"] = 0
            self._save()
        self.state["stage_training_complete"] = True
        self._save()

    def _best_pure_result(self) -> dict[str, Any]:
        return max(
            self.state["results"], key=lambda item: item["wtq_validation_score"]
        )

    def _run_base(self) -> None:
        if any(item["stage"] == "base" for item in self.state["results"]):
            return
        self.state["active_stage"] = "base"
        self.state["stage_training_complete"] = True
        self._save(self._stage_checkpoint("base"))
        score = self._evaluate("validation", "base")
        self._append_result("base", "none", score)
        self.state["active_stage"] = None
        self.state["stage_training_complete"] = False
        self._save()

    def _run_curriculum_levels(self) -> None:
        completed = {item["stage"] for item in self.state["results"]}
        for level in self.run_config.levels:
            stage = f"level_{level}"
            if stage in completed:
                continue
            print(f"\n[Stage L{level}] Training examples: {len(self.synthetic_levels[level])}")
            self.state["next_level"] = level
            self._train_stage(
                stage,
                SFTRecordDataset(self.synthetic_levels[level]),
                self.run_config.epochs_for_level(level),
                level,
            )
            self._save(self._stage_checkpoint(stage))
            score = self._evaluate("validation", stage)
            self._append_result(stage, f"synthetic level {level}", score)
            self.state.update(
                {
                    "next_level": level + 1,
                    "active_stage": None,
                    "epoch": 0,
                    "next_batch_index": 0,
                    "stage_loss_sum": 0.0,
                    "stage_batches_seen": 0,
                    "stage_training_complete": False,
                }
            )
            self._save()
        self.state["phase"] = "curriculum_complete"
        self._save()

    def _run_mixed_phase(self) -> None:
        if not self.run_config.run_mixed_phase or self.state.get("mixed_result"):
            return
        if self.state.get("phase") != "mixed":
            best = self._best_pure_result()
            load_stage_model(best["checkpoint"], self.model)
            self.optimizer = self._new_optimizer()
            self.state.update(
                {
                    "phase": "mixed",
                    "active_stage": None,
                    "epoch": 0,
                    "next_batch_index": 0,
                    "stage_loss_sum": 0.0,
                    "stage_batches_seen": 0,
                    "stage_training_complete": False,
                    "mixed_source_stage": best["stage"],
                }
            )
            self._save()
        synthetic_records = [
            record
            for level in self.run_config.levels
            for record in self.synthetic_levels[level]
        ]
        wtq_records = wtq_training_records(
            self.wtq_dataset["train"], self.experiment_config.data
        )
        mixed_dataset = build_mixed_dataset(
            synthetic_records,
            wtq_records,
            self.run_config.synthetic_to_wtq_ratio,
            self.run_config.seed,
            self.run_config.mixed_total_examples,
        )
        self._train_stage(
            "mixed",
            mixed_dataset,
            self.run_config.mixed_epochs,
            99,
        )
        self._save(self._stage_checkpoint("mixed"))
        score = self._evaluate("validation", "mixed")
        base_score = float(self.state["results"][0]["wtq_validation_score"])
        self.state["mixed_result"] = {
            "stage": "mixed",
            "source_stage": self.state["mixed_source_stage"],
            "synthetic_to_wtq_ratio": list(
                self.run_config.synthetic_to_wtq_ratio
            ),
            "wtq_validation_score": score,
            "delta_vs_base": score - base_score,
            "checkpoint": str(self._stage_checkpoint("mixed")),
        }
        write_json(self.state["mixed_result"], self.results_dir / "mixed_result.json")
        self.state["phase"] = "mixed_complete"
        self._save()

    def _selected_result(self) -> dict[str, Any]:
        candidates = [self._best_pure_result()]
        if self.state.get("mixed_result"):
            candidates.append(self.state["mixed_result"])
        return max(candidates, key=lambda item: item["wtq_validation_score"])

    def _run_final_test(self) -> None:
        if not self.run_config.run_final_test:
            return
        final_path = self.results_dir / "final_test.json"
        if final_path.is_file():
            print(f"[final test] Existing result retained: {final_path}")
            return
        selected = self._selected_result()
        load_stage_model(selected["checkpoint"], self.model)
        score = self._evaluate("test", f"selected_{selected['stage']}")
        result = {
            "selected_stage": selected["stage"],
            "selected_validation_score": selected["wtq_validation_score"],
            "wtq_test_denotation_accuracy": score,
            "checkpoint": selected["checkpoint"],
        }
        write_json(result, final_path)
        self.state["final_test"] = result
        self._save()

    def run(self) -> dict[str, Any]:
        self._run_base()
        self._run_curriculum_levels()
        self._run_mixed_phase()
        self._run_final_test()
        self._write_results()
        selected = self._selected_result()
        base_score = float(self.state["results"][0]["wtq_validation_score"])
        print("\n" + "=" * 88)
        print(f"BEST STAGE: {selected['stage']}")
        print(
            f"BEST WTQ VALIDATION: {selected['wtq_validation_score']:.4f} | "
            f"DELTA VS BASE: {selected['wtq_validation_score'] - base_score:+.4f}"
        )
        print("=" * 88)
        return self.state
