from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-1.7B"
    freeze_backbone: bool = True
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    dataset: str = "stanfordnlp/wikitablequestions"
    revision: str = "refs/pr/4"
    max_rows: int = 32
    max_cols: int = 8
    max_cell_tokens: int = 32
    max_question_tokens: int = 256
    max_answer_tokens: int = 64
    num_workers: int = 0
    max_table_tokens: int = 2048
    answer_mode: str = "first"
    answer_separator: str = " | "
    table_selection: str = "leading"
    selection_neighbor_radius: int = 1


@dataclass
class CellEncoderConfig:
    pooling: str = "mean"
    cell_dim: int = 256
    mlp_type: str = "single"
    deep_hidden_dim: int = 512


@dataclass
class CNNConfig:
    channels: int = 256
    depth: int = 2
    kernel_size: int = 3
    residual: bool = True


@dataclass
class CrossAttentionConfig:
    insertion_layer: int = 13
    num_heads: int = 8
    gate_init: float = 0.1
    dropout: float = 0.0


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"


@dataclass
class Structure2DConfig:
    dropout: float = 0.05
    initial_scale: float = 0.1
    use_row_embeddings: bool = True
    use_column_embeddings: bool = True
    use_cell_type_embeddings: bool = True


@dataclass
class CNNResidualConfig:
    insertion_layer: int = 13
    gate_init: float = 0.0
    dropout: float = 0.05
    use_row_embeddings: bool = True
    use_column_embeddings: bool = True
    use_cell_type_embeddings: bool = True


@dataclass
class GNNConfig:
    depth: int = 2
    dropout: float = 0.05
    insertion_layer: int = 8
    gate_init: float = 0.0
    use_row_edges: bool = True
    use_column_edges: bool = True
    use_header_edges: bool = True
    use_row_embeddings: bool = True
    use_column_embeddings: bool = True
    use_cell_type_embeddings: bool = True


@dataclass
class EvaluationConfig:
    primary_metric: str = "exact_match"
    official_data_dir: str | None = None
    official_cache_dir: str | None = None


@dataclass
class TrainingConfig:
    bf16: bool = True
    batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 10
    seed: int = 42
    log_every: int = 20
    eval_every_epochs: int = 1
    checkpoint_every_steps: int = 100
    early_stopping_patience: int | None = 1
    early_stopping_min_delta: float = 0.0
    auto_resume: bool = True
    resume_from_checkpoint: str | None = None
    max_train_examples: int | None = None
    max_validation_examples: int | None = None
    output_dir: str = "outputs/baseline"
    mirror_output_dir: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 32


@dataclass
class ExperimentConfig:
    experiment_type: str = "cnn"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cell_encoder: CellEncoderConfig = field(default_factory=CellEncoderConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    structure_2d: Structure2DConfig = field(default_factory=Structure2DConfig)
    cnn_residual: CNNResidualConfig = field(default_factory=CNNResidualConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        supported_experiments = {
            "cnn",
            "continuous_prefix",
            "serialized",
            "structured_2d",
            "serialized_cnn_residual",
            "serialized_gnn_residual",
        }
        if self.experiment_type not in supported_experiments:
            choices = ", ".join(sorted(supported_experiments))
            raise ValueError(f"experiment_type must be one of: {choices}")
        if self.cell_encoder.pooling not in {"mean", "max", "attention"}:
            raise ValueError("cell_encoder.pooling must be mean, max, or attention")
        if self.cell_encoder.mlp_type not in {"single", "deep"}:
            raise ValueError("cell_encoder.mlp_type must be single or deep")
        if self.cell_encoder.cell_dim not in {128, 256, 512}:
            raise ValueError("cell_encoder.cell_dim must be 128, 256, or 512")
        if self.cnn.depth not in {2, 4, 6}:
            raise ValueError("cnn.depth must be 2, 4, or 6")
        if self.cnn.kernel_size not in {3, 5}:
            raise ValueError("cnn.kernel_size must be 3 or 5")
        if self.data.max_rows < 1 or self.data.max_cols < 1:
            raise ValueError("max_rows and max_cols must be positive")
        if self.data.max_table_tokens < 1:
            raise ValueError("max_table_tokens must be positive")
        if self.data.answer_mode not in {"first", "all"}:
            raise ValueError("answer_mode must be first or all")
        if not self.data.answer_separator:
            raise ValueError("answer_separator cannot be empty")
        if self.data.table_selection not in {"leading", "question_relevance"}:
            raise ValueError("table_selection must be leading or question_relevance")
        if self.data.selection_neighbor_radius < 0:
            raise ValueError("selection_neighbor_radius cannot be negative")
        if not 0 <= self.cross_attention.gate_init < 1:
            raise ValueError("gate_init must be in [0, 1)")
        if self.lora.rank < 1:
            raise ValueError("lora.rank must be positive")
        if self.lora.alpha < 1:
            raise ValueError("lora.alpha must be positive")
        if not 0 <= self.lora.dropout < 1:
            raise ValueError("lora.dropout must be in [0, 1)")
        if not self.lora.target_modules or not all(
            isinstance(module, str) and module for module in self.lora.target_modules
        ):
            raise ValueError("lora.target_modules must contain module names")
        if self.lora.bias not in {"none", "all", "lora_only"}:
            raise ValueError("lora.bias must be none, all, or lora_only")
        if self.lora.enabled and not self.model.freeze_backbone:
            raise ValueError("LoRA requires model.freeze_backbone: true")
        if not 0 <= self.structure_2d.dropout < 1:
            raise ValueError("structure_2d.dropout must be in [0, 1)")
        if self.structure_2d.initial_scale < 0:
            raise ValueError("structure_2d.initial_scale cannot be negative")
        if self.cnn_residual.insertion_layer < 0:
            raise ValueError("cnn_residual.insertion_layer cannot be negative")
        if not 0 <= self.cnn_residual.gate_init < 1:
            raise ValueError("cnn_residual.gate_init must be in [0, 1)")
        if not 0 <= self.cnn_residual.dropout < 1:
            raise ValueError("cnn_residual.dropout must be in [0, 1)")
        if self.gnn.depth not in {1, 2, 3, 4}:
            raise ValueError("gnn.depth must be 1, 2, 3, or 4")
        if not 0 <= self.gnn.dropout < 1:
            raise ValueError("gnn.dropout must be in [0, 1)")
        if self.gnn.insertion_layer < 0:
            raise ValueError("gnn.insertion_layer cannot be negative")
        if not 0 <= self.gnn.gate_init < 1:
            raise ValueError("gnn.gate_init must be in [0, 1)")
        if not any(
            (
                self.gnn.use_row_edges,
                self.gnn.use_column_edges,
                self.gnn.use_header_edges,
            )
        ):
            raise ValueError("At least one GNN edge relation must be enabled")
        if self.evaluation.primary_metric not in {
            "exact_match",
            "denotation_accuracy",
        }:
            raise ValueError(
                "evaluation.primary_metric must be exact_match or "
                "denotation_accuracy"
            )
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.training.eval_every_epochs < 1:
            raise ValueError("eval_every_epochs must be positive")
        if self.training.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps cannot be negative")
        if (
            self.training.early_stopping_patience is not None
            and self.training.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience cannot be negative")


_SECTIONS: dict[str, type[Any]] = {
    "model": ModelConfig,
    "data": DataConfig,
    "cell_encoder": CellEncoderConfig,
    "cnn": CNNConfig,
    "cross_attention": CrossAttentionConfig,
    "lora": LoRAConfig,
    "structure_2d": Structure2DConfig,
    "cnn_residual": CNNResidualConfig,
    "gnn": GNNConfig,
    "evaluation": EvaluationConfig,
    "training": TrainingConfig,
    "generation": GenerationConfig,
}


def _strict_dataclass(cls: type[Any], values: dict[str, Any], section: str) -> Any:
    valid = cls.__dataclass_fields__.keys()
    unknown = set(values) - set(valid)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"experiment_type", *_SECTIONS}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {"experiment_type": raw.get("experiment_type", "cnn")}
    for name, cls in _SECTIONS.items():
        kwargs[name] = _strict_dataclass(cls, raw.get(name, {}), name)
    config = ExperimentConfig(**kwargs)
    config.validate()
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
