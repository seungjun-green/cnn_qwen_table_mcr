from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .cell_encoder import CellEncoder
from .config import ExperimentConfig
from .cross_attention import GatedCrossAttention
from .data import Table
from .table_cnn import TableCNN, TableProjector


def _decoder_layers(language_model: nn.Module) -> nn.ModuleList:
    candidates = [
        getattr(getattr(language_model, "model", None), "layers", None),
        getattr(
            getattr(getattr(language_model, "model", None), "model", None),
            "layers",
            None,
        ),
        getattr(getattr(language_model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise TypeError("Could not locate decoder layers on the loaded language model")


class TableCNNQwen(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: Any,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.config_bundle = config
        hidden_size = int(language_model.config.hidden_size)
        self.cell_encoder = CellEncoder(
            tokenizer=tokenizer,
            embedding_dim=hidden_size,
            cell_dim=config.cell_encoder.cell_dim,
            pooling=config.cell_encoder.pooling,
            mlp_type=config.cell_encoder.mlp_type,
            max_rows=config.data.max_rows,
            max_cols=config.data.max_cols,
            max_cell_tokens=config.data.max_cell_tokens,
            deep_hidden_dim=config.cell_encoder.deep_hidden_dim,
        )
        self.table_cnn = TableCNN(
            cell_dim=config.cell_encoder.cell_dim,
            channels=config.cnn.channels,
            depth=config.cnn.depth,
            kernel_size=config.cnn.kernel_size,
            residual=config.cnn.residual,
        )
        self.projector = TableProjector(config.cnn.channels, hidden_size)
        self.cross_attention = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=config.cross_attention.num_heads,
            gate_init=config.cross_attention.gate_init,
            dropout=config.cross_attention.dropout,
        )
        self.backbone_frozen = config.model.freeze_backbone
        if self.backbone_frozen:
            self.language_model.requires_grad_(False)

        layers = _decoder_layers(language_model)
        layer_index = config.cross_attention.insertion_layer
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"insertion_layer={layer_index} is outside the model's {len(layers)} layers"
            )
        self._active_memory: torch.Tensor | None = None
        self._active_mask: torch.Tensor | None = None
        self._hook_handle = layers[layer_index].register_forward_hook(
            self._injection_hook
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.language_model.eval()
        return self

    def _injection_hook(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        del module, inputs
        if self._active_memory is None or self._active_mask is None:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            injected = self.cross_attention(
                hidden, self._active_memory, self._active_mask
            )
            return (injected, *output[1:])
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Unsupported decoder layer output: {type(output).__name__}"
            )
        return self.cross_attention(output, self._active_memory, self._active_mask)

    @contextmanager
    def _table_context(
        self, memory: torch.Tensor, mask: torch.Tensor
    ) -> Iterator[None]:
        if self._active_memory is not None:
            raise RuntimeError("Nested table-memory contexts are not supported")
        self._active_memory = memory
        self._active_mask = mask
        try:
            yield
        finally:
            self._active_memory = None
            self._active_mask = None

    def encode_tables(
        self, tables: Sequence[Table]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        embeddings = self.language_model.get_input_embeddings()
        cell_grid, cell_mask = self.cell_encoder(tables, embeddings)
        cnn_output = self.table_cnn(cell_grid, cell_mask)
        memory, memory_mask = self.projector(cnn_output, cell_mask)
        shapes = {
            "cell_grid": tuple(cell_grid.shape),
            "cnn_output": tuple(cnn_output.shape),
            "flattened_cnn": (
                cnn_output.shape[0],
                cnn_output.shape[2] * cnn_output.shape[3],
                cnn_output.shape[1],
            ),
            "table_memory": tuple(memory.shape),
        }
        return memory, memory_mask, shapes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        memory, memory_mask, _ = self.encode_tables(tables)
        with self._table_context(memory, memory_mask):
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        memory, memory_mask, _ = self.encode_tables(tables)
        with self._table_context(memory, memory_mask):
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )


class ContinuousPrefixQwen(nn.Module):
    """Condition one causal decoder on learned table-prefix embeddings."""

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: Any,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.config_bundle = config
        hidden_size = int(language_model.config.hidden_size)
        self.cell_encoder = CellEncoder(
            tokenizer=tokenizer,
            embedding_dim=hidden_size,
            cell_dim=config.cell_encoder.cell_dim,
            pooling=config.cell_encoder.pooling,
            mlp_type=config.cell_encoder.mlp_type,
            max_rows=config.data.max_rows,
            max_cols=config.data.max_cols,
            max_cell_tokens=config.data.max_cell_tokens,
            deep_hidden_dim=config.cell_encoder.deep_hidden_dim,
        )
        self.table_cnn = TableCNN(
            cell_dim=config.cell_encoder.cell_dim,
            channels=config.cnn.channels,
            depth=config.cnn.depth,
            kernel_size=config.cnn.kernel_size,
            residual=config.cnn.residual,
        )
        self.projector = TableProjector(config.cnn.channels, hidden_size)
        self.backbone_frozen = config.model.freeze_backbone
        if self.backbone_frozen:
            self.language_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.language_model.eval()
        return self

    def encode_tables(
        self, tables: Sequence[Table]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        embeddings = self.language_model.get_input_embeddings()
        cell_grid, cell_mask = self.cell_encoder(tables, embeddings)
        cnn_output = self.table_cnn(cell_grid, cell_mask)
        memory, memory_mask = self.projector(cnn_output, cell_mask)
        prefix, prefix_mask = self._compact_prefix(memory, memory_mask)
        shapes = {
            "cell_grid": tuple(cell_grid.shape),
            "cnn_output": tuple(cnn_output.shape),
            "flattened_cnn": (
                cnn_output.shape[0],
                cnn_output.shape[2] * cnn_output.shape[3],
                cnn_output.shape[1],
            ),
            "table_memory": tuple(memory.shape),
            "table_prefix": tuple(prefix.shape),
        }
        return prefix, prefix_mask, shapes

    @staticmethod
    def _compact_prefix(
        memory: torch.Tensor, memory_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack real cells into a left-padded prefix immediately before the prompt."""
        lengths = memory_mask.sum(dim=1)
        maximum = int(lengths.max().item())
        if maximum < 1:
            raise ValueError("Every table must contain at least one valid cell")
        prefix = memory.new_zeros(memory.shape[0], maximum, memory.shape[-1])
        prefix_mask = memory_mask.new_zeros(memory.shape[0], maximum)
        for batch_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            start = maximum - length
            prefix[batch_index, start:] = memory[batch_index, memory_mask[batch_index]]
            prefix_mask[batch_index, start:] = True
        return prefix, prefix_mask

    def _decoder_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        table_prefix, table_mask, _ = self.encode_tables(tables)
        token_embeddings = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([table_prefix, token_embeddings], dim=1)
        combined_mask = torch.cat(
            [table_mask.to(attention_mask.dtype), attention_mask], dim=1
        )
        combined_labels = None
        if labels is not None:
            prefix_labels = labels.new_full(table_mask.shape, -100)
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        return inputs_embeds, combined_mask, combined_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds, combined_mask, combined_labels = self._decoder_inputs(
            input_ids, attention_mask, tables, labels
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        inputs_embeds, combined_mask, _ = self._decoder_inputs(
            input_ids, attention_mask, tables
        )
        generated_tokens = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            **generation_kwargs,
        )
        # Decoder-only Hugging Face generation returns only newly generated token
        # IDs when the context is supplied through inputs_embeds. Preserve the
        # standard generate() contract expected by the evaluator.
        return torch.cat([input_ids, generated_tokens], dim=1)


class SerializedTableQwen(nn.Module):
    def __init__(self, language_model: nn.Module, freeze_backbone: bool) -> None:
        super().__init__()
        self.language_model = language_model
        self.backbone_frozen = freeze_backbone
        if freeze_backbone:
            self.language_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.language_model.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table] | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        del tables
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table] | None = None,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        del tables
        return self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )


def load_tokenizer(config: ExperimentConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, trust_remote_code=config.model.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(
    config: ExperimentConfig,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    language_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    language_model.config.pad_token_id = tokenizer.pad_token_id
    language_model.generation_config.do_sample = False
    language_model.generation_config.temperature = None
    language_model.generation_config.top_p = None
    language_model.generation_config.top_k = None
    if config.experiment_type == "cnn":
        model: nn.Module = TableCNNQwen(language_model, tokenizer, config)
    elif config.experiment_type == "continuous_prefix":
        model = ContinuousPrefixQwen(language_model, tokenizer, config)
    else:
        model = SerializedTableQwen(language_model, config.model.freeze_backbone)
    return model.to(device=device, dtype=dtype)


def load_trainable_checkpoint(model: nn.Module, path: str | Path) -> None:
    from .checkpointing import load_trainable_state_dict

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" in state:
        state = state["model_state"]
    load_trainable_state_dict(model, state)
