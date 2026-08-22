from __future__ import annotations

import math
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
from .table_gnn import RelationalTableGNN


def _decoder_layers(language_model: nn.Module) -> nn.ModuleList:
    models = [language_model]
    get_base_model = getattr(language_model, "get_base_model", None)
    if callable(get_base_model):
        models.append(get_base_model())
    for model in models:
        candidates = [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(
                getattr(getattr(model, "model", None), "model", None),
                "layers",
                None,
            ),
            getattr(getattr(model, "transformer", None), "h", None),
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
        self.lora_enabled = config.lora.enabled
        if self.backbone_frozen and not self.lora_enabled:
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
        if self.backbone_frozen and not self.lora_enabled:
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


class SerializedCNNResidualQwen(nn.Module):
    """Add cell-aligned 2D CNN features to a serialized Qwen decoder layer."""

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
        cell_dim = config.cell_encoder.cell_dim
        self.cell_encoder = CellEncoder(
            tokenizer=tokenizer,
            embedding_dim=hidden_size,
            cell_dim=cell_dim,
            pooling=config.cell_encoder.pooling,
            mlp_type=config.cell_encoder.mlp_type,
            max_rows=config.data.max_rows,
            max_cols=config.data.max_cols,
            max_cell_tokens=config.data.max_cell_tokens,
            deep_hidden_dim=config.cell_encoder.deep_hidden_dim,
        )
        self.row_embeddings = nn.Embedding(config.data.max_rows, cell_dim)
        self.column_embeddings = nn.Embedding(config.data.max_cols, cell_dim)
        self.cell_type_embeddings = nn.Embedding(2, cell_dim)
        for embedding in (
            self.row_embeddings,
            self.column_embeddings,
            self.cell_type_embeddings,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        self.structure_dropout = nn.Dropout(config.cnn_residual.dropout)
        self.table_cnn = TableCNN(
            cell_dim=cell_dim,
            channels=config.cnn.channels,
            depth=config.cnn.depth,
            kernel_size=config.cnn.kernel_size,
            residual=config.cnn.residual,
        )
        self.projector = TableProjector(config.cnn.channels, hidden_size)
        self.residual_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        gate_init = float(config.cnn_residual.gate_init)
        raw_gate = 0.0 if gate_init == 0 else math.atanh(gate_init)
        self.residual_gate = nn.Parameter(torch.tensor(raw_gate))
        self.backbone_frozen = config.model.freeze_backbone
        self.lora_enabled = config.lora.enabled
        if self.backbone_frozen and not self.lora_enabled:
            self.language_model.requires_grad_(False)

        layers = _decoder_layers(language_model)
        layer_index = config.cnn_residual.insertion_layer
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"cnn_residual.insertion_layer={layer_index} is outside the "
                f"model's {len(layers)} layers"
            )
        self.insertion_layer = layer_index
        self._active_token_residual: torch.Tensor | None = None
        self._hook_handle = layers[layer_index].register_forward_hook(
            self._injection_hook
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen and not self.lora_enabled:
            self.language_model.eval()
        return self

    def _add_structure(
        self, cell_grid: torch.Tensor, cell_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, rows, columns, _ = cell_grid.shape
        device = cell_grid.device
        structure = torch.zeros_like(cell_grid)
        residual_config = self.config_bundle.cnn_residual
        if residual_config.use_row_embeddings:
            row_ids = torch.arange(rows, device=device).view(1, rows, 1)
            structure = structure + self.row_embeddings(row_ids)
        if residual_config.use_column_embeddings:
            column_ids = torch.arange(columns, device=device).view(1, 1, columns)
            structure = structure + self.column_embeddings(column_ids)
        if residual_config.use_cell_type_embeddings:
            cell_types = torch.ones(rows, columns, dtype=torch.long, device=device)
            cell_types[0] = 0
            structure = structure + self.cell_type_embeddings(cell_types)
        mask = cell_mask.view(batch_size, rows, columns, 1).to(cell_grid.dtype)
        return (cell_grid + self.structure_dropout(structure)) * mask

    def encode_tables(
        self, tables: Sequence[Table]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        embeddings = self.language_model.get_input_embeddings()
        cell_grid, cell_mask = self.cell_encoder(tables, embeddings)
        structured_grid = self._add_structure(cell_grid, cell_mask)
        cnn_output = self.table_cnn(structured_grid, cell_mask)
        memory, memory_mask = self.projector(cnn_output, cell_mask)
        shapes = {
            "cell_grid": tuple(cell_grid.shape),
            "structured_grid": tuple(structured_grid.shape),
            "cnn_output": tuple(cnn_output.shape),
            "flattened_cnn": (
                cnn_output.shape[0],
                cnn_output.shape[2] * cnn_output.shape[3],
                cnn_output.shape[1],
            ),
            "table_memory": tuple(memory.shape),
        }
        return memory, memory_mask, shapes

    def _align_residual(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        table_cell_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        indices = table_cell_indices.to(memory.device)
        if indices.shape != attention_mask.shape:
            raise ValueError(
                "table_cell_indices must have the same shape as attention_mask"
            )
        if memory.shape[0] != indices.shape[0]:
            raise ValueError("Table memory and token alignment batch sizes differ")
        safe_indices = indices.clamp(min=0)
        gathered = memory.gather(
            1, safe_indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        )
        valid = indices.ge(0)
        valid = valid & attention_mask.to(device=memory.device, dtype=torch.bool)
        valid_cells = memory_mask.gather(1, safe_indices)
        valid = valid & valid_cells
        residual = self.residual_norm(gathered)
        return self.structure_dropout(residual) * valid.unsqueeze(-1).to(memory.dtype)

    def _injection_hook(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        del module, inputs
        residual = self._active_token_residual
        if residual is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                f"Unsupported decoder layer output: {type(hidden).__name__}"
            )
        # The prompt prefill has the aligned sequence length. Cached generation
        # later passes only the newly generated token, which receives no residual.
        if hidden.shape[1] != residual.shape[1]:
            return output
        if hidden.shape[0] != residual.shape[0]:
            if hidden.shape[0] % residual.shape[0]:
                raise ValueError(
                    "Cannot expand the structural residual to generation beams"
                )
            residual = residual.repeat_interleave(
                hidden.shape[0] // residual.shape[0], dim=0
            )
        injected = hidden + torch.tanh(self.residual_gate) * residual
        if isinstance(output, tuple):
            return (injected, *output[1:])
        return injected

    @contextmanager
    def _residual_context(self, residual: torch.Tensor) -> Iterator[None]:
        if self._active_token_residual is not None:
            raise RuntimeError("Nested structural residual contexts are not supported")
        self._active_token_residual = residual
        try:
            yield
        finally:
            self._active_token_residual = None

    def _token_residual(
        self,
        tables: Sequence[Table],
        table_cell_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_mask, _ = self.encode_tables(tables)
        return self._align_residual(
            memory, memory_mask, table_cell_indices, attention_mask
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        table_cell_indices: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        residual = self._token_residual(
            tables, table_cell_indices, attention_mask
        )
        with self._residual_context(residual):
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
        table_cell_indices: torch.Tensor,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        residual = self._token_residual(
            tables, table_cell_indices, attention_mask
        )
        with self._residual_context(residual):
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )


class SerializedGNNResidualQwen(SerializedCNNResidualQwen):
    """Add cell-aligned relational GNN features to a serialized Qwen layer."""

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: Any,
        config: ExperimentConfig,
    ) -> None:
        nn.Module.__init__(self)
        self.language_model = language_model
        self.config_bundle = config
        hidden_size = int(language_model.config.hidden_size)
        cell_dim = config.cell_encoder.cell_dim
        self.cell_encoder = CellEncoder(
            tokenizer=tokenizer,
            embedding_dim=hidden_size,
            cell_dim=cell_dim,
            pooling=config.cell_encoder.pooling,
            mlp_type=config.cell_encoder.mlp_type,
            max_rows=config.data.max_rows,
            max_cols=config.data.max_cols,
            max_cell_tokens=config.data.max_cell_tokens,
            deep_hidden_dim=config.cell_encoder.deep_hidden_dim,
        )
        self.row_embeddings = nn.Embedding(config.data.max_rows, cell_dim)
        self.column_embeddings = nn.Embedding(config.data.max_cols, cell_dim)
        self.cell_type_embeddings = nn.Embedding(2, cell_dim)
        for embedding in (
            self.row_embeddings,
            self.column_embeddings,
            self.cell_type_embeddings,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        self.structure_dropout = nn.Dropout(config.gnn.dropout)
        self.table_gnn = RelationalTableGNN(
            hidden_size=cell_dim,
            depth=config.gnn.depth,
            dropout=config.gnn.dropout,
            use_row_edges=config.gnn.use_row_edges,
            use_column_edges=config.gnn.use_column_edges,
            use_header_edges=config.gnn.use_header_edges,
        )
        self.projector = TableProjector(cell_dim, hidden_size)
        self.residual_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        gate_init = float(config.gnn.gate_init)
        raw_gate = 0.0 if gate_init == 0 else math.atanh(gate_init)
        self.residual_gate = nn.Parameter(torch.tensor(raw_gate))
        self.backbone_frozen = config.model.freeze_backbone
        self.lora_enabled = config.lora.enabled
        if self.backbone_frozen and not self.lora_enabled:
            self.language_model.requires_grad_(False)

        layers = _decoder_layers(language_model)
        layer_index = config.gnn.insertion_layer
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"gnn.insertion_layer={layer_index} is outside the model's "
                f"{len(layers)} layers"
            )
        self.insertion_layer = layer_index
        self._active_token_residual: torch.Tensor | None = None
        self._hook_handle = layers[layer_index].register_forward_hook(
            self._injection_hook
        )

    def _add_structure(
        self, cell_grid: torch.Tensor, cell_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, rows, columns, _ = cell_grid.shape
        device = cell_grid.device
        structure = torch.zeros_like(cell_grid)
        if self.config_bundle.gnn.use_row_embeddings:
            row_ids = torch.arange(rows, device=device).view(1, rows, 1)
            structure = structure + self.row_embeddings(row_ids)
        if self.config_bundle.gnn.use_column_embeddings:
            column_ids = torch.arange(columns, device=device).view(1, 1, columns)
            structure = structure + self.column_embeddings(column_ids)
        if self.config_bundle.gnn.use_cell_type_embeddings:
            cell_types = torch.ones(rows, columns, dtype=torch.long, device=device)
            cell_types[0] = 0
            structure = structure + self.cell_type_embeddings(cell_types)
        mask = cell_mask.view(batch_size, rows, columns, 1).to(cell_grid.dtype)
        return (cell_grid + self.structure_dropout(structure)) * mask

    def encode_tables(
        self, tables: Sequence[Table]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        embeddings = self.language_model.get_input_embeddings()
        cell_grid, cell_mask = self.cell_encoder(tables, embeddings)
        structured_grid = self._add_structure(cell_grid, cell_mask)
        gnn_output = self.table_gnn(structured_grid, cell_mask)
        memory, memory_mask = self.projector(
            gnn_output.permute(0, 3, 1, 2), cell_mask
        )
        shapes = {
            "cell_grid": tuple(cell_grid.shape),
            "structured_grid": tuple(structured_grid.shape),
            "gnn_output": tuple(gnn_output.shape),
            "graph_nodes": (
                gnn_output.shape[0],
                gnn_output.shape[1] * gnn_output.shape[2],
                gnn_output.shape[3],
            ),
            "table_memory": tuple(memory.shape),
        }
        return memory, memory_mask, shapes


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
        self.lora_enabled = config.lora.enabled
        if self.backbone_frozen and not self.lora_enabled:
            self.language_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen and not self.lora_enabled:
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


class Structured2DQwen(nn.Module):
    """Feed lexical table tokens to Qwen with learned 2D structural adapters."""

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: Any,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.config_bundle = config
        hidden_size = int(language_model.config.hidden_size)
        self.row_embeddings = nn.Embedding(config.data.max_rows, hidden_size)
        self.column_embeddings = nn.Embedding(config.data.max_cols, hidden_size)
        self.cell_type_embeddings = nn.Embedding(3, hidden_size)
        for embedding in (
            self.row_embeddings,
            self.column_embeddings,
            self.cell_type_embeddings,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        self.structural_scale = nn.Parameter(
            torch.tensor(float(config.structure_2d.initial_scale))
        )
        self.structural_dropout = nn.Dropout(config.structure_2d.dropout)
        self.backbone_frozen = config.model.freeze_backbone
        self.lora_enabled = config.lora.enabled
        if self.backbone_frozen and not self.lora_enabled:
            self.language_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen and not self.lora_enabled:
            self.language_model.eval()
        return self

    def _tokenize_table(
        self, table: Table
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        token_ids: list[int] = []
        row_ids: list[int] = []
        column_ids: list[int] = []
        cell_types: list[int] = []
        budget = self.config_bundle.data.max_table_tokens

        def add_piece(text: str, row: int, column: int, cell_type: int) -> bool:
            piece = list(self.tokenizer.encode(text, add_special_tokens=False))
            remaining = budget - len(token_ids)
            if remaining <= 0:
                return False
            piece = piece[:remaining]
            token_ids.extend(piece)
            row_ids.extend([row] * len(piece))
            column_ids.extend([column] * len(piece))
            cell_types.extend([cell_type] * len(piece))
            return len(token_ids) < budget

        if not add_piece("Table:\n", 0, 0, 0):
            return token_ids, row_ids, column_ids, cell_types
        for column, header in enumerate(table.header):
            separator = "" if column == 0 else " | "
            if not add_piece(f"{separator}{header}", 0, column, 1):
                return token_ids, row_ids, column_ids, cell_types
        if not add_piece("\n", 0, 0, 0):
            return token_ids, row_ids, column_ids, cell_types
        for row_index, row in enumerate(table.rows, start=1):
            if not add_piece(f"Row {row_index}: ", row_index, 0, 0):
                break
            for column, value in enumerate(row):
                separator = "" if column == 0 else " | "
                if not add_piece(
                    f"{separator}{value}", row_index, column, 2
                ):
                    break
            if len(token_ids) >= budget or not add_piece("\n", row_index, 0, 0):
                break
        return token_ids, row_ids, column_ids, cell_types

    def encode_tables(
        self, tables: Sequence[Table]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        encoded = [self._tokenize_table(table) for table in tables]
        maximum = max((len(item[0]) for item in encoded), default=0)
        if maximum < 1:
            raise ValueError("Every structured table must produce at least one token")
        device = self.language_model.get_input_embeddings().weight.device
        token_tensor = torch.zeros(len(tables), maximum, dtype=torch.long, device=device)
        row_tensor = torch.zeros_like(token_tensor)
        column_tensor = torch.zeros_like(token_tensor)
        type_tensor = torch.zeros_like(token_tensor)
        mask = torch.zeros(len(tables), maximum, dtype=torch.bool, device=device)
        for batch_index, (tokens, rows, columns, types) in enumerate(encoded):
            length = len(tokens)
            start = maximum - length
            token_tensor[batch_index, start:] = torch.tensor(tokens, device=device)
            row_tensor[batch_index, start:] = torch.tensor(rows, device=device)
            column_tensor[batch_index, start:] = torch.tensor(columns, device=device)
            type_tensor[batch_index, start:] = torch.tensor(types, device=device)
            mask[batch_index, start:] = True

        prefix = self.language_model.get_input_embeddings()(token_tensor)
        structure = torch.zeros_like(prefix)
        structure_config = self.config_bundle.structure_2d
        if structure_config.use_row_embeddings:
            structure = structure + self.row_embeddings(row_tensor)
        if structure_config.use_column_embeddings:
            structure = structure + self.column_embeddings(column_tensor)
        if structure_config.use_cell_type_embeddings:
            structure = structure + self.cell_type_embeddings(type_tensor)
        prefix = prefix + self.structural_scale * self.structural_dropout(structure)
        prefix = prefix * mask.unsqueeze(-1)
        shapes = {
            "table_prefix": tuple(prefix.shape),
            "table_tokens": tuple(token_tensor.shape),
        }
        return prefix, mask, shapes

    def _decoder_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tables: Sequence[Table],
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        table_prefix, table_mask, _ = self.encode_tables(tables)
        prompt_embeddings = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([table_prefix, prompt_embeddings], dim=1)
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
        return torch.cat([input_ids, generated_tokens], dim=1)


class SerializedTableQwen(nn.Module):
    def __init__(
        self, language_model: nn.Module, freeze_backbone: bool, lora_enabled: bool
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.backbone_frozen = freeze_backbone
        self.lora_enabled = lora_enabled
        if freeze_backbone and not lora_enabled:
            self.language_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen and not self.lora_enabled:
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
    if config.lora.enabled:
        from peft import LoraConfig, TaskType, get_peft_model

        language_model = get_peft_model(
            language_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                bias=config.lora.bias,
            ),
        )
    if config.experiment_type == "cnn":
        model: nn.Module = TableCNNQwen(language_model, tokenizer, config)
    elif config.experiment_type == "continuous_prefix":
        model = ContinuousPrefixQwen(language_model, tokenizer, config)
    elif config.experiment_type == "structured_2d":
        model = Structured2DQwen(language_model, tokenizer, config)
    elif config.experiment_type == "serialized_cnn_residual":
        model = SerializedCNNResidualQwen(language_model, tokenizer, config)
    elif config.experiment_type == "serialized_gnn_residual":
        model = SerializedGNNResidualQwen(language_model, tokenizer, config)
    else:
        model = SerializedTableQwen(
            language_model,
            config.model.freeze_backbone,
            config.lora.enabled,
        )
    return model.to(device=device, dtype=dtype)


def load_trainable_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    from .checkpointing import load_trainable_state_dict

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    load_trainable_state_dict(model, state)
    return checkpoint if "model_state" in checkpoint else {}
