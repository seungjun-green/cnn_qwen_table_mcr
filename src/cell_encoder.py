from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .data import Table, normalize_table, truncate_table
from .pooling import TokenPooler


class CellEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: Any,
        embedding_dim: int,
        cell_dim: int,
        pooling: str,
        mlp_type: str,
        max_rows: int,
        max_cols: int,
        max_cell_tokens: int,
        deep_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_cell_tokens = max_cell_tokens
        self.pooler = TokenPooler(embedding_dim, pooling)
        if mlp_type == "single":
            self.mlp = nn.Sequential(nn.Linear(embedding_dim, cell_dim), nn.GELU())
        elif mlp_type == "deep":
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, deep_hidden_dim),
                nn.GELU(),
                nn.Linear(deep_hidden_dim, cell_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")
        self.cell_dim = cell_dim

    def _flatten_cells(
        self, tables: Sequence[Table]
    ) -> tuple[list[str], list[tuple[int, int, int]], torch.Tensor]:
        texts: list[str] = []
        positions: list[tuple[int, int, int]] = []
        mask = torch.zeros(len(tables), self.max_rows, self.max_cols, dtype=torch.bool)
        for batch_index, raw_table in enumerate(tables):
            table = truncate_table(
                normalize_table(raw_table), self.max_rows, self.max_cols
            )
            if not table.header:
                raise ValueError(f"Table at batch index {batch_index} has no columns")
            grid = [table.header, *table.rows]
            for row_index, row in enumerate(grid):
                for col_index, value in enumerate(row):
                    texts.append(value)
                    positions.append((batch_index, row_index, col_index))
                    mask[batch_index, row_index, col_index] = True
        return texts, positions, mask

    def forward(
        self, tables: Sequence[Table], embedding_layer: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts, positions, cell_mask = self._flatten_cells(tables)
        device = embedding_layer.weight.device
        dtype = embedding_layer.weight.dtype
        grid = torch.zeros(
            len(tables),
            self.max_rows,
            self.max_cols,
            self.cell_dim,
            device=device,
            dtype=dtype,
        )
        if not texts:
            return grid, cell_mask.to(device)
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_cell_tokens,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        token_mask = tokenized["attention_mask"].to(device).bool()
        embeddings = embedding_layer(input_ids)
        pooled = self.pooler(embeddings, token_mask)
        vectors = self.mlp(pooled)
        batch_ids, row_ids, col_ids = zip(*positions)
        grid[
            torch.tensor(batch_ids, device=device),
            torch.tensor(row_ids, device=device),
            torch.tensor(col_ids, device=device),
        ] = vectors
        return grid, cell_mask.to(device)
