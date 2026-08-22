from __future__ import annotations

import torch
from torch import nn


class RelationalTableGNN(nn.Module):
    """Relation-aware message passing over cell nodes in a table grid."""

    relation_names = ("row", "column", "header")

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        dropout: float,
        *,
        use_row_edges: bool = True,
        use_column_edges: bool = True,
        use_header_edges: bool = True,
    ) -> None:
        super().__init__()
        enabled = {
            "row": use_row_edges,
            "column": use_column_edges,
            "header": use_header_edges,
        }
        self.enabled_relations = tuple(
            name for name in self.relation_names if enabled[name]
        )
        if not self.enabled_relations:
            raise ValueError("At least one GNN edge relation must be enabled")
        self.self_linears = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        )
        self.relation_linears = nn.ModuleList(
            nn.ModuleDict(
                {
                    name: nn.Linear(hidden_size, hidden_size, bias=False)
                    for name in self.enabled_relations
                }
            )
            for _ in range(depth)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(depth))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def relation_adjacencies(
        cell_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return target-by-source adjacency matrices for each edge type."""
        batch_size, rows, columns = cell_mask.shape
        device = cell_mask.device
        row_ids = (
            torch.arange(rows, device=device)
            .view(rows, 1)
            .expand(rows, columns)
            .reshape(-1)
        )
        column_ids = (
            torch.arange(columns, device=device)
            .view(1, columns)
            .expand(rows, columns)
            .reshape(-1)
        )
        valid_nodes = cell_mask.reshape(batch_size, -1).bool()
        valid_pairs = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
        node_count = rows * columns
        not_self = ~torch.eye(node_count, dtype=torch.bool, device=device).unsqueeze(0)
        same_row = row_ids.view(-1, 1).eq(row_ids.view(1, -1)).unsqueeze(0)
        same_column = column_ids.view(-1, 1).eq(
            column_ids.view(1, -1)
        ).unsqueeze(0)
        target_header = row_ids.eq(0).view(1, -1, 1)
        source_header = row_ids.eq(0).view(1, 1, -1)
        both_body = ~target_header & ~source_header
        one_header = target_header ^ source_header
        return {
            "row": valid_pairs & not_self & same_row,
            "column": valid_pairs & not_self & same_column & both_body,
            "header": valid_pairs & not_self & same_column & one_header,
        }

    def forward(
        self, cell_grid: torch.Tensor, cell_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, rows, columns, hidden_size = cell_grid.shape
        node_mask = cell_mask.reshape(batch_size, -1).bool()
        hidden = cell_grid.reshape(batch_size, -1, hidden_size)
        adjacencies = self.relation_adjacencies(cell_mask)
        for self_linear, relation_linears, norm in zip(
            self.self_linears, self.relation_linears, self.norms
        ):
            update = self_linear(hidden)
            for relation_name in self.enabled_relations:
                adjacency = adjacencies[relation_name].to(hidden.dtype)
                degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
                neighbor_mean = torch.bmm(adjacency, hidden) / degree
                update = update + relation_linears[relation_name](neighbor_mean)
            update = update / (len(self.enabled_relations) + 1)
            hidden = norm(hidden + self.dropout(self.activation(update)))
            hidden = hidden * node_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden.reshape(batch_size, rows, columns, hidden_size)
