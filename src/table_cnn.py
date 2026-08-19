from __future__ import annotations

import torch
from torch import nn


class TableCNN(nn.Module):
    def __init__(
        self,
        cell_dim: int,
        channels: int,
        depth: int,
        kernel_size: int,
        residual: bool,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_projection = (
            nn.Identity() if cell_dim == channels else nn.Conv2d(cell_dim, channels, 1)
        )
        self.convolutions = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size, padding=padding)
            for _ in range(depth)
        )
        self.activation = nn.GELU()
        self.residual = residual

    def forward(self, grid: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        mask = cell_mask.unsqueeze(1).to(grid.dtype)
        hidden = self.input_projection(grid.permute(0, 3, 1, 2)) * mask
        residual = hidden
        for convolution in self.convolutions:
            hidden = self.activation(convolution(hidden)) * mask
        if self.residual:
            hidden = (hidden + residual) * mask
        return hidden


class TableProjector(nn.Module):
    def __init__(self, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, hidden_size)

    def forward(
        self, cnn_output: torch.Tensor, cell_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cells = cnn_output.permute(0, 2, 3, 1).flatten(1, 2)
        flat_mask = cell_mask.flatten(1)
        memory = self.projection(cells) * flat_mask.unsqueeze(-1).to(cells.dtype)
        return memory, flat_mask
