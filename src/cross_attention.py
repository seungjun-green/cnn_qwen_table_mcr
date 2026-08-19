from __future__ import annotations

import torch
from torch import nn


class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        gate_init: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        table_memory: torch.Tensor,
        table_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            hidden_states,
            table_memory,
            table_memory,
            key_padding_mask=~table_mask.bool(),
            need_weights=False,
        )
        return hidden_states + torch.tanh(self.alpha) * attended
