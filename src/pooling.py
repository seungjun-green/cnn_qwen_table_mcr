from __future__ import annotations

import torch
from torch import nn


class TokenPooler(nn.Module):
    def __init__(self, hidden_size: int, method: str) -> None:
        super().__init__()
        if method not in {"mean", "max", "attention"}:
            raise ValueError(f"Unknown pooling method: {method}")
        self.method = method
        self.scorer = (
            nn.Linear(hidden_size, 1, bias=False) if method == "attention" else None
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        valid = mask.any(dim=1, keepdim=True)
        if self.method == "mean":
            weights = mask.unsqueeze(-1).to(embeddings.dtype)
            pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        elif self.method == "max":
            minimum = torch.finfo(embeddings.dtype).min
            pooled = (
                embeddings.masked_fill(~mask.unsqueeze(-1), minimum).max(dim=1).values
            )
        else:
            assert self.scorer is not None
            scores = self.scorer(embeddings).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).masked_fill(~mask, 0)
            pooled = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        return pooled.masked_fill(~valid, 0)
