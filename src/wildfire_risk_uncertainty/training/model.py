"""Shared two-head MLP for supplied BP and conditional FLP probabilities."""

import torch
from torch import nn


class RiskMLP(nn.Module):
    def __init__(self, inputs: int, depth: int = 2, width: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = inputs
        for _ in range(depth):
            layers.extend((nn.Linear(size, width), nn.ReLU()))
            size = width
        self.backbone = nn.Sequential(*layers)
        self.bp_head = nn.Linear(size, 1)
        self.flp_head = nn.Linear(size, 6)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.bp_head(hidden).squeeze(-1), self.flp_head(hidden)

    def probabilities(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bp_logits, flp_logits = self(features)
        return bp_logits.sigmoid(), flp_logits.softmax(dim=-1)
