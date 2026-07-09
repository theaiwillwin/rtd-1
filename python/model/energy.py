"""
Explicit potential energy head V(Psi).

Low energy  = coherent, factual, on-constraint hidden state.
High energy = incoherent, out-of-distribution, or unsafe state.

Used in training loss: L_total = L_lm + lambda * V.mean()   (lambda ~ 0.01)
Used at inference: monitor for spikes indicating collapse or boundary escape.
"""

import torch
import torch.nn as nn


class PotentialEnergyHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),   nn.GELU(),
            nn.Linear(dim // 2, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)            # [B,T]

    def sequence_energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=-1)        # [B]
