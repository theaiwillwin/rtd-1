"""
Epsilon manifold — discrete cognitive modes.

Each mode is a learned perturbation vector in tangent space. Switching
modes is a structural shift in how the model processes input, not a
superficial sampling-temperature nudge.

Modes:
  0 = neutral     (default inference)
  1 = technical   (precise, lower entropy)
  2 = creative    (higher exploration)
  3 = skeptical   (higher constraint weight)
"""

import torch
import torch.nn as nn


class EpsilonManifold(nn.Module):
    def __init__(self, dim: int, num_modes: int = 4, temperature: float = 1.0):
        super().__init__()
        self.num_modes   = num_modes
        self.temperature = temperature
        self.modes       = nn.Embedding(num_modes, dim)
        self.transition_cost = nn.Parameter(torch.ones(num_modes, num_modes) * 2.0)
        nn.init.zeros_(self.transition_cost.diagonal())

    def forward(self, x: torch.Tensor, mode: int = 0) -> torch.Tensor:
        p = self.modes(torch.tensor(mode, device=x.device))
        return x + p.unsqueeze(0).unsqueeze(0)

    def transition_prob(self, from_mode: int, to_mode: int) -> float:
        d = self.transition_cost[from_mode, to_mode]
        return torch.exp(-d / self.temperature).item()
