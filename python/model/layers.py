"""
CWD layer: attention + refinement + memory + momentum + FFN.
All linear projections octonion-compressed.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import FanoSparseAttention, FractalRefinement
from .memory import PersistentMemory, CognitiveMomentum
from .octonion import OctonionLinear


class FanoCWDLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, fractal_iters: int,
                 mem_size: int, mem_momentum: float):
        super().__init__()
        self.attn     = FanoSparseAttention(dim, num_heads)
        self.refine   = FractalRefinement(dim, num_iters=fractal_iters)
        self.memory   = PersistentMemory(dim, mem_size, mem_momentum)
        self.momentum = CognitiveMomentum(dim, beta=0.9)
        self.norm1    = nn.LayerNorm(dim)
        self.norm2    = nn.LayerNorm(dim)
        self.ffn      = nn.Sequential(
            OctonionLinear(dim, 4 * dim), nn.GELU(),
            OctonionLinear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, curvature: Optional[float] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), curvature=curvature)
        x = self.refine(x)
        x = x + self.memory(x)
        x = self.momentum(x)
        x = x + self.ffn(self.norm2(x))
        return x

    def detach_persistent(self):
        self.memory.detach()
        self.momentum.detach()

    def reset_persistent(self):
        self.memory.reset()
        self.momentum.velocity.zero_()
