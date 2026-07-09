"""
Fano sparse attention.

Sparsity pattern derived from the Fano plane's PRIMARY LINE assignment,
not the union of all lines through a point. The Fano plane is a complete
projective geometry: every pair of points shares exactly one line, so a
"union of all lines through point i" criterion gives full density (each
point reaches all others through some line). The fix: assign each point
to exactly ONE primary line, giving each block exactly 3/7 ~ 42.9%
attention density.

Primary line assignment (each of the 7 Fano lines is owned by exactly
one point — verified to cover all 7 lines with no duplicates):
  Point 0 -> {0,1,2}   Point 1 -> {1,3,5}   Point 2 -> {2,4,5}
  Point 3 -> {0,3,4}   Point 4 -> {1,4,6}   Point 5 -> {0,5,6}
  Point 6 -> {2,3,6}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

from .octonion import OctonionLinear


class FanoSparseAttention(nn.Module):
    FANO_PRIMARY_LINE = {
        0: (0, 1, 2),
        1: (1, 3, 5),
        2: (2, 4, 5),
        3: (0, 3, 4),
        4: (1, 4, 6),
        5: (0, 5, 6),
        6: (2, 3, 6),
    }

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.q_proj   = OctonionLinear(dim, dim)
        self.k_proj   = OctonionLinear(dim, dim)
        self.v_proj   = OctonionLinear(dim, dim)
        self.out_proj = OctonionLinear(dim, dim)
        self.curvature_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, curvature: Optional[float] = None) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if curvature is not None:
            scores = scores * (1.0 + self.curvature_scale * float(curvature))

        fano_mask   = self._build_fano_mask(T, x.device)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores      = scores.masked_fill(causal_mask | ~fano_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

    def _build_fano_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask   = torch.zeros(T, T, dtype=torch.bool, device=device)
        bsizes = self._block_sizes(T)
        starts = [sum(bsizes[:i]) for i in range(7)]
        for point, line in self.FANO_PRIMARY_LINE.items():
            if point >= len(starts):
                continue
            qs, qe = starts[point], starts[point] + bsizes[point]
            for kb in line:
                if kb >= len(starts):
                    continue
                ks, ke = starts[kb], starts[kb] + bsizes[kb]
                mask[qs:qe, ks:ke] = True
        mask.fill_diagonal_(True)
        return mask

    @staticmethod
    def _block_sizes(T: int) -> List[int]:
        """Distribute T tokens into 7 blocks as evenly as possible."""
        base, rem = divmod(T, 7)
        return [base + 1 if i < rem else base for i in range(7)]

    def density(self, T: int) -> float:
        return self._build_fano_mask(T, torch.device('cpu')).float().mean().item()


class FractalRefinement(nn.Module):
    """Iterative Newton-style refinement within a single forward pass."""

    def __init__(self, dim: int, num_iters: int = 2):
        super().__init__()
        self.num_iters = num_iters
        self.refine    = nn.Linear(dim, dim)
        self.gate      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for _ in range(self.num_iters):
            x = residual + torch.sigmoid(self.gate(x)) * self.refine(x)
        return x
