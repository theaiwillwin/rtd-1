"""
Cognitive momentum (discretized wave inertia) and persistent memory bank.

CognitiveMomentum implements the inertia term of the discretized wave
equation u(t+1) = 2u(t) - u(t-1) + c^2*laplacian(u(t)) in a compressed
form: a velocity buffer that persists across batches, giving the hidden
state directional momentum rather than reacting fresh each forward pass.

PersistentMemory is a slow-moving memory bank updated via momentum
(not backprop) that layers read from via cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CognitiveMomentum(nn.Module):
    def __init__(self, dim: int, beta: float = 0.9):
        super().__init__()
        self.beta = beta
        self.register_buffer("velocity", torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pooled = x.mean(dim=(0, 1), keepdim=True)
        self.velocity = self.beta * self.velocity + (1.0 - self.beta) * (pooled - self.velocity)
        return x + self.velocity.expand(B, T, D)

    def detach(self):
        self.velocity = self.velocity.detach()


class PersistentMemory(nn.Module):
    def __init__(self, dim: int, mem_size: int = 64, momentum: float = 0.99):
        super().__init__()
        self.mem_size = mem_size
        self.momentum = momentum
        self.register_buffer("memory_bank", torch.randn(1, mem_size, dim) * 0.01)
        self.q_proj      = nn.Linear(dim, dim)
        self.k_proj      = nn.Linear(dim, dim)
        self.v_proj      = nn.Linear(dim, dim)
        self.out_proj    = nn.Linear(dim, dim)
        self.update_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q    = self.q_proj(x)
        K    = self.k_proj(self.memory_bank.expand(B, -1, -1))
        V    = self.v_proj(self.memory_bank.expand(B, -1, -1))
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        out  = torch.matmul(attn, V)

        update = self.update_proj(x.mean(dim=(0, 1), keepdim=True)).detach()
        with torch.no_grad():
            self.memory_bank.mul_(self.momentum)
            self.memory_bank.add_((1.0 - self.momentum) * update.expand(1, self.mem_size, -1))
        return self.out_proj(out)

    def detach(self):
        self.memory_bank.detach_()

    def reset(self):
        self.memory_bank.zero_()
        self.memory_bank.add_(torch.randn_like(self.memory_bank) * 0.01)
