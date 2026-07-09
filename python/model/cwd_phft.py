"""
CWD-PHFT — top-level model.

Forward pipeline:
  tokens -> HammingEmbedding -> clamp_tangent_norm -> expmap0
  -> PersistentHyperbolicField -> logmap0 -> EpsilonManifold
  -> N x FanoCWDLayer -> LayerNorm -> lm_head (tied to embed weights)
                                   -> PotentialEnergyHead

Established hyperparameters:
  vocab_size=50257, dim=512, num_layers=6, num_heads=8, mem_size=64,
  field_decay=0.997, mem_momentum=0.99, fractal_iters=2, num_modes=4,
  mode_temp=1.0, seq_len=256, batch_size=8, lr=3e-4, weight_decay=0.01,
  grad_clip=1.0, identity_lr_scale=0.01, curvature 0.1->2.0 over 10k steps.

CRITICAL CALLS:
  model.detach_persistent_state()  -- after EVERY batch, no exceptions
  model.reset_persistent_state()   -- ONLY at conversation boundaries
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .lorentz import LorentzManifold
from .hamming import HammingEmbedding
from .field import PersistentHyperbolicField
from .epsilon import EpsilonManifold
from .layers import FanoCWDLayer
from .energy import PotentialEnergyHead


class CWDPHFT(nn.Module):
    def __init__(
        self,
        vocab_size:    int   = 50257,
        dim:           int   = 512,
        num_layers:    int   = 6,
        num_heads:     int   = 8,
        mem_size:      int   = 64,
        field_decay:   float = 0.997,
        mem_momentum:  float = 0.99,
        fractal_iters: int   = 2,
        num_modes:     int   = 4,
        mode_temp:     float = 1.0,
    ):
        super().__init__()
        self.dim        = dim
        self.num_layers = num_layers

        self.embed       = HammingEmbedding(vocab_size, dim)
        self.field       = PersistentHyperbolicField(dim, decay=field_decay)
        self.epsilon     = EpsilonManifold(dim, num_modes, mode_temp)
        self.layers      = nn.ModuleList([
            FanoCWDLayer(dim, num_heads, fractal_iters, mem_size, mem_momentum)
            for _ in range(num_layers)
        ])
        self.norm_out    = nn.LayerNorm(dim)
        self.lm_head     = nn.Linear(dim, vocab_size, bias=False)
        self.energy_head = PotentialEnergyHead(dim)

        self.lm_head.weight = self.embed.base_embed.weight  # weight tying

        self.register_buffer("current_curvature", torch.tensor(0.1))

    def forward(
        self,
        input_ids:      torch.Tensor,
        curvature:      Optional[float] = None,
        decay_override: Optional[float] = None,
        mode:           int             = 0,
        return_energy:  bool            = False,
    ):
        c = curvature if curvature is not None else self.current_curvature.item()

        x     = self.embed(input_ids)
        x     = LorentzManifold.clamp_tangent_norm(x)
        x_hyp = LorentzManifold.expmap0(x)
        x_hyp = self.field(x_hyp, decay_override=decay_override)
        x     = LorentzManifold.logmap0(x_hyp)
        x     = self.epsilon(x, mode=mode)

        for layer in self.layers:
            x = layer(x, curvature=c)

        x      = self.norm_out(x)
        logits = self.lm_head(x)
        energy = self.energy_head(x)

        if return_energy:
            return logits, energy
        return logits

    def detach_persistent_state(self):
        self.field.detach()
        for layer in self.layers:
            layer.detach_persistent()

    def reset_persistent_state(self):
        self.field.reset()
        for layer in self.layers:
            layer.reset_persistent()

    def update_curvature(self, step: int, start=0.1, end=2.0,
                          warmup=10000, max_change=0.001):
        target = start + (end - start) * min(1.0, step / warmup)
        delta  = max(-max_change, min(max_change, target - self.current_curvature.item()))
        self.current_curvature += delta

    def get_field_state_dict(self) -> Dict:
        state = {
            "field_identity":   self.field.identity_state.clone(),
            "field_context":    self.field.context_state.clone(),
            "field_step_count": self.field.step_count.clone(),
            "curvature":        self.current_curvature.clone(),
        }
        for i, layer in enumerate(self.layers):
            state[f"layer_{i}_memory"]   = layer.memory.memory_bank.clone()
            state[f"layer_{i}_velocity"] = layer.momentum.velocity.clone()
        return state

    def load_field_state_dict(self, state: Dict):
        self.field.identity_state.copy_(state["field_identity"])
        self.field.context_state.copy_(state["field_context"])
        self.field.step_count.copy_(state["field_step_count"])
        self.current_curvature.copy_(state["curvature"])
        for i, layer in enumerate(self.layers):
            layer.memory.memory_bank.copy_(state[f"layer_{i}_memory"])
            layer.momentum.velocity.copy_(state[f"layer_{i}_velocity"])

    def build_optimizer(self, base_lr: float = 3e-4, weight_decay: float = 0.01):
        """Two-group optimizer — identity field gets base_lr * 0.01."""
        identity_ids   = set(id(p) for p in self.field.identity_update_proj.parameters())
        base_params    = [p for p in self.parameters() if id(p) not in identity_ids]
        identity_params = [p for p in self.parameters() if id(p) in identity_ids]
        return torch.optim.AdamW([
            {'params': base_params,     'lr': base_lr},
            {'params': identity_params, 'lr': base_lr * 0.01},
        ], weight_decay=weight_decay)
