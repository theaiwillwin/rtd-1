"""
Persistent Hyperbolic Field — conversational memory substrate.

Dual-field design:
  Identity field: slow-moving (gets lr*0.01 in optimizer). Encodes
                  "who/what the model is" — stable across conversations.
  Context field:  fast-moving, gated decay. Encodes "what's currently
                  being discussed" — decays at conversation boundaries.

decay_override semantics:
  None -> normal field_decay (within a turn)
  0.5  -> soft decay (turn boundary, partial forgetting)
  0.0  -> hard reset (new conversation, full forgetting)

CRITICAL: call detach() after every training/inference batch to cut the
autograd graph while preserving the actual state values. Call reset()
ONLY at genuine conversation boundaries — never mid-conversation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .lorentz import LorentzManifold


class PersistentHyperbolicField(nn.Module):
    def __init__(self, dim: int, decay: float = 0.997):
        super().__init__()
        self.decay = decay
        self.register_buffer("identity_state", torch.zeros(1, 1, dim + 1))
        self.register_buffer("context_state",  torch.zeros(1, 1, dim + 1))
        self.identity_state = LorentzManifold.project_to_hyperboloid(self.identity_state)
        self.context_state  = LorentzManifold.project_to_hyperboloid(self.context_state)

        # Separate modules so the optimizer can assign identity a lower LR
        self.identity_update_proj = nn.Linear(dim, dim)
        self.context_update_proj  = nn.Linear(dim, dim)
        self.boundary_gate = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.LayerNorm(dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1),   nn.Sigmoid(),
        )
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.last_metrics: Dict = {}

    def forward(self, x_hyp: torch.Tensor,
                decay_override: Optional[float] = None) -> torch.Tensor:
        B, T, _ = x_hyp.shape

        total     = LorentzManifold.mobius_add_approx(self.identity_state, self.context_state)
        x_mod_hyp = LorentzManifold.mobius_add_approx(x_hyp, total.expand(B, T, -1))
        x_mod     = LorentzManifold.logmap0(x_mod_hyp)

        pooled    = x_mod.mean(dim=(0, 1), keepdim=True)
        delta_id  = self.identity_update_proj(pooled)
        delta_ctx = self.context_update_proj(pooled)
        gate      = self.boundary_gate(pooled)

        # Identity update — standard decay, strict clamp (max_norm=0.1)
        id_tan = LorentzManifold.logmap0(self.identity_state) * self.decay
        self.identity_state = LorentzManifold.project_to_hyperboloid(
            LorentzManifold.mobius_add_approx(
                LorentzManifold.expmap0(id_tan),
                LorentzManifold.expmap0(LorentzManifold.clamp_tangent_norm(delta_id, 0.1))
            )
        )

        # Context update — gated decay, looser clamp (max_norm=0.5)
        eff_decay = self.decay * gate
        if decay_override is not None:
            eff_decay = eff_decay * torch.tensor(
                decay_override, dtype=gate.dtype, device=gate.device
            ).clamp(0.0, 1.0)
        ctx_tan = LorentzManifold.logmap0(self.context_state) * eff_decay
        self.context_state = LorentzManifold.project_to_hyperboloid(
            LorentzManifold.mobius_add_approx(
                LorentzManifold.expmap0(ctx_tan),
                LorentzManifold.expmap0(LorentzManifold.clamp_tangent_norm(delta_ctx, 0.5))
            )
        )

        new_total = LorentzManifold.mobius_add_approx(self.identity_state, self.context_state)
        result = LorentzManifold.project_to_hyperboloid(
            LorentzManifold.mobius_add_approx(x_hyp, new_total.expand(B, T, -1))
        )

        with torch.no_grad():
            self.last_metrics = {
                "delta_identity": delta_id.norm().item(),
                "delta_context":  delta_ctx.norm().item(),
                "gate_value":     gate.item(),
                "id_radius":      LorentzManifold.logmap0(self.identity_state).norm().item(),
                "ctx_radius":     LorentzManifold.logmap0(self.context_state).norm().item(),
            }

        self.step_count += 1
        return result

    def detach(self):
        """Cut grad graph — state values survive, gradients do not propagate."""
        self.identity_state = self.identity_state.detach()
        self.context_state  = self.context_state.detach()

    def reset(self):
        """Hard reset — use ONLY at conversation boundaries."""
        self.identity_state = LorentzManifold.project_to_hyperboloid(torch.zeros_like(self.identity_state))
        self.context_state  = LorentzManifold.project_to_hyperboloid(torch.zeros_like(self.context_state))
        self.step_count.zero_()

    @property
    def state(self) -> torch.Tensor:
        return LorentzManifold.mobius_add_approx(self.identity_state, self.context_state)
