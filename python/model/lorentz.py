"""
Lorentz manifold operations for hyperbolic geometry.

The model represents token states on a hyperboloid (Lorentz model of
hyperbolic space) rather than flat Euclidean space. This file provides
the core maps between tangent space (where neural net layers operate)
and the hyperboloid (where geometric structure lives).

Stability notes:
  MAX_TANGENT_NORM=2.0 and MAX_SPACE_NORM=10.0 prevent cosh/sinh/acosh
  blowups. Round-trip expmap0->logmap0 only holds EXACTLY for inputs
  with norm <= MAX_TANGENT_NORM — anything larger is correctly clamped,
  which is intended behavior, not a bug.
"""

import torch
import math


class LorentzManifold:
    MAX_TANGENT_NORM = 2.0
    MAX_SPACE_NORM   = 10.0

    @staticmethod
    def minkowski_dot(x: torch.Tensor, y: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """Lorentz inner product: -x0*y0 + sum(xi*yi for i>=1)."""
        return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=keepdim)

    @staticmethod
    def clamp_tangent_norm(v: torch.Tensor, max_norm: float = None) -> torch.Tensor:
        """Clamp tangent vector norm to prevent exp/sinh explosion."""
        if max_norm is None:
            max_norm = LorentzManifold.MAX_TANGENT_NORM
        norm  = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return v * scale

    @staticmethod
    def project_to_hyperboloid(x: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """Project arbitrary vector onto the hyperboloid x0^2 - ||x_space||^2 = 1/c."""
        x_space    = x[..., 1:]
        space_norm = torch.norm(x_space, dim=-1, keepdim=True).clamp(min=eps)
        scale      = torch.clamp(LorentzManifold.MAX_SPACE_NORM / space_norm, max=1.0)
        x_space    = x_space * scale
        norm_sq    = (x_space ** 2).sum(dim=-1, keepdim=True)
        x_time     = torch.sqrt(1.0 / c + norm_sq + eps)
        return torch.cat([x_time, x_space], dim=-1)

    @staticmethod
    def expmap0(v: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """Exponential map from origin: tangent space R^d -> hyperboloid."""
        v      = LorentzManifold.clamp_tangent_norm(v)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
        sc     = math.sqrt(c)
        x_time  = torch.cosh(sc * v_norm) / sc
        x_space = torch.sinh(sc * v_norm) / (sc * v_norm + eps) * v
        return torch.cat([x_time, x_space], dim=-1)

    @staticmethod
    def logmap0(x: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """Logarithmic map from hyperboloid back to tangent space at origin."""
        x_time     = x[..., :1].clamp(min=1.0 + eps, max=1e6)
        x_space    = x[..., 1:]
        space_norm = torch.norm(x_space, dim=-1, keepdim=True).clamp(min=eps)
        sc         = math.sqrt(c)
        coeff      = torch.acosh((sc * x_time).clamp(min=1.0 + eps, max=1e6))
        coeff      = (coeff / (sc * space_norm + eps)).clamp(max=10.0)
        return LorentzManifold.clamp_tangent_norm(coeff * x_space)

    @staticmethod
    def mobius_add_approx(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Approximate Mobius addition via logmap -> sum -> expmap."""
        return LorentzManifold.expmap0(
            LorentzManifold.clamp_tangent_norm(
                LorentzManifold.logmap0(x, c) + LorentzManifold.logmap0(y, c)
            ), c
        )

    @staticmethod
    def geodesic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """Geodesic distance on the hyperboloid."""
        inner = -LorentzManifold.minkowski_dot(x, y, keepdim=False)
        return torch.acosh(inner.clamp(min=1.0 + eps, max=1e6)) / math.sqrt(c)


if __name__ == "__main__":
    # Quick sanity check — see python/tests/test_lorentz.py for the full suite
    v = torch.randn(4, 512)
    v = v / v.norm(dim=-1, keepdim=True) * 1.0   # norm=1.0, inside valid domain
    err = (LorentzManifold.logmap0(LorentzManifold.expmap0(v)) - v).norm(dim=-1).mean().item()
    print(f"Round-trip error (norm=1.0 input): {err:.2e}  (target < 1e-3)")
