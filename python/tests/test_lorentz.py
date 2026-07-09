"""
Lorentz manifold tests.

IMPORTANT: round-trip expmap0->logmap0 only holds exactly for inputs
with norm <= MAX_TANGENT_NORM (2.0). Testing with larger norms will
correctly show a "failure" that is actually the clamp working as
intended — not a bug. Always test round-trip with in-domain vectors.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.lorentz import LorentzManifold


def test_roundtrip_within_domain():
    """Round-trip must hold for norms strictly inside [0, MAX_TANGENT_NORM)."""
    for target_norm in [0.1, 0.5, 1.0, 1.5, 1.9]:
        v = torch.randn(4, 512)
        v = v / v.norm(dim=-1, keepdim=True) * target_norm
        err = (LorentzManifold.logmap0(LorentzManifold.expmap0(v)) - v).norm(dim=-1).mean().item()
        assert err < 1e-3, f"Round-trip failed at norm={target_norm}: error={err}"


def test_out_of_domain_clamps_correctly():
    """Vectors exceeding MAX_TANGENT_NORM should be clamped, not error."""
    v_large = torch.randn(4, 512) * 0.5  # norm ~11.3, way outside domain
    mapped  = LorentzManifold.logmap0(LorentzManifold.expmap0(v_large))
    norms   = torch.norm(mapped, dim=-1)
    assert (norms <= LorentzManifold.MAX_TANGENT_NORM + 1e-5).all()


def test_geodesic_symmetry():
    x1 = LorentzManifold.project_to_hyperboloid(torch.randn(4, 513))
    x2 = LorentzManifold.project_to_hyperboloid(torch.randn(4, 513))
    d12 = LorentzManifold.geodesic_distance(x1, x2)
    d21 = LorentzManifold.geodesic_distance(x2, x1)
    assert (d12 - d21).abs().max().item() < 1e-5


def test_hyperboloid_constraint():
    x = LorentzManifold.project_to_hyperboloid(torch.randn(4, 513))
    constraint = (x[..., 0]**2 - (x[..., 1:]**2).sum(-1) - 1.0).abs().max().item()
    assert constraint < 1e-4


def test_no_nan_on_extreme_input():
    big = LorentzManifold.expmap0(torch.randn(4, 512) * 100.0)
    assert not torch.isnan(big).any().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
