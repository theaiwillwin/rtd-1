import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.octonion import OctonionLinear, OCTONION_SIGN, OCTONION_INDEX


def test_anticommutativity():
    """ei*ej = -ej*ei for i != j, both nonzero imaginary units."""
    for i in range(1, 8):
        for j in range(1, 8):
            if i != j:
                assert OCTONION_SIGN[i, j].item() == -OCTONION_SIGN[j, i].item()


def test_non_associativity():
    """Octonions are deliberately non-associative — confirm this, don't fix it."""
    S, I = OCTONION_SIGN, OCTONION_INDEX
    ab_s, ab_i = S[1, 2], I[1, 2]
    abc_s, abc_i = ab_s * S[ab_i, 3], I[ab_i, 3]
    bc_s, bc_i = S[2, 3], I[2, 3]
    a_bc_s = S[1, bc_i] * bc_s
    assert (abc_s != a_bc_s).item(), "Octonions should be non-associative"


def test_compression_ratio():
    std_params = 512 * 512 + 512
    oct_params = sum(p.numel() for p in OctonionLinear(512, 512).parameters())
    ratio = std_params / oct_params
    assert 7.5 < ratio < 8.5, f"Expected ~8x compression, got {ratio:.2f}x"


def test_forward_shape_and_stability():
    layer = OctonionLinear(512, 512)
    x = torch.randn(2, 16, 512)
    y = layer(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
