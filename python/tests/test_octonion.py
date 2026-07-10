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
    """Octonions are deliberately non-associative — confirm this, don't fix it.

    NOTE: the triple must NOT lie on a single Fano line — units on one line
    generate a quaternion subalgebra, which IS associative. (e1,e2,e3) sits
    on line (0,1,2) and associates; (e1,e2,e5) does not.
    """
    S, I = OCTONION_SIGN, OCTONION_INDEX
    ab_s, ab_i = S[1, 2], I[1, 2]
    abc_s, abc_i = ab_s * S[ab_i, 5], I[ab_i, 5]
    bc_s, bc_i = S[2, 5], I[2, 5]
    a_bc_s = S[1, bc_i] * bc_s
    assert (abc_s != a_bc_s).item(), "Octonions should be non-associative"


def test_compression_ratio():
    std_params = 512 * 512 + 512
    oct_params = sum(p.numel() for p in OctonionLinear(512, 512).parameters())
    ratio = std_params / oct_params
    assert 7.5 < ratio < 8.5, f"Expected ~8x compression, got {ratio:.2f}x"


def test_forward_matches_naive_reference():
    """The matmul-folded forward must equal direct octonion multiplication:
    out[o,k] = sum_i sum_{a,b: index[a,b]=k} sign[a,b] * w[o,i,a] * x[i,b]."""
    torch.manual_seed(0)
    layer = OctonionLinear(16, 24, bias=True)
    x = torch.randn(3, 5, 16)

    ref = torch.zeros(3, 5, layer.out_oct, 8)
    x_oct = x.view(3, 5, layer.in_oct, 8)
    for a in range(8):
        for b in range(8):
            k = OCTONION_INDEX[a, b].item()
            s = OCTONION_SIGN[a, b].item()
            ref[..., :, k] += s * torch.einsum(
                'oi,...i->...o', layer.weight[:, :, a], x_oct[..., b]
            )
    ref = ref.view(3, 5, 24) + layer.bias

    assert torch.allclose(layer(x), ref, atol=1e-5)


def test_identity_octonion_weight_is_identity_map():
    """Weight = e0 on a single 8-dim block must reproduce the input."""
    layer = OctonionLinear(8, 8, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, 0, 0] = 1.0
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), x, atol=1e-6)


def test_forward_shape_and_stability():
    layer = OctonionLinear(512, 512)
    x = torch.randn(2, 16, 512)
    y = layer(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
