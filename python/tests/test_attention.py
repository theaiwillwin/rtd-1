import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.attention import FanoSparseAttention


def test_mask_density_target():
    """Primary-line mask should give density ~3/7 ~ 0.4286, not the
    buggy union-of-lines density of ~0.756."""
    attn = FanoSparseAttention(512, num_heads=8)
    for T in [32, 64, 128, 256]:
        d = attn.density(T)
        assert 0.40 <= d <= 0.46, f"T={T}: density={d:.4f} outside [0.40, 0.46]"


def test_all_fano_lines_covered_exactly_once():
    lines_used = [frozenset(v) for v in FanoSparseAttention.FANO_PRIMARY_LINE.values()]
    assert len(lines_used) == len(set(lines_used)), "Duplicate primary line assignment"
    expected_lines = {
        frozenset(l) for l in
        [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
    }
    assert set(lines_used) == expected_lines, "Not all Fano lines covered"


def test_forward_no_nan():
    attn = FanoSparseAttention(512, num_heads=8)
    x = torch.randn(2, 256, 512)
    out = attn(x, curvature=0.1)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
