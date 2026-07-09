import torch
import pytest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.cwd_phft import CWDPHFT


@pytest.fixture
def small_model():
    """Smaller config for fast tests."""
    return CWDPHFT(
        vocab_size=1000, dim=64, num_layers=2, num_heads=4,
        mem_size=16, field_decay=0.997, mem_momentum=0.99,
        fractal_iters=2, num_modes=4,
    )


def test_forward_pass_shape(small_model):
    ids = torch.randint(0, 1000, (2, 64))
    logits = small_model(ids)
    assert logits.shape == (2, 64, 1000)
    assert not torch.isnan(logits).any()


def test_energy_head(small_model):
    ids = torch.randint(0, 1000, (2, 64))
    logits, energy = small_model(ids, return_energy=True)
    assert energy.shape == (2, 64)


def test_modes_produce_different_output(small_model):
    ids = torch.randint(0, 1000, (1, 32))
    outs = []
    for m in range(4):
        small_model.detach_persistent_state()
        outs.append(small_model(ids, mode=m))
    for i in range(4):
        for j in range(i+1, 4):
            assert not torch.allclose(outs[i], outs[j], atol=1e-4)


def test_field_persists_across_batches(small_model):
    small_model.reset_persistent_state()
    states = []
    for _ in range(3):
        _ = small_model(torch.randint(0, 1000, (1, 32)))
        states.append(small_model.field.identity_state.clone().detach())
        small_model.detach_persistent_state()
    for i in range(len(states) - 1):
        assert not torch.allclose(states[i], states[i+1], atol=1e-7)


def test_reset_clears_field(small_model):
    _ = small_model(torch.randint(0, 1000, (1, 32)))
    before = small_model.field.identity_state.clone()
    small_model.reset_persistent_state()
    after = small_model.field.identity_state.clone()
    assert not torch.allclose(before, after, atol=1e-5)
    assert small_model.field.step_count.item() == 0


def test_gradient_flow(small_model):
    small_model.train()
    small_model.reset_persistent_state()
    ids = torch.randint(0, 1000, (2, 32))
    logits, energy = small_model(ids, return_energy=True)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 1000), ids[:, 1:].reshape(-1)
    ) + 0.01 * energy.mean()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in small_model.layers[0].attn.parameters()
    )
    assert has_grad, "Attention layer received no gradient"


def test_checkpoint_roundtrip(small_model, tmp_path):
    small_model.eval()
    ckpt_path = tmp_path / "test.pt"
    torch.save({
        "model_state_dict": small_model.state_dict(),
        "field_state": small_model.get_field_state_dict(),
    }, ckpt_path)

    model2 = CWDPHFT(
        vocab_size=1000, dim=64, num_layers=2, num_heads=4,
        mem_size=16, field_decay=0.997, mem_momentum=0.99,
        fractal_iters=2, num_modes=4,
    )
    ckpt = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.load_field_state_dict(ckpt["field_state"])
    model2.eval()

    ids = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        out1 = small_model(ids)
        out2 = model2(ids)
    assert (out1 - out2).abs().max().item() < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
