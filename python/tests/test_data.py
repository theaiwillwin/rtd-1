"""
Data pipeline tests — run fully offline via a stub tokenizer.

The real WikiText-2 download (load_wikitext2_tokens) needs network access
to huggingface.co and is exercised on the training machine; everything
downstream of tokenization is covered here.
"""

import torch
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.data import (
    ConversationFormatter, SequentialLMLoader,
    is_doc_header, synthetic_tokens, tokens_from_lines,
)


class StubTokenizer:
    """One token per character — deterministic, no network."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) % 1000 for c in text])


def test_doc_header_detection():
    assert is_doc_header(" = Robert Boulter = \n")
    assert is_doc_header("= Valkyria Chronicles III =")
    assert not is_doc_header(" = = Career = = \n")      # section, not article
    assert not is_doc_header("plain prose line")
    assert not is_doc_header("")


def test_tokens_from_lines_marks_doc_starts():
    lines = [" = Article One = ", "some text here", " = Article Two = ", "more text"]
    tokens, doc_starts = tokens_from_lines(lines, StubTokenizer())
    assert tokens.dtype == torch.long
    assert doc_starts.dtype == torch.bool
    assert tokens.numel() == doc_starts.numel()
    assert doc_starts.sum().item() == 2
    assert doc_starts[0].item()  # first article starts at token 0


def test_sequential_loader_shapes_and_order():
    tokens = torch.arange(1000, dtype=torch.long)
    doc_starts = torch.zeros(1000, dtype=torch.bool)
    loader = SequentialLMLoader(tokens, doc_starts, batch_size=4, seq_len=10)
    batches = list(loader)
    assert len(batches) == len(loader) == 25
    for b in batches:
        assert b["input_ids"].shape == (4, 10)
    # Each lane must be a contiguous stream across steps (stateful batching)
    lane0 = torch.cat([b["input_ids"][0] for b in batches])
    assert (lane0 == torch.arange(250)).all()


def test_sequential_loader_doc_frac():
    tokens = torch.zeros(400, dtype=torch.long)
    doc_starts = torch.zeros(400, dtype=torch.bool)
    doc_starts[5] = True     # falls in lane 0, window 0 (batch_size=2 -> shard of 200)
    loader = SequentialLMLoader(tokens, doc_starts, batch_size=2, seq_len=10)
    fracs = [b["new_doc_frac"] for b in loader]
    assert fracs[0] == 0.5   # one of two lanes hits a boundary
    assert all(f == 0.0 for f in fracs[1:])


def test_synthetic_tokens():
    tokens, doc_starts = synthetic_tokens(n_tokens=5000, vocab_size=100)
    assert tokens.numel() == doc_starts.numel() == 5000
    assert tokens.max().item() < 100
    assert doc_starts.any().item()


def test_conversation_formatter_boundaries():
    fmt = ConversationFormatter(StubTokenizer())
    conv = fmt.encode([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "bye"},
    ])
    assert conv.input_ids.dim() == 2 and conv.input_ids.shape[0] == 1
    overrides = [o for _, o in conv.boundaries]
    assert overrides == [0.0, 0.5, 0.5]   # hard reset at start, soft decay per turn
    offsets = [t for t, _ in conv.boundaries]
    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    assert offsets[-1] < conv.input_ids.shape[1]
    assert "User: hello" in conv.text and "Assistant: hi there" in conv.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
