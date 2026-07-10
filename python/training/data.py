"""
WikiText-2 dataloader and conversation formatter for CWD-PHFT.

Persistent-state note: CWD-PHFT carries state across batches (hyperbolic
field, memory banks, momentum velocity), so data must arrive IN CORPUS
ORDER, never shuffled. SequentialLMLoader uses the standard stateful-LM
batching scheme: the token stream is split into `batch_size` contiguous
shards and step t serves the t-th window of every shard, so each batch
lane is a continuous stream across steps.

Document boundaries are surfaced per batch as `new_doc_frac` (fraction
of lanes whose window contains the start of a new article). The training
loop maps this to the field's decay_override semantics:
  new_doc_frac == 0 -> None (normal decay within a document)
  new_doc_frac  > 0 -> 0.5  (soft decay, partial forgetting)
Hard reset (0.0 / reset_persistent_state) is reserved for genuine
conversation boundaries, e.g. between dialogue episodes.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch


# WikiText article titles look like " = Robert Boulter = " (single '=' pair);
# section headings use doubled markers ("= = Career = =") and are NOT doc starts.
_DOC_HEADER_RE = re.compile(r"^=\s+[^=].*[^=]\s+=$")


def is_doc_header(line: str) -> bool:
    return bool(_DOC_HEADER_RE.match(line.strip()))


def get_tokenizer(name: str = "gpt2"):
    """GPT-2 BPE tokenizer — vocab_size=50257, matching the model default."""
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(name)


def tokens_from_lines(lines, tokenizer):
    """Tokenize an iterable of text lines into one long stream.

    Returns:
      tokens:     LongTensor [N] — the concatenated corpus
      doc_starts: BoolTensor [N] — True at the first token of each article
    """
    ids: List[int] = []
    starts: List[int] = []
    for line in lines:
        if is_doc_header(line):
            starts.append(len(ids))
        if line.strip():
            ids.extend(tokenizer(line, add_special_tokens=False).input_ids)

    tokens = torch.tensor(ids, dtype=torch.long)
    doc_starts = torch.zeros(len(ids), dtype=torch.bool)
    for s in starts:
        if s < len(ids):
            doc_starts[s] = True
    return tokens, doc_starts


def load_wikitext2_tokens(split: str = "train", tokenizer=None):
    """Tokenize WikiText-2 (raw) into one long stream. See tokens_from_lines."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return tokens_from_lines(ds["text"], tokenizer or get_tokenizer())


def synthetic_tokens(n_tokens: int = 100_000, vocab_size: int = 1000,
                     avg_doc_len: int = 2000, seed: int = 0):
    """Random token stream with fake document boundaries — offline smoke tests."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_tokens,), generator=g)
    doc_starts = torch.zeros(n_tokens, dtype=torch.bool)
    doc_starts[::avg_doc_len] = True
    return tokens, doc_starts


class SequentialLMLoader:
    """Stateful-LM batching: `batch_size` contiguous shards, served in order.

    Yields dicts:
      input_ids:    LongTensor [batch_size, seq_len]
      new_doc_frac: float — fraction of lanes starting a new article here
    """

    def __init__(self, tokens: torch.Tensor, doc_starts: torch.Tensor,
                 batch_size: int, seq_len: int):
        n_windows = tokens.numel() // (batch_size * seq_len)
        assert n_windows > 0, (
            f"Corpus of {tokens.numel()} tokens too small for "
            f"batch_size={batch_size} x seq_len={seq_len}"
        )
        n_keep = n_windows * batch_size * seq_len
        self.seq_len   = seq_len
        self.n_windows = n_windows
        # [batch_size, n_windows * seq_len] — each row is a contiguous shard
        self.tokens     = tokens[:n_keep].view(batch_size, -1)
        self.doc_starts = doc_starts[:n_keep].view(batch_size, -1)

    def __len__(self) -> int:
        return self.n_windows

    def __iter__(self) -> Iterator[Dict]:
        for t in range(self.n_windows):
            lo, hi = t * self.seq_len, (t + 1) * self.seq_len
            yield {
                "input_ids":    self.tokens[:, lo:hi],
                "new_doc_frac": self.doc_starts[:, lo:hi].any(dim=-1).float().mean().item(),
            }


@dataclass
class FormattedConversation:
    """Token ids for one conversation plus the field-control schedule."""
    input_ids:      torch.Tensor            # [1, T]
    text:           str
    # (token_offset, decay_override) pairs: 0.0 at conversation start,
    # 0.5 at each subsequent turn boundary. Segments between entries run
    # with decay_override=None (normal within-turn decay).
    boundaries:     List[tuple] = field(default_factory=list)


class ConversationFormatter:
    """Render chat turns to text/token ids with field-boundary markers.

    Produces the decay_override schedule matching PersistentHyperbolicField:
      conversation start -> 0.0 (hard forgetting of prior context)
      each turn boundary -> 0.5 (soft decay, partial forgetting)
      within a turn      -> None
    """

    def __init__(self, tokenizer=None,
                 role_prefixes: Optional[Dict[str, str]] = None,
                 turn_sep: str = "\n"):
        self.tokenizer     = tokenizer
        self.role_prefixes = role_prefixes or {"user": "User: ", "assistant": "Assistant: "}
        self.turn_sep      = turn_sep

    def format_text(self, turns: List[Dict[str, str]]) -> str:
        parts = [self.role_prefixes.get(t["role"], f"{t['role'].title()}: ") + t["content"]
                 for t in turns]
        return self.turn_sep.join(parts) + self.turn_sep

    def encode(self, turns: List[Dict[str, str]]) -> FormattedConversation:
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer()
        ids: List[int] = []
        boundaries = [(0, 0.0)]  # conversation start -> hard forgetting
        for i, turn in enumerate(turns):
            if i > 0:
                boundaries.append((len(ids), 0.5))  # turn boundary -> soft decay
            prefix = self.role_prefixes.get(turn["role"], f"{turn['role'].title()}: ")
            ids.extend(self.tokenizer(prefix + turn["content"] + self.turn_sep,
                                      add_special_tokens=False).input_ids)
        return FormattedConversation(
            input_ids=torch.tensor([ids], dtype=torch.long),
            text=self.format_text(turns),
            boundaries=boundaries,
        )
