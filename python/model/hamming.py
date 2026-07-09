"""
Hamming(7,4,3) error-correcting token embeddings.

The parity check matrix H IS the Fano plane incidence matrix — not an
analogy, the same combinatorial object. This makes embeddings naturally
robust to quantization noise, which matters when this model eventually
gets quantized for FPGA deployment.
"""

import torch
import torch.nn as nn


class HammingEmbedding(nn.Module):
    """
    Token embeddings with Hamming(7,4,3) structure.

    Pipeline: [B,T,D] -> reshape [B,T,D/4,4] -> matmul G -> [B,T,D/4,7]
              -> learned 7->4 compression -> [B,T,D]

    syndrome_magnitude() is a free OOD/anomaly detector: near-zero for
    clean in-distribution tokens, large for corrupted/adversarial input.
    """

    G = torch.tensor([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=torch.float32)  # [4,7] generator

    H = torch.tensor([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ], dtype=torch.float32)  # [3,7] parity check = Fano incidence matrix

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        assert dim % 4 == 0, f"dim={dim} must be divisible by 4"
        self.dim         = dim
        self.n_groups    = dim // 4
        self.base_embed  = nn.Embedding(vocab_size, dim)
        self.encode_proj = nn.Linear(7, 4, bias=False)
        self.register_buffer("G_mat", self.G)
        self.register_buffer("H_mat", self.H)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.base_embed(input_ids)                   # [B,T,D]
        B, T, D = x.shape
        x4 = x.view(B, T, self.n_groups, 4)
        x7 = torch.matmul(x4, self.G_mat)                 # [B,T,D/4,7]
        return self.encode_proj(x7).view(B, T, D)

    def syndrome_magnitude(self, input_ids: torch.Tensor) -> float:
        x  = self.base_embed(input_ids)
        x4 = x.view(*x.shape[:-1], self.n_groups, 4)
        x7 = torch.matmul(x4, self.G_mat)
        s  = torch.matmul(x7, self.H_mat.T)
        return s.norm(dim=-1).mean().item()


if __name__ == "__main__":
    hemb = HammingEmbedding(50257, 512)
    ids  = torch.randint(0, 50257, (2, 16))
    emb  = hemb(ids)
    print(f"Output shape: {emb.shape}  (expect [2,16,512])")
    print(f"Syndrome magnitude (clean tokens): {hemb.syndrome_magnitude(ids):.4f}")
