"""
Fano plane structure and octonion-compressed linear layers.

The Fano plane (7 points, 7 lines, 3 points/line, 3 lines/point) IS the
multiplication table for octonions. OctonionLinear exploits this to get
8x parameter compression vs nn.Linear at the same in/out dimensions.

Init note: weight std = 1/sqrt(in_features), NOT a flat small constant.
A flat std=0.02 init causes activation variance to collapse through
depth, producing near-uniform logits with high cross-entropy loss at
initialization (~80+ instead of the expected ln(vocab_size) ~ 10.8).
"""

import torch
import torch.nn as nn
import math


FANO_LINES = [
    (0, 1, 2), (0, 3, 4), (0, 5, 6),
    (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5),
]


def build_octonion_table():
    """Build 8x8 sign and index tables for octonion multiplication.

    Rules:
      e0 * ei = ei * e0 = ei      (real unit)
      ei * ei = -e0                (imaginary squares to -1)
      For each Fano line {a,b,c} -> i=a+1, j=b+1, k=c+1:
        ei*ej=+ek, ej*ek=+ei, ek*ei=+ej   (cyclic)
        ej*ei=-ek, ek*ej=-ei, ei*ek=-ej   (anti-cyclic)
    """
    sign  = torch.zeros(8, 8)
    index = torch.zeros(8, 8, dtype=torch.long)
    for i in range(8):
        sign[0, i] = sign[i, 0] = 1.0
        index[0, i] = index[i, 0] = i
    for i in range(1, 8):
        sign[i, i]  = -1.0
        index[i, i] = 0
    for (a, b, c) in FANO_LINES:
        i, j, k = a + 1, b + 1, c + 1
        sign[i, j] = 1.0;  index[i, j] = k
        sign[j, k] = 1.0;  index[j, k] = i
        sign[k, i] = 1.0;  index[k, i] = j
        sign[j, i] = -1.0; index[j, i] = k
        sign[k, j] = -1.0; index[k, j] = i
        sign[i, k] = -1.0; index[i, k] = j
    return sign, index


OCTONION_SIGN, OCTONION_INDEX = build_octonion_table()


class OctonionLinear(nn.Module):
    """
    8x parameter-compressed drop-in replacement for nn.Linear.

    Standard Linear(512,512):  262,144 params
    OctonionLinear(512,512):    32,768 params

    in_features and out_features must both be divisible by 8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features  % 8 == 0, f"in_features={in_features} must be divisible by 8"
        assert out_features % 8 == 0, f"out_features={out_features} must be divisible by 8"
        self.in_oct  = in_features  // 8
        self.out_oct = out_features // 8

        # Kaiming-style init — keeps activation variance ~1.0 through depth
        std = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(self.out_oct, self.in_oct, 8) * std)
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        sign, index = build_octonion_table()
        self.register_buffer("sign",  sign)
        self.register_buffer("index", index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_dims = x.shape[:-1]
        x_oct  = x.view(*B_dims, self.in_oct, 8)
        # products: [..., out_oct, in_oct, 8, 8]
        products = (
            self.weight.unsqueeze(-1) *
            x_oct.unsqueeze(-4).unsqueeze(-1) *
            self.sign
        )
        out = torch.zeros(*B_dims, self.out_oct, 8, device=x.device, dtype=x.dtype)
        for k in range(8):
            mask = (self.index == k)
            out[..., k] = products[..., mask].view(
                *B_dims, self.out_oct, self.in_oct, -1
            ).sum(dim=(-1, -2))
        out = out.view(*B_dims, self.out_oct * 8)
        if self.bias is not None:
            out = out + self.bias
        return out


if __name__ == "__main__":
    std_params = 512 * 512 + 512
    oct_params = sum(p.numel() for p in OctonionLinear(512, 512).parameters())
    print(f"Standard Linear(512,512): {std_params:,} params")
    print(f"OctonionLinear(512,512):  {oct_params:,} params")
    print(f"Compression: {std_params/oct_params:.2f}x  (target ~8x)")
