```
∂²ψₚᵃ/∂t² = c² · Δ_seq(ψ)ₚᵃ + c_chan² · Δ_chanᵃ(hₜ)
             - κ · ∂ψₚᵃ/∂t - ψₚᵃ
             - λ · K_{bc}^a · ψₚ^b ⊙ ψₚ^c
             + β · (ψₚᵃ - (ψₚ^a)³)
             + εₚ(t)
```

Where:

- `Δ_seq` is the Laplacian over the sequence dimension (adjacent tokens)
- `Δ_chan` is the Fano graph Laplacian over the 7 channels
- `c²` is the sequence propagation speed
- `c_chan²` is the channel coupling speed
- `εₚ(t)` is per-position noise
