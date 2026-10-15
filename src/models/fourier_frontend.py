"""FourierFrontEnd: random-Fourier-feature (RFF) control for the quantum layer.

    (x,t) -> [cos(w_k . (x,t) + b_k)]_{k=1..width}  (FIXED) -> classical post-net -> u

A VQC with R_Y angle encoding of raw coordinates is a trigonometric polynomial with
integer frequencies (|w| <= 3 per coordinate without entanglement, <= 6 with the
linear CNOT chain; see verify_claims.py). This control replaces it with *fixed*,
untrained cosine features whose integer frequencies are drawn uniformly from the
same range, feeding the identical post-net. If it matches the QAPINN, the quantum
layer's contribution is reproducible by a classical Fourier feature map
(cf. Landman et al. 2023; Tancik et al. 2020).

Frequencies and phases are drawn from a generator seeded by ``feature_seed``
(default: the run seed is used via torch's global RNG at construction time).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.pinn import PINNMLP


class FourierFrontEnd(nn.Module):
    def __init__(self, width: int = 4, max_freq: int = 6, post_hidden: list[int] | None = None,
                 in_dim: int = 2, out_dim: int = 1):
        super().__init__()
        post_hidden = [16] if post_hidden is None else list(post_hidden)
        self.in_dim, self.out_dim = in_dim, out_dim
        self.width, self.max_freq = width, max_freq
        W = torch.randint(-max_freq, max_freq + 1, (width, in_dim)).float()
        zero = (W == 0).all(dim=1)                      # avoid constant features
        W[zero, 0] = 1.0
        self.register_buffer("W", W)
        self.register_buffer("b", torch.rand(width) * 2 * math.pi)
        self.post = PINNMLP([width, *post_hidden, out_dim])

    def features(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cos(X @ self.W.T + self.b)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.post(self.features(X))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict:
        return {"type": "rff", "width": self.width, "max_freq": self.max_freq,
                "frequencies": self.W.tolist(),
                "post": [self.width, *[l.out_features for l in self.post.linears]],
                "num_params": self.num_parameters()}


def build_rff_from_config(cfg: dict) -> FourierFrontEnd:
    return FourierFrontEnd(width=cfg.get("width", 4), max_freq=cfg.get("max_freq", 6),
                           post_hidden=cfg.get("post_hidden", [16]),
                           in_dim=cfg.get("in_dim", 2), out_dim=cfg.get("out_dim", 1))
