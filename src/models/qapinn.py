"""QAPINN (Shah et al. 2024, ref [3]): first hidden layer replaced by a VQC.

    X=(x,t) --[fixed coord map]--> VQC (quantum first hidden layer)
            --> classical post-net (MLP) --> u

Design choices that match ref [3] and keep the Fourier picture clean:
  - NO trainable pre-net: the VQC encodes the raw coordinates, so the accessible
    frequency band lives directly in (x,t) — required for the spectral analysis.
  - Classical post-net (not a bare linear readout): supplies the flexibility a
    linear readout lacked (which made a pure-VQC model collapse to the trivial
    solution). The post-net is just a tanh MLP (reuses PINNMLP).
  - Fixed round-robin coord map distributes the in_dim inputs across n_qubits
    (e.g. 4 qubits, 2 inputs -> encode x,t,x,t). Non-trainable, so this is
    genuinely "a VQC as the first layer," not a learned embedding.

Reuses QuantumLayer (all ablation knobs) from vqc.py unchanged.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from src.models.pinn import PINNMLP
from src.models.vqc import QuantumLayer, VQCConfig


class QAPINN(nn.Module):
    def __init__(
        self,
        vqc: VQCConfig,
        post_hidden: list[int] | None = None,
        in_dim: int = 2,
        out_dim: int = 1,
    ):
        super().__init__()
        post_hidden = [16] if post_hidden is None else list(post_hidden)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.vqc = QuantumLayer(vqc)
        # fixed (non-trainable) map: qubit i encodes input coordinate (i mod in_dim)
        self.register_buffer(
            "coord_map", torch.tensor([i % in_dim for i in range(vqc.n_qubits)])
        )
        # classical post-net: vqc measurements -> ... -> u
        self.post = PINNMLP([self.vqc.out_dim, *post_hidden, out_dim])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = X[:, self.coord_map]        # (batch, n_qubits): raw coords -> encoding slots
        q = self.vqc(z)                 # (batch, vqc_out)
        return self.post(q)             # (batch, out_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        vqc_n = sum(p.numel() for p in self.vqc.parameters() if p.requires_grad)
        post_n = sum(p.numel() for p in self.post.parameters() if p.requires_grad)
        return {"vqc": vqc_n, "post": post_n, "total": vqc_n + post_n}

    def architecture(self) -> dict:
        return {
            "type": "qapinn",
            "vqc": asdict(self.vqc.cfg),
            "coord_map": self.coord_map.tolist(),
            "vqc_out_dim": self.vqc.out_dim,
            "post": [self.vqc.out_dim, *[l.out_features for l in self.post.linears]],
            "param_breakdown": self.parameter_breakdown(),
        }
