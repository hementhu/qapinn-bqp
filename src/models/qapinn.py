"""Hybrid QAPINN: classical pre-net -> quantum layer -> classical post-net.

    (x, t) --pre--> latent (n_qubits) --VQC--> measurements --post--> u

Reuses the exact residual/loss/trainer machinery of the classical PINN (it is
just another ``model(xt) -> u`` callable), so classical-vs-QAPINN ablations are
controlled comparisons.

Dimension reconciliation with CLAUDE.md's sketch (pre 2->16, VQC 4 qubits,
post 4->16->1): the pre-net's final width is pinned to ``n_qubits`` so it emits
exactly one angle per qubit; the post-net input is pinned to the VQC output dim
(which depends on the measurement knob).
"""
from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from src.models.quantum_layer import QuantumLayer, VQCConfig


def _mlp(dims: list[int], final_act: bool) -> nn.Sequential:
    """Tanh MLP over the given widths; Xavier init, optional final tanh."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2 or final_act:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class HybridPINN(nn.Module):
    """Classical -> quantum -> classical PINN.

    Args:
        vqc: VQC configuration (qubits, depth, entanglement, measurement, ...).
        pre_hidden: hidden widths of the pre-net between input(2) and n_qubits.
        post_hidden: hidden widths of the post-net between vqc-out and 1.
        in_dim: coordinate dimension (2 for Burgers (x, t)).
        out_dim: solution dimension (1 for scalar u).
    """

    def __init__(
        self,
        vqc: VQCConfig,
        pre_hidden: list[int] | None = None,
        post_hidden: list[int] | None = None,
        in_dim: int = 2,
        out_dim: int = 1,
    ):
        super().__init__()
        pre_hidden = [16] if pre_hidden is None else list(pre_hidden)
        post_hidden = [16] if post_hidden is None else list(post_hidden)

        self.quantum = QuantumLayer(vqc)
        # tanh on the pre-net output bounds encoding angles to (-1, 1)*input_scale
        self.pre = _mlp([in_dim, *pre_hidden, vqc.n_qubits], final_act=True)
        self.post = _mlp([self.quantum.out_dim, *post_hidden, out_dim], final_act=False)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        z = self.pre(xt)
        q = self.quantum(z)
        return self.post(q)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "pre": count(self.pre),
            "quantum": count(self.quantum),
            "post": count(self.post),
            "total": self.num_parameters(),
        }

    def architecture(self) -> dict:
        """Full, serializable description of the model (for reproducibility)."""
        return {
            "vqc": asdict(self.quantum.cfg),
            "pre": [m.in_features for m in self.pre if isinstance(m, nn.Linear)]
                   + [self.quantum.n_qubits],
            "post_out_dim": 1,
            "vqc_out_dim": self.quantum.out_dim,
            "param_breakdown": self.parameter_breakdown(),
        }


def build_qapinn_from_config(cfg: dict) -> "HybridPINN":
    """Construct a HybridPINN from a parsed YAML dict (see qapinn_baseline.yaml)."""
    vqc = VQCConfig(**cfg["vqc"])
    return HybridPINN(
        vqc,
        pre_hidden=cfg.get("pre_hidden", [16]),
        post_hidden=cfg.get("post_hidden", [16]),
    )
