"""Full-quantum PINN: coordinates encoded DIRECTLY into the VQC.

    (x, t) --RY re-upload--> [VQC ansatz] --measure <Zi>--> Linear(n->1) --> u

No classical pre-net: the physical coordinates (x, t) are the encoded angles, so
the Schuld/Sweke/Meyer Fourier result applies directly to (x, t) — the accessible
frequencies in each coordinate are set by the re-upload count. The only classical
part is a linear readout of the measured observables (a learned combination of
Fourier modes). This is the theory-pure counterpart to the hybrid HybridPINN.

Coordinate-to-qubit map is round-robin: with n_qubits=4, in_dim=2 the qubits
encode (x, t, x, t). Encoding angle on a qubit is ``input_scale * coordinate``.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from src.models.quantum_layer import QuantumLayer, VQCConfig


class QuantumPINN(nn.Module):
    def __init__(self, vqc: VQCConfig, in_dim: int = 2, out_dim: int = 1):
        super().__init__()
        self.quantum = QuantumLayer(vqc)
        self.in_dim = in_dim
        # Fixed (non-trainable) round-robin map coordinate -> qubit encoding slot,
        # so encoded angles are the raw coordinates and frequencies stay clean.
        self.register_buffer(
            "coord_map", torch.tensor([i % in_dim for i in range(vqc.n_qubits)])
        )
        self.readout = nn.Linear(self.quantum.out_dim, out_dim)
        nn.init.xavier_normal_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        z = xt[:, self.coord_map]          # (batch, n_qubits): each col a coordinate
        q = self.quantum(z)                # (batch, vqc_out)
        return self.readout(q)             # (batch, out_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "quantum": count(self.quantum),
            "readout": count(self.readout),
            "total": self.num_parameters(),
        }

    def architecture(self) -> dict:
        return {
            "type": "quantum",
            "vqc": asdict(self.quantum.cfg),
            "coord_map": self.coord_map.tolist(),
            "vqc_out_dim": self.quantum.out_dim,
            "param_breakdown": self.parameter_breakdown(),
        }


def build_quantum_pinn_from_config(cfg: dict) -> "QuantumPINN":
    """Construct a QuantumPINN from a parsed YAML dict (see quantum_baseline.yaml)."""
    return QuantumPINN(VQCConfig(**cfg["vqc"]))
