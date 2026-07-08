"""Variational quantum circuit (VQC) layer for the QAPINN, in PennyLane.

Designed around the challenge's four ablation knobs so each is a config change:

    encoding      : angle encoding (RY), with optional data re-uploading
    depth         : number of variational layers (n_layers)
    entanglement  : none | linear | ring | all_to_all
    measurement   : single <Z0> | multi <Zi> | zz correlators

Built as a hand-wired QNode (interface="torch", diff_method="backprop") wrapped
in an nn.Module, rather than qml.qnn.TorchLayer, so that the second-order
autograd needed for the PDE residual (u_xx) flows reliably, and so the circuit
is directly available for Fourier / gradient-variance analysis later.

Mathematical note (Schuld/Sweke/Meyer 2021): with L re-uploads of a scalar via
RY, the model is a truncated Fourier series with integer frequencies in
{-L, ..., L}. Re-upload count therefore controls accessible frequency content —
the lever we use to match encoding depth to a PDE's spectral demands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pennylane as qml
import torch
import torch.nn as nn


@dataclass
class VQCConfig:
    n_qubits: int = 4
    n_layers: int = 3                     # variational/ansatz depth
    reupload: bool = True                 # re-encode data each layer
    entanglement: str = "linear"          # none | linear | ring | all_to_all
    measurement: str = "multi"            # single | multi | zz
    input_scale: float = 1.0              # angle = input_scale * feature


def _entangle(entanglement: str, n_qubits: int) -> None:
    """Apply the chosen entangling pattern of CNOTs (in place on the tape)."""
    if entanglement == "none" or n_qubits < 2:
        return
    if entanglement == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif entanglement == "ring":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    elif entanglement == "all_to_all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    else:
        raise ValueError(f"unknown entanglement '{entanglement}'")


def _measurements(measurement: str, n_qubits: int) -> list:
    """Return the list of observables to measure -> defines the output dim."""
    if measurement == "single":
        return [qml.PauliZ(0)]
    if measurement == "multi":
        return [qml.PauliZ(i) for i in range(n_qubits)]
    if measurement == "zz":
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
        for i in range(n_qubits - 1):
            obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
        return obs
    raise ValueError(f"unknown measurement '{measurement}'")


def output_dim(cfg: VQCConfig) -> int:
    return len(_measurements(cfg.measurement, cfg.n_qubits))


class QuantumLayer(nn.Module):
    """A trainable VQC mapping (batch, n_qubits) angles -> (batch, output_dim).

    Trainable weights: (n_layers, n_qubits, 2) — an RY and RZ rotation per qubit
    per layer. Data (the pre-net features) is angle-encoded via RY at the start
    of each layer when ``reupload`` is True, else only once.
    """

    def __init__(self, cfg: VQCConfig):
        super().__init__()
        self.cfg = cfg
        self.n_qubits = cfg.n_qubits
        self.out_dim = output_dim(cfg)

        self.dev = qml.device("default.qubit", wires=cfg.n_qubits)
        obs = _measurements(cfg.measurement, cfg.n_qubits)

        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # inputs: (..., n_qubits) angles ; weights: (n_layers, n_qubits, 2)
            for layer in range(cfg.n_layers):
                if cfg.reupload or layer == 0:
                    for w in range(cfg.n_qubits):
                        qml.RY(cfg.input_scale * inputs[..., w], wires=w)
                for w in range(cfg.n_qubits):
                    qml.RY(weights[layer, w, 0], wires=w)
                    qml.RZ(weights[layer, w, 1], wires=w)
                _entangle(cfg.entanglement, cfg.n_qubits)
            return [qml.expval(o) for o in obs]

        self.qnode = qml.QNode(
            circuit, self.dev, interface="torch", diff_method="backprop"
        )

        # Small-angle init keeps the circuit near identity (helps early training
        # and mirrors the barren-plateau-aware initialization we study later).
        w = 0.1 * torch.randn(cfg.n_layers, cfg.n_qubits, 2)
        self.weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # qnode returns a list of (batch,) tensors -> stack to (batch, out_dim)
        out = self.qnode(x, self.weights)
        out = torch.stack(out, dim=-1)
        return out.to(x.dtype)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
