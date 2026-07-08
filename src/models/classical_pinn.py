"""Classical fully-connected PINN (the baseline / parameter-matched control).

A standard MLP: tanh activations (smooth high-order derivatives, needed for the
PDE residual) and Xavier init. Kept deliberately generic — the residual and loss
functions take any ``model(xt) -> u`` callable, so the QAPINN can reuse the same
training machinery by swapping this module for a hybrid classical+VQC model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PINNMLP(nn.Module):
    """Fully-connected network u(x, t) with tanh activations and Xavier init.

    Args:
        layers: layer widths including input and output, e.g. [2, 20, 20, 20, 1]
            for a 2-input (x, t), 1-output (u) net with three hidden layers.
    """

    def __init__(self, layers: list[int]):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must have at least [in_dim, out_dim]")
        self.layers_spec = list(layers)
        self.linears = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )
        self.act = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-normal weights, zero biases (standard PINN initialization)."""
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """Map input coordinates (..., in_dim) -> output (..., out_dim)."""
        h = xt
        for lin in self.linears[:-1]:
            h = self.act(lin(h))
        return self.linears[-1](h)  # linear output layer

    def num_parameters(self) -> int:
        """Total number of trainable parameters (for fair QAPINN comparison)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
