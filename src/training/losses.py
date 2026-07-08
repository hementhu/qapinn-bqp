"""Composite PINN loss: PDE residual + soft IC/BC penalties.

Soft (penalty) enforcement of IC/BC — deliberately not hard-encoded — so the
identical loss machinery serves both the classical PINN and the QAPINN, keeping
the ablation apples-to-apples (a hard-constrained output would be awkward for the
VQC's bounded range).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from src.pdes.burgers import NU_DEFAULT, pde_residual

Model = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class LossWeights:
    """Relative weights for the three loss terms."""
    pde: float = 1.0
    ic: float = 1.0
    bc: float = 1.0


@dataclass
class LossTerms:
    """Individual (unweighted) loss components + the weighted total, for logging."""
    total: torch.Tensor
    pde: torch.Tensor
    ic: torch.Tensor
    bc: torch.Tensor


def burgers_loss(
    model: Model,
    xt_interior: torch.Tensor,
    xt_ic: torch.Tensor,
    u_ic: torch.Tensor,
    xt_bc: torch.Tensor,
    u_bc: torch.Tensor,
    weights: LossWeights = LossWeights(),
    nu: float = NU_DEFAULT,
) -> LossTerms:
    """Compute PDE + IC + BC MSE losses and the weighted total.

    Returns a LossTerms bundle so the trainer can log each component separately
    (the balance between terms is itself an object of study).
    """
    residual = pde_residual(model, xt_interior, nu=nu)
    loss_pde = residual.pow(2).mean()
    loss_ic = F.mse_loss(model(xt_ic), u_ic)
    loss_bc = F.mse_loss(model(xt_bc), u_bc)

    total = weights.pde * loss_pde + weights.ic * loss_ic + weights.bc * loss_bc
    return LossTerms(total=total, pde=loss_pde, ic=loss_ic, bc=loss_bc)
