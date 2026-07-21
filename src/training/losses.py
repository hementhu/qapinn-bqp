"""PDE-agnostic composite loss: residual + constraint (IC/BC/...) penalties.

Soft (penalty) enforcement of all constraints, so the identical loss serves the
classical PINN and the QAPINN — the only difference between them is the model.
The loss walks spec.residual and spec.constraints; it never names a PDE, an
equation count, or a constraint type. That is what makes it run Burgers, heat,
and 3-equation Navier-Stokes with no change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.pdes.base import Model, PDESpec


@dataclass
class LossWeights:
    pde: float = 1.0
    constraints: float = 1.0     # single weight applied to the summed constraint loss


@dataclass
class LossTerms:
    total: torch.Tensor
    pde: torch.Tensor
    constraints: torch.Tensor                       # summed constraint loss
    per_constraint: dict = field(default_factory=dict)  # name -> loss (for logging)


def pde_loss(
    model: Model,
    spec: PDESpec,
    X_interior: torch.Tensor,
    constraint_points: list[tuple[torch.Tensor, torch.Tensor]],
    weights: LossWeights = LossWeights(),
) -> LossTerms:
    """Residual MSE (over all equation columns) + summed constraint MSE."""
    residual = spec.residual(model, X_interior)             # (N, n_eqns)
    loss_pde = residual.pow(2).mean()                       # mean over points AND equations

    per = {}
    for c, (Xc, target) in zip(spec.constraints, constraint_points):
        pred = model(Xc)[:, list(c.components)]             # select constrained outputs
        per[c.name] = F.mse_loss(pred, target)
    loss_con = sum(per.values())

    total = weights.pde * loss_pde + weights.constraints * loss_con
    return LossTerms(total=total, pde=loss_pde, constraints=loss_con, per_constraint=per)
