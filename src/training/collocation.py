"""PDE-agnostic point sampling: interior collocation + constraint points.

Interior points are Latin-Hypercube-sampled inside the domain box (better space-
filling than i.i.d. uniform, standard for PINNs) and carry requires_grad=True
because they feed the residual autograd. Constraint points come from each
Constraint's own sampler (defined in the PDE module). Everything is seeded.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc

from src.pdes.base import PDESpec


def sample_interior(spec: PDESpec, n: int, seed: int, requires_grad: bool = True) -> torch.Tensor:
    """Latin-Hypercube collocation points in the domain box. Shape (n, in_dim)."""
    unit = qmc.LatinHypercube(d=spec.in_dim, seed=seed).random(n)   # (n, in_dim) in [0,1)
    lows = np.array([lo for lo, _ in spec.domain])
    highs = np.array([hi for _, hi in spec.domain])
    X = lows + unit * (highs - lows)
    return torch.tensor(X, dtype=torch.float32, requires_grad=requires_grad)


def sample_constraints(spec: PDESpec, n: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each constraint, return (X_points, target_values), sampled once.

    Uses a distinct sub-seed per constraint so different faces get different
    points. Points are held fixed for the run (L-BFGS needs a deterministic
    objective, and fixed points keep runs reproducible).
    """
    out = []
    for i, c in enumerate(spec.constraints):
        Xc = c.sample(n, seed + i)
        out.append((Xc, c.target(Xc)))
    return out
