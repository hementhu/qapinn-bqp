"""Sampling of collocation, initial-condition, and boundary points for Burgers.

All sampling is driven by an explicit ``torch.Generator`` so runs are
reproducible from a seed. Interior points use Latin Hypercube sampling (better
space-filling than i.i.d. uniform, standard for PINNs); IC/BC use structured
edges. Column convention throughout: (x, t).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc

from src.pdes.burgers import (
    X_MIN, X_MAX, T_MIN, T_MAX, initial_condition,
)


def _to_tensor(a: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32, requires_grad=requires_grad)


def sample_interior(n: int, seed: int, requires_grad: bool = True) -> torch.Tensor:
    """Latin Hypercube collocation points in the open domain, shape (n, 2).

    requires_grad=True by default because these feed the PDE residual (autograd).
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n)  # (n, 2) in [0, 1)
    x = X_MIN + unit[:, 0] * (X_MAX - X_MIN)
    t = T_MIN + unit[:, 1] * (T_MAX - T_MIN)
    xt = np.stack([x, t], axis=1)
    return _to_tensor(xt, requires_grad=requires_grad)


def sample_ic(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial-condition points at t=0. Returns (xt (n,2), u_target (n,1)).

    u(x, 0) = -sin(pi x).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=n)
    t = np.zeros(n)
    xt = np.stack([x, t], axis=1)
    u = initial_condition(x).reshape(-1, 1)
    return _to_tensor(xt), _to_tensor(u)


def sample_bc(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Boundary points at x=+-1. Returns (xt (n,2), u_target (n,1)).

    u(-1, t) = u(1, t) = 0. Half the points on each wall.
    """
    rng = np.random.default_rng(seed)
    n_left = n // 2
    n_right = n - n_left
    t = rng.uniform(T_MIN, T_MAX, size=n)
    x = np.concatenate([np.full(n_left, X_MIN), np.full(n_right, X_MAX)])
    xt = np.stack([x, t], axis=1)
    u = np.zeros((n, 1))
    return _to_tensor(xt), _to_tensor(u)
