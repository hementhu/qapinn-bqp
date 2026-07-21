"""Loss-landscape XAI — dimension 4: the geometry each model optimizes in.

2-D slice of the training loss around the trained minimum along two random,
filter-normalized directions (Li et al. 2018 — normalizing each direction block
to its weight-tensor norm makes the picture scale-invariant and comparable across
models). Flat, wide minima vs sharp ones is a standard "how it learns / how well
it generalizes" signal, compared classical vs QAPINN.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pdes.base import get_pde
from src.training.collocation import sample_interior, sample_constraints
from src.training.losses import LossWeights, pde_loss


def _filter_normalized_direction(params, generator):
    """Random direction with each param-tensor block scaled to that tensor's norm."""
    d = []
    for p in params:
        r = torch.randn(p.shape, generator=generator)
        r = r / (r.norm() + 1e-12) * (p.norm() + 1e-12)   # filter normalization
        d.append(r)
    return d


def loss_landscape(model, cfg, span: float = 1.0, n: int = 21, seed: int = 0):
    """2-D loss surface around the trained minimum. Returns alpha, beta, loss (n,n).

    cfg is a TrainConfig (for pde + point counts). The model is perturbed in place
    and restored afterward.
    """
    spec = get_pde(cfg.pde)
    weights = LossWeights(pde=cfg.w_pde, constraints=cfg.w_constraints)
    X_int = sample_interior(spec, cfg.n_interior, seed=cfg.seed + 1)
    con_pts = sample_constraints(spec, cfg.n_constraint, seed=cfg.seed + 2)

    params = [p for p in model.parameters() if p.requires_grad]
    theta = [p.detach().clone() for p in params]
    g = torch.Generator().manual_seed(seed)
    d1 = _filter_normalized_direction(theta, g)
    d2 = _filter_normalized_direction(theta, g)

    grid = np.linspace(-span, span, n)
    loss = np.zeros((n, n))
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            with torch.no_grad():
                for p, t, u, v in zip(params, theta, d1, d2):
                    p.copy_(t + a * u + b * v)
            loss[i, j] = pde_loss(model, spec, X_int, con_pts, weights).total.item()
    with torch.no_grad():                                  # restore
        for p, t in zip(params, theta):
            p.copy_(t)
    return grid, grid, loss


def plot_landscape(alpha, beta, loss, out_path, title: str = "Loss landscape"):
    A, B = np.meshgrid(alpha, beta)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(A, B, np.log10(loss.T + 1e-12), levels=30, cmap="viridis")
    ax.contour(A, B, np.log10(loss.T + 1e-12), levels=12, colors="k", linewidths=0.3, alpha=0.4)
    ax.plot(0, 0, "r*", markersize=12)                     # the trained minimum
    ax.set_xlabel("direction 1"); ax.set_ylabel("direction 2"); ax.set_title(title)
    fig.colorbar(cs, ax=ax, label="log10 loss")
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
