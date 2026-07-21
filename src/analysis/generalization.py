"""Generalization XAI — dimension 5: behaviour OUTSIDE the training domain.

The model was trained on t in [t0, t1]; we score it against the exact solution on
an extended time window t in [t0, t_max] and report relative L2 vs t. The jump at
the training boundary t1 is the extrapolation penalty. A Fourier-type model
extrapolates (quasi-)periodically, so this is where a QAPINN's spectral prior can
help or hurt — a concrete, testable prediction (and a listed challenge metric).

Uses spec.exact (pointwise, valid outside the training domain).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def generalization_curve(model, spec, t_max: float = 1.5, nx: int = 128, nt: int = 150):
    """Relative L2 vs t on [t0, t_max]. Returns t (nt,), rel_l2_per_t (nt,), t_train.

    Requires spec.exact. t_train is the training-domain upper bound (extrapolation
    starts beyond it).
    """
    if spec.exact is None:
        raise ValueError(f"pde '{spec.name}' has no exact() for extrapolation scoring")
    (x0, x1), (t0, t1) = spec.domain[0], spec.domain[1]
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t_max, nt)

    rel = np.zeros(nt)
    for j, tj in enumerate(t):
        X = torch.tensor(np.stack([x, np.full_like(x, tj)], 1), dtype=torch.float32)
        with torch.no_grad():
            pred = model(X)[:, 0].cpu().numpy()
        true = spec.exact(x, np.full_like(x, tj))
        denom = np.linalg.norm(true)
        rel[j] = np.linalg.norm(pred - true) / denom if denom > 1e-9 else np.linalg.norm(pred)
    return t, rel, t1


def plot_generalization(curves: dict, out_path, t_train: float):
    """Relative-L2-vs-t curves (one per model), with the extrapolation boundary marked."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (t, rel) in curves.items():
        ax.semilogy(t, rel, label=name)
    ax.axvline(t_train, color="k", ls="--", lw=1)
    ax.text(t_train, ax.get_ylim()[1], " train | extrapolate", va="top", fontsize=9)
    ax.set_xlabel("t"); ax.set_ylabel("relative L2 (log)")
    ax.set_title("In-domain vs extrapolation error"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
