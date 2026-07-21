"""PDE-agnostic plotting: solution-field comparison and loss-component curves."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def plot_solution(model, spec, out_path, nx: int = 301, nt: int = 151) -> None:
    """3-panel field (reference | prediction | |error|) + 3 time-slice panels.

    PDE-agnostic: uses spec.reference for ground truth and the (t0,t1) domain for
    slice locations. Scalar-output (out_dim=1) PDEs.
    """
    x, t, U = spec.reference(nx=nx, nt=nt)                    # (nt, nx)
    XX, TT = np.meshgrid(x, t)
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    P = model(X)[:, 0].cpu().numpy().reshape(U.shape)
    err = np.abs(P - U)

    fig = plt.figure(figsize=(14, 7))
    for k, (title, F, cmap) in enumerate(
        [("reference", U, "rainbow"), ("prediction", P, "rainbow"), ("|error|", err, "inferno")]
    ):
        ax = fig.add_subplot(2, 3, k + 1)
        pc = ax.pcolormesh(t, x, F.T, cmap=cmap, shading="auto")
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
        fig.colorbar(pc, ax=ax)

    for i, frac in enumerate((0.25, 0.5, 0.75)):
        j = int(frac * (nt - 1))
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.plot(x, U[j], "k-", lw=2, label="reference")
        ax.plot(x, P[j], "r--", lw=1.5, label="model")
        ax.set_title(f"t = {t[j]:.2f}"); ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_loss_history(history_csv, out_path) -> None:
    """Plot loss components (total, pde, constraints) from a RunLogger history.csv."""
    rows = list(csv.DictReader(open(history_csv)))
    if not rows:
        return
    it = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in ("total", "pde", "constraints"):
        if key in rows[0]:
            ax.semilogy(it, [float(r[key]) for r in rows], label=key)
    ax.set_xlabel("logged step"); ax.set_ylabel("loss (log)")
    ax.set_title("Training loss components"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
