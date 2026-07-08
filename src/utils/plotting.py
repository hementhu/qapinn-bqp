"""Plotting: predicted vs reference field, absolute error, and time slices."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pdes.burgers import NU_DEFAULT, cole_hopf_reference, reference_field


@torch.no_grad()
def plot_solution(
    model: torch.nn.Module,
    out_path: str | Path,
    nx: int = 401,
    nt: int = 201,
    nu: float = NU_DEFAULT,
) -> None:
    """Save a 3-panel field comparison + a 3-slice panel next to it.

    Top row: reference | prediction | |error|.  Bottom row: snapshots at
    t = 0.25, 0.5, 0.75 (prediction vs reference).
    """
    x, t, U = reference_field(nx=nx, nt=nt, nu=nu)  # (nt, nx)
    XX, TT = np.meshgrid(x, t)
    xt = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1), dtype=torch.float32)
    P = model(xt).cpu().numpy().reshape(U.shape)
    err = np.abs(P - U)

    fig = plt.figure(figsize=(14, 7))
    fields = [("reference", U, "rainbow"), ("PINN prediction", P, "rainbow"),
              ("|error|", err, "inferno")]
    for k, (title, F, cmap) in enumerate(fields):
        ax = fig.add_subplot(2, 3, k + 1)
        pc = ax.pcolormesh(t, x, F.T, cmap=cmap, shading="auto")
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
        fig.colorbar(pc, ax=ax)

    for tq in (0.25, 0.5, 0.75):
        ax = fig.add_subplot(2, 3, 3 + [0.25, 0.5, 0.75].index(tq) + 1)
        uref = cole_hopf_reference(x, np.full_like(x, tq), nu=nu)
        xt_s = torch.tensor(np.stack([x, np.full_like(x, tq)], axis=1), dtype=torch.float32)
        upred = model(xt_s).cpu().numpy().ravel()
        ax.plot(x, uref, "k-", lw=2, label="reference")
        ax.plot(x, upred, "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {tq}"); ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_loss_history(history_csv: str | Path, out_path: str | Path) -> None:
    """Plot the loss-component curves from a RunLogger history.csv."""
    import csv

    rows = list(csv.DictReader(open(history_csv)))
    if not rows:
        return
    it = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in ("total", "pde", "ic", "bc"):
        ax.semilogy(it, [float(r[key]) for r in rows], label=key)
    ax.set_xlabel("logged step"); ax.set_ylabel("loss (log)")
    ax.set_title("Training loss components"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
