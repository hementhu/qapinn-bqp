"""VQC-internal XAI — dimension 3: open up the quantum layer itself.

The quantum-specific views the classical-only XAI cannot give, and which the
challenge explicitly asks for ("what is gained by ... entanglement, types of
output extraction"):

  - qubit_z_fields     : each qubit's <Z_i> as a function of (x,t) — the feature
                         each qubit computes (the quantum analogue of a neuron's
                         receptive field).
  - entanglement_field : von Neumann entropy of a qubit bipartition across the
                         (x,t) domain — how entangled the encoded state is, and
                         where. The direct handle on "what entanglement buys".
  - qubit_dominant_freq: the spatial frequency each qubit specializes in.

Operates on a QAPINN (uses its .vqc and fixed .coord_map to feed raw (x,t)).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _grid(spec, nx: int, nt: int):
    (x0, x1), (t0, t1) = spec.domain[0], spec.domain[1]
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    XX, TT = np.meshgrid(x, t)
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    return x, t, X


def qubit_z_fields(qapinn, spec, nx: int = 96, nt: int = 48):
    """Each qubit's <Z> over (x,t). Returns x, t, fields (nt*nx, n_qubits)."""
    x, t, X = _grid(spec, nx, nt)
    z = qapinn.vqc.per_qubit_z(X[:, qapinn.coord_map])       # (N, n_qubits)
    return x, t, z.cpu().numpy()


def entanglement_field(qapinn, spec, nx: int = 32, nt: int = 16):
    """Entanglement entropy over (x,t). Coarse grid (entropy is looped per point).

    Returns x, t, ent (nt, nx).
    """
    x, t, X = _grid(spec, nx, nt)
    ent = qapinn.vqc.entanglement_entropy(X[:, qapinn.coord_map])   # (N,)
    return x, t, ent.cpu().numpy().reshape(nt, nx)


def qubit_dominant_freq(fields: np.ndarray, nx: int, nt: int) -> np.ndarray:
    """Dominant spatial harmonic per qubit (argmax of spatial FFT, ignoring DC)."""
    dom = np.zeros(fields.shape[1], dtype=int)
    for j in range(fields.shape[1]):
        f = fields[:, j].reshape(nt, nx)
        mag = np.abs(np.fft.rfft(f, axis=1)).mean(axis=0)
        mag[0] = 0.0
        dom[j] = int(np.argmax(mag))
    return dom


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_qubit_fields(x, t, fields: np.ndarray, out_path, nx: int, nt: int):
    """Grid of each qubit's <Z>(x,t) — 'what each qubit sees'."""
    nq = fields.shape[1]
    cols = min(nq, 4)
    rows = int(np.ceil(nq / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), squeeze=False)
    for j in range(rows * cols):
        ax = axes[j // cols][j % cols]
        if j < nq:
            f = fields[:, j].reshape(nt, nx)
            pc = ax.pcolormesh(t, x, f.T, cmap="PuOr", shading="auto", vmin=-1, vmax=1)
            ax.set_title(f"qubit {j}  <Z>", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-qubit expectation <Z_i>(x,t)")
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_entanglement_field(x, t, ent: np.ndarray, out_path):
    """Heatmap of entanglement entropy across (x,t)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    pc = ax.pcolormesh(t, x, ent.T, cmap="viridis", shading="auto")
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.set_title("Entanglement entropy S(x,t)")
    fig.colorbar(pc, ax=ax, label="von Neumann entropy")
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
