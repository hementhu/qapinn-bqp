"""Offloading analysis (the novel classical-side mechanism).

Hypothesis: the quantum front-end hands the classical post-net a ready-made
frequency basis (high-frequency features), so the post-net does not have to build
one --- freeing its neurons to specialise. A classical front-end (Linear+tanh) can
only supply smooth, low-frequency features, so its post-net must work harder.

We test this by measuring the FREQUENCY CONTENT OF THE FRONT-END OUTPUT --- the
features handed to the post-net --- for three front-ends on the same task:
    QAPINN  : VQC measurements
    CFE     : classical Linear+tanh front-end (the clean control)
    PINN    : first hidden layer of a plain PINN

If the VQC front-end carries markedly higher frequencies than the classical ones,
that is the offloading, made quantitative.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _grid(spec, nx, nt):
    (x0, x1), (t0, t1) = spec.domain[0], spec.domain[1]
    x = np.linspace(x0, x1, nx); t = np.linspace(t0, t1, nt)
    XX, TT = np.meshgrid(x, t)
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    return x, t, X


@torch.no_grad()
def frontend_field(model, spec, nx: int = 96, nt: int = 48):
    """Features the front-end hands to the post-net, over (x,t). Returns (N, width)."""
    _, _, X = _grid(spec, nx, nt)
    if hasattr(model, "vqc"):                      # QAPINN: VQC measurements
        f = model.vqc(X[:, model.coord_map])
    elif hasattr(model, "front"):                  # CFE: Linear+tanh features
        f = model.frontend_features(X)
    else:                                          # plain PINN: first hidden layer
        f = torch.tanh(model.linears[0](X))
    return f.cpu().numpy()


def feature_spectra(field: np.ndarray, nx: int, nt: int) -> np.ndarray:
    """Per-feature spatial spectrum (|rfft| along x, averaged over t). (width, n_k)."""
    specs = []
    for j in range(field.shape[1]):
        fj = field[:, j].reshape(nt, nx)
        specs.append(np.abs(np.fft.rfft(fj, axis=1)).mean(axis=0))
    return np.array(specs)


def frontend_bandwidth(model, spec, nx: int = 96, nt: int = 48, thresh: float = 0.05) -> int:
    """Highest harmonic the front-end supplies (max over its features, ignoring DC)."""
    f = frontend_field(model, spec, nx, nt)
    s = feature_spectra(f, nx, nt)
    s[:, 0] = 0.0
    total = s.max(axis=0)
    sig = np.where(total > thresh * total.max())[0] if total.max() > 0 else []
    return int(sig.max()) if len(sig) else 0


def frontend_energy_by_freq(model, spec, nx: int = 96, nt: int = 48) -> np.ndarray:
    """Total front-end energy per harmonic (summed over features, normalised)."""
    f = frontend_field(model, spec, nx, nt)
    s = feature_spectra(f, nx, nt)
    s[:, 0] = 0.0
    e = s.sum(axis=0)
    return e / (e.sum() + 1e-12)


def plot_frontend_spectra(models: dict, spec, out_path, nx: int = 96, nt: int = 48, kmax: int = 16):
    """Compare the front-end energy spectrum across model types."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, m in models.items():
        e = frontend_energy_by_freq(m, spec, nx, nt)[:kmax + 1]
        bw = frontend_bandwidth(m, spec, nx, nt)
        ax.plot(np.arange(len(e)), e, "o-", label=f"{name} (bandwidth {bw})")
    ax.set_xlabel("harmonic k"); ax.set_ylabel("front-end energy fraction")
    ax.set_title("Frequency content the front-end supplies to the post-net")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
