"""Neuron-level XAI — dimension 2: what each classical neuron encodes, and how
that differs with vs without the quantum layer (the challenge's Fig 3, deepened).

For every hidden neuron in the classical part of the network we extract its
activation over the whole (x,t) domain (its "receptive field"), then:
  - neuron_spectra : FFT each field -> the dominant spatial frequency each neuron
    specializes in. Ties the neuron view to the Fourier story: does the quantum
    layer make downstream neurons carry higher/different frequencies?
  - effective_rank : SVD of the activation matrix -> how many DISTINCT features
    the layer really uses (feeds the parameter-reduction explanation).

Works uniformly on the classical PINN (its hidden layers) and the QAPINN (its
classical post-net), so the two are directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _hidden_linears(model) -> list:
    """The hidden Linear layers whose (post-tanh) outputs are the 'neurons'.

    QAPINN -> its classical post-net; classical PINN -> itself. Excludes the
    final (output) Linear, which has no activation.
    """
    mlp = model.post if hasattr(model, "post") else model
    return list(mlp.linears[:-1])


def _grid(spec, nx: int, nt: int):
    (x0, x1), (t0, t1) = spec.domain[0], spec.domain[1]
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    XX, TT = np.meshgrid(x, t)
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    return x, t, X


@torch.no_grad()
def activation_fields(model, spec, nx: int = 96, nt: int = 48) -> tuple:
    """Per-neuron activations over the (x,t) grid, per hidden layer.

    Returns (x, t, fields) where fields[layer_i] has shape (n_points, n_neurons),
    n_points = nt*nx (row-major, reshape to (nt,nx) for a field).
    """
    x, t, X = _grid(spec, nx, nt)
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, lin in enumerate(_hidden_linears(model)):
        def hook(_m, _inp, out, i=i):
            captured[i] = torch.tanh(out).detach()      # post-activation neuron output
        hooks.append(lin.register_forward_hook(hook))
    model(X)
    for h in hooks:
        h.remove()
    fields = {f"layer{i + 1}": captured[i].cpu().numpy() for i in sorted(captured)}
    return x, t, fields


def neuron_spectra(field: np.ndarray, nx: int, nt: int) -> np.ndarray:
    """Dominant spatial harmonic per neuron.

    field: (nt*nx, n_neurons). For each neuron, reshape to (nt,nx), take the
    spatial FFT magnitude averaged over t, and return argmax harmonic k.
    """
    n_neurons = field.shape[1]
    dom = np.zeros(n_neurons, dtype=int)
    for j in range(n_neurons):
        f = field[:, j].reshape(nt, nx)
        mag = np.abs(np.fft.rfft(f, axis=1)).mean(axis=0)
        mag[0] = 0.0                                     # ignore the DC/mean component
        dom[j] = int(np.argmax(mag))
    return dom


def effective_rank(field: np.ndarray) -> float:
    """Entropy-based effective rank of the (centered) activation matrix.

    eff_rank = exp(-sum p_i log p_i), p_i = sigma_i / sum(sigma). A layer whose
    neurons are near-duplicates has low effective rank; diverse features -> high.
    """
    A = field - field.mean(axis=0, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_receptive_fields(x, t, field: np.ndarray, out_path, nx: int, nt: int, max_neurons: int = 12):
    """Grid of neuron activation fields over (x,t) — the literal 'what each neuron sees'."""
    n = min(field.shape[1], max_neurons)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), squeeze=False)
    for j in range(rows * cols):
        ax = axes[j // cols][j % cols]
        if j < n:
            f = field[:, j].reshape(nt, nx)
            ax.pcolormesh(t, x, f.T, cmap="RdBu_r", shading="auto", vmin=-1, vmax=1)
            ax.set_title(f"neuron {j}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Neuron receptive fields (activation over x,t)")
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_neuron_frequency_comparison(dom_by_model: dict, out_path, kmax: int = 12):
    """Histogram of neurons' dominant frequencies, one series per model.

    Shifts right (toward higher k) for a model whose neurons carry higher
    frequencies — the key classical-vs-QAPINN neuron comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, kmax + 1.5, 1)
    for name, dom in dom_by_model.items():
        ax.hist(dom, bins=bins, alpha=0.55, label=f"{name} (eff.freq {np.mean(dom):.1f})")
    ax.set_xlabel("neuron dominant harmonic k"); ax.set_ylabel("# neurons")
    ax.set_title("What frequency each neuron specializes in"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
