"""Spectral XAI — dimension 1: what frequencies the model builds, and *when*.

Two views:
  - STATIC : the realized Fourier spectrum of the trained field vs the reference
    (which harmonics the model captured, and the per-frequency error). On the
    domain x in [-1,1] the rfft index k is exactly the k-th harmonic, so k lines
    up with the VQC's accessible-frequency language {-L..L}.
  - DYNAMIC (F-principle) : from the training-time spectrum series (Monitor),
    when each frequency converges -> the order in which frequencies are learned.

These power the "which frequencies, learned when, classical vs QAPINN" figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #
@torch.no_grad()
def output_field(model, spec, nx: int = 256, nt: int = 100):
    """Model prediction on a uniform grid. Returns x, t, U (nt, nx)."""
    (x0, x1), (t0, t1) = spec.domain[0], spec.domain[1]
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    XX, TT = np.meshgrid(x, t)
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    U = model(X)[:, 0].cpu().numpy().reshape(nt, nx)
    return x, t, U


# --------------------------------------------------------------------------- #
# Static spectra
# --------------------------------------------------------------------------- #
def spatial_spectrum(U: np.ndarray) -> np.ndarray:
    """Per-harmonic magnitude: |rfft along x|, averaged over t. -> (nx//2+1,).

    Index k = k-th spatial harmonic over the domain (matches the VQC frequency k).
    """
    return np.abs(np.fft.rfft(U, axis=1)).mean(axis=0)


def spacetime_spectrum(U: np.ndarray) -> np.ndarray:
    """Full 2-D magnitude spectrum |rfft2(U)| over (t, x). -> (nt, nx//2+1)."""
    return np.abs(np.fft.rfft2(U))


def realized_spectrum(model, spec, nx: int = 256, nt: int = 100):
    """Trained-model spatial spectrum + reference spatial spectrum.

    Returns k (harmonics), mag_model, mag_reference.
    """
    _, _, Up = output_field(model, spec, nx, nt)
    xg, tg, Ur = spec.reference(nx=nx, nt=nt)
    k = np.arange(Up.shape[1] // 2 + 1)
    return k, spatial_spectrum(Up), spatial_spectrum(Ur)


def spectral_error(model, spec, nx: int = 256, nt: int = 100):
    """Per-harmonic magnitude error |model - reference|. Returns k, err."""
    k, mm, mr = realized_spectrum(model, spec, nx, nt)
    return k, np.abs(mm - mr)


# --------------------------------------------------------------------------- #
# Dynamic: the F-principle (frequency learning order) from a Monitor .npz
# --------------------------------------------------------------------------- #
def fprinciple_convergence(monitor_path, spec, halve_frac: float = 0.5, floor_ratio: float = 0.05):
    """F-principle via per-frequency relative-error reduction (the standard view).

    For each harmonic k active in the reference, track the relative error
    e_k(t) = |model_k(t) - ref_k| / ref_k over training, and report the first
    iteration at which it drops to ``halve_frac`` of its initial value. Low
    frequencies reach this before high ones — that ORDER is the F-principle, and
    the halving-time is robust to the model not being fully converged (unlike an
    absolute-magnitude threshold). Inactive reference harmonics -> NaN.

    Returns dict: k, conv_iter (NaN if inactive/never), ref_mag, relerr (T,n_k),
    iters, values.
    """
    d = np.load(monitor_path, allow_pickle=True)
    iters = d["spectrum_iters"]                       # (T,)
    vals = d["spectrum_values"]                       # (T, n_k)
    n_k = vals.shape[1]
    nx = 2 * (n_k - 1)

    _, _, Ur = spec.reference(nx=nx, nt=50)
    ref = spatial_spectrum(Ur)                        # (n_k,)
    active = ref >= floor_ratio * ref.max()

    relerr = np.abs(vals - ref[None, :]) / (ref[None, :] + 1e-12)   # (T, n_k)
    conv = np.full(n_k, np.nan)
    for j in range(n_k):
        if not active[j]:
            continue
        target = halve_frac * relerr[0, j]
        reached = np.where(relerr[:, j] <= target)[0]
        if reached.size:
            conv[j] = iters[reached[0]]
    return {"k": np.arange(n_k), "conv_iter": conv, "ref_mag": ref,
            "relerr": relerr, "iters": iters, "values": vals}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_spectrum(model, spec, out_path, nx: int = 256, nt: int = 100, kmax: int = 20):
    """Bar plot: model vs reference spatial spectrum (first kmax harmonics)."""
    k, mm, mr = realized_spectrum(model, spec, nx, nt)
    k, mm, mr = k[:kmax + 1], mm[:kmax + 1], mr[:kmax + 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.4
    ax.bar(k - w / 2, mr, w, label="reference", color="k", alpha=0.6)
    ax.bar(k + w / 2, mm, w, label="model", color="tab:red", alpha=0.8)
    ax.set_xlabel("spatial harmonic k"); ax.set_ylabel("|coefficient|")
    ax.set_title("Realized Fourier spectrum"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_fprinciple(monitor_path, spec, out_path, kmax: int = 16):
    """Heatmap of per-frequency magnitude over training (each row normalized).

    A convergence 'wave' sweeping from low k (bottom) upward = low frequencies
    learned first (the F-principle). Compare this between classical and QAPINN.
    """
    r = fprinciple_convergence(monitor_path, spec)
    iters = r["iters"]
    relerr = r["relerr"][:, :kmax + 1]                 # per-frequency relative error
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    im = a1.pcolormesh(iters, np.arange(relerr.shape[1]), relerr.T,
                       cmap="magma_r", shading="auto", vmin=0, vmax=1)
    a1.set_xlabel("training iteration"); a1.set_ylabel("harmonic k")
    a1.set_title("Per-frequency relative error (dark = converged)")
    fig.colorbar(im, ax=a1)
    ci = r["conv_iter"][:kmax + 1]
    a2.plot(np.arange(len(ci)), ci, "o-")
    a2.set_xlabel("harmonic k"); a2.set_ylabel("iter to halve rel-error")
    a2.set_title("Frequency learning order (F-principle)"); a2.grid(alpha=0.3)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
