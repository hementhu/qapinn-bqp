"""Training-time probes — the DYNAMIC half of the XAI suite.

A probe is a callable ``probe(model, spec, iteration) -> (value, labels)`` with
two attributes:
    name       : series name in the Monitor
    needs_grad : if True the trainer calls it right after backward() (grads live)

Dynamic quantities (only observable during training):
  - spectrum_probe     : spatial frequency content of the output over training
                         -> powers the F-principle (which frequencies learned when)
  - grad_variance_probe: per-layer gradient variance -> barren-plateau / trainability
"""
from __future__ import annotations

import numpy as np
import torch


def spectrum_probe(nx: int = 128, nt: int = 20):
    """Probe the model's SPATIAL spectrum, averaged over time.

    Builds a fixed (x, t) grid from the domain, evaluates the model, takes the
    magnitude of the spatial FFT (along x) and averages over t. Returns the
    per-wavenumber magnitude (length nx//2+1). Watching this over training shows
    the order in which spatial frequencies appear (the F-principle).
    """
    def probe(model, spec, iteration):
        (x0, x1), (t0, t1) = spec.domain[0], spec.domain[1]
        x = np.linspace(x0, x1, nx)
        t = np.linspace(t0, t1, nt)
        XX, TT = np.meshgrid(x, t)
        X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
        with torch.no_grad():
            u = model(X)[:, 0].cpu().numpy().reshape(nt, nx)   # first output component
        mag = np.abs(np.fft.rfft(u, axis=1)).mean(axis=0)      # (nx//2+1,)
        labels = [f"k{k}" for k in range(mag.shape[0])]
        return mag, labels

    probe.name = "spectrum"
    probe.needs_grad = False
    return probe


def grad_variance_probe():
    """Per-parameter-tensor gradient variance (called after backward()).

    Var[dL/dtheta] per named parameter. A collapsing variance with depth/qubits is
    the barren-plateau / trainability signature; a sudden drop can flag the
    black-hole collapse.
    """
    def probe(model, spec, iteration):
        names, vars_ = [], []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            names.append(name)
            # population variance (unbiased=False) -> 0, not NaN, for size-1 tensors
            vars_.append(g.var(unbiased=False).item())
            # per-qubit and per-layer breakdown of the VQC weight tensor
            # (shape n_layers, n_qubits, 2) -> which qubit/layer is plateauing
            if name.endswith("vqc.weights") and g.ndim == 3:
                for q in range(g.shape[1]):
                    names.append(f"{name}:qubit{q}")
                    vars_.append(g[:, q, :].var(unbiased=False).item())
                for layer in range(g.shape[0]):
                    names.append(f"{name}:layer{layer}")
                    vars_.append(g[layer].var(unbiased=False).item())
        return np.array(vars_), names

    probe.name = "grad_variance"
    probe.needs_grad = True
    return probe
