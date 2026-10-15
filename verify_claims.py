"""Back the paper's spectral claims with measurements (complements verify_pipeline.py).

1. Band = depth for an isolated qubit, L = 1..6.
2. Exact x-frequency content of the trained VQC features, and the realised output
   bandwidth of every converged model (cv_*, big_*).
3. F-principle: within each converged run, do low harmonics reach half their
   initial relative error before high ones? (Spearman rho of k vs halving iter.)

"Significant" = magnitude >= THRESH x the largest harmonic, DC excluded.
Writes report/claims.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from scipy import stats

from src.models.factory import build_model
from src.models.vqc import QuantumLayer, VQCConfig
from src.pdes.base import get_pde
from src.analysis.spectrum import output_field, spatial_spectrum, fprinciple_convergence

R = Path("results")
THRESH = 1e-2


def max_harmonic(mag, thresh=THRESH):
    mag = np.asarray(mag)[1:]                      # drop DC
    idx = np.where(mag >= thresh * mag.max())[0]
    return int(idx.max() + 1) if idx.size else 0


def bandwidth(mag, frac=0.99):
    """Smallest K whose harmonics 1..K hold `frac` of the non-DC spectral energy."""
    e = np.asarray(mag)[1:] ** 2
    return int(np.searchsorted(np.cumsum(e) / e.sum(), frac) + 1)


def isolated_band(L):
    torch.manual_seed(1)
    q = QuantumLayer(VQCConfig(n_qubits=1, n_layers=L, reupload=True,
                               entanglement="none", measurement="single", input_scale=1.0))
    N = 256
    s = (torch.arange(N) / N * 2 * np.pi).reshape(-1, 1)
    z = q.per_qubit_z(s)[:, 0].detach().numpy()
    mag = np.abs(np.fft.rfft(z))
    return int(np.where(mag > 1e-6 * mag.max())[0].max())


def feature_x_frequencies(model, t=0.5, N=256):
    """Exact max x-frequency of each VQC readout: features are 2pi-periodic in x
    (integer frequencies), so sampling x over [0, 2pi) gives a leakage-free FFT."""
    x = torch.arange(N) / N * 2 * np.pi
    X = torch.stack([x, torch.full_like(x, t)], 1)
    with torch.no_grad():
        Z = model.vqc(X[:, model.coord_map]).numpy()
    ks = []
    for i in range(Z.shape[1]):
        mag = np.abs(np.fft.rfft(Z[:, i]))
        ks.append(int(np.where(mag > 1e-6 * mag.max())[0].max()))
    return ks


def load(run):
    raw = json.load(open(R / run / "config.json"))["raw"]
    m = build_model(raw)
    m.load_state_dict(torch.load(R / run / "model.pt"))
    m.eval()
    return m, raw


def feature_field(model, spec, nx=256, nt=100):
    """VQC per-qubit <Z> on the evaluation grid -> (n_qubits, nt, nx)."""
    (x0, x1), (t0, t1) = spec.domain
    XX, TT = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(t0, t1, nt))
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    if not hasattr(model, "vqc"):
        return None
    with torch.no_grad():
        Z = model.vqc(X[:, model.coord_map])
    return Z.numpy().T.reshape(-1, nt, nx)


def main():
    out = {"threshold": THRESH}
    out["isolated_band"] = {L: isolated_band(L) for L in range(1, 7)}
    print("isolated qubit max harmonic, L=1..6:", out["isolated_band"])

    spec = get_pde("burgers")
    ref = spatial_spectrum(spec.reference(nx=256, nt=100)[2])
    out["reference"] = dict(max_k=max_harmonic(ref), bw99=bandwidth(ref))
    print("Burgers reference:", out["reference"])

    out["feature_x_frequencies"] = {}
    for run in ["cv_qd3_none_s0", "cv_qd3_lin_s0", "big_qd3_none_s0", "big_qd3_lin_s0"]:
        m, _ = load(run)
        out["feature_x_frequencies"][run] = feature_x_frequencies(m)
        print(f"{run:18s} max x-frequency per VQC readout: {out['feature_x_frequencies'][run]}")

    rows = []
    runs = [p.name for pat in ("cv_*", "big_*") for p in sorted(R.glob(pat)) if (p / "model.pt").exists()]
    for run in runs:
        m, raw = load(run)
        with torch.no_grad():
            _, _, U = output_field(m, spec)
        mag = spatial_spectrum(U)
        model_k, model_bw = max_harmonic(mag), bandwidth(mag)
        rho = None
        if (R / run / "monitor.npz").exists():
            fp = fprinciple_convergence(R / run / "monitor.npz", spec)
            ok = ~np.isnan(fp["conv_iter"])
            if ok.sum() > 2:
                rho = float(stats.spearmanr(fp["k"][ok], fp["conv_iter"][ok]).statistic)
        rel = json.load(open(R / run / "summary.json"))["rel_l2"]
        rows.append(dict(run=run, rel_l2=rel, model_max_k=model_k, model_bw99=model_bw,
                         fprinciple_rho=rho))
        print(f"{run:18s} rel={rel:.3f}  model k<={model_k:3d} bw99={model_bw:3d}  F-principle rho={rho}")
    out["converged_models"] = rows
    Path("report/claims.json").write_text(json.dumps(out, indent=2))
    print("wrote report/claims.json")


if __name__ == "__main__":
    main()
