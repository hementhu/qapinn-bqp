"""Why does entanglement HURT at convergence? Compare the converged
cv_qd3_none (no entanglement) vs cv_qd3_lin (linear entanglement) QAPINNs.

Hypothesis: entanglement correlates the per-qubit Fourier features that feed the
classical post-net, lowering the rank of the basis the readout can use -> worse fit.

For each config (3 seeds) we measure, on a dense (x,t) grid, the VQC feature matrix
z (n_points x 4):
  - mean |pairwise correlation| between the 4 features   (higher = more redundant)
  - effective rank of z                                  (lower = poorer basis)
  - effective rank of the post-net hidden activations
  - per-feature max Fourier harmonic (x-sweep, t=0.5)
  - mean entanglement entropy (sanity: none ~ 0)
Writes report/mechanism.md + a bar figure.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from src.models.factory import build_model
from src.pdes.base import get_pde
from src.analysis.vqc_internals import entanglement_field

R = Path("results")
SEEDS = (0, 1, 2)
CONFIGS = {"none": "cv_qd3_none", "linear": "cv_qd3_lin"}


def eff_rank(A: np.ndarray) -> float:
    """Entropy-based effective rank of columns of A (n_points x d)."""
    A = A - A.mean(0, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_abs_corr(A: np.ndarray) -> float:
    """Mean |correlation| over distinct feature pairs."""
    C = np.corrcoef(A.T)
    d = C.shape[0]
    iu = np.triu_indices(d, k=1)
    return float(np.nanmean(np.abs(C[iu])))


def max_harm(vec: np.ndarray, thr=0.02) -> int:
    m = np.abs(np.fft.rfft(vec - vec.mean()))
    nz = np.where(m > thr * m.max())[0]
    return int(nz.max()) if nz.size else 0


def analyze(run: str):
    raw = json.loads((R / run / "config.json").read_text())["raw"]
    m = build_model(raw)
    m.load_state_dict(torch.load(R / run / "model.pt"))
    m.eval()
    # dense grid
    xs = np.linspace(-1, 1, 60); ts = np.linspace(0, 1, 40)
    XX, TT = np.meshgrid(xs, ts)
    X = torch.tensor(np.stack([XX.ravel(), TT.ravel()], 1), dtype=torch.float32)
    # post-net hidden activation hook
    acts = {}
    h = m.post.linears[0].register_forward_hook(lambda mod, i, o: acts.__setitem__("h", o.detach()))
    with torch.no_grad():
        z = m.vqc(X[:, m.coord_map]).numpy()          # (n, 4) VQC features
        _ = m(X)
    post_h = torch.tanh(acts["h"]).numpy()            # (n, 16) post-net hidden
    h.remove()
    # per-feature spectrum on an x-sweep at t=0.5
    xr = torch.linspace(-1, 1, 256).reshape(-1, 1)
    Xr = torch.cat([xr, 0.5 * torch.ones_like(xr)], 1)
    with torch.no_grad():
        zr = m.vqc(Xr[:, m.coord_map]).numpy()
    harms = [max_harm(zr[:, i]) for i in range(zr.shape[1])]
    # entanglement entropy
    try:
        _, _, e = entanglement_field(m, get_pde("burgers"), nx=24, nt=12)
        ent = float(np.nanmean(e))
    except Exception:
        ent = float("nan")
    return dict(feat_corr=mean_abs_corr(z), feat_rank=eff_rank(z),
                post_rank=eff_rank(post_h), harms=harms, ent=ent,
                rel=json.load(open(R / run / "summary.json"))["rel_l2"])


def main():
    res = {}
    for tag, base in CONFIGS.items():
        rows = [analyze(f"{base}_s{s}") for s in SEEDS if (R / f"{base}_s{s}" / "model.pt").exists()]
        agg = lambda k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows]))
        res[tag] = dict(rel=agg("rel"), feat_corr=agg("feat_corr"),
                        feat_rank=agg("feat_rank"), post_rank=agg("post_rank"),
                        ent=agg("ent"),
                        harms=[r["harms"] for r in rows], n=len(rows))

    lines = ["# Mechanism: why entanglement hurts (converged, n=3)", "",
             "Comparing cv_qd3_none (no entanglement) vs cv_qd3_lin (linear), 3 seeds each.",
             "",
             "| metric | none | linear | direction |",
             "|---|---|---|---|"]
    def fmt(t, k): m, s = res[t][k]; return f"{m:.3f}±{s:.3f}"
    def line(name, k, worse_if_higher, better_tag_expected):
        n = res["none"][k][0]; l = res["linear"][k][0]
        arrow = "linear higher" if l > n else "none higher"
        lines.append(f"| {name} | {fmt('none',k)} | {fmt('linear',k)} | {arrow} |")
    line("rel-L2 (worse=higher)", "rel", True, "none")
    line("mean |feature corr| (redundancy)", "feat_corr", True, "none")
    line("VQC feature eff-rank (basis richness)", "feat_rank", False, "none")
    line("post-net hidden eff-rank", "post_rank", False, "none")
    line("mean entanglement entropy", "ent", None, None)
    lines += ["",
              f"none feature max-harmonics per seed: {res['none']['harms']}",
              f"linear feature max-harmonics per seed: {res['linear']['harms']}",
              "",
              "## Reading",
              f"- Entanglement entropy: none={fmt('none','ent')} (~0 sanity check), "
              f"linear={fmt('linear','ent')} (genuinely entangled).",
              "- If linear shows HIGHER feature correlation and LOWER feature/post eff-rank "
              "than none, the hypothesis holds: entanglement redundifies the per-qubit "
              "features, giving the post-net a poorer basis -> worse fit.", ""]
    Path("report/mechanism.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (k, ttl) in zip(ax, [("feat_corr", "mean |feature correlation|\n(higher = redundant)"),
                                 ("feat_rank", "VQC feature effective rank\n(higher = richer basis)"),
                                 ("post_rank", "post-net hidden effective rank")]):
        vals = [res["none"][k][0], res["linear"][k][0]]
        errs = [res["none"][k][1], res["linear"][k][1]]
        a.bar(["none", "linear"], vals, yerr=errs, capsize=5,
              color=["tab:green", "tab:red"])
        a.set_title(ttl); a.set_xlabel("entanglement")
    fig.suptitle("Why entanglement hurts: feature-basis analysis (converged, n=3)")
    fig.tight_layout()
    out = Path("results/ablation_summary/mechanism.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
