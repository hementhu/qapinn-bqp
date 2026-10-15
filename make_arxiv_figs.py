"""Build the figures for the arXiv version into report/arxiv/figs/.

Reads results/*/summary.json and report/claims.json (run verify_claims.py first).
Also copies the reused figures from results/ so the arXiv source is self-contained.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

R = Path("results")
OUT = Path("report/arxiv/figs")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
Q, C, F = "#c0392b", "#2c3e50", "#7f8c8d"


def rel(run):
    return json.load(open(R / run / "summary.json"))["rel_l2"]


def seeds(base):
    return [rel(f"{base}_s{i}") for i in range(9) if (R / f"{base}_s{i}" / "summary.json").exists()]


def strip(ax, groups, title):
    rng = np.random.default_rng(0)
    for i, (label, base, color) in enumerate(groups):
        v = np.array(seeds(base))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=color, alpha=0.8, zorder=3)
        ax.errorbar(i + 0.28, v.mean(), yerr=v.std(ddof=1), fmt="_", color=color, ms=10, capsize=3)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=30, ha="right")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_title(title)
    ax.set_ylim(bottom=0)


# Figure: per-seed results at both scales
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
strip(axes[0], [("classical", "cv_cls", C), ("CFE", "cv_cfe", C), ("RFF-16", "cv_rff16", F),
                ("RFF-4", "cv_rff4", F), ("QAPINN d5", "cv_qd5_lin", Q),
                ("QAPINN d3 none", "cv_qd3_none", Q), ("QAPINN d3 lin", "cv_qd3_lin", Q)],
      "small (~121 params)")
strip(axes[1], [("QAPINN none", "big_qd3_none", Q), ("QAPINN lin", "big_qd3_lin", Q),
                ("depth-matched", "big_cls_deep", C), ("CFE", "big_cfe", C),
                ("RFF-16", "big_rff16", F), ("RFF-4", "big_rff4", F),
                ("balanced", "big_cls_bal", C), ("shallow", "big_cls", C)],
      "larger (~1100-1273 params)")
fig.tight_layout()
fig.savefig(OUT / "seeds.pdf")

# Figure: realised output bandwidth vs error
claims = json.load(open("report/claims.json"))
rows = claims["converged_models"]
ref_bw = claims["reference"]["bw99"]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
families = [("small QAPINN", lambda r: r.startswith("cv_qd"), Q, "o"),
            ("small classical / CFE", lambda r: r.startswith(("cv_cls", "cv_cfe")), C, "o"),
            ("fixed Fourier features", lambda r: "_rff" in r, F, "s"),
            ("larger QAPINN", lambda r: r.startswith("big_qd"), Q, "^"),
            ("larger classical / CFE", lambda r: r.startswith("big_c"), C, "^")]
# y-limit: fixed-feature width-4 runs reach ~0.9
for label, pred, color, marker in families:
    sel = [r for r in rows if pred(r["run"])]
    ax.scatter([r["model_bw99"] for r in sel], [r["rel_l2"] for r in sel], s=16, color=color,
               marker=marker, alpha=0.75, label=label,
               facecolors="none" if marker == "o" and color == C else color)
ax.axvline(ref_bw, color="grey", ls="--", lw=0.8)
ax.text(ref_bw - 0.3, 0.97, "reference", rotation=90, ha="right", va="top", color="grey", fontsize=7)
ax.set_xlabel(r"output bandwidth $K_{99}$ (harmonics)")
ax.set_ylabel(r"relative $L_2$ error")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.savefig(OUT / "bandwidth.pdf")

small_q = [r for r in rows if r["run"].startswith("cv_qd")]
print(f"n models in bandwidth plot: {len(rows)}")
rho = stats.spearmanr([r["model_bw99"] for r in small_q], [r["rel_l2"] for r in small_q])
allr = stats.spearmanr([r["model_bw99"] for r in rows], [r["rel_l2"] for r in rows])
print(f"small QAPINN bw vs error: Spearman rho={rho.statistic:.2f} p={rho.pvalue:.2g} (n={len(small_q)})")
print(f"all models bw vs error:   Spearman rho={allr.statistic:.2f} p={allr.pvalue:.2g} (n={len(rows)})")

# Reused figures (self-contained source for arXiv)
for src in ["reference/burgers_reference.png",
            "ablation_summary/tier3_sweeps.png"]:
    shutil.copy(R / src, OUT / Path(src).name)
print("wrote", sorted(p.name for p in OUT.iterdir()))
