"""Seed-expansion runner (for parallelising across machines).

Runs the headline configs (entanglement sweep + matched classical) at a given set
of seeds, so different machines can cover different seeds with no overlap.

    python run_seeds.py 3 4 5      # laptop 2 covers seeds 3,4,5
    python run_seeds.py preflight 3 4 5

Laptop 1 (run_stability.py) already covers seeds 1-2; combine by copying the
results/seed*/ folders together and re-running the seed aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path

from run_ablation import qapinn_cfg, classical_cfg
from src.models.factory import build_model, describe

ENTANGLEMENTS = ("none", "linear", "ring", "all_to_all")


def seed_configs(seed: int):
    q = [qapinn_cfg(f"seed{seed}_ent_{e}", vqc_over={"entanglement": e}) for e in ENTANGLEMENTS]
    for c in q:
        c["seed"] = seed
    cls = classical_cfg(121); cls["name"] = f"seed{seed}_cls_121"; cls["seed"] = seed
    return q, [cls]


def preflight(seeds):
    print(f"=== SEED-EXPANSION PRE-FLIGHT (seeds {seeds}) ===")
    total = 0
    for s in seeds:
        q, c = seed_configs(s)
        for cfg in q + c:
            total += 1
        print(f"  seed {s}: {len(q)} QAPINN + {len(c)} classical  ({[cfg['name'] for cfg in q]})")
    print(f"Total: {total} runs ({sum(1 for s in seeds)*len(ENTANGLEMENTS)} QAPINN ~35min each)")
    # sanity: all 121 params
    q, c = seed_configs(seeds[0])
    print("param check:", {cfg["name"].split('_')[-1]: describe(build_model(cfg))["num_params"] for cfg in q})


def run(seeds):
    import run as R

    def done(n):
        return (Path("results") / n / "summary.json").exists()

    for s in seeds:
        q, c = seed_configs(s)
        for cfg in c + q:
            if done(cfg["name"]):
                print(f"skip {cfg['name']} (done)"); continue
            R.run_config(dict(cfg))
    print("=== SEED EXPANSION COMPLETE ===")


def aggregate():
    """Mean +/- std across ALL available seeds for the entanglement sweep.

    Pools seed 0 (from the ablation: abl_ent_* / abl_baseline) with seeds 1..N
    (seed{N}_ent_*). Reports whether the entanglement differences survive the
    seed-to-seed spread, and plots error bars.
    """
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rl(name):
        p = Path("results") / name / "summary.json"
        return json.load(open(p))["rel_l2"] if p.exists() else None

    # seed 0 uses the ablation names; seeds >=1 use seed{n}_ent_{e}
    def names(ent):
        base = "abl_baseline" if ent == "linear" else f"abl_ent_{ent}"
        return [base] + [f"seed{s}_ent_{ent}" for s in range(1, 9)]

    print("=== ENTANGLEMENT sweep across seeds (mean +/- std) ===")
    stats = {}
    for e in ENTANGLEMENTS:
        vals = [v for v in (rl(n) for n in names(e)) if v is not None]
        stats[e] = (np.mean(vals), np.std(vals), len(vals))
        print(f"  {e:12} n={len(vals)}  rel_L2 = {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    means = np.array([stats[e][0] for e in ENTANGLEMENTS])
    stds = np.array([stats[e][1] for e in ENTANGLEMENTS])
    spread = means.max() - means.min()
    typ_std = float(np.mean(stds))
    verdict = ("WITHIN noise -> entanglement effect NOT significant"
               if spread < 2 * typ_std else
               "EXCEEDS noise -> entanglement effect may be real")
    print(f"\n  between-entanglement spread = {spread:.3f}; typical seed std = {typ_std:.3f}")
    print(f"  VERDICT: {verdict}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(list(ENTANGLEMENTS), means, yerr=stds, capsize=5, color="tab:blue", alpha=0.8)
    ax.set_ylabel("rel L2 (mean +/- std over seeds)")
    ax.set_title(f"Entanglement vs accuracy across seeds\n{verdict}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    out = Path("results/ablation_summary"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "entanglement_seeds.png", dpi=110); plt.close(fig)
    print(f"-> {out}/entanglement_seeds.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "preflight":
        preflight([int(x) for x in args[1:]])
    elif args and args[0] == "aggregate":
        aggregate()
    else:
        run([int(x) for x in args])
