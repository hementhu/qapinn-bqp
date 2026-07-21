"""Converged re-run (the audit keystone).

The Tier-1/Tier-3 ablations trained with L-BFGS DISABLED (lbfgs_iters=0), only
1500 collocation points and 2000 Adam steps, so every Burgers model sat at
rel-L2 ~0.47 -- the shock was unresolved by everyone and the ablation differences
were within seed noise. Nothing could be concluded from failed models.

This re-runs a SMALL HONEST CORE with L-BFGS ON, to convergence, across 3 seeds,
so any surviving difference is between *resolved* models with error bars. It also
directly tests the one live lead (does entanglement hurt at convergence?) with
seeds instead of n=1.

Core (5 configs x 3 seeds = 15 runs), names prefixed `cv_`:
    cv_cls        classical PINN, 121 params (matched control)
    cv_cfe        classical front-end control (fair structural control)
    cv_qd3_lin    QAPINN depth-3, entanglement=linear (the reference QAPINN)
    cv_qd3_none   QAPINN depth-3, entanglement=none   (entanglement lead)
    cv_qd5_lin    QAPINN depth-5, entanglement=linear (more band)

Two-laptop split (no overlap):
    laptop1  ->  cv_cls, cv_cfe            (classical/CFE, faster)   6 runs
    laptop2  ->  cv_qd3_lin, cv_qd3_none, cv_qd5_lin  (quantum, slow) 9 runs

Usage:
    python run_converged.py preflight
    python run_converged.py laptop1     # this machine's slice
    python run_converged.py laptop2     # other machine's slice
    python run_converged.py all         # everything on one machine
    python run_converged.py aggregate   # mean +/- std tables once runs exist
"""
from __future__ import annotations

import sys
from pathlib import Path

from run_ablation import qapinn_cfg, classical_cfg

SEEDS = (0, 1, 2)

# Converged training: L-BFGS ON, more collocation. These override BASE_TRAIN.
CONV = dict(n_interior=4000, adam_iters=2500, lbfgs_iters=400, checkpoint_every=0)


def _conv(cfg: dict, seed: int) -> dict:
    cfg = dict(cfg)
    cfg.update(CONV)
    cfg["seed"] = seed
    cfg["name"] = f"{cfg['name']}_s{seed}"
    return cfg


def _cfe(seed: int) -> dict:
    # classical front-end: Linear(2->4)+tanh -> same post-net [16] (109 params,
    # the structural fair control; matches the earlier cfe_burgers width).
    cfg = dict(name="cv_cfe", model="cfe", width=4, post_hidden=[16], pde="burgers",
               n_constraint=150, adam_lr=0.01, seed=seed, log_every=100, probe_every=0)
    cfg.update(CONV)
    cfg["name"] = f"cv_cfe_s{seed}"
    return cfg


def configs(slice_name: str) -> list[dict]:
    cls = lambda s: _conv({**classical_cfg(121), "name": "cv_cls"}, s)
    qd3_lin = lambda s: _conv({**qapinn_cfg("cv_qd3_lin"), }, s)
    qd3_none = lambda s: _conv({**qapinn_cfg("cv_qd3_none", vqc_over={"entanglement": "none"})}, s)
    qd5_lin = lambda s: _conv({**qapinn_cfg("cv_qd5_lin", vqc_over={"n_layers": 5})}, s)

    laptop1 = [f(s) for s in SEEDS for f in (cls,)] + [_cfe(s) for s in SEEDS]
    laptop2 = [f(s) for s in SEEDS for f in (qd3_lin, qd3_none, qd5_lin)]
    if slice_name == "laptop1":
        return laptop1
    if slice_name == "laptop2":
        return laptop2
    return laptop1 + laptop2                       # "all"


def preflight():
    from src.models.factory import build_model, describe
    print("=== CONVERGED RE-RUN PRE-FLIGHT ===")
    print(f"training override: {CONV}   seeds={SEEDS}\n")
    for lab in ("laptop1", "laptop2"):
        cfgs = configs(lab)
        print(f"[{lab}] {len(cfgs)} runs:")
        seen = set()
        for c in cfgs:
            base = c["name"].rsplit("_s", 1)[0]
            if base in seen:
                continue
            seen.add(base)
            p = describe(build_model(c))["num_params"]
            print(f"    {base:14} params={p}  model={c.get('model')}")
        print()
    print("param note: QAPINN depth-3=121, depth-5=137, classical=121, cfe=109 (structural control)")


def run(slice_name: str):
    import run as R

    def done(n):
        return (Path("results") / n / "summary.json").exists()

    cfgs = configs(slice_name)
    print(f"=== CONVERGED RE-RUN: {slice_name} ({len(cfgs)} runs) ===")
    for cfg in cfgs:
        if done(cfg["name"]):
            print(f"skip {cfg['name']} (done)"); continue
        R.run_config(dict(cfg))
    print(f"=== {slice_name} COMPLETE ===")


def aggregate():
    import json
    import numpy as np

    def rl(name):
        p = Path("results") / name / "summary.json"
        return json.load(open(p))["rel_l2"] if p.exists() else None

    groups = ["cv_cls", "cv_cfe", "cv_qd3_lin", "cv_qd3_none", "cv_qd5_lin"]
    print("=== CONVERGED CORE: mean +/- std over seeds ===")
    stats = {}
    for g in groups:
        vals = [v for v in (rl(f"{g}_s{s}") for s in SEEDS) if v is not None]
        if vals:
            stats[g] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
            print(f"  {g:14} n={len(vals)}  rel_L2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        else:
            print(f"  {g:14} (no runs yet)")

    # the two questions this settles
    if {"cv_qd3_lin", "cv_qd3_none"} <= stats.keys():
        a, sa, _ = stats["cv_qd3_none"]; b, sb, _ = stats["cv_qd3_lin"]
        d = b - a
        sig = abs(d) > 2 * (sa**2 + sb**2) ** 0.5 / max(len(SEEDS), 1) ** 0.5
        print(f"\n  entanglement lead: none={a:.3f} vs linear={b:.3f}  (delta={d:+.3f})")
        print(f"    -> {'entanglement HURTS (survives seeds)' if d>0 and sig else 'within seed noise' if not sig else 'entanglement helps'}")
    if {"cv_qd3_lin", "cv_cfe", "cv_cls"} <= stats.keys():
        print(f"\n  quantum vs fair controls: qapinn={stats['cv_qd3_lin'][0]:.3f}  "
              f"cfe={stats['cv_cfe'][0]:.3f}  classical={stats['cv_cls'][0]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    if cmd == "preflight":
        preflight()
    elif cmd == "aggregate":
        aggregate()
    elif cmd in ("laptop1", "laptop2", "all"):
        run(cmd)
    else:
        print(__doc__)
