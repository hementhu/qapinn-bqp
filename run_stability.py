"""Stability batch: (1) convergence (long training) and (2) seeds on the one
contested result (entanglement). Everything else in the ablation is expected, so
we only spend rigor here.

    python run_stability.py preflight
    python run_stability.py run          # convergence first (the gate), then seeds

(1) Convergence: baseline + entanglement-none QAPINN + matched classical, 8000
    iters, dense checkpoints -> do the ablation orderings survive training, or are
    they undertraining artifacts?
(2) Seeds: entanglement sweep across seeds 1-2 (seed 0 already exists) -> is the
    "entanglement does not help" result real or initialisation noise?
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.models.factory import build_model, describe
from run_ablation import qapinn_cfg, classical_cfg


def converge_configs():
    q = []
    for name, ent in (("conv_qapinn_base", "linear"), ("conv_qapinn_entnone", "none")):
        c = qapinn_cfg(name, vqc_over={"entanglement": ent})
        c.update(adam_iters=8000, checkpoint_every=1000, log_every=200)
        q.append(c)
    cls = classical_cfg(121); cls["name"] = "conv_cls_121"
    cls.update(adam_iters=8000, checkpoint_every=1000, log_every=200)
    return q, [cls]


def seed_configs():
    q, cls = [], []
    for e in ("none", "linear", "ring", "all_to_all"):
        for s in (1, 2):
            c = qapinn_cfg(f"seed{s}_ent_{e}", vqc_over={"entanglement": e}); c["seed"] = s
            q.append(c)
    for s in (1, 2):
        c = classical_cfg(121); c["name"] = f"seed{s}_cls_121"; c["seed"] = s
        cls.append(c)
    return q, cls


def preflight():
    cq, cc = converge_configs()
    sq, sc = seed_configs()
    print("=== STABILITY BATCH PRE-FLIGHT ===\n")
    print("CONVERGENCE (8000 iters, dense checkpoints):")
    for c in cq + cc:
        print(f"  {c['name']:22} {describe(build_model(c))['num_params']:>4}p  "
              f"seed={c['seed']} iters={c['adam_iters']}")
    print("\nSEEDS (entanglement sweep, seeds 1-2, 2000 iters):")
    for c in sq + sc:
        e = c.get('vqc', {}).get('entanglement', 'n/a')
        print(f"  {c['name']:22} {describe(build_model(c))['num_params']:>4}p  seed={c['seed']} ent={e}")
    nq = len(cq) + len(sq)
    print(f"\nTotal: {len(cc)+len(sc)} classical (~20s) + 2 long QAPINN (~2.3h) "
          f"+ {len(sq)} seed QAPINN (~35min)")
    print("Est. wall time ~9-10 h (convergence first, then seeds).")


def run():
    import run as R
    cq, cc = converge_configs()
    sq, sc = seed_configs()

    def done(name):
        return (Path("results") / name / "summary.json").exists()

    def go(cfgs, tag):
        print(f"=== {tag} ({len(cfgs)}) ===")
        for c in cfgs:
            if done(c["name"]):
                print(f"  skip {c['name']} (done)"); continue
            R.run_config(dict(c))

    go(cc + sc, "classical controls")
    go(cq, "convergence QAPINN (the gate)")
    go(sq, "seed QAPINN")
    print("=== STABILITY BATCH COMPLETE ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "run"])
    {"preflight": preflight, "run": run}[ap.parse_args().cmd]()
