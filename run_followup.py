"""Follow-up experiments after the converged core:

(A) MECHANISM firm-up -- 3 more seeds (3,4,5) of the converged depth-3 pair
    (cv_qd3_none / cv_qd3_lin) to shrink the seed variance on the feature metrics.
    NB: this tightens error bars but will not resolve a metric *conflict*.

(B) LARGER-MODEL regime -- the load-bearing question: does "classical wins /
    entanglement hurts" survive once the model is big enough (and trained hard
    enough) to actually resolve the shock? Bigger post-net ([32,32]), more
    collocation (6000) and more L-BFGS (600), 3 seeds:
        big_cls        classical, matched params
        big_qd3_lin    QAPINN depth-3 linear
        big_qd3_none   QAPINN depth-3 no-entanglement

Two-laptop split:
    laptop1 -> mechanism seeds (6 small quantum) + big_cls (fast classical)
    laptop2 -> big QAPINN (big_qd3_lin, big_qd3_none x3 = 6 larger quantum)

    python run_followup.py preflight | laptop1 | laptop2 | all | aggregate
"""
from __future__ import annotations
import sys
from pathlib import Path

from run_ablation import qapinn_cfg, classical_cfg
from run_converged import CONV

MECH_SEEDS = (3, 4, 5)
BIG_SEEDS = (0, 1, 2)
BIG = dict(n_interior=6000, adam_iters=3000, lbfgs_iters=600, checkpoint_every=0)
BIG_POST = [32, 32]


def _seeded(cfg, seed, override):
    cfg = dict(cfg); cfg.update(override); cfg["seed"] = seed
    cfg["name"] = f"{cfg['name']}_s{seed}"
    return cfg


def mech_configs():
    out = []
    for s in MECH_SEEDS:
        out.append(_seeded(qapinn_cfg("cv_qd3_none", vqc_over={"entanglement": "none"}), s, CONV))
        out.append(_seeded(qapinn_cfg("cv_qd3_lin"), s, CONV))
    return out


def _big_qd3(ent, tag):
    return qapinn_cfg(f"big_qd3_{tag}", vqc_over={"entanglement": ent}, post_hidden=BIG_POST)


def _big_params():
    from src.models.factory import build_model, describe
    return describe(build_model(_big_qd3("linear", "lin")))["num_params"]


def big_quantum_configs():
    out = []
    for s in BIG_SEEDS:
        out.append(_seeded(_big_qd3("linear", "lin"), s, BIG))
        out.append(_seeded(_big_qd3("none", "none"), s, BIG))
    return out


def big_classical_configs():
    p = _big_params()
    c = classical_cfg(p); c["name"] = "big_cls"
    return [_seeded(c, s, BIG) for s in BIG_SEEDS]


CONFIRM_SEEDS = (3, 4, 5)   # extend the larger-model core to n=6


def _big_classical_seeds(seeds):
    p = _big_params()
    c = classical_cfg(p); c["name"] = "big_cls"
    return [_seeded(c, s, BIG) for s in seeds]


def slice_configs(name):
    if name == "laptop1":
        return mech_configs() + big_classical_configs()
    if name == "laptop2":
        return big_quantum_configs()
    # n=6 confirmation of the larger-model result (seeds 3-5):
    if name == "laptop1c":   # no-entanglement quantum (3) + all classical (fast)
        return ([_seeded(_big_qd3("none", "none"), s, BIG) for s in CONFIRM_SEEDS]
                + _big_classical_seeds(CONFIRM_SEEDS))
    if name == "laptop2c":   # linear-entanglement quantum (3)
        return [_seeded(_big_qd3("linear", "lin"), s, BIG) for s in CONFIRM_SEEDS]
    return mech_configs() + big_classical_configs() + big_quantum_configs()


def preflight():
    from src.models.factory import build_model, describe
    p = _big_params()
    print("=== FOLLOW-UP PRE-FLIGHT ===")
    print(f"mechanism: cv_qd3_(none|lin) seeds {MECH_SEEDS}  (converged training, post[16], 121 params)")
    print(f"larger-model: big_qd3_(lin|none) params={p}, post{BIG_POST}, train={BIG}")
    print(f"             big_cls matched to {p} params")
    for lab in ("laptop1", "laptop2"):
        cfgs = slice_configs(lab)
        print(f"\n[{lab}] {len(cfgs)} runs: {sorted(set(c['name'].rsplit('_s',1)[0] for c in cfgs))}")
    print("\nEST wall-clock: small quantum ~2.5 h/run, big quantum ~4-5 h/run, classical ~min.")
    print("  laptop1 ~ 6 small quantum (~15 h) + 3 classical (fast)")
    print("  laptop2 ~ 6 big quantum (~24 h). Skips finished runs; safe to stop/restart.")


def run(name):
    import run as R
    done = lambda n: (Path("results") / n / "summary.json").exists()
    cfgs = slice_configs(name)
    print(f"=== FOLLOW-UP: {name} ({len(cfgs)} runs) ===")
    for cfg in cfgs:
        if done(cfg["name"]):
            print(f"skip {cfg['name']} (done)"); continue
        R.run_config(dict(cfg))
    print(f"=== {name} COMPLETE ===")


def aggregate():
    import json, numpy as np
    def m(base, seeds):
        v = [json.load(open(f"results/{base}_s{s}/summary.json"))["rel_l2"]
             for s in seeds if (Path("results") / f"{base}_s{s}" / "summary.json").exists()]
        return (np.mean(v), np.std(v), len(v)) if v else (float("nan"), 0, 0)

    print("=== MECHANISM firm-up: converged pair, seeds 0-2 (core) + 3-5 (new) ===")
    for base in ("cv_qd3_none", "cv_qd3_lin"):
        mu, sd, n = m(base, (0, 1, 2, 3, 4, 5))
        print(f"  {base:14} n={n}  rel_L2 = {mu:.4f} +/- {sd:.4f}")

    print("\n=== LARGER-MODEL regime (seeds 0-2) ===")
    for base in ("big_cls", "big_qd3_none", "big_qd3_lin"):
        mu, sd, n = m(base, BIG_SEEDS)
        print(f"  {base:14} n={n}  rel_L2 = {mu:.4f} +/- {sd:.4f}")
    print("\n  Questions: does classical still beat QAPINN? does none still beat linear?")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    {"preflight": preflight, "aggregate": aggregate}.get(cmd, lambda: run(cmd))()
