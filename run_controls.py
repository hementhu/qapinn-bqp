"""Pre-publication controls (all classical, cheap):

(A) Random-Fourier-feature (RFF) front end, fixed integer frequencies |w| <= 6 (the
    VQC's measured range), identical post-net and protocol as the QAPINN:
        cv_rff4 / cv_rff16    small scale, post [16]      seeds 0-5
        big_rff4 / big_rff16  larger scale, post [32,32]  seeds 0-5
(B) Seed expansion of the classical baselines/controls to n=10 (seeds reuse the exact
    raw config of the existing seed-0 run, so the protocol is unchanged).

    python run_controls.py list | rff | seeds | all
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

R = Path("results")
SMALL = dict(pde="burgers", n_interior=4000, n_constraint=150, adam_iters=2500, adam_lr=0.01,
             lbfgs_iters=400, log_every=100, probe_every=25, checkpoint_every=0)
LARGE = dict(SMALL, n_interior=6000, adam_iters=3000, lbfgs_iters=600)
SEEDS = range(6)
EXTRA = {"cv_cls": range(3, 10), "cv_cfe": range(3, 10),
         "big_cls": range(6, 10), "big_cls_bal": range(6, 10),
         "big_cls_deep": range(6, 10), "big_cfe": range(6, 10)}


def rff_configs():
    out = []
    for scale, base, post in (("cv", SMALL, [16]), ("big", LARGE, [32, 32])):
        for width in (4, 16):
            for s in SEEDS:
                out.append(dict(name=f"{scale}_rff{width}_s{s}", model="rff", width=width,
                                max_freq=6, post_hidden=post, seed=s, **base))
    return out


def seed_configs():
    out = []
    for base, seeds in EXTRA.items():
        raw = json.load(open(R / f"{base}_s0" / "config.json"))["raw"]
        for s in seeds:
            out.append(dict(raw, name=f"{base}_s{s}", seed=s))
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    cfgs = {"rff": rff_configs, "seeds": seed_configs,
            "all": lambda: rff_configs() + seed_configs(), "list": lambda: rff_configs() + seed_configs()}[what]()
    if what == "list":
        for c in cfgs:
            print(c["name"], "(done)" if (R / c["name"] / "summary.json").exists() else "")
        print(len(cfgs), "runs"); return
    import run as RUN
    for c in cfgs:
        if (R / c["name"] / "summary.json").exists():
            print("skip", c["name"]); continue
        RUN.run_config(dict(c))
    print("=== controls COMPLETE ===")


if __name__ == "__main__":
    main()
