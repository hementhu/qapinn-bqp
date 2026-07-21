"""nu-sweep sufficiency test (#3): is BANDWIDTH a sufficient statistic for the
quantum advantage across a PDE family?

Fix the QAPINN (4 qubits, depth 3) and its matched classical control, and vary the
Burgers viscosity nu -> the solution bandwidth changes continuously (nu=0.1 -> bw 3,
nu=0.003 -> bw 22). For each nu we measure the advantage (classical rel-L2 minus
QAPINN rel-L2) and the solution bandwidth. Plotting advantage vs bandwidth, and
overlaying the heat point (bandwidth 1), tests the collapse: if different problems
fall on one curve, bandwidth is a sufficient statistic.

    python run_nusweep.py preflight | run | aggregate

The baseline nu (0.01/pi, bw 22) reuses abl_baseline / abl_cls_121.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_ablation import BASE_VQC, BASE_TRAIN, classical_cfg
from src.analysis.spectrum import spatial_spectrum
from src.models.factory import build_model, describe
from src.pdes.base import get_pde

NU_POINTS = [(0.1, "01"), (0.05, "005"), (0.02, "002"), (0.01, "001")]   # baseline reused


def qapinn_nu(nu, tag):
    return dict(name=f"nu_qapinn_{tag}", model="qapinn", vqc=dict(BASE_VQC),
                post_hidden=[16], pde_params={"nu": nu}, **BASE_TRAIN)


def cls_nu(nu, tag):
    c = classical_cfg(121); c["name"] = f"nu_cls_{tag}"
    c["pde_params"] = {"nu": nu}
    return c


def configs():
    q = [qapinn_nu(nu, tag) for nu, tag in NU_POINTS]
    c = [cls_nu(nu, tag) for nu, tag in NU_POINTS]
    return q, c


def bandwidth(nu):
    spec = get_pde("burgers", nu=nu)
    _, _, U = spec.reference(nx=256, nt=60)
    m = spatial_spectrum(U); m[0] = 0
    return int(np.where(m > 0.02 * m.max())[0].max())


def preflight():
    q, c = configs()
    print("=== NU-SWEEP PRE-FLIGHT (fixed QAPINN depth-3, vary nu) ===\n")
    print(f"{'nu':>8} {'bandwidth':>10} {'qapinn':>16} {'classical':>14}")
    for (nu, tag), qc, cc in zip(NU_POINTS, q, c):
        print(f"{nu:>8.3f} {bandwidth(nu):>10} {qc['name']:>16} {cc['name']:>14} "
              f"({describe(build_model(qc))['num_params']}p)")
    print(f"\nbaseline nu={0.01/np.pi:.4f} bw={bandwidth(0.01/np.pi)} reuses abl_baseline / abl_cls_121")
    print(f"Total: {len(q)} QAPINN (~35min) + {len(c)} classical (~20s)")


def run():
    import run as R
    q, c = configs()

    def done(n):
        return (Path("results") / n / "summary.json").exists()

    for cfg in c + q:
        if done(cfg["name"]):
            print(f"skip {cfg['name']} (done)"); continue
        R.run_config(dict(cfg))
    print("=== NU-SWEEP COMPLETE ===")


def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rl(n):
        return json.load(open(f"results/{n}/summary.json"))["rel_l2"]

    # points: (bandwidth, qapinn rel-L2, classical rel-L2, label)
    pts = []
    for nu, tag in NU_POINTS:
        pts.append((bandwidth(nu), rl(f"nu_qapinn_{tag}"), rl(f"nu_cls_{tag}"), f"nu={nu}"))
    pts.append((bandwidth(0.01 / np.pi), rl("abl_baseline"), rl("abl_cls_121"), "nu=0.003 (base)"))
    pts.append((1, rl("heat_qapinn"), rl("heat_classical"), "heat"))
    pts.sort()

    bw = [p[0] for p in pts]
    adv = [p[2] - p[1] for p in pts]           # classical - QAPINN (positive = quantum better)
    qap = [p[1] for p in pts]; cls = [p[2] for p in pts]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.plot(bw, qap, "o-", label="QAPINN"); a1.plot(bw, cls, "s--", label="classical")
    a1.axvline(3, ls=":", color="gray"); a1.text(3, a1.get_ylim()[1], " QAPINN band (depth 3)", va="top", fontsize=8)
    a1.set_xlabel("solution bandwidth"); a1.set_ylabel("rel L2"); a1.set_title("Accuracy vs solution bandwidth")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(bw, adv, "o-", color="tab:red"); a2.axhline(0, ls=":", color="gray")
    for x, y, _, lab in [(p[0], p[2]-p[1], p[1], p[3]) for p in pts]:
        a2.annotate(lab, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a2.set_xlabel("solution bandwidth"); a2.set_ylabel("advantage (classical - QAPINN rel L2)")
    a2.set_title("Quantum advantage vs bandwidth (collapse test)"); a2.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("results/ablation_summary"); out.mkdir(exist_ok=True)
    fig.savefig(out / "nusweep.png", dpi=110); plt.close(fig)

    table = {p[3]: {"bandwidth": p[0], "qapinn": round(p[1], 3), "classical": round(p[2], 3),
                    "advantage": round(p[2] - p[1], 3)} for p in pts}
    (out / "nusweep.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    print(f"-> {out}/nusweep.png + .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "run", "aggregate"])
    {"preflight": preflight, "run": run, "aggregate": aggregate}[ap.parse_args().cmd]()
