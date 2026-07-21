"""Tier-1 ablation on Burgers: depth + entanglement (fixed band) + post-net width,
each QAPINN paired with a param-matched classical control.

    python run_ablation.py preflight    # build all, print param table, NO runs (guardrail)
    python run_ablation.py run           # run all (classical controls first, then QAPINNs)
    python run_ablation.py aggregate     # build per-sweep comparison figures from results

Baseline (held while varying one knob): 4 qubits, depth 3, linear entanglement,
multi-Z measurement, input_scale 1, post-net [16], 2000 Adam iters, seed 0.

Matched classical control per distinct QAPINN param count P: a single-hidden PINN
[2, (P-1)//4, 1], which hits P exactly (all QAPINN counts are ≡ 1 mod 4) and is
structurally the right control — 'classical first layer' vs 'VQC first layer'.
Entanglement variants share the identical VQC init (same shape + same seed), so the
only difference is the wiring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.models.factory import build_model, describe

BASE_VQC = dict(n_qubits=4, n_layers=3, reupload=True, entanglement="linear",
                measurement="multi", input_scale=1.0)
BASE_TRAIN = dict(pde="burgers", n_interior=1500, n_constraint=150, adam_iters=2000,
                  adam_lr=0.01, lbfgs_iters=0, seed=0, log_every=100,
                  probe_every=25, checkpoint_every=500)
ABL_DIR = Path("experiments/ablation")


def qapinn_cfg(name, vqc_over=None, post_hidden=(16,)):
    vqc = dict(BASE_VQC)
    if vqc_over:
        vqc.update(vqc_over)
    return dict(name=name, model="qapinn", vqc=vqc, post_hidden=list(post_hidden), **BASE_TRAIN)


def _match_arch(params: int):
    """Smallest-imbalance [2,a,b,1] with exactly `params` weights, else closest 1-hidden.

    For [2,a,b,1]: params = 3a + ab + 2b + 1. A single-hidden [2,H,1] = 4H+1 only
    hits params == 1 (mod 4); qubit/measurement ablations produce other residues,
    so we search two-hidden layouts for an exact match (most balanced a,b).
    """
    best = None
    for a in range(1, params):
        num = params - 1 - 3 * a               # b = num / (a+2)
        if num < a + 2:
            break
        if num % (a + 2) == 0:
            b = num // (a + 2)
            score = abs(a - b)
            if best is None or score < best[0]:
                best = (score, [2, a, b, 1])
    return best[1] if best else [2, round((params - 1) / 4), 1]


def classical_cfg(params: int):
    return dict(name=f"abl_cls_{params}", model="classical",
                layers=_match_arch(params), **BASE_TRAIN)


def tier1_qapinn_configs():
    cfgs = [qapinn_cfg("abl_baseline")]                         # depth3 / linear / post[16]
    for L in (1, 2, 5):
        cfgs.append(qapinn_cfg(f"abl_depth_{L}", vqc_over={"n_layers": L}))
    for e in ("none", "ring", "all_to_all"):
        cfgs.append(qapinn_cfg(f"abl_ent_{e}", vqc_over={"entanglement": e}))
    for ph, tag in (([], "lin"), ([8], "8"), ([32], "32")):
        cfgs.append(qapinn_cfg(f"abl_post_{tag}", post_hidden=ph))
    return cfgs


def tier3_qapinn_configs():
    """Remaining VQC knobs: qubits, measurement, input_scale (baseline held)."""
    cfgs = []
    for q in (3, 5, 6):                              # 4 is baseline
        cfgs.append(qapinn_cfg(f"abl_qubits_{q}", vqc_over={"n_qubits": q}))
    for meas in ("single", "zz"):                   # multi is baseline
        cfgs.append(qapinn_cfg(f"abl_meas_{meas}", vqc_over={"measurement": meas}))
    for sc, tag in ((0.5, "05"), (3.14159, "pi"), (6.28318, "2pi")):   # 1.0 baseline
        cfgs.append(qapinn_cfg(f"abl_scale_{tag}", vqc_over={"input_scale": sc}))
    return cfgs


def all_configs(tier="1"):
    """QAPINN configs + the matched classical control for each distinct param count."""
    q = {"1": tier1_qapinn_configs(), "3": tier3_qapinn_configs(),
         "all": tier1_qapinn_configs() + tier3_qapinn_configs()}[tier]
    counts = sorted({describe(build_model(c))["num_params"] for c in q})
    classicals = [classical_cfg(p) for p in counts]
    return q, classicals, counts


def preflight(tier="1"):
    q, classicals, counts = all_configs(tier)
    print(f"=== TIER-{tier} ABLATION PRE-FLIGHT (Burgers) ===\n")
    print(f"{'QAPINN config':20} {'params':>7}  knob")
    for c in q:
        p = describe(build_model(c))["num_params"]
        v = c["vqc"]
        knob = f"depth={v['n_layers']} ent={v['entanglement']} post={c['post_hidden']}"
        print(f"{c['name']:20} {p:>7}  {knob}")
    print(f"\nDistinct param counts -> matched classical controls: {counts}")
    for c in classicals:
        p = describe(build_model(c))["num_params"]
        ok = "OK" if p == int(c['name'].split('_')[-1]) else "MISMATCH!"
        print(f"  {c['name']:16} layers={c['layers']} -> {p} params  [{ok}]")
    print(f"\nTotal: {len(q)} QAPINN runs (~35 min each) + {len(classicals)} classical (~20 s each)")
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    for c in q + classicals:
        (ABL_DIR / f"{c['name']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"configs written to {ABL_DIR}/")


def run(tier="1"):
    import run as R
    q, classicals, _ = all_configs(tier)

    def done(name):                                  # skip already-completed runs
        return (Path("results") / name / "summary.json").exists()

    print(f"=== running classical controls ({len(classicals)}) ===")
    for c in classicals:
        if done(c["name"]):
            print(f"  skip {c['name']} (done)"); continue
        R.run_config(dict(c))
    print(f"=== running QAPINN ablations ({len(q)}) ===")
    for c in q:
        if done(c["name"]):
            print(f"  skip {c['name']} (done)"); continue
        R.run_config(dict(c))
    print("=== ABLATION RUN COMPLETE ===")


def aggregate():
    """Build per-sweep comparison figures from the completed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from src.analysis import spectrum as S, vqc_internals as V
    from src.models.factory import build_model as bm
    from src.pdes.base import get_pde

    spec = get_pde("burgers")
    out = Path("results/ablation_summary"); out.mkdir(parents=True, exist_ok=True)

    def rel_l2(name):
        return json.load(open(f"results/{name}/summary.json"))["rel_l2"]

    def params(name):
        return json.load(open(f"results/{name}/summary.json"))["num_params"]

    def max_freq(name):                          # highest harmonic the model actually builds
        raw = yaml.safe_load(open(ABL_DIR / f"{name}.yaml"))
        m = bm(raw); m.load_state_dict(torch.load(f"results/{name}/model.pt")); m.eval()
        k, mm, _ = S.realized_spectrum(m, spec)
        sig = np.where(mm > 0.05 * mm.max())[0]
        return int(sig.max())

    def mean_ent(name):
        raw = yaml.safe_load(open(ABL_DIR / f"{name}.yaml"))
        m = bm(raw); m.load_state_dict(torch.load(f"results/{name}/model.pt")); m.eval()
        _, _, e = V.entanglement_field(m, spec, nx=24, nt=12)
        return float(e.mean())

    # --- depth sweep ---
    depths = [1, 2, 3, 5]
    qn = {1: "abl_depth_1", 2: "abl_depth_2", 3: "abl_baseline", 5: "abl_depth_5"}
    q_rl = [rel_l2(qn[d]) for d in depths]
    c_rl = [rel_l2(f"abl_cls_{params(qn[d])}") for d in depths]
    q_mf = [max_freq(qn[d]) for d in depths]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(depths, q_rl, "o-", label="QAPINN"); a1.plot(depths, c_rl, "s--", label="matched classical")
    a1.set_xlabel("depth (n_layers)"); a1.set_ylabel("rel L2"); a1.set_title("Accuracy vs depth"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(depths, q_mf, "o-", color="tab:red"); a2.plot(depths, depths, "k:", label="freq = depth (theory)")
    a2.set_xlabel("depth"); a2.set_ylabel("max realized harmonic"); a2.set_title("Realized bandwidth vs depth"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "depth_sweep.png", dpi=110); plt.close(fig)

    # --- entanglement sweep (band fixed) ---
    ents = ["none", "linear", "ring", "all_to_all"]
    en = {"linear": "abl_baseline", "none": "abl_ent_none", "ring": "abl_ent_ring", "all_to_all": "abl_ent_all_to_all"}
    e_rl = [rel_l2(en[e]) for e in ents]; e_ent = [mean_ent(en[e]) for e in ents]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.bar(ents, e_rl, color="tab:blue"); a1.set_ylabel("rel L2"); a1.set_title("Accuracy vs entanglement (band FIXED)"); a1.tick_params(axis="x", rotation=20)
    a2.bar(ents, e_ent, color="tab:purple"); a2.set_ylabel("mean entanglement entropy"); a2.set_title("Entanglement vs wiring"); a2.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "entanglement_sweep.png", dpi=110); plt.close(fig)

    # --- post-net width sweep ---
    posts = [("lin", "abl_post_lin"), ("8", "abl_post_8"), ("16", "abl_baseline"), ("32", "abl_post_32")]
    p_lbl = [p[0] for p in posts]; p_rl = [rel_l2(p[1]) for p in posts]
    p_cl = [rel_l2(f"abl_cls_{params(p[1])}") for p in posts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_lbl, p_rl, "o-", label="QAPINN"); ax.plot(p_lbl, p_cl, "s--", label="matched classical")
    ax.set_xlabel("post-net hidden width"); ax.set_ylabel("rel L2"); ax.set_title("Accuracy vs quantum/classical split"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "post_sweep.png", dpi=110); plt.close(fig)

    table = {"depth": dict(depths=depths, qapinn=q_rl, classical=c_rl, max_freq=q_mf),
             "entanglement": dict(types=ents, rel_l2=e_rl, entropy=e_ent),
             "post": dict(widths=p_lbl, qapinn=p_rl, classical=p_cl)}
    (out / "ablation_summary.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    print(f"-> {out}/  (depth_sweep, entanglement_sweep, post_sweep .png)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "run", "aggregate"])
    ap.add_argument("--tier", default="1", choices=["1", "3", "all"])
    a = ap.parse_args()
    if a.cmd == "aggregate":
        aggregate()
    else:
        {"preflight": preflight, "run": run}[a.cmd](a.tier)
