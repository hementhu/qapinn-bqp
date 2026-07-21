"""Classical-front-end (CFE) control comparison, done properly and saved.

Trains a CFE (classical front-end, identical post-net) for each QAPINN we compare
against, then compares the POST-NET neurons: accuracy, diversity (normalised
effective rank of the 16 post-net neurons), and post-net gradient variance.

    python analyze_cfe.py

Pairs: Burgers depth (CFE-w4 vs depth-3, depth-5), heat (CFE-w4 vs heat_qapinn),
Burgers qubits (CFE-w{3,5,6} vs qubits-{3,5,6}). Saves everything to
results/ablation_summary/cfe_comparison.json + .png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import run as R
from src.analysis.neurons import activation_fields, effective_rank, neuron_spectra
from src.models.factory import build_model
from src.pdes.base import get_pde

NX, NT = 96, 48
TRAIN = dict(n_interior=1500, n_constraint=150, adam_iters=2000, adam_lr=1e-2,
             lbfgs_iters=0, seed=0, log_every=100, probe_every=25, checkpoint_every=500)


def cfe_cfg(name, width, pde):
    return dict(name=name, model="cfe", width=width, post_hidden=[16], pde=pde, **TRAIN)


def ensure(cfg):
    if not (Path("results") / cfg["name"] / "summary.json").exists():
        R.run_config(dict(cfg))


def metrics(name, pde):
    spec = get_pde(pde)
    raw = json.load(open(f"results/{name}/config.json"))["raw"]
    m = build_model(raw); m.load_state_dict(torch.load(f"results/{name}/model.pt")); m.eval()
    _, _, fields = activation_fields(m, spec, NX, NT)
    last = list(fields.values())[-1]                       # 16 post-net neurons
    er = effective_rank(last) / last.shape[1]
    d = np.load(f"results/{name}/monitor.npz", allow_pickle=True)
    labels = list(d["grad_variance_labels"]); vals = d["grad_variance_values"]
    idx = [i for i, l in enumerate(labels) if "post.linears" in l and "weight" in l]
    gv = float(vals[:, idx].mean()) if idx else float("nan")
    rl = json.load(open(f"results/{name}/summary.json"))["rel_l2"]
    return dict(rel_l2=round(rl, 3), norm_effrank=round(er, 3), postnet_gradvar=gv)


def main():
    # train the CFE controls we need
    ensure(cfe_cfg("cfe_heat", 4, "heat"))
    for w in (3, 5, 6):
        ensure(cfe_cfg(f"cfe_q{w}", w, "burgers"))

    pairs = [
        ("Burgers depth-3", "cfe_burgers", "abl_baseline", "burgers"),
        ("Burgers depth-5", "cfe_burgers", "abl_depth_5", "burgers"),
        ("Heat",            "cfe_heat",    "heat_qapinn",  "heat"),
        ("Burgers 3 qubits","cfe_q3",      "abl_qubits_3", "burgers"),
        ("Burgers 5 qubits","cfe_q5",      "abl_qubits_5", "burgers"),
        ("Burgers 6 qubits","cfe_q6",      "abl_qubits_6", "burgers"),
    ]

    table = {}
    print(f"{'comparison':18} {'':8} {'rel_L2':>7} {'nEffRank':>9} {'gradVar':>9}")
    for label, cfe, qap, pde in pairs:
        cm, qm = metrics(cfe, pde), metrics(qap, pde)
        table[label] = {"cfe": cm, "qapinn": qm}
        print(f"{label:18} {'CFE':8} {cm['rel_l2']:>7.3f} {cm['norm_effrank']:>9} {cm['postnet_gradvar']:>9.2e}")
        print(f"{'':18} {'QAPINN':8} {qm['rel_l2']:>7.3f} {qm['norm_effrank']:>9} {qm['postnet_gradvar']:>9.2e}")

    Path("results/ablation_summary").mkdir(parents=True, exist_ok=True)
    json.dump(table, open("results/ablation_summary/cfe_comparison.json", "w"), indent=2)

    # figure: gradient variance (quantum vs classical front-end) across comparisons
    labels = list(table.keys())
    cfe_gv = [table[k]["cfe"]["postnet_gradvar"] for k in labels]
    qap_gv = [table[k]["qapinn"]["postnet_gradvar"] for k in labels]
    cfe_er = [table[k]["cfe"]["norm_effrank"] for k in labels]
    qap_er = [table[k]["qapinn"]["norm_effrank"] for k in labels]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    xx = np.arange(len(labels)); w = 0.38
    a1.bar(xx - w/2, cfe_gv, w, label="CFE (classical front)"); a1.bar(xx + w/2, qap_gv, w, label="QAPINN")
    a1.set_yscale("log"); a1.set_ylabel("post-net grad variance"); a1.set_title("Gradient amplification (quantum vs classical front-end)")
    a1.set_xticks(xx); a1.set_xticklabels(labels, rotation=25, ha="right"); a1.legend()
    a2.bar(xx - w/2, cfe_er, w, label="CFE"); a2.bar(xx + w/2, qap_er, w, label="QAPINN")
    a2.set_ylabel("post-net norm. effective rank"); a2.set_title("Neuron diversity (quantum vs classical front-end)")
    a2.set_xticks(xx); a2.set_xticklabels(labels, rotation=25, ha="right"); a2.legend()
    fig.tight_layout(); fig.savefig("results/ablation_summary/cfe_comparison.png", dpi=110); plt.close(fig)
    print("\n-> results/ablation_summary/cfe_comparison.json + .png")


if __name__ == "__main__":
    main()
