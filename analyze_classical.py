"""(1) Table of all ablation runs. (2) Classical-side analysis: how the quantum
layer changes the DOWNSTREAM CLASSICAL net's neurons/weights/gradients — the
challenge's Task-3 question (not 'how good is the VQC' but 'what does it do to
the classical PINN behind it').

    python analyze_classical.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.analysis.neurons import activation_fields, effective_rank, neuron_spectra
from src.models.factory import build_model
from src.pdes.base import get_pde

RES = Path("results")
NX, NT = 96, 48


def load(name):
    raw = json.load(open(RES / name / "config.json"))["raw"]
    s = json.load(open(RES / name / "summary.json"))
    return raw, s


def build_loaded(name):
    raw = yaml.safe_load(open(f"experiments/ablation/{name}.yaml")) if (Path("experiments/ablation") / f"{name}.yaml").exists() \
        else json.load(open(RES / name / "config.json"))["raw"]
    m = build_model(raw)
    m.load_state_dict(torch.load(RES / name / "model.pt"))
    m.eval()
    return m


def classical_mlp(model):
    """The classical part: post-net for QAPINN, the whole MLP for a classical PINN."""
    return model.post if hasattr(model, "post") else model


def weight_mean_abs(model):
    mlp = classical_mlp(model)
    w = torch.cat([l.weight.detach().flatten() for l in mlp.linears])
    return w.abs().mean().item()


def classical_neuron_stats(model, spec):
    """Normalized effective rank + mean dominant frequency of the classical neurons."""
    _, _, fields = activation_fields(model, spec, NX, NT)
    if not fields:
        return None, None
    last = list(fields.values())[-1]
    er = effective_rank(last)
    return er / last.shape[1], float(neuron_spectra(last, NX, NT).mean())   # normalized, freq


def postnet_grad_variance(name):
    """Mean gradient variance of the CLASSICAL layers over training (from monitor)."""
    p = RES / name / "monitor.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    labels = list(d["grad_variance_labels"])
    vals = d["grad_variance_values"]                       # (T, n_params)
    # classical layers: 'post.linears.*' (QAPINN) or 'linears.*' (classical), weights only
    idx = [i for i, l in enumerate(labels)
           if ("post.linears" in l or l.startswith("linears")) and "weight" in l]
    if not idx:
        return None
    return float(vals[:, idx].mean())


# --------------------------------------------------------------------------- #
# (1) ABLATION TABLE
# --------------------------------------------------------------------------- #
def ablation_table():
    print("=" * 78)
    print("ALL RUNS")
    print("=" * 78)
    groups = {
        "BASE (heat/burgers x classical/qapinn)":
            ["heat_classical", "heat_qapinn", "burgers_classical", "burgers_qapinn"],
        "DEPTH sweep (Burgers QAPINN)":
            ["abl_depth_1", "abl_depth_2", "abl_baseline", "abl_depth_5"],
        "ENTANGLEMENT sweep (band fixed, depth 3)":
            ["abl_ent_none", "abl_baseline", "abl_ent_ring", "abl_ent_all_to_all"],
        "POST-NET width sweep":
            ["abl_post_lin", "abl_post_8", "abl_baseline", "abl_post_32"],
        "MATCHED CLASSICAL CONTROLS":
            [f"abl_cls_{p}" for p in (29, 73, 105, 113, 121, 137, 217)],
    }
    for title, names in groups.items():
        print(f"\n--- {title} ---")
        print(f"{'run':22} {'model':9} {'knob':22} {'params':>6} {'rel_L2':>9}")
        for n in names:
            if not (RES / n / "summary.json").exists():
                continue
            raw, s = load(n)
            if raw["model"] == "qapinn":
                v = raw["vqc"]
                knob = f"d{v['n_layers']} {v['entanglement'][:4]} m={v['measurement'][:3]} p={raw['post_hidden']}"
            else:
                knob = f"layers={raw['layers']}"
            print(f"{n:22} {raw['model']:9} {knob:22} {s['num_params']:>6} {s['rel_l2']:>9.3e}")


# --------------------------------------------------------------------------- #
# (2) CLASSICAL-SIDE ANALYSIS
# --------------------------------------------------------------------------- #
def classical_side():
    spec = get_pde("burgers")
    print("\n" + "=" * 78)
    print("HOW THE QUANTUM LAYER CHANGES THE CLASSICAL NET (Task 3)")
    print("=" * 78)

    # baseline pair: QAPINN vs param-matched classical
    print("\n--- baseline: QAPINN (VQC front-end) vs matched classical PINN ---")
    print(f"{'model':16} {'classical-neurons':>18} {'norm_effrank':>13} {'neuron_freq':>12} {'mean|W|':>9} {'grad_var':>10}")
    rows = {}
    for name, tag in [("abl_baseline", "QAPINN post-net"), ("abl_cls_121", "classical PINN")]:
        m = build_loaded(name)
        er_n, freq = classical_neuron_stats(m, spec)
        rows[tag] = dict(effrank=er_n, freq=freq, w=weight_mean_abs(m), gv=postnet_grad_variance(name))
        mlp = classical_mlp(m)
        n_neu = mlp.linears[0].out_features
        print(f"{tag:16} {n_neu:>18} {er_n:>13.3f} {freq:>12.2f} {rows[tag]['w']:>9.4f} "
              f"{(rows[tag]['gv'] or float('nan')):>10.2e}")

    # across the depth sweep: does the classical post-net change as the VQC provides more?
    print("\n--- classical post-net vs VQC depth (does the post-net adapt as the VQC gives more?) ---")
    print(f"{'depth':>6} {'norm_effrank':>13} {'neuron_freq':>12} {'mean|W|':>9} {'grad_var':>10}")
    depth_map = {1: "abl_depth_1", 2: "abl_depth_2", 3: "abl_baseline", 5: "abl_depth_5"}
    dd, effs, ws, gvs = [], [], [], []
    for d, name in depth_map.items():
        m = build_loaded(name)
        er_n, freq = classical_neuron_stats(m, spec)
        gv = postnet_grad_variance(name)
        dd.append(d); effs.append(er_n); ws.append(weight_mean_abs(m)); gvs.append(gv)
        print(f"{d:>6} {er_n:>13.3f} {freq:>12.2f} {ws[-1]:>9.4f} {(gv or float('nan')):>10.2e}")

    # figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.bar(list(rows.keys()), [rows[k]["effrank"] for k in rows], color=["tab:red", "tab:gray"])
    a1.set_ylabel("normalized effective rank"); a1.set_title("Classical-neuron diversity: quantum vs classical front-end")
    a2.plot(dd, effs, "o-", label="norm eff-rank"); a2.set_xlabel("VQC depth"); a2.set_ylabel("post-net norm eff-rank")
    a2.set_title("Does the classical post-net adapt to VQC depth?"); a2.grid(alpha=0.3)
    fig.tight_layout()
    out = RES / "ablation_summary"; out.mkdir(exist_ok=True)
    fig.savefig(out / "classical_response.png", dpi=110); plt.close(fig)
    print(f"\n-> {out}/classical_response.png")


if __name__ == "__main__":
    ablation_table()
    classical_side()
