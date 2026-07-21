"""Reconstruct per-neuron / per-qubit evolution from a run's saved checkpoints.

    python analyze_checkpoints.py results/burgers_qapinn

Rebuilds the model from config.json, loads each checkpoint in order, and runs the
static neuron + (for QAPINN) qubit analysis on each -> how individual features
evolve during training, WITHOUT the cost of logging every neuron/qubit live.

Writes results/<name>/evolution.npz and figures/evolution.png:
    - neuron effective rank vs iteration       (feature diversity growing)
    - mean neuron dominant frequency vs iter    (F-principle at the neuron level)
    - (QAPINN) mean entanglement vs iteration
    - (QAPINN) mean qubit dominant frequency vs iteration
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis import neurons as N
from src.analysis import vqc_internals as V
from src.models.factory import build_model
from src.pdes.base import get_pde

NX, NT = 96, 48


def _iter_of(name: str, adam_iters: int) -> int:
    if "final" in name:
        return adam_iters
    m = re.search(r"_(\d+)", name)
    return int(m.group(1)) if m else 0


def analyze(run_dir: str) -> dict:
    run_dir = Path(run_dir)
    raw = json.load(open(run_dir / "config.json"))["raw"]
    spec = get_pde(raw["pde"])
    adam_iters = int(raw.get("adam_iters", 0))

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"),
                   key=lambda p: _iter_of(p.name, adam_iters))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {run_dir/'checkpoints'}")

    is_q = raw.get("model") == "qapinn"
    iters, eff_rank, neuron_freq, ent_mean, qubit_freq = [], [], [], [], []

    for ck in ckpts:
        model = build_model(raw)
        model.load_state_dict(torch.load(ck))
        model.eval()
        iters.append(_iter_of(ck.name, adam_iters))

        _, _, fields = N.activation_fields(model, spec, NX, NT)
        last = list(fields.values())[-1]
        eff_rank.append(N.effective_rank(last))
        neuron_freq.append(float(N.neuron_spectra(last, NX, NT).mean()))

        if is_q:
            _, _, zf = V.qubit_z_fields(model, spec, NX, NT)
            qubit_freq.append(float(V.qubit_dominant_freq(zf, NX, NT).mean()))
            _, _, e = V.entanglement_field(model, spec, nx=24, nt=12)
            ent_mean.append(float(e.mean()))
        print(f"  {ck.name}: eff_rank={eff_rank[-1]:.2f} neuron_freq={neuron_freq[-1]:.2f}"
              + (f" ent={ent_mean[-1]:.2f} qubit_freq={qubit_freq[-1]:.2f}" if is_q else ""))

    iters = np.array(iters)
    out = {"iters": iters, "eff_rank": np.array(eff_rank), "neuron_freq": np.array(neuron_freq)}
    if is_q:
        out["ent_mean"] = np.array(ent_mean)
        out["qubit_freq"] = np.array(qubit_freq)
    np.savez_compressed(run_dir / "evolution.npz", **out)

    # figure
    panels = 2 + (2 if is_q else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.5 * panels, 4))
    axes = np.atleast_1d(axes)
    axes[0].plot(iters, out["eff_rank"], "o-"); axes[0].set_title("neuron effective rank")
    axes[1].plot(iters, out["neuron_freq"], "o-"); axes[1].set_title("mean neuron dominant freq")
    if is_q:
        axes[2].plot(iters, out["ent_mean"], "o-", color="tab:purple"); axes[2].set_title("mean entanglement")
        axes[3].plot(iters, out["qubit_freq"], "o-", color="tab:green"); axes[3].set_title("mean qubit dominant freq")
    for ax in axes:
        ax.set_xlabel("training iteration"); ax.grid(alpha=0.3)
    fig.suptitle(f"Feature evolution — {run_dir.name}")
    fig.tight_layout()
    (run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(run_dir / "figures" / "evolution.png", dpi=110)
    plt.close(fig)
    print(f"-> {run_dir/'evolution.npz'} + figures/evolution.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="results/<name> directory of a completed run")
    args = ap.parse_args()
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
