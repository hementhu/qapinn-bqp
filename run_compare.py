"""Comparison framework: train N models from N YAML configs and compare them.

    python run_compare.py experiments/classical_control.yaml \
                          experiments/quantum_baseline.yaml \
                          experiments/qapinn_baseline.yaml

Trains each config (any model kind) and writes to results/<name>/, then produces
a joint results/compare/ folder with:
  - comparison.png  : reference + each field + each |error| + a t=0.5 slice
  - loss.png        : all training-loss curves overlaid
  - comparison.json : metrics table (params, rel-L2, residual, time) per model

Fair comparison is the caller's responsibility: give the configs the same
training budget (points/iters/seed) so the only variable is the model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.pdes.burgers import cole_hopf_reference, reference_field
from src.models.factory import build_model, describe
from src.training.trainer import train
from src.utils.logging import RunLogger
from src.utils.seeding import seed_everything
from run import build_train_config


def _field(model, x, t):
    XX, TT = np.meshgrid(x, t)
    xt = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        return model(xt).cpu().numpy().reshape(len(t), len(x))


def comparison_figure(models: dict, out_path: Path, nx=301, nt=151):
    x, t, U = reference_field(nx=nx, nt=nt)
    fields = {name: _field(m, x, t) for name, m in models.items()}
    ncol = 1 + len(models)

    fig = plt.figure(figsize=(4.6 * ncol, 9))
    # Row 1: reference + predictions
    panels = [("reference", U)] + list(fields.items())
    for k, (title, F) in enumerate(panels):
        ax = fig.add_subplot(3, ncol, k + 1)
        pc = ax.pcolormesh(t, x, F.T, cmap="rainbow", shading="auto", vmin=-1, vmax=1)
        ax.set_title(title); ax.set_xlabel("t"); ax.set_ylabel("x")
        fig.colorbar(pc, ax=ax)
    # Row 2: |error| under each prediction
    for k, (name, F) in enumerate(fields.items()):
        ax = fig.add_subplot(3, ncol, ncol + 2 + k)
        pc = ax.pcolormesh(t, x, np.abs(F - U).T, cmap="inferno", shading="auto")
        ax.set_title(f"|error| {name}"); ax.set_xlabel("t"); ax.set_ylabel("x")
        fig.colorbar(pc, ax=ax)
    # Row 3: slice at t=0.5
    ax = fig.add_subplot(3, 1, 3)
    ax.plot(x, cole_hopf_reference(x, np.full_like(x, 0.5)), "k-", lw=2.5, label="reference")
    for name, m in models.items():
        xt = torch.tensor(np.stack([x, np.full_like(x, 0.5)], 1), dtype=torch.float32)
        with torch.no_grad():
            ax.plot(x, m(xt).numpy().ravel(), "--", lw=1.8, label=name)
    ax.set_title("slice at t = 0.5"); ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def loss_figure(logs: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, csv_path in logs.items():
        rows = list(csv.DictReader(open(csv_path)))
        ax.semilogy(range(len(rows)), [float(r["total"]) for r in rows], label=name)
    ax.set_xlabel("logged step"); ax.set_ylabel("total loss (log)")
    ax.set_title("Training loss (compare)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="two or more YAML config paths")
    ap.add_argument("--results-root", default="results", type=str)
    ap.add_argument("--out", default="results/compare", type=str)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models: dict = {}
    logs: dict = {}
    table: dict = {}

    for path in args.configs:
        raw = yaml.safe_load(Path(path).read_text())
        name = raw.get("name", Path(path).stem)
        cfg = build_train_config(raw)
        run_dir = Path(args.results_root) / name
        logger = RunLogger(run_dir)

        seed_everything(cfg.seed)
        model = build_model(raw)
        info = describe(model)
        print(f"training [{name}] model={raw.get('model','classical')} params={info['num_params']} ...")
        summary = train(model, cfg, logger)
        summary.update(info)
        logger.write_summary(summary)
        torch.save(model.state_dict(), run_dir / "model.pt")
        print(f"  [{name}] rel_L2={summary['rel_l2']:.4e} time={summary['train_time_s']:.0f}s")

        models[name] = model
        logs[name] = run_dir / "history.csv"
        table[name] = {"model": raw.get("model", "classical"),
                       "params": info["num_params"],
                       **{k: summary[k] for k in ("rel_l2", "final_total", "final_pde", "train_time_s")}}

    comparison_figure(models, out / "comparison.png")
    loss_figure(logs, out / "loss.png")
    (out / "comparison.json").write_text(json.dumps(table, indent=2, default=str))
    print(json.dumps(table, indent=2, default=str))
    print(f"compare artifacts -> {out}")


if __name__ == "__main__":
    main()
