"""Universal single-model runner. Trains any PINN variant from one YAML config.

    python run.py experiments/burgers_baseline.yaml     # classical (tuned)
    python run.py experiments/quantum_baseline.yaml     # full quantum
    python run.py experiments/qapinn_baseline.yaml      # hybrid

The config's ``model:`` field selects the architecture (classical|quantum|hybrid).
Writes results/<name>/: config.json (full resolved arch + train config),
summary.json (metrics), history.csv, solution.png, loss.png, model.pt.
Reproducible from config + seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.models.factory import build_model, describe
from src.training.trainer import TrainConfig, train
from src.utils.logging import RunLogger
from src.utils.plotting import plot_loss_history, plot_solution
from src.utils.seeding import seed_everything

_TRAIN_KEYS = {
    "layers", "n_interior", "n_ic", "n_bc", "adam_iters", "adam_lr",
    "lbfgs_iters", "nu", "seed", "log_every", "eval_nx", "eval_nt",
}


def build_train_config(raw: dict) -> TrainConfig:
    return TrainConfig(**{k: raw[k] for k in _TRAIN_KEYS if k in raw})


def run_one(config_path: str, results_root: str = "results") -> dict:
    raw = yaml.safe_load(Path(config_path).read_text())
    name = raw.get("name", Path(config_path).stem)

    cfg = build_train_config(raw)
    run_dir = Path(results_root) / name
    logger = RunLogger(run_dir)

    seed_everything(cfg.seed)
    model = build_model(raw)
    info = describe(model)
    print(f"[{name}] model={raw.get('model','classical')} params={info['num_params']}")

    (run_dir / "config.json").write_text(
        json.dumps({"name": name, "model": raw.get("model", "classical"),
                    **info, "train": raw}, indent=2, default=str)
    )

    summary = train(model, cfg, logger)
    summary.update(info)
    logger.write_summary(summary)

    torch.save(model.state_dict(), run_dir / "model.pt")
    plot_solution(model, run_dir / "solution.png")
    plot_loss_history(run_dir / "history.csv", run_dir / "loss.png")

    print(f"[{name}] DONE rel_L2={summary['rel_l2']:.3e} "
          f"final_total={summary['final_total']:.3e} time={summary['train_time_s']:.0f}s "
          f"-> {run_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="path to a YAML experiment config")
    ap.add_argument("--results-root", default="results", type=str)
    args = ap.parse_args()
    run_one(args.config, args.results_root)


if __name__ == "__main__":
    main()
