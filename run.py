"""Run one experiment from a YAML config: train + full instrumentation + figures.

    python run.py experiments/burgers_qapinn.yaml

Writes everything to results/<name>/:
    config.json  summary.json  history.csv  monitor.npz  model.pt  checkpoints/
    figures/  solution loss spectrum fprinciple neurons (qubits entanglement) landscape generalization

Reproducible from config + seed. The config's `model` field selects classical|qapinn;
`pde` selects burgers|heat. A failing figure is logged and skipped, never aborts the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.analysis import generalization as A_gen
from src.analysis import landscape as A_land
from src.analysis import neurons as A_neu
from src.analysis import spectrum as A_spec
from src.analysis import vqc_internals as A_vqc
from src.analysis.probes import grad_variance_probe, spectrum_probe
from src.models.factory import build_model, describe
from src.pdes.base import get_pde
from src.training.trainer import TrainConfig, train
from src.utils.logging import RunLogger
from src.utils.plotting import plot_loss_history, plot_solution
from src.utils.seeding import seed_everything

_TRAIN_KEYS = {
    "pde", "pde_params", "n_interior", "n_constraint", "adam_iters", "adam_lr",
    "lbfgs_iters", "w_pde", "w_constraints", "seed", "log_every", "eval_nx",
    "eval_nt", "probe_every", "checkpoint_every",
}


def build_train_config(raw: dict) -> TrainConfig:
    return TrainConfig(**{k: raw[k] for k in _TRAIN_KEYS if k in raw})


def _make_figures(model, spec, cfg, run_dir: Path) -> None:
    figs = run_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    NX, NT = 96, 48

    def safe(name, fn):
        try:
            fn()
            print(f"    fig: {name}")
        except Exception as e:                       # a bad figure never aborts the run
            print(f"    fig: {name} SKIPPED ({type(e).__name__}: {e})")

    safe("solution", lambda: plot_solution(model, spec, figs / "solution.png"))
    safe("loss", lambda: plot_loss_history(run_dir / "history.csv", figs / "loss.png"))
    safe("spectrum", lambda: A_spec.plot_spectrum(model, spec, figs / "spectrum.png"))
    if (run_dir / "monitor.npz").exists():
        safe("fprinciple", lambda: A_spec.plot_fprinciple(run_dir / "monitor.npz", spec, figs / "fprinciple.png"))

    def _neurons():
        x, t, fields = A_neu.activation_fields(model, spec, NX, NT)
        if not fields:                           # bare-linear post-net: no hidden neurons
            print("    fig: neurons (no hidden layers — nothing to plot)"); return
        last = list(fields)[-1]
        A_neu.plot_receptive_fields(x, t, fields[last], figs / "neurons.png", NX, NT)
    safe("neurons", _neurons)

    if hasattr(model, "vqc"):                        # QAPINN-only quantum internals
        def _qubits():
            x, t, zf = A_vqc.qubit_z_fields(model, spec, NX, NT)
            A_vqc.plot_qubit_fields(x, t, zf, figs / "qubits.png", NX, NT)
        safe("qubits", _qubits)

        def _entangle():
            ex, et, e = A_vqc.entanglement_field(model, spec, nx=32, nt=16)
            A_vqc.plot_entanglement_field(ex, et, e, figs / "entanglement.png")
        safe("entanglement", _entangle)

    def _landscape():
        a, b, loss = A_land.loss_landscape(model, cfg, span=1.0, n=15)
        A_land.plot_landscape(a, b, loss, figs / "landscape.png", f"{cfg.pde} loss landscape")
    safe("landscape", _landscape)

    if spec.exact is not None:
        def _gen():
            t, rel, tt = A_gen.generalization_curve(model, spec)
            A_gen.plot_generalization({"model": (t, rel)}, figs / "generalization.png", tt)
        safe("generalization", _gen)


def run_config(raw: dict, results_root: str = "results") -> dict:
    """Run one experiment from an already-parsed config dict (see run_one)."""
    name = raw.get("name", "run")
    cfg = build_train_config(raw)
    spec = get_pde(cfg.pde, **cfg.pde_params)   # figures must use the run's PDE params (e.g. nu), not defaults
    run_dir = Path(results_root) / name
    logger = RunLogger(run_dir)

    seed_everything(cfg.seed)
    model = build_model(raw)
    info = describe(model)
    print(f"[{name}] pde={cfg.pde} model={raw.get('model','classical')} params={info['num_params']}")
    (run_dir / "config.json").write_text(json.dumps({"name": name, **info, "raw": raw}, indent=2, default=str))

    probes = (spectrum_probe(), grad_variance_probe()) if cfg.probe_every > 0 else ()
    summary = train(model, cfg, logger, probes=probes)
    summary.update(info)
    logger.write_summary(summary)
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"[{name}] trained rel_L2={summary['rel_l2']:.3e} time={summary['train_time_s']:.0f}s -> figures")
    _make_figures(model, spec, cfg, run_dir)
    print(f"[{name}] DONE -> {run_dir}")
    return summary


def run_one(config_path: str, results_root: str = "results") -> dict:
    """Run one experiment from a YAML config file path."""
    raw = yaml.safe_load(Path(config_path).read_text())
    raw.setdefault("name", Path(config_path).stem)
    return run_config(raw, results_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a YAML experiment config")
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()
    run_one(args.config, args.results_root)


if __name__ == "__main__":
    main()
