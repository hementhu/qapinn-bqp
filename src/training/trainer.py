"""PDE-agnostic training + evaluation.

Two-stage optimization: Adam (find the basin) then L-BFGS (polish). Points are
sampled once and held fixed (L-BFGS needs a deterministic objective; fixed points
keep runs reproducible). The trainer resolves the PDE from cfg.pde and touches
nothing PDE-specific beyond the spec.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from src.analysis.monitor import Monitor
from src.pdes.base import get_pde
from src.training.collocation import sample_interior, sample_constraints
from src.training.losses import LossWeights, pde_loss
from src.utils.logging import RunLogger
from src.utils.seeding import seed_everything

Model = torch.nn.Module


@dataclass
class TrainConfig:
    pde: str = "burgers"
    pde_params: dict = field(default_factory=dict)   # e.g. {"nu": 0.05} for the nu-sweep
    n_interior: int = 10000
    n_constraint: int = 200          # points per constraint (IC, each wall, ...)
    adam_iters: int = 5000
    adam_lr: float = 1e-3
    lbfgs_iters: int = 500
    w_pde: float = 1.0
    w_constraints: float = 1.0
    seed: int = 0
    log_every: int = 100
    eval_nx: int = 256
    eval_nt: int = 100
    probe_every: int = 0             # 0 = off; else run probes every N Adam iters
    checkpoint_every: int = 0        # 0 = off; else save a checkpoint every N iters


@torch.no_grad()
def relative_l2(model: Model, spec, cfg: TrainConfig) -> float:
    """Relative L2 vs the exact reference on a dense grid (scalar 1-D-time PDEs).

    Assumes reference returns (g1, g2, U) with U shape (len(g2), len(g1)) and a
    scalar output. Multi-output PDEs (cavity) will need their own eval later.
    """
    if spec.reference is None:                        # no field reference (e.g. cavity)
        return float("nan")
    g1, g2, U = spec.reference(nx=cfg.eval_nx, nt=cfg.eval_nt)
    A, B = np.meshgrid(g1, g2)
    X = torch.tensor(np.stack([A.ravel(), B.ravel()], axis=1), dtype=torch.float32)
    pred = model(X).cpu().numpy().reshape(U.shape)
    return float(np.linalg.norm(pred - U) / np.linalg.norm(U))


def _scalars(terms) -> dict:
    d = dict(total=terms.total.item(), pde=terms.pde.item(),
             constraints=terms.constraints.item())
    d.update({k: v.item() for k, v in terms.per_constraint.items()})
    return d


def train(
    model: Model,
    cfg: TrainConfig,
    logger: RunLogger | None = None,
    probes: tuple = (),
) -> dict:
    """Train `model` in place on cfg.pde. Returns a summary dict.

    probes: optional training-time probes (see src.analysis.probes). Their
    time-series are saved to <run_dir>/monitor.npz; grad-needing probes run
    right after backward() so gradients are live.
    """
    seed_everything(cfg.seed)
    spec = get_pde(cfg.pde, **cfg.pde_params)
    weights = LossWeights(pde=cfg.w_pde, constraints=cfg.w_constraints)

    X_int = sample_interior(spec, cfg.n_interior, seed=cfg.seed + 1)
    con_pts = sample_constraints(spec, cfg.n_constraint, seed=cfg.seed + 2)

    run_dir = Path(logger.run_dir) if logger is not None else None
    monitor = Monitor() if probes else None

    def loss_fn():
        return pde_loss(model, spec, X_int, con_pts, weights)

    def log(stage, it, terms):
        if logger is not None:
            logger.log_step(stage=stage, iter=it, **_scalars(terms))

    def run_probes(it):
        if monitor is None or cfg.probe_every <= 0 or it % cfg.probe_every != 0:
            return
        for pr in probes:                       # called after backward -> grads live
            val, labels = pr(model, spec, it)
            monitor.record(pr.name, it, val, labels)

    def checkpoint(tag):
        if run_dir is None or cfg.checkpoint_every <= 0:
            return
        d = run_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / f"ckpt_{tag}.pt")

    tail = {}
    t0 = time.perf_counter()

    # -- Stage 1: Adam --------------------------------------------------------
    adam = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    for it in range(cfg.adam_iters):
        adam.zero_grad()
        terms = loss_fn()
        terms.total.backward()
        run_probes(it)                          # grads are live here
        adam.step()
        if cfg.checkpoint_every > 0 and it % cfg.checkpoint_every == 0:
            checkpoint(f"adam_{it}")
        if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
            log("adam", it, terms)
            tail = dict(stage="adam", iter=it, **_scalars(terms))

    # -- Stage 2: L-BFGS ------------------------------------------------------
    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
            line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12,
        )
        state = {"it": 0}

        def closure():
            lbfgs.zero_grad()
            terms = loss_fn()
            terms.total.backward()
            if state["it"] % cfg.log_every == 0:
                log("lbfgs", state["it"], terms)
            state["it"] += 1
            return terms.total

        lbfgs.step(closure)
        terms = loss_fn()
        log("lbfgs", state["it"], terms)
        tail = dict(stage="lbfgs", iter=state["it"], **_scalars(terms))

    wall = time.perf_counter() - t0

    checkpoint("final")
    if monitor is not None and run_dir is not None:
        monitor.save(run_dir / "monitor.npz")

    summary = {
        "rel_l2": relative_l2(model, spec, cfg),
        "final_total": tail.get("total"),
        "final_pde": tail.get("pde"),
        "final_constraints": tail.get("constraints"),
        "train_time_s": wall,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config": asdict(cfg),
    }
    if logger is not None:
        logger.write_summary(summary)
    return summary
