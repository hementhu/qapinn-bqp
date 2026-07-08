"""Training + evaluation for the Burgers PINN.

Two-stage optimization: Adam (find the basin) then L-BFGS (polish to the low
relative-L2 that Adam alone stalls short of). Collocation points are sampled
once and held fixed — L-BFGS needs a deterministic objective, and fixed points
keep runs bit-reproducible. Everything is driven by ``TrainConfig`` + a seed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import torch

from src.pdes.burgers import NU_DEFAULT, reference_field
from src.training.collocation import sample_interior, sample_ic, sample_bc
from src.training.losses import LossWeights, burgers_loss
from src.utils.logging import RunLogger
from src.utils.seeding import seed_everything

Model = torch.nn.Module


@dataclass
class TrainConfig:
    layers: list[int] = field(default_factory=lambda: [2, 20, 20, 20, 1])
    n_interior: int = 10000
    n_ic: int = 200
    n_bc: int = 200
    adam_iters: int = 5000
    adam_lr: float = 1e-3
    lbfgs_iters: int = 500
    w_pde: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    nu: float = NU_DEFAULT
    seed: int = 0
    log_every: int = 100
    eval_nx: int = 256
    eval_nt: int = 100


def _build_points(cfg: TrainConfig):
    """Sample the fixed collocation / IC / BC sets (distinct sub-seeds)."""
    xt_int = sample_interior(cfg.n_interior, seed=cfg.seed + 1)
    xt_ic, u_ic = sample_ic(cfg.n_ic, seed=cfg.seed + 2)
    xt_bc, u_bc = sample_bc(cfg.n_bc, seed=cfg.seed + 3)
    return xt_int, xt_ic, u_ic, xt_bc, u_bc


@torch.no_grad()
def relative_l2(model: Model, cfg: TrainConfig) -> float:
    """Relative L2 error of the model vs the Cole-Hopf reference on a dense grid."""
    x, t, U = reference_field(nx=cfg.eval_nx, nt=cfg.eval_nt, nu=cfg.nu)  # U: (nt, nx)
    XX, TT = np.meshgrid(x, t)
    xt = torch.tensor(
        np.stack([XX.ravel(), TT.ravel()], axis=1), dtype=torch.float32
    )
    pred = model(xt).cpu().numpy().reshape(U.shape)
    return float(np.linalg.norm(pred - U) / np.linalg.norm(U))


def train(
    model: Model,
    cfg: TrainConfig,
    logger: RunLogger | None = None,
) -> dict:
    """Train ``model`` in place. Returns a summary dict of final metrics."""
    seed_everything(cfg.seed)
    weights = LossWeights(pde=cfg.w_pde, ic=cfg.w_ic, bc=cfg.w_bc)
    xt_int, xt_ic, u_ic, xt_bc, u_bc = _build_points(cfg)

    def closure_loss():
        return burgers_loss(
            model, xt_int, xt_ic, u_ic, xt_bc, u_bc, weights=weights, nu=cfg.nu
        )

    def _scalars(terms):
        return dict(
            total=terms.total.item(), pde=terms.pde.item(),
            ic=terms.ic.item(), bc=terms.bc.item(),
        )

    def log(stage: str, it: int, terms):
        if logger is not None:
            logger.log_step(stage=stage, iter=it, **_scalars(terms))

    history_tail = {}
    t0 = time.perf_counter()

    # -- Stage 1: Adam ------------------------------------------------------
    adam = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    for it in range(cfg.adam_iters):
        adam.zero_grad()
        terms = closure_loss()
        terms.total.backward()
        adam.step()
        if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
            log("adam", it, terms)
            history_tail = dict(stage="adam", iter=it, **_scalars(terms))

    # -- Stage 2: L-BFGS ----------------------------------------------------
    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=cfg.lbfgs_iters,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )
        state = {"it": 0}

        def closure():
            lbfgs.zero_grad()
            terms = closure_loss()
            terms.total.backward()
            if state["it"] % cfg.log_every == 0:
                log("lbfgs", state["it"], terms)
            state["it"] += 1
            return terms.total

        lbfgs.step(closure)
        terms = closure_loss()
        log("lbfgs", state["it"], terms)
        history_tail = dict(stage="lbfgs", iter=state["it"], **_scalars(terms))

    wall = time.perf_counter() - t0
    rel_l2 = relative_l2(model, cfg)
    summary = {
        "rel_l2": rel_l2,
        "final_pde": history_tail.get("pde"),
        "final_ic": history_tail.get("ic"),
        "final_bc": history_tail.get("bc"),
        "final_total": history_tail.get("total"),
        "train_time_s": wall,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config": asdict(cfg),
    }
    if logger is not None:
        logger.write_summary(summary)
    return summary
