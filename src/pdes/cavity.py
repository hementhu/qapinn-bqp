"""2-D lid-driven cavity: steady incompressible Navier-Stokes (the real CFD case).

    u u_x + v u_y + p_x - (1/Re)(u_xx + u_yy) = 0        (x-momentum)
    u v_x + v v_y + p_y - (1/Re)(v_xx + v_yy) = 0        (y-momentum)
    u_x + v_y = 0                                        (continuity)

Unit square (x,y) in [0,1]^2. Top lid moves: u=1, v=0 at y=1; no-slip (u=v=0) on
the other three walls. Pressure has no Dirichlet BC, so it is pinned at one
reference point (p=0 at the origin). Re=100 (well-benchmarked).

No closed-form solution; validated against Ghia, Ghia & Shin (1982) centerline
velocity profiles (hard-coded below for Re=100). So SPEC.reference is None and
scoring uses ghia_error() instead of a field relative-L2.

Input columns: X[:,0]=x, X[:,1]=y. Output columns: (u, v, p).
"""
from __future__ import annotations

import numpy as np
import torch

from src.pdes.base import Constraint, Model, PDESpec

RE_DEFAULT: float = 100.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
IDX_X, IDX_Y = 0, 1


# --------------------------------------------------------------------------- #
# Residual: 3 equations -> (N, 3)
# --------------------------------------------------------------------------- #
def residual(model: Model, X: torch.Tensor, re: float = RE_DEFAULT) -> torch.Tensor:
    nu = 1.0 / re
    out = model(X)                                     # (N,3) = (u,v,p)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    ones = torch.ones_like(u)

    def grad(y, keep=True):
        return torch.autograd.grad(y, X, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True)[0]

    ug, vg, pg = grad(u), grad(v), grad(p)
    u_x, u_y = ug[:, IDX_X:IDX_X + 1], ug[:, IDX_Y:IDX_Y + 1]
    v_x, v_y = vg[:, IDX_X:IDX_X + 1], vg[:, IDX_Y:IDX_Y + 1]
    p_x, p_y = pg[:, IDX_X:IDX_X + 1], pg[:, IDX_Y:IDX_Y + 1]
    u_xx = grad(u_x)[:, IDX_X:IDX_X + 1]
    u_yy = grad(u_y)[:, IDX_Y:IDX_Y + 1]
    v_xx = grad(v_x)[:, IDX_X:IDX_X + 1]
    v_yy = grad(v_y)[:, IDX_Y:IDX_Y + 1]

    mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    cont = u_x + v_y
    return torch.cat([mom_x, mom_y, cont], dim=1)      # (N,3)


# --------------------------------------------------------------------------- #
# Constraints: lid, three no-slip walls, pressure reference
# --------------------------------------------------------------------------- #
def _tensor(a):
    return torch.tensor(a, dtype=torch.float32)


def _edge_sampler(fixed_axis: int, fixed_val: float):
    """Points along an edge: one coordinate fixed, the other uniform in [0,1]."""
    def sample(n: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        free = rng.uniform(0.0, 1.0, n)
        cols = [free, free]
        cols[fixed_axis] = np.full(n, fixed_val)
        return _tensor(np.stack(cols, axis=1))
    return sample


def _lid_target(X):
    n = X.shape[0]
    return _tensor(np.stack([np.ones(n), np.zeros(n)], axis=1))   # u=1, v=0


def _noslip_target(X):
    return torch.zeros((X.shape[0], 2), dtype=X.dtype)            # u=0, v=0


def _pref_sampler(n: int, seed: int) -> torch.Tensor:
    return _tensor(np.zeros((n, 2)))                              # origin (0,0)


def _pref_target(X):
    return torch.zeros((X.shape[0], 1), dtype=X.dtype)           # p=0


CONSTRAINTS = (
    Constraint("lid",         _edge_sampler(IDX_Y, Y_MAX), _lid_target,    components=(0, 1)),
    Constraint("wall_bottom", _edge_sampler(IDX_Y, Y_MIN), _noslip_target, components=(0, 1)),
    Constraint("wall_left",   _edge_sampler(IDX_X, X_MIN), _noslip_target, components=(0, 1)),
    Constraint("wall_right",  _edge_sampler(IDX_X, X_MAX), _noslip_target, components=(0, 1)),
    Constraint("p_ref",       _pref_sampler,               _pref_target,   components=(2,)),
)


SPEC = PDESpec(
    name="cavity",
    in_dim=2,
    out_dim=3,
    domain=((X_MIN, X_MAX), (Y_MIN, Y_MAX)),
    residual=lambda model, X: residual(model, X, RE_DEFAULT),
    constraints=CONSTRAINTS,
    reference=None,          # no closed form; validate via ghia_error()
    exact=None,
)


# --------------------------------------------------------------------------- #
# Ghia et al. (1982) benchmark, Re=100 (for validation)
# --------------------------------------------------------------------------- #
# u-velocity along the vertical centerline x=0.5:
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                   0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
                   -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
                   0.78871, 0.84123, 1.0])
# v-velocity along the horizontal centerline y=0.5:
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                   0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
GHIA_V = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
                   0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                   -0.05906, 0.0])


@torch.no_grad()
def ghia_error(model) -> dict:
    """Relative-L2 of model centerline profiles vs Ghia (Re=100).

    Returns errors and the profiles (for plotting).
    """
    # u along x=0.5
    Xu = torch.tensor(np.stack([np.full_like(GHIA_Y, 0.5), GHIA_Y], 1), dtype=torch.float32)
    u_pred = model(Xu)[:, 0].cpu().numpy()
    # v along y=0.5
    Xv = torch.tensor(np.stack([GHIA_X, np.full_like(GHIA_X, 0.5)], 1), dtype=torch.float32)
    v_pred = model(Xv)[:, 1].cpu().numpy()
    eu = np.linalg.norm(u_pred - GHIA_U) / np.linalg.norm(GHIA_U)
    ev = np.linalg.norm(v_pred - GHIA_V) / np.linalg.norm(GHIA_V)
    return {"u_rel_l2": float(eu), "v_rel_l2": float(ev), "rel_l2": float(0.5 * (eu + ev)),
            "y": GHIA_Y, "u_pred": u_pred, "u_ghia": GHIA_U,
            "x": GHIA_X, "v_pred": v_pred, "v_ghia": GHIA_V}
