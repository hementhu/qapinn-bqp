"""Viscous Burgers' equation as a PDESpec.

    u_t + u u_x = nu u_xx,   x in [-1,1], t in [0,1],  nu = 0.01/pi
    IC:  u(x,0) = -sin(pi x)
    BC:  u(-1,t) = u(1,t) = 0

This is the reference example of how a concrete PDE fills the PDE-agnostic
interface in base.py:
    - residual  -> one column (N,1): the single Burgers equation
    - IC + two walls -> three Constraints (IC is NOT special; it is the
      constraint on the t=0 face)
    - reference -> the exact Cole-Hopf solution (self-generated ground truth)

Input column convention: X[:,0] = x, X[:,1] = t.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import roots_hermite

from src.pdes.base import Constraint, Model, PDESpec

NU_DEFAULT: float = 0.01 / np.pi
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
IDX_X, IDX_T = 0, 1


# --------------------------------------------------------------------------- #
# Initial condition and the exact Cole-Hopf reference solution
# --------------------------------------------------------------------------- #
def initial_condition(x: np.ndarray) -> np.ndarray:
    """u(x, 0) = -sin(pi x)."""
    return -np.sin(np.pi * x)


def _antiderivative_ic(y: np.ndarray) -> np.ndarray:
    """G(y) = integral_0^y u0(s) ds = (cos(pi y) - 1)/pi, for u0 = -sin(pi s)."""
    return (np.cos(np.pi * y) - 1.0) / np.pi


def cole_hopf_reference(
    x: np.ndarray, t: np.ndarray, nu: float = NU_DEFAULT, n_quad: int = 200
) -> np.ndarray:
    """Exact u(x,t) via the Cole-Hopf transform, evaluated by Gauss-Hermite quad.

    u = [int (x-y)/t exp(-F/2nu) dy] / [int exp(-F/2nu) dy],
        F(x,y,t) = (x-y)^2/2t + G(y).
    Substituting y = x - 2 sqrt(nu t) z turns the Gaussian factor into the
    Gauss-Hermite weight exp(-z^2). scipy's roots_hermite is stable at large
    n_quad (numpy's hermgauss overflows for this tiny nu). A log-sum-exp shift
    keeps exp() from overflowing. t=0 returns the IC exactly.
    """
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    x, t = np.broadcast_arrays(x, t)
    out = np.empty(x.shape, np.float64)

    zero = t <= 0.0
    out[zero] = initial_condition(x[zero])
    pos = ~zero
    if not np.any(pos):
        return out

    xp, tp = x[pos], t[pos]
    z, w = roots_hermite(n_quad)                       # weight exp(-z^2)
    y = xp[:, None] - 2.0 * np.sqrt(nu * tp)[:, None] * z[None, :]
    a = -_antiderivative_ic(y) / (2.0 * nu)            # (npts, nquad)
    a -= a.max(axis=1, keepdims=True)                  # log-sum-exp stabilization
    expo = np.exp(a)
    coeff = (xp[:, None] - y) / tp[:, None]            # (x-y)/t; common factor cancels
    out[pos] = np.sum(w * coeff * expo, axis=1) / np.sum(w * expo, axis=1)
    return out


def reference_field(nx: int = 256, nt: int = 100, nu: float = NU_DEFAULT):
    """Exact solution on a mesh. Returns x (nx,), t (nt,), U (nt, nx)."""
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    XX, TT = np.meshgrid(x, t)                          # (nt, nx)
    U = cole_hopf_reference(XX.ravel(), TT.ravel(), nu=nu).reshape(XX.shape)
    return x, t, U


# --------------------------------------------------------------------------- #
# PDE residual  (one equation -> one column)
# --------------------------------------------------------------------------- #
def residual(model: Model, X: torch.Tensor, nu: float = NU_DEFAULT) -> torch.Tensor:
    """r = u_t + u u_x - nu u_xx at points X (N,2), requires_grad=True. -> (N,1)."""
    u = model(X)                                        # (N,1)
    grad_u = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]                                                # (N,2) = (u_x, u_t)
    u_x = grad_u[:, IDX_X:IDX_X + 1]
    u_t = grad_u[:, IDX_T:IDX_T + 1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, IDX_X:IDX_X + 1]
    return u_t + u * u_x - nu * u_xx


# --------------------------------------------------------------------------- #
# Constraints: IC (t=0 face) + two walls (x=+-1 faces)
# --------------------------------------------------------------------------- #
def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)


def _sample_ic(n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, n)
    return _tensor(np.stack([x, np.zeros(n)], axis=1))          # t = 0 face


def _target_ic(X: torch.Tensor) -> torch.Tensor:
    x = X[:, IDX_X:IDX_X + 1]
    return -torch.sin(np.pi * x)                                # (n,1)


def _make_wall(x_value: float):
    def sample(n: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        t = rng.uniform(T_MIN, T_MAX, n)
        return _tensor(np.stack([np.full(n, x_value), t], axis=1))
    return sample


def _target_zero(X: torch.Tensor) -> torch.Tensor:
    return torch.zeros((X.shape[0], 1), dtype=X.dtype)


CONSTRAINTS = (
    Constraint("ic", _sample_ic, _target_ic, components=(0,)),
    Constraint("wall_left", _make_wall(X_MIN), _target_zero, components=(0,)),
    Constraint("wall_right", _make_wall(X_MAX), _target_zero, components=(0,)),
)


def make_spec(nu: float = NU_DEFAULT) -> PDESpec:
    """Build a Burgers spec for a given viscosity nu (the bandwidth knob).

    Smaller nu -> sharper shock -> broader solution spectrum. The IC/BCs do not
    depend on nu, so only the residual and the (Cole-Hopf) reference are rewired.
    """
    return PDESpec(
        name="burgers",
        in_dim=2,
        out_dim=1,
        domain=((X_MIN, X_MAX), (T_MIN, T_MAX)),
        residual=lambda model, X: residual(model, X, nu),
        constraints=CONSTRAINTS,
        reference=lambda nx=256, nt=100: reference_field(nx, nt, nu),
        exact=lambda x, t: cole_hopf_reference(x, t, nu),
    )


SPEC = make_spec()                        # default nu = 0.01/pi
