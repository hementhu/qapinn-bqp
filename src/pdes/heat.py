"""1-D heat equation as a PDESpec — the smooth, narrow-band control for Burgers.

    u_t = alpha u_xx,   x in [-1,1], t in [0,1],  alpha = 1/pi^2
    IC:  u(x,0) = -sin(pi x)         <-- IDENTICAL to Burgers
    BC:  u(-1,t) = u(1,t) = 0

With alpha = 1/pi^2 the exact solution is a single Fourier mode decaying in time:

    u(x,t) = -sin(pi x) * exp(-t)

Same IC, domain, and BCs as Burgers, so the ONLY difference between the two
problems is the operator (no nonlinear u*u_x here). Heat's solution therefore
stays at frequency k=1 for all t, while Burgers spreads to high harmonics — the
clean spectral contrast the study is built on.
"""
from __future__ import annotations

import numpy as np
import torch

from src.pdes.base import Constraint, Model, PDESpec

ALPHA_DEFAULT: float = 1.0 / np.pi**2
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
IDX_X, IDX_T = 0, 1


def initial_condition(x: np.ndarray) -> np.ndarray:
    """u(x, 0) = -sin(pi x)  (identical to Burgers)."""
    return -np.sin(np.pi * x)


def exact(x: np.ndarray, t: np.ndarray, alpha: float = ALPHA_DEFAULT) -> np.ndarray:
    """Pointwise exact solution u = -sin(pi x) e^{-pi^2 alpha t} (any x, t)."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    return -np.sin(np.pi * x) * np.exp(-(np.pi**2) * alpha * t)


def reference_field(nx: int = 256, nt: int = 100, alpha: float = ALPHA_DEFAULT):
    """Exact solution u = -sin(pi x) e^{-t} on a mesh. Returns x, t, U (nt, nx).

    (alpha is fixed to 1/pi^2 so the time constant is exactly 1; passed through
    for interface symmetry with burgers.reference_field.)
    """
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    XX, TT = np.meshgrid(x, t)                               # (nt, nx)
    U = -np.sin(np.pi * XX) * np.exp(-(np.pi**2) * alpha * TT)
    return x, t, U


def residual(model: Model, X: torch.Tensor, alpha: float = ALPHA_DEFAULT) -> torch.Tensor:
    """r = u_t - alpha u_xx at points X (N,2), requires_grad=True. -> (N,1)."""
    u = model(X)
    grad_u = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]                                                     # (N,2) = (u_x, u_t)
    u_x = grad_u[:, IDX_X:IDX_X + 1]
    u_t = grad_u[:, IDX_T:IDX_T + 1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, IDX_X:IDX_X + 1]
    return u_t - alpha * u_xx


def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)


def _sample_ic(n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, n)
    return _tensor(np.stack([x, np.zeros(n)], axis=1))       # t = 0 face


def _target_ic(X: torch.Tensor) -> torch.Tensor:
    return -torch.sin(np.pi * X[:, IDX_X:IDX_X + 1])         # (n,1)


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


SPEC = PDESpec(
    name="heat",
    in_dim=2,
    out_dim=1,
    domain=((X_MIN, X_MAX), (T_MIN, T_MAX)),
    residual=lambda model, X: residual(model, X, ALPHA_DEFAULT),
    constraints=CONSTRAINTS,
    reference=reference_field,
    exact=exact,                         # enables extrapolation scoring beyond t=1
)
