"""Viscous Burgers' equation: residual, IC/BC, and an exact reference solution.

PDE (1D viscous Burgers):
    u_t + u * u_x = nu * u_xx,   x in [-1, 1], t in [0, 1]
    IC:  u(x, 0) = -sin(pi x)
    BC:  u(-1, t) = u(1, t) = 0
    nu = 0.01 / pi   (default)

The exact reference is generated in-repo via the Cole-Hopf transformation
(original code, no downloaded datasets), so it satisfies the challenge's
"self-generated data" rule and doubles as ground truth for relative-L2 scoring.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.special import roots_hermite

NU_DEFAULT: float = 0.01 / np.pi
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Input column convention for every (x, t) tensor fed to a model: col 0 = x, 1 = t.
IDX_X, IDX_T = 0, 1


def pde_residual(
    model: Callable[[torch.Tensor], torch.Tensor],
    xt: torch.Tensor,
    nu: float = NU_DEFAULT,
) -> torch.Tensor:
    """Viscous-Burgers residual  r = u_t + u*u_x - nu*u_xx  at collocation points.

    Derivatives are taken by autograd (``create_graph=True`` so the residual is
    itself differentiable for training). ``xt`` must have ``requires_grad=True``
    and columns (x, t). Works with any model — classical MLP or hybrid QAPINN.

    Args:
        model: maps xt (N, 2) -> u (N, 1).
        xt: collocation points (N, 2), requires_grad=True.
        nu: viscosity.

    Returns:
        Residual tensor (N, 1). A perfect solution gives all-zeros.
    """
    u = model(xt)  # (N, 1)
    grad_u = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]  # (N, 2) = (u_x, u_t)
    u_x = grad_u[:, IDX_X : IDX_X + 1]
    u_t = grad_u[:, IDX_T : IDX_T + 1]

    u_xx = torch.autograd.grad(
        u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, IDX_X : IDX_X + 1]

    return u_t + u * u_x - nu * u_xx


def initial_condition(x: np.ndarray) -> np.ndarray:
    """IC: u(x, 0) = -sin(pi x)."""
    return -np.sin(np.pi * x)


def _antiderivative_ic(y: np.ndarray) -> np.ndarray:
    """G(y) = integral_0^y u0(s) ds  for u0(s) = -sin(pi s).

    integral_0^y -sin(pi s) ds = (cos(pi y) - 1) / pi.
    Appears inside the Cole-Hopf exponent.
    """
    return (np.cos(np.pi * y) - 1.0) / np.pi


def cole_hopf_reference(
    x: np.ndarray,
    t: np.ndarray,
    nu: float = NU_DEFAULT,
    n_quad: int = 200,
) -> np.ndarray:
    """Exact u(x, t) for viscous Burgers with u0 = -sin(pi x), via Cole-Hopf.

    The Cole-Hopf solution is a ratio of Gaussian-weighted integrals:

        u(x,t) = [ int (x-y)/t * exp(-F/(2nu)) dy ]
                 / [ int         exp(-F/(2nu)) dy ]
        F(x,y,t) = (x-y)^2 / (2t) + G(y)

    Substituting y = x - 2*sqrt(nu*t)*z turns the Gaussian factor into the
    Gauss-Hermite weight exp(-z^2), so the integrals are evaluated exactly by
    Gauss-Hermite quadrature. A log-sum-exp shift keeps exp() from overflowing
    for small nu (the exponent reaches ~100 at nu = 0.01/pi).

    Args:
        x, y grids are broadcast together; x and t are 1D arrays of the same
            length OR broadcastable — output has their broadcast shape.
        nu: kinematic viscosity.
        n_quad: number of Gauss-Hermite nodes (200 -> ~machine precision here).

    Returns:
        u evaluated at each (x, t). Points with t == 0 return the IC exactly.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x, t = np.broadcast_arrays(x, t)
    out = np.empty(x.shape, dtype=np.float64)

    # t == 0 is singular in the substitution; return the IC directly.
    zero = t <= 0.0
    out[zero] = initial_condition(x[zero])

    pos = ~zero
    if not np.any(pos):
        return out

    xp = x[pos]
    tp = t[pos]

    # Gauss-Hermite nodes/weights for the weight exp(-z^2).
    # scipy's roots_hermite is stable for large n_quad; numpy's hermgauss
    # overflows (Hermite polynomials blow up) for n_quad >~ 150.
    z, w = roots_hermite(n_quad)                      # (n_quad,), (n_quad,)

    # y_i(x,t) = x - 2 sqrt(nu t) z_i   -> shape (n_points, n_quad)
    scale = 2.0 * np.sqrt(nu * tp)[:, None]           # (n_points, 1)
    y = xp[:, None] - scale * z[None, :]              # (n_points, n_quad)

    # Exponent a_i = -G(y_i) / (2 nu); shift by row-max for stability.
    a = -_antiderivative_ic(y) / (2.0 * nu)           # (n_points, n_quad)
    a -= a.max(axis=1, keepdims=True)
    expo = np.exp(a)                                  # in (0, 1]

    # Coefficient c_i = (x - y_i)/t = 2 z_i sqrt(nu/t); common 2 sqrt(nu t)
    # cancels between numerator and denominator, so we keep the raw form.
    coeff = (xp[:, None] - y) / tp[:, None]           # (n_points, n_quad)

    num = np.sum(w[None, :] * coeff * expo, axis=1)
    den = np.sum(w[None, :] * expo, axis=1)
    out[pos] = num / den
    return out


def grid(nx: int = 256, nt: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Uniform (x, t) mesh over the domain. Returns 1D x and t axes."""
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    return x, t


def reference_field(
    nx: int = 256, nt: int = 100, nu: float = NU_DEFAULT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full reference solution on a mesh.

    Returns:
        x (nx,), t (nt,), U (nt, nx) with U[j, i] = u(x_i, t_j).
    """
    x, t = grid(nx, nt)
    XX, TT = np.meshgrid(x, t)  # (nt, nx)
    U = cole_hopf_reference(XX.ravel(), TT.ravel(), nu=nu).reshape(XX.shape)
    return x, t, U
