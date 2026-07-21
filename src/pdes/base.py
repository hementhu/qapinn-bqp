"""PDE specification + registry — the PDE-agnostic interface for the whole bench.

The whole point of this file is that 1-D time-dependent PDEs and 2-D steady CFD
share ONE interface, so collocation / losses / trainer / analysis never special-
case a particular PDE. The keystone decision: an initial condition is NOT special
— it is just another entry in a uniform list of ``Constraint``s (a sampler on some
face of the domain + a target there + which output components it fixes).

Two shapes both fit:
    Burgers/heat : in_dim=2 (x,t), out_dim=1 (u),      residual -> (N,1),
                   constraints = [IC at t=0, walls at x=+-1]
    Cavity (NS)  : in_dim=2 (x,y), out_dim=3 (u,v,p),  residual -> (N,3),
                   constraints = [lid u=1 v=0, no-slip walls, pressure reference]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

# A model maps input coordinates X (N, in_dim) -> output (N, out_dim).
Model = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Constraint:
    """One prescribed condition on a face of the domain (IC, wall, lid, p-ref...).

    Attributes:
        name: identifier for logging (e.g. "ic", "wall_left", "lid", "p_ref").
        sample: (n_points, seed) -> X of shape (n_points, in_dim) on this face.
        target: X -> prescribed values of shape (n_points, len(components)).
        components: which output columns this constraint fixes, e.g. (0,) fixes u;
            (0, 1) fixes (u, v) for a no-slip wall; (2,) fixes p for a reference.
    """
    name: str
    sample: Callable[[int, int], torch.Tensor]
    target: Callable[[torch.Tensor], torch.Tensor]
    components: tuple[int, ...]


@dataclass(frozen=True)
class PDESpec:
    """Everything the pipeline needs to know about a PDE, PDE-agnostically.

    Attributes:
        name: registry key.
        in_dim: number of input coordinates (2 for both (x,t) and (x,y)).
        out_dim: number of solution components (1 for u; 3 for (u,v,p)).
        domain: per-input (min, max) bounds, length == in_dim.
        residual: (model, X) -> (N, n_eqns). n_eqns is 1 for Burgers/heat, 3 for
            incompressible NS (2 momentum + continuity). The trainer takes the MSE
            over all columns, so multi-equation systems need no special handling.
        constraints: uniform list of IC/BC/lid/p-ref conditions (see Constraint).
        reference: (…) -> (grids, U) exact/benchmark solution for scoring, or None
            when no ground truth is available.
        exact: pointwise exact solution exact(x, t) -> u (numpy), used to score
            OUTSIDE the training domain (generalization). None if unavailable.
    """
    name: str
    in_dim: int
    out_dim: int
    domain: tuple[tuple[float, float], ...]
    residual: Callable[[Model, torch.Tensor], torch.Tensor]
    constraints: tuple[Constraint, ...]
    reference: Callable[..., tuple] | None = None
    exact: Callable[..., "object"] | None = None

    def __post_init__(self):
        if len(self.domain) != self.in_dim:
            raise ValueError(
                f"domain has {len(self.domain)} bounds but in_dim={self.in_dim}"
            )


def get_pde(name: str, **params) -> PDESpec:
    """Resolve a PDE spec by name, optionally parameterised (e.g. nu for burgers).

    ``params`` lets a config vary a physical parameter (the bandwidth knob for the
    nu-sweep): get_pde("burgers", nu=0.05). Empty params -> the module default SPEC.
    """
    if name == "burgers":
        from src.pdes import burgers
        return burgers.make_spec(**params) if params else burgers.SPEC
    if name == "heat":
        from src.pdes import heat
        return heat.SPEC
    if name == "cavity":
        from src.pdes import cavity
        return cavity.SPEC
    raise ValueError(f"unknown pde '{name}' (expected: burgers | heat | cavity)")
