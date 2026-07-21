"""ClassicalFrontEnd: the clean control for the classical-side mechanism study.

    (x,t) -> Linear(in_dim -> width) -> tanh -> classical post-net -> u

Structurally identical to the QAPINN (fixed-width front-end feeding an identical
post-net) but with a CLASSICAL front-end in place of the VQC. Comparing the two
therefore isolates the effect of a *quantum* vs *classical* front-end on the SAME
downstream post-net --- removing the confound in the earlier QAPINN-vs-pure-PINN
comparison (which differed in post-net structure). ``width`` is set to the QAPINN's
measurement count so the post-net input dimension matches exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.pinn import PINNMLP


class ClassicalFrontEnd(nn.Module):
    def __init__(self, width: int = 4, post_hidden: list[int] | None = None,
                 in_dim: int = 2, out_dim: int = 1):
        super().__init__()
        post_hidden = [16] if post_hidden is None else list(post_hidden)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.front = nn.Linear(in_dim, width)          # classical analogue of the VQC
        nn.init.xavier_normal_(self.front.weight); nn.init.zeros_(self.front.bias)
        self.act = nn.Tanh()
        self.post = PINNMLP([width, *post_hidden, out_dim])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        f = self.act(self.front(X))                    # front-end features (batch, width)
        return self.post(f)

    def frontend_features(self, X: torch.Tensor) -> torch.Tensor:
        """The features the front-end hands to the post-net (for offloading analysis)."""
        return self.act(self.front(X))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict:
        return {"type": "cfe", "width": self.width,
                "post": [self.width, *[l.out_features for l in self.post.linears]],
                "num_params": self.num_parameters()}


def build_cfe_from_config(cfg: dict) -> "ClassicalFrontEnd":
    return ClassicalFrontEnd(
        width=cfg.get("width", 4),
        post_hidden=cfg.get("post_hidden", [16]),
        in_dim=cfg.get("in_dim", 2),
        out_dim=cfg.get("out_dim", 1),
    )
