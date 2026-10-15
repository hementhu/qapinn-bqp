"""Model factory: build a classical PINN or the ref [3] QAPINN from a config dict.

Dispatch on the ``model`` field:
    classical -> PINNMLP   (needs ``layers``)
    qapinn    -> QAPINN    (needs ``vqc``; optional ``post_hidden``, in/out dims)
"""
from __future__ import annotations

import torch.nn as nn

from src.models.classical_frontend import build_cfe_from_config
from src.models.fourier_frontend import build_rff_from_config
from src.models.pinn import PINNMLP
from src.models.qapinn import QAPINN
from src.models.vqc import VQCConfig

MODEL_KINDS = ("classical", "qapinn", "cfe", "rff")


def build_model(cfg: dict) -> nn.Module:
    kind = cfg.get("model", "classical")
    if kind == "classical":
        if "layers" not in cfg:
            raise ValueError("classical model requires a 'layers' field")
        return PINNMLP(cfg["layers"])
    if kind == "qapinn":
        return QAPINN(
            VQCConfig(**cfg["vqc"]),
            post_hidden=cfg.get("post_hidden", [16]),
            in_dim=cfg.get("in_dim", 2),
            out_dim=cfg.get("out_dim", 1),
        )
    if kind == "cfe":                       # classical-front-end control
        return build_cfe_from_config(cfg)
    if kind == "rff":                       # random-Fourier-feature control
        return build_rff_from_config(cfg)
    raise ValueError(f"unknown model '{kind}', expected one of {MODEL_KINDS}")


def describe(model: nn.Module) -> dict:
    """Uniform, serializable description of any model (arch if available)."""
    info = {"num_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if hasattr(model, "architecture"):
        info["architecture"] = model.architecture()
    elif hasattr(model, "layers_spec"):
        info["architecture"] = {"type": "classical", "layers": model.layers_spec}
    return info
