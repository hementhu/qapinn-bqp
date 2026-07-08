"""Model factory: build any PINN variant from a parsed YAML config dict.

Dispatch on the config's ``model`` field:
    classical -> PINNMLP              (needs ``layers``)
    quantum   -> QuantumPINN          (needs ``vqc``)           [full quantum]
    hybrid    -> HybridPINN           (needs ``vqc``, pre/post) [classical->VQC->classical]

Also exposes helpers shared by run.py / run_compare.py so every entrypoint
builds and serializes models identically.
"""
from __future__ import annotations

import torch.nn as nn

from src.models.classical_pinn import PINNMLP
from src.models.qapinn import build_qapinn_from_config
from src.models.quantum_pinn import build_quantum_pinn_from_config

MODEL_KINDS = ("classical", "quantum", "hybrid")


def build_model(cfg: dict) -> nn.Module:
    kind = cfg.get("model", "classical")
    if kind == "classical":
        if "layers" not in cfg:
            raise ValueError("classical model requires a 'layers' field")
        return PINNMLP(cfg["layers"])
    if kind == "quantum":
        return build_quantum_pinn_from_config(cfg)
    if kind == "hybrid":
        return build_qapinn_from_config(cfg)
    raise ValueError(f"unknown model '{kind}', expected one of {MODEL_KINDS}")


def describe(model: nn.Module) -> dict:
    """Uniform, serializable description of any model (arch if available)."""
    info = {"num_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if hasattr(model, "architecture"):
        info["architecture"] = model.architecture()
    elif hasattr(model, "layers_spec"):
        info["architecture"] = {"type": "classical", "layers": model.layers_spec}
    return info
