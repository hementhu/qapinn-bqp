"""Regenerate the nu-sweep run figures with the CORRECT per-nu reference.

The old run.py built the figure spec with get_pde(cfg.pde) (default nu), so the
nu-sweep solution/spectrum/generalization figures were drawn against the wrong
(default-viscosity) reference. run.py is now fixed; this reruns the figure step
for the already-trained nu runs using their saved model.pt + config.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

from run import _make_figures, build_train_config
from src.models.factory import build_model
from src.pdes.base import get_pde

RESULTS = Path("results")
NU_RUNS = [p.name for p in RESULTS.iterdir()
           if p.is_dir() and p.name.startswith("nu_")]

for name in sorted(NU_RUNS):
    run_dir = RESULTS / name
    if not (run_dir / "model.pt").exists():
        print(f"[{name}] no model.pt yet (run in progress) -> skip")
        continue
    cfgj = json.loads((run_dir / "config.json").read_text())
    raw = cfgj["raw"]
    cfg = build_train_config(raw)
    spec = get_pde(cfg.pde, **cfg.pde_params)            # correct per-nu reference
    model = build_model(raw)
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    print(f"[{name}] nu={cfg.pde_params.get('nu')} -> regenerating figures")
    _make_figures(model, spec, cfg, run_dir)
print("=== nu figures regenerated against correct per-nu reference ===")
