"""Correctness audit of the pipeline — independent checks of every load-bearing
computation. Run: python verify_pipeline.py   (prints PASS/FAIL per check).

Each check validates a computation against an INDEPENDENT ground truth (analytic
derivatives, finite differences, known Fourier theory), not against itself.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import numpy as np
import torch

from src.models.factory import build_model, describe
from src.pdes import burgers, heat
from src.pdes.base import get_pde

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. Burgers residual is computed correctly (autograd vs ANALYTIC residual)
#    Test fn u = -sin(pi x) e^{-t}; compute the Burgers residual analytically and
#    compare to what burgers.residual() produces via autograd.
# --------------------------------------------------------------------------- #
def test_fn(X):
    return -torch.sin(np.pi * X[:, 0:1]) * torch.exp(-X[:, 1:2])

X = torch.rand(50, 2, requires_grad=True)
x, t = X[:, 0:1], X[:, 1:2]
nu = burgers.NU_DEFAULT
u = -torch.sin(np.pi * x) * torch.exp(-t)
u_t = torch.sin(np.pi * x) * torch.exp(-t)
u_x = -np.pi * torch.cos(np.pi * x) * torch.exp(-t)
u_xx = (np.pi**2) * torch.sin(np.pi * x) * torch.exp(-t)
analytic = (u_t + u * u_x - nu * u_xx).detach()
autograd = burgers.residual(test_fn, X).detach()
err = (analytic - autograd).abs().max().item()
check("Burgers residual = analytic (autograd u_t+u·u_x-nu·u_xx)", err < 1e-4, f"max err {err:.1e}")

# 2. Heat residual is correct (exact heat solution -> residual ~ 0)
Xh = torch.rand(50, 2, requires_grad=True)
r_heat = heat.residual(test_fn, Xh).abs().max().item()
check("Heat residual of exact solution ~ 0", r_heat < 1e-4, f"max |r| {r_heat:.1e}")

# 3. Autograd first/second derivatives vs FINITE DIFFERENCES on a known fn
xp = torch.tensor([[0.3, 0.4]], requires_grad=True)
up = test_fn(xp)
g = torch.autograd.grad(up, xp, create_graph=True)[0]
ux_ad = g[0, 0].item()
h = 1e-4
with torch.no_grad():
    fp = test_fn(torch.tensor([[0.3 + h, 0.4]])).item()
    fm = test_fn(torch.tensor([[0.3 - h, 0.4]])).item()
ux_fd = (fp - fm) / (2 * h)
check("Autograd u_x = finite-difference u_x", abs(ux_ad - ux_fd) < 1e-3, f"AD {ux_ad:.4f} vs FD {ux_fd:.4f}")

# --------------------------------------------------------------------------- #
# 4. THE core theory claim: a VQC with L re-uploads is a Fourier series with
#    frequencies bounded by L. 1-qubit controlled test: sweep the input, FFT the
#    output, confirm NO energy above harmonic L.
# --------------------------------------------------------------------------- #
from src.models.vqc import QuantumLayer, VQCConfig

def max_frequency(L):
    torch.manual_seed(1)
    q = QuantumLayer(VQCConfig(n_qubits=1, n_layers=L, reupload=True,
                               entanglement="none", measurement="single", input_scale=1.0))
    N = 128
    s = (torch.arange(N) / N * 2 * np.pi).reshape(-1, 1)   # [0, 2pi) — endpoint EXCLUDED
    z = q.per_qubit_z(s)[:, 0].numpy()
    mag = np.abs(np.fft.rfft(z))
    signif = np.where(mag > 1e-6 * mag.max())[0]
    return signif.max()

f2, f4 = max_frequency(2), max_frequency(4)
check("VQC(L=2) max frequency = 2 (Fourier bandwidth = depth)", f2 == 2, f"max freq {f2}")
check("VQC(L=4) max frequency = 4 (Fourier bandwidth = depth)", f4 == 4, f"max freq {f4}")

# --------------------------------------------------------------------------- #
# 5. relative-L2 metric formula (identical fields -> 0; scaled -> known ratio)
# --------------------------------------------------------------------------- #
from src.training.trainer import relative_l2, TrainConfig

class ExactModel(torch.nn.Module):
    def forward(self, X):
        u = heat.exact(X[:, 0].numpy(), X[:, 1].numpy())
        return torch.tensor(u, dtype=torch.float32)[:, None]

rl = relative_l2(ExactModel(), get_pde("heat"), TrainConfig(pde="heat", eval_nx=64, eval_nt=40))
check("relative_l2 of the exact solution ~ 0", rl < 1e-5, f"rel_L2 {rl:.1e}")

# --------------------------------------------------------------------------- #
# 6. Matched parameters across the 4 experiment configs
# --------------------------------------------------------------------------- #
import yaml
params = {}
for c in ["heat_classical", "heat_qapinn", "burgers_classical", "burgers_qapinn"]:
    raw = yaml.safe_load(open(f"experiments/{c}.yaml"))
    params[c] = describe(build_model(raw))["num_params"]
matched = len(set(params.values())) == 1
check("All 4 experiment configs have matched params", matched, str(params))

# --------------------------------------------------------------------------- #
# 7. Reproducibility: same config+seed -> identical result
# --------------------------------------------------------------------------- #
from src.training.trainer import train
def quick():
    from src.utils.seeding import seed_everything
    seed_everything(0)
    m = build_model({"model": "classical", "layers": [2, 7, 11, 1]})
    return train(m, TrainConfig(pde="heat", n_interior=300, n_constraint=50,
                                adam_iters=100, lbfgs_iters=0, seed=0), logger=None)["rel_l2"]
a, b = quick(), quick()
check("Reproducible: same seed -> identical rel_L2", a == b, f"{a:.6e} == {b:.6e}")

print(f"\n{sum(results)}/{len(results)} checks passed"
      + ("  ✓ ALL CORRECT" if all(results) else "  <-- FAILURES ABOVE").replace("✓", "OK"))
