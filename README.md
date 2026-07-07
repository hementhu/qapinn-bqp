# QAPINN-BQP — Quantum-Assisted PINNs for CFD

WISER Summer Program 2026 · BQP Challenge · deadline **Aug 7, 2026**.

Investigating how a variational quantum circuit (VQC) layer changes the learning
behaviour of a physics-informed neural network (PINN) — **not** a claim that
QAPINN beats classical PINN, but *when / why / how* the quantum layer changes
learning, with experimental + mathematical support.

## Status

| Component | State |
|---|---|
| Repo scaffold + seeding/logging utils | ✅ |
| Burgers classical PINN | 🚧 in progress |
| Cole–Hopf reference solution | 🚧 in progress |
| Quantum layer (PennyLane, `interface="torch"`) | ⬜ |
| Ablation matrix (encoding / depth / entanglement / measurement) | ⬜ |
| Fourier-spectrum + barren-plateau analysis | ⬜ |
| LDC (2D Navier–Stokes) QAPINN | ⬜ |

## Setup (Windows / WSL2, CPU-only)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   WSL/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Reproducibility

Every experiment runs from a YAML config + fixed seed. Seeds are logged to each
run's `summary.json`; loss curves to `history.csv`. Re-running the same config
reproduces results to floating-point tolerance.

## Layout

```
src/pdes/        # burgers.py, ldc.py — residuals, BCs, reference solutions
src/models/      # classical_pinn.py, quantum_layer.py, qapinn.py
src/training/    # trainer.py, losses.py, collocation.py
src/analysis/    # fourier_spectrum.py, gradient_variance.py, xai.py
src/utils/       # seeding.py, logging.py, plotting
experiments/     # one YAML per ablation run
results/         # logs, metrics, figures (heavy artifacts gitignored)
notebooks/       # exploration only — nothing load-bearing
report/          # technical report source
```
