# QAPINN-BQP — Quantum-Assisted PINNs for CFD

**WISER Global Quantum+AI Program 2026 · BQP Industry Challenge**
*"Explainability of the quantum layer in a Physics-Informed Neural Network."*

---

## TL;DR (headline finding)

We ask **when, why, and how** replacing a PINN's first hidden layer with a
variational quantum circuit (VQC) changes learning — the challenge's actual ask,
*not* proving the quantum model wins. Using a verified bench, matched + structural +
depth-matched classical controls, multi-seed statistics, and two adversarial
self-audits, we find:

> **No statistically significant quantum-specific advantage** for QAPINNs on 1-D
> viscous Burgers, at either model scale, against a *fairly-chosen* classical
> baseline. Apparent advantages are **architectural** (depth, post-net width) and
> reproducible classically; there is **no robust entanglement effect**.

Crucially, we also explain **why** (see the *Mechanistic Interpretation* section of
the report): the VQC is a fixed Fourier feature map that sits *upstream* of the
classical post-net, which is the real representational bottleneck — so a classical
front end spanning a comparable space does equally well.

The transferable deliverable is a **four-gate falsification protocol** for claimed
QAPINN advantages (converge → fair controls → multi-seed significance →
features-vs-model) that caught **three of our own initial over-claims**.

---

## 1. The challenge / problem

The BQP challenge asks us to explain — with XAI and mathematics — *when, why, and how*
replacing a PINN's first hidden layer with a VQC changes its learning, and to distil a
design methodology for problem-specific quantum circuits. The reference architecture
(Shah et al., IEEE QCE 2024) reports ~20% fewer trainable parameters at maintained
accuracy on 1-D Burgers; our job is to **explain that behaviour**, not reproduce the
headline number.

## 2. Our approach

1. Build a **verified, parameter-matched, PDE-agnostic bench** (Burgers + heat, each
   with an exact reference) so any difference is attributable to the quantum layer, not
   capacity or tooling.
2. Instrument it with a **5-dimension explainability (XAI) suite**.
3. Ablate the VQC knobs and compare against **increasingly strict classical controls**.
4. **Adversarially audit our own results** and retract anything that does not survive
   fair controls, convergence, and multi-seed statistics.

## 3. Methods, tools, and approaches considered

- **Classical:** tanh-MLP PINN (Raissi et al.); Adam + L-BFGS training; exact
  Cole–Hopf reference via Gauss–Hermite quadrature.
- **Quantum:** VQC first layer in **PennyLane** (`default.qubit` statevector,
  `interface="torch"`, `diff_method="backprop"` for 2nd-order autograd through the
  circuit); RY angle re-uploading; entanglement/measurement/qubit/scale ablations.
- **Hybrid:** the reference QAPINN — `(x,t) → VQC → classical post-net → u`.
- **Controls:** parameter-matched classical, a **structural control** (CFE: identical
  post-net fed by a classical front end), and a **depth-matched** classical net.
- **XAI:** spectral / F-principle, neuron receptive fields & effective rank, VQC
  internals (per-qubit ⟨Z⟩, von-Neumann entanglement entropy), loss landscape,
  generalisation.
- **Statistics:** multi-seed means ± std, t-tests, and honest under-powered caveats.
- **Stack:** Python, PyTorch (CPU), PennyLane, NumPy/SciPy, Matplotlib, LaTeX.

## 4. Why we selected our final approach

The literature's QAPINN claims are single-group, non-compute-matched, and rarely
reproduced. We judged the honest, defensible contribution to be a **rigorously
controlled explainability study** rather than another advantage claim. Every time a
positive result appeared, we subjected it to a stricter control — and it dissolved.
That pattern *is* the finding, and the discipline that produced it is the contribution.

## 5. Results / findings

At matched parameters, across two model scales and three control types (n up to 6):

| regime | outcome |
|---|---|
| Small (121p) | classical PINN wins outright |
| Resolved (1273p) | QAPINN beats a *shallow* classical (p=0.003) but **ties** a depth-matched classical (p=0.51) and a structural CFE control (p=0.36) |
| Entanglement | no robust effect (an n=3 p≈0.02 signal **deflated** at n=6, p=0.41 — retracted) |
| Encoding scale 2π | deterministic collapse to the trivial solution (**aliasing**, verified) |
| "Band = depth" | true for the VQC *features*, not the model (tanh post-net reaches k≈22) |

**Mechanistic "why":** the quantum layer is a fixed Fourier feature map upstream of the
classical post-net (the real bottleneck); its useful content is equally available to a
classical front end, and its extra structure (entanglement) is redundant once a
flexible post-net follows. The shock defeats every model via classical spectral bias
(F-principle), which the quantum layer neither causes nor cures.

Full numbers: [`report/master_results.md`](report/master_results.md) (~100 runs).
Full write-up: [`report/qapinn_report.pdf`](report/qapinn_report.pdf) and the extended
version with roadmap.

## 6. Limitations & recommended next steps

- **Under-powered:** the decisive comparisons are *failures to reject* under high seed
  variance (n ≤ 6; same-seed L-BFGS reruns are non-deterministic). More seeds (5–8)
  would firm the nulls.
- **Scope:** 1-D Burgers + heat only; no real CFD (lid-driven cavity is built in
  `src/pdes/cavity.py` but not run); statevector simulation only (no shot noise /
  hardware) — parameter reduction is **not** resource reduction (~85× slower simulated).
- **One reasoned (unmeasured) claim:** the *why* of entanglement neutrality is grounded
  in theory + controls; our direct mechanism probe was inconclusive.
- **Next:** still-larger scale, more seeds, cross-PDE checks, a probability-vector
  readout, and — only if an advantage ever survives all four falsification gates — a
  validated circuit-design methodology.

## 7. Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/mac: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python verify_pipeline.py         # 8/8 correctness checks
python run.py <config>            # single run
python run_converged.py all       # the converged core (L-BFGS on, multi-seed)
python run_followup.py all        # larger-model + fair controls
python make_master_table.py       # regenerate report/master_results.md
```
Every run is driven by a config + fixed seed; metrics/figures land in `results/<run>/`.
Heavy artifacts (`*.pt`, `*.npz`, the 500-page journal PDF) are gitignored; the
converged-core weights are kept for cross-machine analysis.

## 8. Repository layout

```
src/pdes/        base.py (PDESpec/Constraint), burgers.py, heat.py, cavity.py
src/models/      pinn.py, vqc.py, qapinn.py, classical_frontend.py (CFE), factory.py
src/training/    trainer.py, losses.py, collocation.py
src/analysis/    spectrum, neurons, vqc_internals, landscape, generalization, monitor, probes
run*.py          runners (ablation / converged / followup / nusweep / seeds / stability)
verify_pipeline.py   8/8 correctness audit
report/          qapinn_report.pdf (+ extended), master_results.md, literature_review.md,
                 methodology.md, xai_methods.md, mechanism.md, ai_usage_log.md
results/         per-run metrics, configs, figures
```

## 9. Team members & contributions

> **ACTION REQUIRED (fill before submitting):**

- **[Full name] ([email/GitHub])** — *lead:* research questions, hypotheses, scope,
  go/no-go decisions, adversarial framing, verification.
- *[Add any other team members and their specific contributions, or state "solo
  project" if applicable.]*

## 10. AI-assistant usage (disclosure)

Per the challenge's requirement to document AI usage and defend all work as the team's
own: this project used **Claude Code (Anthropic, Claude Opus 4.8)** extensively for
implementation (bench, models, XAI suite, runners), running and aggregating
experiments, literature search, statistical analysis, the adversarial self-audits, and
drafting the documentation and reports. The **human lead directed all scientific
decisions** — research questions, hypotheses, scope, the repeated Socratic reframing of
the thesis, and every go/no-go — and is responsible for understanding and defending the
methods, mathematics, and results. A continuous log is in
[`report/ai_usage_log.md`](report/ai_usage_log.md). Load-bearing derivations
(Cole–Hopf reference, PDE-residual/autograd formulation, the VQC-as-Fourier-series
result) should be re-derived and owned by the team before defending the submission.

## Contact
Questions: [team contact].
