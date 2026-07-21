# Methodology & Research Plan

**Project:** QAPINN-BQP — WISER Global Quantum+AI Program 2026 · BQP Challenge
**Challenge:** *Explainability of the quantum layer in a Physics-Informed Neural Network*
**Status:** Research design (pre-execution). Companion to [literature_review.md](literature_review.md).
**Date:** 2026-07-16

---

## 1. Problem

The challenge asks us to **explain — with XAI and mathematics — when, why, and how replacing a PINN's first hidden layer with a variational quantum circuit (VQC) changes its learning**, and to distil that into a design methodology for problem-specific quantum circuits. The stated primary goal is *not* to show QAPINNs beat PINNs, but to explain the learning dynamics.

The reference QAPINN (Shah et al. 2024 [3]) is a classical PINN with its **first hidden layer replaced by a VQC**:

```
(x,t) → VQC  [quantum first hidden layer] → classical hidden layers → u(x,t)
```

It was reported to *maintain/improve* accuracy on 1-D viscous Burgers while cutting trainable parameters by up to ~20% (3/4/5-qubit variants). **Our job is to explain that behaviour**, not reproduce the headline number.

### What is already known vs what we contribute
- **Known (not our contribution):** a VQC's output *is* a truncated Fourier series in its input; the accessible frequencies are fixed by the encoding, the coefficients by the trainable gates (Schuld–Sweke–Meyer [4]). Depth/encoding set the band. This is our *analysis language*, not a result we re-prove.
- **Open / our contribution:**
  1. **Explainability (novel):** *how* the quantum first layer reshapes the downstream **classical** neurons, features, and loss landscape — never done for QAPINNs.
  2. **Design methodology (novel):** turning the descriptive Fourier theory into a *predictive* recipe (PDE spectrum → circuit config) for a class of PDEs.
  3. **Mechanism attribution (supporting hinge):** *what* drives the change — the frequency band, the coefficient machinery (entanglement/measurement), or the loss landscape. Load-bearing mainly if it comes out **against** the naive "band explains everything" intuition.

---

## 2. Central questions (the when / why / how)

| Challenge word | Our operational question |
|---|---|
| **HOW** (learning changes) | How do the classical layers' neurons/features and the loss landscape differ *with vs without* the quantum first layer? (XAI) |
| **WHY** (it changes) | Is the change driven by the VQC's **frequency band**, its **coefficients/entanglement**, or the **loss-landscape** it induces? (mechanism attribution) |
| **WHEN** (it helps / hurts) | For which PDEs (spectral content) does the quantum layer help, do nothing, or hurt — and what is gained vs lost? (ablations across a PDE family) |
| **Design** (the deliverable) | Given a *new* PDE, how do we prescribe the circuit (qubits, depth, encoding, entanglement, measurement) from cheap information? (methodology) |

---

## 3. The five work pillars

**P1 — Solve.** Classical PINN and QAPINN (ref [3] architecture) on a PDE family: **heat** (smooth), **Burgers** (shock), and **one real CFD problem** (2-D lid-driven cavity or channel flow — the judging criteria explicitly reward real CFD). Matched trainable-parameter counts so any difference is attributable to the quantum layer, not capacity. *Substrate, not the point.*

**P2 — Explain (XAI) — the headline.** Show *how* the quantum first layer changes the rest of the network:
- **Neuron/feature analysis** in the classical hidden layers, with vs without the quantum layer (their Fig 3): activation distributions, feature spectra, per-neuron contribution.
- **Loss-landscape visualization** (2-D random-direction slices around the minimum), classical vs QAPINN.
- **Learned-function spectrum:** FFT the trained output and the layer features to see what frequency content each model actually builds.

**P3 — Attribute the mechanism (the hinge — open, not assumed).** The VQC output is provably Fourier; the open question is *what part* drives the learning change. Three candidate routes:
- **Band** (expressivity): set by depth/encoding.
- **Coefficients / entanglement** (expressivity, reachability [3-Mhiri]): entanglement + measurement shape which coefficients are realizable.
- **Loss landscape** (trainability): entanglement-driven barren plateaus, trivial-collapse basins — two circuits with identical Fourier power can train very differently.

**Test — fix the band, move the rest:** hold depth/encoding fixed (band fixed), then vary entanglement (none/linear/ring/all-to-all) and measurement (⟨Z⟩ vs ⟨ZZ⟩ vs probability vector).
- Advantage/behaviour **invariant** → the **band** is the driver (Fourier picture predictive).
- Advantage/behaviour **moves** → **coefficients/entanglement** (if expressivity metrics move) or **landscape** (if only trainability metrics move) carry it — a measured, named conclusion, not a guess.

**P4 — Gained vs lost (both, honestly).** Ablate the challenge's named knobs — qubits, entanglement type, measurement (expectation values vs probability vector) — and report:
- **Gained:** parameter reduction (quantify vs ref [3]'s ~20%), accuracy where it holds, generalization.
- **Lost:** training time, memory, trainability failures (trivial collapse), regimes where quantum is *neutral or worse* (e.g. smooth PDEs, cavity flow per literature).
- **Generalization to unseen domains** (a listed metric; ref [1] is on when PINNs generalize): test extrapolation outside the training domain — a Fourier basis extrapolates *periodically*, a concrete, testable prediction.

**P5 — Design methodology — the deliverable.** Convert the mechanism into a *predictive* recipe:
```
new PDE → estimate spectral content (cheap proxy) → required band → prescribe circuit
          (Reynolds/ν, or coarse FEA/classical solve)     (qubits, depth, encoding,
                                                            entanglement, measurement)
```
Resolving the circularity (§5): we validate the recipe against exact solutions on heat/Burgers, then state it for use *without* the exact solution.

---

## 4. Models and the "clean test" note

- **Classical PINN (cPINN):** tanh MLP. Matched-parameter control.
- **QAPINN (ref [3]):** `(x,t) → VQC → classical hidden layers → u`. Direct `(x,t)` encoding (no classical pre-net), VQC as first hidden layer, classical post-net. Retires our earlier prototypes (a pre-net "hybrid" muddied the band; a linear-readout "full-quantum" was untrainable). This canonical form is both **trainable** (post-net gives flexibility) and **Fourier-clean** (no pre-net warps the input, so the band lives directly in `(x,t)`).
- **VQC knobs (ablation axes):** qubits, re-upload depth, encoding, entanglement structure, measurement operator.

---

## 5. Test problems — one spectral axis

Same IC `u(x,0)=sin(πx)`, same domain, same BCs across the 1-D problems, so the *only* difference is the operator → the solution's spectral content:

| PDE | Operator | Exact reference | Spectrum | Role |
|---|---|---|---|---|
| **Heat** | `u_t = α u_xx` | `sin(πx)e^{-t}` (analytic) | single mode k=1 | smooth control |
| **Burgers** | `u_t + u u_x = ν u_xx` | Cole–Hopf (exact) | shock → high harmonics | sharp case |
| **Burgers `ν`-sweep** | vary `ν` | Cole–Hopf per `ν` | tunes bandwidth low→high | the continuum axis |
| **Real CFD** (LDC / channel) | 2-D incompressible NS | Ghia et al. benchmark | broadband, corner singularities | real-world test |

The `ν`-sweep is the controlled bandwidth axis; heat is the narrowband anchor; the CFD problem is the real-world stress test (and, per literature, a candidate "what is lost" case).

---

## 6. Controls, metrics, reproducibility

- **Matched-budget controls:** cPINN vs QAPINN at equal trainable-parameter count, identical collocation points, optimizer, iteration budget, seed. Multiple seeds with error bars on any headline number (a single-seed result is not a claim).
- **Metrics (the challenge's list):** PDE residual error, relative L2 vs exact reference, training time, trainable-parameter count, memory, explainability/interpretability metrics (neuron/feature/landscape), generalization on unseen domains. Wall-clock reported honestly — simulated quantum is ~85× slower; **parameter reduction ≠ resource reduction.**
- **Reproducibility:** every run from a YAML config + fixed seed; full VQC config serialized to `config.json`. Directly addresses the literature's reproducibility gap (every prior QAPINN claim is single-group, non-reproduced, non-compute-matched — see review §8).

---

## 7. What we claim vs what we don't

**We claim (falsifiably):**
- A mechanistic, XAI-grounded account of *how* the quantum first layer changes classical PINN learning.
- Which factor (band / entanglement-coefficients / landscape) drives that change, measured by controlled ablation.
- A validated, proxy-based methodology for sizing a problem-specific circuit — or evidence that no simple rule exists.
- Both what is gained and what is lost, including neutral/worse regimes.

**We do NOT claim:**
- That QAPINN beats classical PINN (explicitly not the goal).
- A new expressivity theorem — Schuld [4] and Mhiri's coefficient constraint are *used*, not re-proven.
- Resource advantage — the ~85× simulation cost is reported.

---

## 8. Mapping to the challenge

| Challenge task / criterion | Pillar |
|---|---|
| Solve PDEs with PINN and QAPINN | P1 |
| XAI: how learning happens in each network | **P2** |
| How the quantum layer affects classical learning (ref [4]) | P2 + P3 |
| Visualize network *and individual neuron* learning | P2 |
| Real-world CFD problems | P1 (CFD case) |
| Math rigour: actual difference in how they learn | P3 |
| Gained by qubits / entanglement / measurement; parameter reduction | P4 |
| What is lost | P4 |
| Ideal quantum config per problem type | **P5** |
| Design methodology + mathematical basis | **P5** |

---

## 9. Integrity note (AI-assistant usage)

The challenge advises against AI coding assistants and requires that all submitted work be documented, understood, and defensible as the team's own. Accordingly: the team drives all scientific decisions and must be able to explain and defend every result and derivation; AI-assisted scaffolding/analysis is documented; load-bearing code and the mathematical arguments are owned and re-derived by the team. (Tracked separately in the submission's AI-usage disclosure.)

---

## References
Full annotated bibliography in [literature_review.md](literature_review.md). Anchors: [1] Hu, Jagtap, Karniadakis, Kawaguchi, *SIAM J. Sci. Comput.* 44(5) (2022) — generalization. [2] Raissi, Perdikaris, Karniadakis (2017/2019) — PINNs. [3] Shah, Lineswala, Chopra, *IEEE QCE* (2024) — the QAPINN we explain. [4] Schuld, Sweke, Meyer, *Phys. Rev. A* 103, 032430 (2021) — VQC-as-Fourier-series. [Mhiri] *Quantum* 9, 1847 (2025) — coefficient constraints.
