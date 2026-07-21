# XAI Methods — what each analysis module computes, and how to read it

Plain-language reference for the explainability suite in `src/analysis/`. Use it
to verify each method does what it claims and to defend the results. Every method
runs identically on the classical PINN and the QAPINN, so the two are comparable.

The suite has **5 static dimensions** (post-hoc, from a trained model/checkpoint)
plus **dynamic monitoring** (recorded during training).

---

## Dynamic monitoring — `monitor.py`, `probes.py` (+ trainer hook)

**What:** every `probe_every` iterations the trainer runs small probes and stores
their output as a time-series in `<run>/monitor.npz`; it also saves model
checkpoints. This captures quantities that only exist *during* training.

- **`spectrum_probe`** — evaluates the model on a fixed grid, takes the spatial
  FFT of the output, records the magnitude per harmonic k. → how the frequency
  content builds up over training.
- **`grad_variance_probe`** — variance of `dLoss/dtheta` for each parameter
  tensor (run right after `backward()`, grads live). → trainability signal; a
  collapsing variance flags barren-plateau / black-hole trouble.

**Read it:** load the `.npz`; each series is `<name>_iters` (T,) and
`<name>_values` (T, dim).

---

## Dimension 1 — Spectral: `spectrum.py`

**What functions the model builds, and *when*.**

- **`realized_spectrum(model, spec)`** → the trained field's per-harmonic
  magnitude vs the reference's. On x∈[-1,1] the rfft index k IS the k-th harmonic,
  so it lines up with the VQC's accessible frequencies {-L..L}. *Look for:* does
  the model capture the reference's high-k tail (Burgers shock) or only low k?
- **`spectral_error`** → per-harmonic |model − reference|. Where the model fails
  in frequency space.
- **`fprinciple_convergence(monitor, spec)`** → from the training spectrum series,
  the iteration at which each harmonic's *relative error* halves. **Low harmonics
  halve before high ones = the F-principle.** *The key question:* does the quantum
  layer change this order? (`plot_fprinciple` draws the error-reduction heatmap.)

**Why it matters:** this is the mechanistic core — it tests whether behaviour is
governed by the frequency band.

---

## Dimension 2 — Neurons: `neurons.py`

**What each classical neuron encodes, with vs without the quantum layer (Fig 3).**

- **`activation_fields(model, spec)`** → each hidden neuron's activation over the
  whole (x,t) domain (its "receptive field"), via forward hooks. Works on the PINN
  (its hidden layers) and the QAPINN (its classical post-net).
- **`neuron_spectra(field)`** → the dominant spatial frequency each neuron
  specializes in. *Look for:* do QAPINN neurons carry higher/different frequencies?
- **`effective_rank(field)`** → entropy-based count of *distinct* features in a
  layer (SVD of activations). *Look for:* higher effective rank = more diverse
  features per neuron → mechanistic support for parameter reduction.
- Plots: receptive-field grid; per-neuron dominant-frequency histogram (classical
  vs QAPINN).

---

## Dimension 3 — VQC internals: `vqc_internals.py`

**Open up the quantum layer itself (quantum-specific; the literature skips this).**

- **`qubit_z_fields(qapinn, spec)`** → each qubit's ⟨Z⟩ as a function of (x,t) —
  the "receptive field" of a qubit; what feature each qubit computes.
- **`entanglement_field(qapinn, spec)`** → von Neumann entropy of a qubit
  bipartition across the domain (looped per point; coarse grid). *Validated:*
  `entanglement=none` gives ≈0 entropy. *Look for:* where the state is entangled,
  and whether more entanglement correlates with more downstream feature diversity
  → the direct handle on "what entanglement buys".
- **`qubit_dominant_freq`** → the frequency each qubit specializes in.

Relies on `QuantumLayer.per_qubit_z` and `.entanglement_entropy` (analysis-only
views that reuse the exact trained circuit via the shared `_apply_gates`).

---

## Dimension 4 — Loss landscape: `landscape.py`

**The optimization geometry.**

- **`loss_landscape(model, cfg)`** → 2-D slice of the training loss around the
  trained minimum along two random, **filter-normalized** directions (each
  direction block scaled to its weight-tensor norm, Li et al. 2018 — makes it
  scale-invariant and comparable across models). Model perturbed in place, then
  restored. *Validated:* the center (0,0) is the minimum. *Look for:* flat/wide
  (good, generalizes) vs sharp minima, classical vs QAPINN.

---

## Dimension 5 — Generalization: `generalization.py`

**Behaviour outside the training domain.**

- **`generalization_curve(model, spec, t_max)`** → relative L2 vs t on
  [t0, t_max], scored against `spec.exact` (the pointwise exact solution, valid
  beyond the training window). The model was trained on t∈[0,1]; error past t=1 is
  the extrapolation penalty. *Look for:* how fast error grows off-domain, and
  whether the QAPINN's Fourier prior extrapolates (quasi-)periodically —
  differently from the classical net.

---

## How this maps to the challenge
- Task 2 (XAI: how learning happens) → dimensions 1,2,4 + dynamic monitoring.
- Task 3 (how the quantum layer affects classical learning) → dimensions 2,3.
- "What is gained by qubits/entanglement/measurement" → dimension 3 + ablations.
- Generalization metric → dimension 5.
- Mathematical rigour → spectral (1) grounded in the Schuld Fourier result.
