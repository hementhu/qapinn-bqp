# AI-Assistant Usage Log

**Purpose:** Transparent record of how AI tools were used on this project, per the WISER/BQP challenge requirement:
> *"Use of AI or AI coding assistants is not advised; however, teams must clearly document how such tools were used and must be able to explain and defend all submitted work as their own."*

This log is maintained continuously and will be distilled into the AI-usage disclosure in the final submission.

**Tool used:** Claude Code (Anthropic), model Claude Opus 4.8.
**Human lead:** Hemanth — owns all scientific decisions, must be able to explain and defend every result, derivation, and design choice.

---

## Guiding principles (how we draw the line)

- **The human drives the science.** Research questions, hypotheses, scope, architecture choices, and go/no-go decisions are made by the team, not the tool. (The methodology was reached through extended human-led Socratic questioning that repeatedly rejected and tightened AI-proposed framings — see log.)
- **AI is used for:** boilerplate/scaffolding, implementing well-specified numerical routines, literature search/summarization, drafting docs, and acting as a sounding board.
- **AI is NOT used to:** decide what the contribution is, choose the hypothesis, or produce results the team cannot independently explain and verify.
- **Before submission:** load-bearing code and all mathematical arguments must be re-derived, understood, and owned by the team; any result that cannot be defended is cut.

---

## Running log

### Phase 0 — Project setup & context (early July 2026)
- **AI:** Read the team's project brief (CLAUDE.md); flagged that the existing NS/LDC notebook was a broken heat-equation-derived skeleton, not a working LDC PINN; corrected the project status.
- **Human:** Set the overall strategy, timeline, and constraints (in CLAUDE.md, authored by the team).

### Phase 1 — Repo & classical baseline (early July 2026)
- **AI:** Scaffolded the `qapinn-bqp/` repo structure; implemented seeding/logging utilities; implemented the **Cole–Hopf exact reference** for Burgers (Gauss–Hermite quadrature) and validated it to machine precision; implemented the classical Burgers PINN (MLP + PDE residual via autograd), trainer (Adam + L-BFGS), collocation sampling, and plotting.
- **Result:** classical Burgers PINN, rel-L2 2.2e-3, reproducible.
- **Human to own:** the Cole–Hopf derivation and the PINN residual/autograd formulation must be re-derived and understood before submission.

### Phase 2 — Quantum framework (early July 2026)
- **AI:** Implemented the VQC layer in PennyLane (encoding, re-uploading, entanglement, measurement knobs); verified second-order autograd (`u_xx`) through the circuit; built model variants (full-quantum, hybrid) and a config-driven run/compare framework; ran a first matched-parameter comparison.
- **Result (preliminary):** at matched params, a hybrid QAPINN (~237 params) reached rel-L2 0.199 vs an equal-capacity classical control (0.404); full-quantum collapsed to a trivial solution.
- **Note:** these prototype architectures were later **retired** in favour of the ref [3] canonical QAPINN (see Phase 4).

### Phase 3 — Literature review (2026-07-16)
- **AI:** Ran a multi-source deep-research pass (search → fetch → adversarial verification of claims); drafted `literature_review.md` with confidence-tagged, cited findings and explicitly recorded refuted claims.
- **Human to own:** verify the key citations firsthand (Schuld–Sweke–Meyer, Kyriienko et al., Mhiri et al., Shah et al.) and confirm the claims we build on.

### Phase 4 — Research design / methodology (2026-07-16)
- **Human (lead):** Through extended questioning, rejected AI's initial framings as too obvious / too detached from reality / question-begging; forced the thesis to be reframed multiple times:
  - rejected "bandwidth is trivially depth" as obvious,
  - rejected the "point at other suspects" disproof as fishing,
  - rejected a contrived "Fourier-null" bench as too far from reality,
  - rejected fixating on Fourier as *the* mechanism (it could be entanglement/landscape),
  - re-centred the project on **explainability** (the challenge's actual ask) after AI drifted into an "advantage-proof" framing,
  - required the mechanism to be an *open* attribution (band vs entanglement/coefficients vs landscape), with Fourier as analysis language, not assumed answer.
- **AI:** Verified (via web search) that the Fourier picture is the field-standard framing for VQC expressivity; drafted `methodology.md` reflecting the human-decided framing.
- **Ownership:** the research questions, hypotheses, and scope in `methodology.md` are the team's, arrived at by the team's reasoning; AI drafted the prose.

### Phase 5 — Test-bench revamp + XAI suite (2026-07-16)
- **Human (lead):** directed a clean-slate revamp of the codebase into a
  systematic test bench; approved the PDE-agnostic interface design; confirmed the
  ref [3] QAPINN architecture (VQC as first hidden layer, no pre-net); chose to
  build the full 5-dimension XAI suite. Elected to defer detailed review of the
  analysis modules to a final verification pass (this log + `xai_methods.md` are
  the review aids).
- **AI:** removed retired prototype code; built a PDE-agnostic core (base.py
  Constraint/PDESpec, PDE-agnostic collocation/losses/trainer), ported Burgers,
  added the heat equation (validated: residual of exact solution ~1e-8), rebuilt
  the QAPINN to ref [3], and implemented the XAI suite: dynamic monitoring
  (probes + Monitor + checkpoints) and 5 analysis modules (spectrum/F-principle,
  neurons, VQC internals incl. entanglement entropy, loss landscape,
  generalization). Each module tested on real runs; `xai_methods.md` documents
  what each computes and how to read it.
- **Human to own:** the meaning of each XAI method and the interpretation of every
  figure used in the report (see `xai_methods.md` and the verification checklist
  below); re-run and confirm each analysis personally.

### Phase 6 — Pipeline verification (2026-07)
- **AI:** built `verify_pipeline.py` — 8 independent correctness checks (Burgers/heat
  residuals vs analytic, autograd vs finite differences, VQC-as-Fourier band = depth,
  relative-L2 metric, matched params, reproducibility). All 8 pass.
- **Human to own:** confirm each check tests what it claims.

### Phase 7 — Ablations (2026-07)
- **AI:** ran base comparison (heat/Burgers × classical/QAPINN), Tier-1 (depth,
  entanglement, post-net), Tier-3 (qubits, measurement, encoding scale), matched
  classical controls, and a ν-sweep — all single-seed, config-driven, seeded.
- **Result:** initial (single-seed) readings suggested a "bandwidth-gated advantage."

### Phase 8 — Classical-side analysis & the CFE deflation (2026-07)
- **Human (lead):** pushed for a *structurally fair* control, not a plain-PINN baseline.
- **AI:** built the CFE control (classical front end + identical post-net); it
  **refuted** the "quantum diversifies the classical neurons" claim (inconsistent
  across knobs; the earlier effect was a mismatched-control artifact).

### Phase 9 — Convergence, seeds, ν-sweep (2026-07)
- **AI:** long-training runs; 7-seed entanglement statistics (split across two laptops);
  the ν-sweep aggregation. Found the entanglement differences within seed noise at the
  undertrained budget.

### Phase 10 — Documentation (2026-07)
- **AI:** drafted the two reports (submission + extended), the 500-page lab journal,
  and the master results table; **Human:** directed scope, framing, and what to include.

### Phase 11 — First adversarial self-audit (2026-07)
- **Human (lead):** ordered a "hard, tough, fresh-eyes" audit of everything.
- **AI:** ran multi-agent adversarial review (code correctness, experimental design,
  statistics, number consistency, blind spots) and its own verification. It caught real
  problems: **undertraining** (L-BFGS was disabled — every Burgers model stuck at
  ~0.47), a **figure-reference bug** (ν-sweep plots used the wrong reference — fixed), a
  **category error** ("band = depth" applies to the VQC features, not the post-net-
  filtered model — a depth-3 model reaches k≈22), and an unsupported **barren-plateau**
  claim (wrong gradient measured, non-monotone). All fixed/retracted.
- **Ownership:** the audit's findings and the decision to retract were reviewed and
  accepted by the human lead.

### Phase 12 — Converged re-run + larger-model test (the keystone, 2026-07/08)
- **AI:** re-ran a converged core (L-BFGS on, multi-seed) and a larger-model regime
  (~1273 params) against structural (CFE) and depth-matched classical controls, split
  across two laptops via Git. Findings: no significant quantum advantage vs a fair
  baseline; an n=3 "entanglement hurts" signal that **deflated at n=6** and was retracted.

### Phase 13 — Final exhaustive audit + mechanism + README (2026-08)
- **Human (lead):** ordered an exhaustive final audit; noted the reports gave *facts*
  but not *why*, and directed a mechanistic account; directed the README/submission prep.
- **AI:** ran a 4-reviewer final audit (fact-check, consistency, integrity, hostile
  reviewer). It caught a **control-training confound** (fair controls accidentally
  trained at 10× lower LR) — **re-run under the correct protocol** (result held and
  strengthened) — plus internal contradictions from piecewise edits (fixed). Wrote the
  *Mechanistic Interpretation* section (each claim tagged measured/reasoned) and this
  README.
- **Ownership:** all retractions, the confound fix, and the final framing were reviewed
  and accepted by the human lead.

---

## Outstanding items to own before defending the submission
Done by the project (but the human lead should still be able to defend each):
- [x] Final AI-usage disclosure written (README §10, distilled from this log).
- [x] Every headline number verified against the raw `summary.json` logs (final audit,
      0 mismatches); non-defensible/over-stated results retracted (3 of them).
- [x] Pipeline correctness independently audited (8/8) and adversarially re-audited twice.

Still the human lead's to own before a live defence:
- [ ] Re-derive the Cole–Hopf reference solution independently.
- [ ] Re-derive the PDE-residual / autograd formulation for each PDE.
- [ ] Re-derive the VQC-as-Fourier-series result (Schuld) to whiteboard level.
- [ ] Independently verify every citation used in `literature_review.md`.
- [ ] Be able to explain each figure and each statistical test in the reports.
