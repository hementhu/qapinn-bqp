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

---

## Outstanding items to own before submission
- [ ] Re-derive the Cole–Hopf reference solution independently.
- [ ] Re-derive the PDE-residual / autograd formulation for each PDE.
- [ ] Re-derive the VQC-as-Fourier-series result (Schuld [4]) to the level of being able to explain it on a whiteboard.
- [ ] Independently verify every citation used in `literature_review.md`.
- [ ] Re-run and confirm every headline number personally; cut anything not defensible.
- [ ] Write the final AI-usage disclosure (distilled from this log) for the submission.
