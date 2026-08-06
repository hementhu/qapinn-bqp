# QAPINN-BQP — Master Experiment Table

Authoritative `rel_l2` / `num_params` from each run's `summary.json`. `train` = A<Adam iters>(+L<L-BFGS iters>). Status reflects the **post-audit** understanding (some pre-audit claims were undertrained or mis-framed — see notes).

## 0. Converged core — THE KEYSTONE (L-BFGS on, 3 seeds each)
*The audit found every earlier Burgers run was undertrained (L-BFGS off → all ~0.47). These re-runs are converged, so comparisons are between resolved models.*

| config | model | params | rel-L2 (mean ± std, n=3) | finding |
|---|---|---|---|---|
| cv_cls | classical | 121 | **0.248 ± 0.031** | **best of all** — classical wins |
| cv_qd5_lin | QAPINN | 137 | 0.314 ± 0.073 | more depth helps; only ties CFE |
| cv_cfe | CFE control | 109 | 0.315 ± 0.026 | classical front-end control |
| cv_qd3_none | QAPINN | 121 | 0.369 ± 0.022 | **no-entanglement beats linear** |
| cv_qd3_lin | QAPINN | 121 | 0.424 ± 0.005 | reference QAPINN — worst |

**No quantum advantage (small model):** classical (0.248) beats every QAPINN; best QAPINN (depth-5, 0.314) only reaches the CFE control (0.315).  
**Entanglement effect NOT robust:** at n=3 none (0.369) beat linear (0.424) with p≈0.02 -- but this DEFLATED at n=6 (p=0.41). A lucky 3-seed draw; retracted. See §0b for the larger-model test.

## 0b. Larger-model regime (~1273 params, L-BFGS on) + FAIR controls -- THE DECIDER
*The small-model (121p) comparison was between failing models. This repeats it at a resolved scale, AND against fair controls (structural CFE + a depth-matched classical). Result: the apparent quantum advantage over a shallow classical net does NOT survive a fairly-chosen (deeper) classical baseline.*

| config | model | rel-L2 (mean ± std) | vs QAPINN-none |
|---|---|---|---|
| big_qd3_none | QAPINN (no ent) | **0.094 ± 0.075** (n=6) | best (but see below) |
| big_qd3_lin | QAPINN (linear) | 0.128 ± 0.056 (n=6) | p=0.44 (n.s.) |
| big_cls_deep | classical [2,22,22,22,1] | 0.133 ± 0.047 (n=6) | **p=0.35 (n.s.) -- ties QAPINN** |
| big_cfe | structural control | 0.173 ± 0.048 (n=6) | p=0.076 (marginal) |
| big_cls_bal | classical [2,32,32,1] | 0.231 ± 0.026 (n=3) | p=0.03 |
| big_cls | classical [2,16,68,1] shallow | 0.241 ± 0.037 (n=6) | p=0.003 |

**Verdict:** QAPINN beats a *shallow* classical (p=0.003) but a *depth-matched* classical net ties it (0.133 vs 0.094, p=0.35). The advantage is ARCHITECTURAL (depth/post-net width), reproducible classically -- NOT quantum-specific. Entanglement: none vs linear p=0.44 (n.s.).

## 1. Base comparison (matched 121 params, undertrained A2000)
*Heat (smooth, single mode) fits well; Burgers (shock) does not, for either model.*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| heat_classical | classical | 121 | — | A2000 | **0.008** | near-exact; single k=1 mode |
| heat_qapinn | qapinn | 121 | L3,4q,linear | A2000 | **0.006** | near-exact; ties classical |
| burgers_classical | classical | 121 | — | A2000 | **0.433** | shock rounded, amp capped ~0.7 |
| burgers_qapinn | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | ≈ classical failure; no advantage |

## 2. Tier-1: re-upload depth (undertrained, single seed)
*Depth improves the fit — but at A2000 the FULL model still caps at ~k3; the improvement is partly optimisation, not a hard band limit (audit).*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_depth_1 | qapinn | 105 | L1,4q,linear | A2000 | **0.572** | weakest; least entanglement |
| abl_depth_2 | qapinn | 113 | L2,4q,linear | A2000 | **0.493** | ≈ depth-1 (Δ within seed noise) |
| abl_baseline | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | depth-3 reference (=121) |
| abl_depth_5 | qapinn | 137 | L5,4q,linear | A2000 | **0.424** | best of sweep; error band narrower |

## 3. Tier-1: entanglement (band fixed, undertrained, single seed)
*At A2000 accuracy barely moves while entropy swings — LOOKED like a null; convergence later showed entanglement HURTS (see §0).*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_ent_none | qapinn | 121 | L3,4q,none | A2000 | **0.456** | entropy ~0 (validates sanity check) |
| abl_baseline | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | linear (reference) |
| abl_ent_ring | qapinn | 121 | L3,4q,ring | A2000 | **0.485** | higher entropy, ~same accuracy |
| abl_ent_all_to_all | qapinn | 121 | L3,4q,all_to_all | A2000 | **0.488** | highest entropy, worst extrapolation |

## 4. Tier-1: post-net width (undertrained, single seed)
*VQC needs a small readout; beats matched classical for post ≥ [8] (at this undertrained budget).*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_post_lin | qapinn | 29 | L3,4q,linear | A2000 | **0.544** | [] linear readout — weakest |
| abl_post_8 | qapinn | 73 | L3,4q,linear | A2000 | **0.478** | [8] |
| abl_baseline | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | [16] reference |
| abl_post_32 | qapinn | 217 | L3,4q,linear | A2000 | **0.466** | [32] — no further gain |

## 5. Tier-3: qubit count (undertrained, single seed)
*Behaves like depth (bandwidth). Amplitude saturates ~5q while entanglement entropy grows monotonically → diminishing returns.*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_qubits_3 | qapinn | 99 | L3,3q,linear | A2000 | **0.501** | 3q |
| abl_baseline | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | 4q reference |
| abl_qubits_5 | qapinn | 143 | L3,5q,linear | A2000 | **0.468** | 5q |
| abl_qubits_6 | qapinn | 165 | L3,6q,linear | A2000 | **0.443** | 6q best |

## 6. Tier-3: measurement (undertrained, single seed)
*Readout width matters (single-Z bottlenecks); correlator type does not (multi = ZZ).*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_meas_single | qapinn | 73 | L3,4q,linear,single | A2000 | **0.591** | single-Z — weakest |
| abl_baseline | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | multi-Z reference |
| abl_meas_zz | qapinn | 169 | L3,4q,linear,zz | A2000 | **0.472** | ZZ ≈ multi |

## 7. Tier-3: encoding input scale (undertrained, single seed)
*Optimum near 1; scale=2π ALIASES the encoding → collapse to trivial u≈0 (black-hole). Verified: max harmonic jumps to Nyquist at 2π.*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_scale_05 | qapinn | 121 | L3,4q,linear,s0.5 | A2000 | **0.513** | 0.5 — under-covers, over-smooth |
| abl_baseline | qapinn | 121 | L3,4q,linear | A2000 | **0.472** | 1.0 reference |
| abl_scale_pi | qapinn | 121 | L3,4q,linear,s3.14159 | A2000 | **0.516** | π — onset of noise |
| abl_scale_2pi | qapinn | 121 | L3,4q,linear,s6.28318 | A2000 | **1.000** | 2π — **collapse (rel-L2 1.0)** |

## 8. Matched classical controls (undertrained, single seed)
*Param-matched tanh MLPs. Shock quality is NON-monotonic in params (165>217): capacity is NOT the limiter — it's spectral bias.*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| abl_cls_29 | classical | 29 | — | A2000 | **0.532** |  |
| abl_cls_73 | classical | 73 | — | A2000 | **0.533** |  |
| abl_cls_99 | classical | 99 | — | A2000 | **0.470** |  |
| abl_cls_105 | classical | 105 | — | A2000 | **0.522** |  |
| abl_cls_113 | classical | 113 | — | A2000 | **0.502** |  |
| abl_cls_121 | classical | 121 | — | A2000 | **0.535** |  |
| abl_cls_137 | classical | 137 | — | A2000 | **0.500** |  |
| abl_cls_143 | classical | 143 | — | A2000 | **0.541** |  |
| abl_cls_165 | classical | 165 | — | A2000 | **0.465** |  |
| abl_cls_169 | classical | 169 | — | A2000 | **0.515** |  |
| abl_cls_217 | classical | 217 | — | A2000 | **0.505** |  |

## 9. CFE fair controls (undertrained, single seed)
*Identical post-net fed by a classical Linear+tanh front end. Ties QAPINN on Burgers → deflated the 'quantum diversifies neurons' claim.*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| cfe_burgers | cfe | 109 | — | A2000 | **0.474** | ≈ qapinn (no advantage) |
| cfe_heat | cfe | 109 | — | A2000 | **0.013** | near-exact |
| cfe_q3 | cfe | 90 | — | A2000 | **0.496** |  |
| cfe_q5 | cfe | 128 | — | A2000 | **0.464** |  |
| cfe_q6 | cfe | 147 | — | A2000 | **0.467** |  |

## 10. Convergence probes (8000 iter, single seed)
*First hint that entanglement hurts at convergence (later confirmed with seeds in §0).*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| conv_cls_121 | classical | 121 | — | A8000 | **0.376** | classical, long train |
| conv_qapinn_base | qapinn | 121 | L3,4q,linear | A8000 | **0.437** | entangled — stays band-limited |
| conv_qapinn_entnone | qapinn | 121 | L3,4q,none | A8000 | **0.261** | no-ent — crisp shock, matches k1..20 |

## 11. Seed expansion — entanglement across seeds (undertrained A2000)
*At the undertrained budget the entanglement differences are within seed noise (spread < 2×std). Convergence (§0) is where the effect appears.*

| config | model | params | n | train | rel-L2 (mean ± std) | note |
|---|---|---|---|---|---|---|
| ent=none | QAPINN | 121 | 7 seeds | A2000 | **0.459 ± 0.011** | pooled |
| ent=linear | QAPINN | 121 | 7 seeds | A2000 | **0.448 ± 0.022** | pooled |
| ent=ring | QAPINN | 121 | 7 seeds | A2000 | **0.474 ± 0.014** | pooled |
| ent=all_to_all | QAPINN | 121 | 7 seeds | A2000 | **0.462 ± 0.014** | pooled |
| classical | classical | 121 | 7 seeds | A2000 | **0.445 ± 0.041** | control |

## 12. ν-sweep — bandwidth continuum (undertrained, single seed/point)
*Error rises monotonically as ν falls (bandwidth grows). Advantage is V-shaped, NOT a clean collapse — depth-3 QAPINN loses to matched classical across the family. (Figures fixed for a per-ν reference bug.)*

| run | model | params | config | train | rel-L2 | note |
|---|---|---|---|---|---|---|
| nu_cls_01 | classical | 121 | — | A2000 | **0.018** | ν=0.1 smooth — easiest |
| nu_qapinn_01 | qapinn | 121 | L3,4q,linear | A2000 | **0.049** |  |
| nu_cls_005 | classical | 121 | — | A2000 | **0.032** |  |
| nu_qapinn_005 | qapinn | 121 | L3,4q,linear | A2000 | **0.077** |  |
| nu_cls_002 | classical | 121 | — | A2000 | **0.069** |  |
| nu_qapinn_002 | qapinn | 121 | L3,4q,linear | A2000 | **0.165** |  |
| nu_cls_001 | classical | 121 | — | A2000 | **0.141** | ν=0.01 sharp — hardest classical |
| nu_qapinn_001 | qapinn | 121 | L3,4q,linear | A2000 | **0.354** | hardest QAPINN |

## Status ledger — what survived the audit

| claim | status | why |
|---|---|---|
| VQC features = Fourier, band = depth | **robust** | proven + verified L=1..6 (isolated qubit) |
| input_scale 2π collapse = aliasing | **robust** | verified (max harmonic → Nyquist) |
| Heat (in-band) fits; sharp Burgers doesn't at small budget | **robust** | consistent everywhere |
| Entanglement hurts at convergence | **REFUTED at n=6** | p=0.02 at n=3 -> p=0.41 at n=6 (lucky draw) |
| No quantum advantage vs FAIR classical baseline | **robust** | depth-matched classical ties QAPINN (p=0.35, §0b) |
| Apparent large-model advantage is architectural | **robust** | reproducible by a deeper classical net (§0b) |
| Capacity fixes the shock | refuted | classical error non-monotonic in params (§8) |
| 'band = depth' limits the FULL model | **refuted (category error)** | depth-3 reaches k22 with training |
| Depth 'monotonically' helps | overstated | single-seed; some steps within noise |
| Barren-plateau in qubit count | **refuted** | non-monotone + measured wrong gradient |
| Quantum diversifies classical neurons | refuted | died under CFE control (§9) |
| Entanglement 'not significant' | superseded | true at A2000, but HURTS at convergence |
