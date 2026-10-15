"""Build the master experiment table: every run, grouped by family, with the
authoritative rel-L2 / params from summary.json plus a status/observation column
that reflects the post-audit understanding. Writes report/master_results.md.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path("results")

def s(name):
    p = R / name / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    c = d.get("config", {}); a = d.get("architecture", {}); v = a.get("vqc", {}) or {}
    return dict(rel=d["rel_l2"], par=d["num_params"], pde=c.get("pde", "?"),
                adam=c.get("adam_iters"), lbfgs=c.get("lbfgs_iters"), seed=c.get("seed"),
                model=a.get("type", c.get("pde") and "?"),
                nq=v.get("n_qubits"), nl=v.get("n_layers"), ent=v.get("entanglement"),
                meas=v.get("measurement"), scale=v.get("input_scale"))

def row(name, note=""):
    d = s(name)
    if not d:
        return f"| {name} | — | — | — | *missing* | {note} |"
    knob = []
    if d["nl"] is not None: knob.append(f"L{d['nl']}")
    if d["nq"] is not None: knob.append(f"{d['nq']}q")
    if d["ent"]: knob.append(d["ent"])
    if d["meas"] and d["meas"] != "multi": knob.append(d["meas"])
    if d["scale"] not in (None, 1.0): knob.append(f"s{d['scale']}")
    knob = ",".join(knob) or "—"
    tr = f"A{d['adam']}" + (f"+L{d['lbfgs']}" if d["lbfgs"] else "")
    return f"| {name} | {d['model']} | {d['par']} | {knob} | {tr} | **{d['rel']:.3f}** | {note} |"

def mean(names):
    vals = [s(n)["rel"] for n in names if s(n)]
    return (np.mean(vals), np.std(vals, ddof=1), len(vals)) if vals else (float("nan"), 0, 0)

HDR = "| run | model | params | config | train | rel-L2 | note |\n|---|---|---|---|---|---|---|"

out = ["# QAPINN-BQP — Master Experiment Table",
       "",
       "Authoritative `rel_l2` / `num_params` from each run's `summary.json`. "
       "`train` = A<Adam iters>(+L<L-BFGS iters>). Status reflects the **post-audit** "
       "understanding (some pre-audit claims were undertrained or mis-framed — see notes).",
       ""]

# ---- 0. CONVERGED CORE (keystone) ----
m_cls = mean(["cv_cls_s0","cv_cls_s1","cv_cls_s2"])
m_cfe = mean(["cv_cfe_s0","cv_cfe_s1","cv_cfe_s2"])
m_d3l = mean(["cv_qd3_lin_s0","cv_qd3_lin_s1","cv_qd3_lin_s2"])
m_d3n = mean(["cv_qd3_none_s0","cv_qd3_none_s1","cv_qd3_none_s2"])
m_d5  = mean(["cv_qd5_lin_s0","cv_qd5_lin_s1","cv_qd5_lin_s2"])
out += ["## 0. Converged core — THE KEYSTONE (L-BFGS on, 3 seeds each)",
        "*The audit found every earlier Burgers run was undertrained (L-BFGS off → all ~0.47). "
        "These re-runs are converged, so comparisons are between resolved models.*",
        "",
        "| config | model | params | rel-L2 (mean ± sample sd, n=3) | finding |",
        "|---|---|---|---|---|",
        f"| cv_cls | classical | 121 | **{m_cls[0]:.3f} ± {m_cls[1]:.3f}** | **best of all** — classical wins |",
        f"| cv_qd5_lin | QAPINN | 137 | {m_d5[0]:.3f} ± {m_d5[1]:.3f} | more depth helps; only ties CFE |",
        f"| cv_cfe | CFE control | 109 | {m_cfe[0]:.3f} ± {m_cfe[1]:.3f} | classical front-end control |",
        f"| cv_qd3_none | QAPINN | 121 | {m_d3n[0]:.3f} ± {m_d3n[1]:.3f} | n=3 gap did not survive n=6 (see below) |",
        f"| cv_qd3_lin | QAPINN | 121 | {m_d3l[0]:.3f} ± {m_d3l[1]:.3f} | reference QAPINN — worst |",
        "",
        f"**No quantum advantage (small model):** classical ({m_cls[0]:.3f}) beats every QAPINN; "
        f"best QAPINN (depth-5, {m_d5[0]:.3f}) only reaches the CFE control ({m_cfe[0]:.3f}).  \n"
        f"**Entanglement effect NOT robust:** at n=3 none ({m_d3n[0]:.3f}) beat linear ({m_d3l[0]:.3f}) "
        "with Student p≈0.02 (Welch p=0.06) -- and this DEFLATED at n=6 (Welch p=0.42). A lucky 3-seed draw; retracted. "
        "See §0b for the larger-model test.",
        ""]

def bigmean(base):
    import numpy as np
    v = [s(f"{base}_s{i}")["rel"] for i in range(9) if s(f"{base}_s{i}")]
    return (np.mean(v), np.std(v, ddof=1), len(v)) if v else (float("nan"), 0, 0)

def section(title, blurb, rows):
    out.append(f"## {title}")
    if blurb: out.append(f"*{blurb}*"); out.append("")
    out.append(HDR)
    out.extend(rows)
    out.append("")

# ---- 0b. LARGER-MODEL regime + fair controls (THE FINAL TEST) ----
_st = json.load(open("report/stats.json"))["comparisons"]
P = {c["b"]: c["welch_p"] for c in _st if c["a"] == "big_qd3_none"}
bm = {b: bigmean(b) for b in ["big_qd3_none","big_qd3_lin","big_cfe","big_cls_deep","big_cls_bal","big_cls"]}
out += ["## 0b. Larger-model regime (~1273 params, L-BFGS on) + FAIR controls -- THE DECIDER",
        "*The small-model (121p) comparison was between failing models. This repeats it at a "
        "resolved scale, AND against fair controls (structural CFE + a depth-matched classical). "
        "Result: the apparent quantum advantage over a shallow classical net does NOT survive a "
        "fairly-chosen (deeper) classical baseline.*",
        "",
        "| config | model | rel-L2 (mean ± sample sd) | vs QAPINN-none (Welch p, report/stats.json) |",
        "|---|---|---|---|",
        f"| big_qd3_none | QAPINN (no ent) | **{bm['big_qd3_none'][0]:.3f} ± {bm['big_qd3_none'][1]:.3f}** (n={bm['big_qd3_none'][2]}) | best mean (see caveats) |",
        f"| big_qd3_lin | QAPINN (linear) | {bm['big_qd3_lin'][0]:.3f} ± {bm['big_qd3_lin'][1]:.3f} (n={bm['big_qd3_lin'][2]}) | p={P['big_qd3_lin']:.2f} (n.s.) |",
        f"| big_cls_deep | classical [2,22,22,22,1], 1101p | {bm['big_cls_deep'][0]:.3f} ± {bm['big_cls_deep'][1]:.3f} (n={bm['big_cls_deep'][2]}) | **p={P['big_cls_deep']:.2f} (n.s.) -- ties QAPINN** |",
        f"| big_cfe | structural control | {bm['big_cfe'][0]:.3f} ± {bm['big_cfe'][1]:.3f} (n={bm['big_cfe'][2]}) | **p={P['big_cfe']:.2f} (n.s.) -- ties QAPINN** |",
        f"| big_cls_bal | classical [2,32,32,1] | {bm['big_cls_bal'][0]:.3f} ± {bm['big_cls_bal'][1]:.3f} (n={bm['big_cls_bal'][2]}) | p={P['big_cls_bal']:.3f} |",
        f"| big_cls | classical [2,16,68,1] shallow | {bm['big_cls'][0]:.3f} ± {bm['big_cls'][1]:.3f} (n={bm['big_cls'][2]}) | p={P['big_cls']:.3f} |",
        "",
        f"**Verdict:** QAPINN beats a *shallow* classical (p={P['big_cls']:.3f}) but is *statistically "
        f"indistinguishable* from BOTH fair controls -- the structural CFE (p={P['big_cfe']:.2f}) and a "
        f"depth-matched classical net (p={P['big_cls_deep']:.2f}). The gap over shallow classical is ARCHITECTURAL "
        "(depth/post-net width), reproducible classically -- NOT quantum-specific.",
        "",
        "*Integrity notes (from the final audit): (i) all controls are trained under the QAPINN's "
        "EXACT protocol (lr 0.01, n_con 150, Adam 3000 + L-BFGS 600) -- an earlier control-training "
        "confound was found and fixed. (ii) The tying net has FEWER params (1101 vs 1273), which "
        "favours the classical -- strengthening the no-advantage reading. (iii) The result is "
        "UNDERPOWERED: seed variance (~0.08) exceeds the ~0.03 effect, same-seed L-BFGS reruns are "
        "non-deterministic, and one qd3_none seed (0.243) is anomalous (dropping it keeps the tie, "
        "Welch p=0.17). We state the comparison as 'no significant advantage', not a proven equality.*",
        ""]

section("1. Base comparison (matched 121 params, undertrained A2000)",
    "Heat (smooth, single mode) fits well; Burgers (shock) does not, for either model.",
    [row("heat_classical","near-exact; single k=1 mode"),
     row("heat_qapinn","near-exact; ties classical"),
     row("burgers_classical","shock rounded, amp capped ~0.7"),
     row("burgers_qapinn","≈ classical failure; no advantage")])

section("2. Tier-1: re-upload depth (undertrained, single seed)",
    "Depth improves the fit — but at A2000 the FULL model still caps at ~k3; the improvement is "
    "partly optimisation, not a hard band limit (audit).",
    [row("abl_depth_1","weakest; least entanglement"),
     row("abl_depth_2","≈ depth-1 (Δ within seed noise)"),
     row("abl_baseline","depth-3 reference (=121)"),
     row("abl_depth_5","best of sweep; error band narrower")])

section("3. Tier-1: entanglement (band fixed, undertrained, single seed)",
    "At A2000 accuracy barely moves while entropy swings — LOOKED like a null; convergence later "
    "at n=3 looked like entanglement hurts, but this DEFLATED at n=6 (Welch p=0.42) -- retracted; see §0/§0b).",
    [row("abl_ent_none","entropy ~0 (validates sanity check)"),
     row("abl_baseline","linear (reference)"),
     row("abl_ent_ring","higher entropy, ~same accuracy"),
     row("abl_ent_all_to_all","highest entropy, worst extrapolation")])

section("4. Tier-1: post-net width (undertrained, single seed)",
    "VQC needs a small readout; beats matched classical for post ≥ [8] (at this undertrained budget).",
    [row("abl_post_lin","[] linear readout — weakest"),
     row("abl_post_8","[8]"), row("abl_baseline","[16] reference"),
     row("abl_post_32","[32] — no further gain")])

section("5. Tier-3: qubit count (undertrained, single seed)",
    "Behaves like depth (bandwidth). Amplitude saturates ~5q while entanglement entropy grows "
    "monotonically → diminishing returns.",
    [row("abl_qubits_3","3q"), row("abl_baseline","4q reference"),
     row("abl_qubits_5","5q"), row("abl_qubits_6","6q best")])

section("6. Tier-3: measurement (undertrained, single seed)",
    "Readout width matters (single-Z bottlenecks); correlator type does not (multi = ZZ).",
    [row("abl_meas_single","single-Z — weakest"),
     row("abl_baseline","multi-Z reference"),
     row("abl_meas_zz","ZZ ≈ multi")])

section("7. Tier-3: encoding input scale (undertrained, single seed)",
    "Optimum near 1; scale=2π ALIASES the encoding → collapse to trivial u≈0 (black-hole). Verified: "
    "max harmonic jumps to Nyquist at 2π.",
    [row("abl_scale_05","0.5 — under-covers, over-smooth"),
     row("abl_baseline","1.0 reference"),
     row("abl_scale_pi","π — onset of noise"),
     row("abl_scale_2pi","2π — **collapse (rel-L2 1.0)**")])

section("8. Matched classical controls (undertrained, single seed)",
    "Param-matched tanh MLPs. Shock quality is NON-monotonic in params (165>217): capacity is NOT the "
    "limiter — it's spectral bias.",
    [row(n) for n in ["abl_cls_29","abl_cls_73","abl_cls_99","abl_cls_105","abl_cls_113",
                       "abl_cls_121","abl_cls_137","abl_cls_143","abl_cls_165","abl_cls_169","abl_cls_217"]])

section("9. CFE fair controls (undertrained, single seed)",
    "Identical post-net fed by a classical Linear+tanh front end. Ties QAPINN on Burgers → deflated the "
    "'quantum diversifies neurons' claim.",
    [row("cfe_burgers","≈ qapinn (no advantage)"), row("cfe_heat","near-exact"),
     row("cfe_q3"), row("cfe_q5"), row("cfe_q6")])

section("10. Convergence probes (8000 iter, single seed)",
    "An n=3 signal suggested entanglement hurts here; it did NOT survive seed expansion (n=6, Welch p=0.42) -- retracted.",
    [row("conv_cls_121","classical, long train"),
     row("conv_qapinn_base","entangled — stays band-limited"),
     row("conv_qapinn_entnone","no-ent — crisp shock, matches k1..20")])

# 11. seeds
seed_rows = []
for e in ["none","linear","ring","all_to_all"]:
    names = ([f"abl_ent_{e}"] if e!="linear" else ["abl_baseline"]) + [f"seed{i}_ent_{e}" for i in range(1,7)]
    mu,sd,n = mean(names)
    seed_rows.append(f"| ent={e} | QAPINN | 121 | {n} seeds | A2000 | **{mu:.3f} ± {sd:.3f}** | pooled |")
mu,sd,n = mean(["abl_cls_121"]+[f"seed{i}_cls_121" for i in range(1,7)])
seed_rows.append(f"| classical | classical | 121 | {n} seeds | A2000 | **{mu:.3f} ± {sd:.3f}** | control |")
out += ["## 11. Seed expansion — entanglement across seeds (undertrained A2000)",
        "*At the undertrained budget the entanglement differences are within seed noise "
        "(spread < 2×std). Convergence (§0) is where the effect appears.*",
        "",
        "| config | model | params | n | train | rel-L2 (mean ± std) | note |",
        "|---|---|---|---|---|---|---|", *seed_rows, ""]

# 12. nu-sweep
section("12. ν-sweep — bandwidth continuum (undertrained, single seed/point)",
    "Error rises monotonically as ν falls (bandwidth grows). Advantage is V-shaped, NOT a clean collapse "
    "— depth-3 QAPINN loses to matched classical across the family. (Figures fixed for a per-ν reference bug.)",
    [row("nu_cls_01","ν=0.1 smooth — easiest"), row("nu_qapinn_01"),
     row("nu_cls_005"), row("nu_qapinn_005"),
     row("nu_cls_002"), row("nu_qapinn_002"),
     row("nu_cls_001","ν=0.01 sharp — hardest classical"), row("nu_qapinn_001","hardest QAPINN")])

# summary of status
out += ["## Status ledger — what survived the audit",
        "",
        "| claim | status | why |",
        "|---|---|---|",
        "| VQC features = Fourier, band = depth | **robust** | proven + verified L=1..6 (isolated qubit, verify_claims.py) |",
        "| input_scale 2π collapse = aliasing | **robust** | verified (max harmonic → Nyquist) |",
        "| Heat (in-band) fits; sharp Burgers doesn't at small budget | **robust** | consistent everywhere |",
        "| Entanglement hurts at convergence | **REFUTED at n=6** | Student p=0.02 / Welch p=0.06 at n=3 -> p=0.42 at n=6 (lucky draw) |",
        "| No quantum advantage vs FAIR classical baseline | **robust** | depth-matched classical ties QAPINN (p=0.51, §0b) |",
        "| Apparent large-model advantage is architectural | **robust** | reproducible by a deeper classical net (§0b) |",
        "| Capacity fixes the shock | refuted | classical error non-monotonic in params (§8) |",
        "| 'band = depth' limits the FULL model | **soft limit, scale-dependent** | small depth-3 QAPINNs mostly stay at K99=2-6 (ref 13) and error tracks K99 (rho=-0.94); every 1273p model reaches K99 11-14 (verify_claims.py) |",
        "| Depth 'monotonically' helps | overstated | single-seed; some steps within noise |",
        "| Barren-plateau in qubit count | **refuted** | non-monotone + measured wrong gradient |",
        "| Quantum diversifies classical neurons | refuted | died under CFE control (§9) |",
        "| Entanglement 'not significant' | **confirmed** | n.s. at A2000 AND at convergence n=6 (p=0.42) |",
        ""]

Path("report/master_results.md").write_text("\n".join(out), encoding="utf-8")
print("wrote report/master_results.md")
print(f"converged: cls={m_cls[0]:.3f} cfe={m_cfe[0]:.3f} d5={m_d5[0]:.3f} d3none={m_d3n[0]:.3f} d3lin={m_d3l[0]:.3f}")
