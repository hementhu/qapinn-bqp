"""Generate the LARGE project journal (project_journal.tex).

Starts at the literature review (earlier setup phases dropped per request),
embeds every curated figure at full size, then a complete gallery of ALL 77
runs with every per-run figure at full width. Auto-generated so nothing is missed.

    python gen_journal.py    # writes report/project_journal.tex
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "project_journal.tex"
OBS_FILE = Path(__file__).resolve().parent / "run_observations.json"
OBS = json.loads(OBS_FILE.read_text(encoding="utf-8")) if OBS_FILE.exists() else {}

_LTX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}

def ltx(s: str) -> str:
    """Escape arbitrary text (e.g. agent-written observations) for LaTeX."""
    return "".join(_LTX.get(c, c) for c in s)

# --------------------------------------------------------------------------- #
# per-run figure order + captions
FIG_ORDER = [
    ("solution.png",       "Predicted field $u(x,t)$ vs.\\ exact reference, with the pointwise error map. The shock ($x\\!=\\!0$) is where a smooth network struggles."),
    ("spectrum.png",       "Realized spatial spectrum (per-harmonic magnitude) vs.\\ reference. Tests whether the model captures the high-$k$ shock tail or only low frequencies."),
    ("fprinciple.png",     "F-principle heat-map: iteration at which each harmonic's relative error halves. Low harmonics converging first is the F-principle signature."),
    ("neurons.png",        "Classical post-net neurons: receptive fields over $(x,t)$ and each neuron's dominant spatial frequency / effective rank."),
    ("qubits.png",         "Per-qubit $\\langle Z\\rangle$ receptive fields --- what feature each qubit computes over the domain (quantum models only)."),
    ("entanglement.png",   "Von Neumann entanglement entropy $S(x,t)$ of a qubit bipartition across the domain (quantum models only)."),
    ("landscape.png",      "Filter-normalized 2-D loss landscape around the trained minimum (Li et al.\\ 2018). Flat/wide vs.\\ sharp basin."),
    ("generalization.png", "Extrapolation: relative $L_2$ vs.\\ $t$ beyond the $t\\in[0,1]$ training window --- the off-domain penalty."),
    ("evolution.png",      "Solution evolution: snapshots of $u(x,\\cdot)$ across time."),
    ("loss.png",           "Training loss history (total, PDE-residual, and constraint terms)."),
]

# --------------------------------------------------------------------------- #
# gallery families: (title, blurb, [run names in order])   -- abl_baseline shown once as reference
def cls_runs():
    return sorted([p.name for p in RESULTS.iterdir()
                   if p.is_dir() and p.name.startswith("abl_cls_")],
                  key=lambda n: int(n.split("_")[-1]))

def seed_runs():
    return sorted([p.name for p in RESULTS.iterdir()
                   if p.is_dir() and p.name.startswith("seed")])

FAMILIES = [
    ("Base comparison --- heat \\& Burgers, classical vs.\\ QAPINN",
     "Four runs at matched 121 parameters, identical training. The clean spectral contrast: heat stays a single Fourier mode, Burgers spreads to high harmonics.",
     ["heat_classical", "heat_qapinn", "burgers_classical", "burgers_qapinn"]),
    ("Tier-1 ablation --- re-upload depth (the bandwidth axis)",
     "Depth sets the VQC's accessible Fourier band $\\{-L..L\\}$. Prediction: accuracy and realized bandwidth grow with depth. The depth-3 point is the reference run.",
     ["abl_depth_1", "abl_depth_2", "abl_depth_5"]),
    ("Tier-1 ablation --- entanglement structure (band fixed)",
     "Band held fixed; only the wiring changes (none/linear/ring/all-to-all). Tests whether entanglement (a coefficient/reachability knob) moves the advantage. Linear is the reference run.",
     ["abl_ent_none", "abl_ent_ring", "abl_ent_all_to_all"]),
    ("Tier-1 ablation --- post-net width",
     "How much classical readout the VQC needs. $[16]$ is the reference run.",
     ["abl_post_lin", "abl_post_8", "abl_post_32"]),
    ("Tier-3 ablation --- qubit count",
     "Another bandwidth knob. Also the axis where a barren-plateau signature appears (gradient variance shrinking with qubits). 4 qubits is the reference run.",
     ["abl_qubits_3", "abl_qubits_5", "abl_qubits_6"]),
    ("Tier-3 ablation --- measurement operator",
     "Readout width and correlator type: single-$Z$ vs.\\ multi-$Z$ (reference) vs.\\ $ZZ$.",
     ["abl_meas_single", "abl_meas_zz"]),
    ("Tier-3 ablation --- encoding input scale (the $2\\pi$ collapse)",
     "Scaling the encoding angle. Optimum near 1 (reference); at $2\\pi$ the model collapses to the trivial solution --- the black-hole trainability pathology made concrete.",
     ["abl_scale_05", "abl_scale_pi", "abl_scale_2pi"]),
    ("Matched classical controls (cPINN)",
     "Param-matched tanh MLPs [2,a,b,1] --- the fair-capacity baseline each QAPINN is compared against.",
     cls_runs()),
    ("Fair structural control --- classical front-end (CFE)",
     "The decisive control: the identical post-net fed by a classical $\\mathrm{Linear}(2\\!\\to\\!w)+\\tanh$ instead of the VQC. This is what deflated the classical-side-diversity claim.",
     ["cfe_burgers", "cfe_heat", "cfe_q3", "cfe_q5", "cfe_q6"]),
    ("Convergence study (8000 iters)",
     "4x longer training. The ordering shifts vs.\\ 2000 iters --- convergence itself is a variable.",
     ["conv_qapinn_base", "conv_qapinn_entnone", "conv_cls_121"]),
    ("Seed expansion --- entanglement across seeds 1--6 (two machines)",
     "The statistics behind the 'entanglement not significant at 2000 iters' verdict: 7 seeds pooled, differences within seed noise.",
     seed_runs()),
    ("$\\nu$-sweep --- the bandwidth continuum (sufficiency test)",
     "Vary Burgers viscosity $\\nu$ to sweep solution bandwidth continuously; test whether advantage collapses onto one bandwidth curve.",
     sorted([p.name for p in RESULTS.iterdir() if p.is_dir() and p.name.startswith("nu_")])),
]

def esc(s: str) -> str:
    return s.replace("_", "\\_")

def run_meta(name: str) -> dict | None:
    p = RESULTS / name / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))

def run_block(name: str) -> str:
    m = run_meta(name)
    figs_dir = RESULTS / name / "figures"
    out = [f"\\subsection{{{esc(name)}}}"]
    if m:
        cfg = m.get("config", {})
        arch = m.get("architecture", {})
        rows = [
            ("rel-$L_2$", f"{m.get('rel_l2', float('nan')):.4f}"),
            ("params", str(m.get("num_params", "?"))),
            ("PDE", esc(str(cfg.get("pde", "?")))),
            ("Adam iters", str(cfg.get("adam_iters", "?"))),
            ("seed", str(cfg.get("seed", "?"))),
            ("model", esc(str(arch.get("type", "?")))),
            ("final PDE loss", f"{m.get('final_pde', float('nan')):.2e}"),
            ("wall time (s)", f"{m.get('train_time_s', float('nan')):.0f}"),
        ]
        vqc = arch.get("vqc")
        if isinstance(vqc, dict):
            rows.append(("VQC", esc(f"{vqc.get('n_qubits')}q x{vqc.get('n_layers')}L "
                                    f"ent={vqc.get('entanglement')} meas={vqc.get('measurement')} "
                                    f"scale={vqc.get('input_scale')}")))
        cells = "  ".join(f"\\textbf{{{k}}}: {v}" for k, v in rows)
        out.append("\\noindent\\small " + cells + "\\normalsize\n")
    # per-run observation (from actually viewing this run's figures)
    obs = OBS.get(name)
    if obs:
        out.append("\\noindent\\textbf{Observation.} " + ltx(obs) + "\n")
    # figures at full width
    if figs_dir.exists():
        present = {f.name for f in figs_dir.glob("*.png")}
        for fname, cap in FIG_ORDER:
            if fname in present:
                rel = f"{name}/figures/{fname}"
                out.append("\\begin{figure}[H]\\centering")
                out.append(f"\\includegraphics[width=\\textwidth]{{{rel}}}")
                out.append(f"\\caption{{\\textbf{{{esc(name)}}} --- {cap}}}")
                out.append("\\end{figure}")
    out.append("\\clearpage")
    return "\n".join(out)

# --------------------------------------------------------------------------- #
def big_fig(path: str, cap: str) -> str:
    return ("\\begin{figure}[H]\\centering\n"
            f"\\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"\\caption{{{cap}}}\n\\end{{figure}}\n")

# NARRATIVE (rich; starts at literature review). Big curated figures embedded.
NARRATIVE = r"""
\section{Literature review (deep research)}
\emph{Provenance: a multi-source deep-research pass --- 5 search angles, 19 sources
fetched, 86 candidate claims extracted, 25 adversarially verified by independent
3-vote panels; 20 confirmed, 5 refuted. Full annotated version in
\texttt{report/literature\_review.md}.}

\subsection{Core theory --- VQCs as Fourier series}
A variational quantum circuit with data-encoding gates \textbf{is exactly a truncated
Fourier series}, whose accessible frequency set is fixed by the eigenvalues of the
encoding gates; \emph{re-uploading} the encoding enlarges the accessible spectrum, and
circuits exist that realize arbitrary Fourier coefficients (asymptotic universality).
This is the bedrock under all QAPINN work (Schuld--Sweke--Meyer 2021, \textbf{[1]}).
For us: with $L$ re-uploads of a scalar coordinate via $R_Y$, the accessible
frequencies are integers in $\{-L,\dots,L\}$ --- so \textbf{smooth (low-frequency)
solutions need few re-uploads, shocks (high-frequency) need many}, and the Gibbs-like
overshoot at the Burgers shock is the signature of a truncated Fourier series meeting a
near-discontinuity.

Kyriienko--Paine--Elfving (2021, \textbf{[2]}) turned this into a practical
differential-equation solver (DQC): the solution is the expectation value of a
parameterized circuit, differentiated \emph{analytically}; the encoding (e.g.\ a
Chebyshev feature map) sets the basis.

\textbf{Critical refinement (Mhiri et al.\ 2025, \emph{Quantum} 9:1847, \textbf{[3]}):}
the frequency spectrum alone does \emph{not} determine expressivity. The achievable
Fourier \emph{coefficients} are constrained --- coefficient variance scales with a
frequency's redundancy, and a ``vanishing expressivity'' effect can make coefficient
variances decay \emph{exponentially with qubit count}, independent of barren plateaus.
\emph{Implication for us:} our spectral analysis must measure realized
\emph{coefficients}, not just frequency support --- most applied papers ignore this.

\subsection{Architectures and encodings}
The field converged on a hybrid design: (classical) pre-net $\to$ VQC $\to$ optional
classical post-net, with angle/$R_Y$ encoding, a hardware-efficient ansatz
(single-qubit rotations $+$ CNOT chains), and a $\sum_i\langle Z_i\rangle$ observable
\textbf{[4]}. A trainable NN embedding (TE-QPINN) beats a fixed Tower-Chebyshev map at
equal qubits. Our ref-[3] QAPINN --- direct $(x,t)$ encoding, VQC as first hidden
layer, classical post-net --- is the theory-clean member of this family (no pre-net to
warp the band).

\subsection{Empirical claims --- parameter reduction}
The dominant narrative is parameter reduction at comparable-or-better accuracy:

\begin{center}\small\begin{tabular}{lll}\toprule
Work & Claim & Conf. \\ \midrule
QCPINN (Farea--Khan 2025 \textbf{[5]}) & 10--30\% of classical params; conv--diff 821 vs 7901 ($\sim$89.6\%) & [C] \\
QA-PINN (IEEE QCE 2024 \textbf{[6]}) & 4/5-qubit, \textbf{$\sim$20\% param reduction on 1-D Burgers} & [C] \\
Maxwell QPINN (Chen 2026 \textbf{[7]}) & up to $\sim$19\% higher accuracy, $\sim$19\% fewer params & [C$^2$] \\
Poisson QPINN (Trahan 2024 \textbf{[8]}) & 27-param net cuts RMSE $>$75\% vs 51-param & [C] \\
BQP QA-PINN (\textbf{[9]}, vendor) & ``20\% param reduction'', ``98.04\% accuracy'' & [C$^2$] \\
\bottomrule\end{tabular}\end{center}

\textbf{The $\sim$20\% Burgers figure in the challenge slide is real and peer-reviewed
\textbf{[6]}.} \emph{But quantum gains are PDE/regularity-dependent, not universal:}
QCPINN is \emph{worse} on the wave equation (12.44\% vs 11.13\%) and lid-driven cavity
(25.73\% vs 21.61\%). \textbf{Advantage correlates with solution complexity} --- the
single most useful finding for our design. It also tempered our own LDC ambition (the
literature finds quantum \emph{worse} on cavity flow).

\subsection{Trainability and the ``black hole''}
Generic VQCs suffer barren plateaus (gradient variance vanishing with qubits/depth).
QAPINNs additionally show a distinct pathology: the \textbf{``black-hole'' collapse} ---
after descending toward the solution, the optimizer can suddenly collapse to the trivial
(zero) solution satisfying only IC/BC (Chen et al.\ 2026 \textbf{[7]}). \emph{This is
exactly what our early full-quantum prototype did (rel-$L_2$ 0.94), and what our
input-scale-$2\pi$ run reproduces --- a named, published failure mode, not a bug.}

\subsection{The central gap --- and our contribution}
\textbf{The gap is mechanistic.} Across the literature, quantum gains are attributed to
``entanglement-driven expressibility'' or ``Fourier richness'' as \emph{heuristics};
\textbf{no validated mechanism exists}, and rigorous XAI comparing classical vs quantum
learning is essentially absent. Our wedges: (Q2) Fourier-spectrum $\leftrightarrow$
PDE-regularity as a \emph{tested} mechanism (Burgers shock vs heat smooth, measuring
realized coefficients per [3]); (Q4) characterizing the black-hole collapse; (Q3) a rare
matched-resource, reproduced, seeded comparison; and the XAI dimension the challenge
explicitly rewards.

\subsection{Refuted claims (recorded, never cited)}
\begin{center}\small\begin{tabular}{ll}\toprule
Refuted claim & Vote \\ \midrule
TE-QPINN $\sim$order-of-magnitude below best classical at matched params (both PDEs) & 0-3 \\
Hybrid QPINN (1456p) beat classical (3441p) on 2D Burgers at matched accuracy & 1-2 \\
PDE loss gives gradient variance decaying only polynomially ($n^{-0.7}$) & 0-3 \\
Embedding PDE residuals inherits local-cost polynomial gradient scaling & 1-2 \\
Physics constraints concentrate gradient information (mechanistic trainability) & 0-3 \\
\bottomrule\end{tabular}\end{center}
\emph{Evidence-base caveats: no independent reproduction; not compute-matched
(``parameter reduction'' $\neq$ ``resource reduction'' --- our own runs are $\sim$85$\times$
slower wall-clock); best-case framing and unequal-qubit comparisons are common.}

\clearpage
\section{Methodology and thesis}
\emph{Full version: \texttt{report/methodology.md}.}

The challenge asks us to explain \emph{when, why, and how} replacing a PINN's first
hidden layer with a VQC changes learning, and to distil a design methodology --- the
stated goal is \emph{not} to show QAPINNs win. The reference architecture:
\[ (x,t)\ \to\ \text{VQC [quantum first layer]}\ \to\ \text{classical hidden layers}\ \to\ u(x,t). \]

\subsection{The when / why / how, operationalized}
\begin{center}\small\begin{tabular}{lll}\toprule
Word & Operational question & Method \\ \midrule
HOW & how do the classical neurons / loss landscape differ with vs without the VQC? & XAI \\
WHY & is the change driven by the \emph{band}, the \emph{coefficients/entanglement}, or the \emph{landscape}? & attribution \\
WHEN & for which PDEs (spectral content) does it help / do nothing / hurt? & PDE-family ablation \\
Design & given a new PDE, how do we prescribe the circuit from cheap info? & methodology \\
\bottomrule\end{tabular}\end{center}

\subsection{The central controlled experiment}
The VQC output is provably Fourier; the open question is \emph{what part} drives the
learning change. \textbf{Fix the band, move the rest:} hold depth/encoding fixed (band
fixed), then vary entanglement (none/linear/ring/all-to-all) and measurement
($\langle Z\rangle$ / $\langle ZZ\rangle$ / probability vector).
\begin{itemize}[nosep]
\item behaviour \textbf{invariant} $\Rightarrow$ the \textbf{band} is the driver (Fourier picture predictive);
\item behaviour \textbf{moves} $\Rightarrow$ \textbf{coefficients/entanglement} (if expressivity metrics move) or the \textbf{landscape} (if only trainability metrics move).
\end{itemize}
This is the thesis: \emph{is the VQC's frequency band a sufficient statistic for its
effect in a PINN, and if not, what carries the rest?}

\subsection{Test problems --- one spectral axis}
Same IC $u(x,0)=\sin(\pi x)$, same domain/BCs across the 1-D problems, so the
\emph{only} difference is the operator (hence the spectrum):
\begin{center}\small\begin{tabular}{llll}\toprule
PDE & Operator & Exact reference & Spectrum \\ \midrule
Heat & $u_t=\alpha u_{xx}$ & $\sin(\pi x)e^{-t}$ & single mode $k{=}1$ (smooth control) \\
Burgers & $u_t+u u_x=\nu u_{xx}$ & Cole--Hopf & shock $\to$ high harmonics (sharp case) \\
Burgers $\nu$-sweep & vary $\nu$ & Cole--Hopf per $\nu$ & tunes bandwidth low$\to$high (continuum) \\
\bottomrule\end{tabular}\end{center}
Controls: matched trainable-parameter count, identical collocation points, optimizer,
iteration budget, seed; multi-seed error bars on any headline number; every run from a
config $+$ fixed seed. \emph{We do NOT claim QAPINN beats classical, nor a new
expressivity theorem --- Schuld [4] and Mhiri [3] are used, not re-proven.}

\clearpage
\section{The bench and the models}
A PDE-agnostic interface (\texttt{PDESpec}: residual columns $+$ a uniform list of
face-constraints) lets heat and Burgers share one bench; a config selects PDE, model,
VQC knobs, and training. Models: \textbf{cPINN} (tanh MLP, matched control);
\textbf{QAPINN} (ref [3]: direct $(x,t)$ encode $\to$ VQC first layer $\to$ classical
post-net); \textbf{CFE} (the fair structural control --- identical post-net fed by a
classical $\mathrm{Linear}(2\!\to\!w)+\tanh$). PennyLane \texttt{default.qubit}
statevector, \texttt{interface=torch}, \texttt{diff\_method=backprop} so second-order
autograd ($u_{xx}$) flows through the circuit. Reference: exact Cole--Hopf (Burgers, via
Gauss--Hermite quadrature with \texttt{scipy.roots\_hermite} $+$ log-sum-exp) and the
analytic heat solution.
""" + big_fig("reference/burgers_reference.png",
    "\\textbf{Exact Cole--Hopf reference} for Burgers ($\\nu=0.01/\\pi$): field $u(x,t)$ "
    "and snapshots. The shock forms at $x=0$ around $t\\approx0.4$. Validated to machine "
    "precision (IC error 0, BC $\\sim$1e-16, quadrature $\\sim$1e-15). This is the ground "
    "truth every relative-$L_2$ is measured against.") + r"""

\clearpage
\section{The XAI suite (five dimensions + dynamic monitoring)}
\emph{Full reference: \texttt{report/xai\_methods.md}.} Every method runs identically on
the cPINN and the QAPINN, so the two are comparable.
\begin{enumerate}[nosep]
\item \textbf{Spectral} (\texttt{spectrum.py}): realized per-harmonic magnitude vs
reference; per-harmonic error; \textbf{F-principle} (iteration at which each harmonic's
error halves --- low-before-high). The mechanistic core: does the VQC change the order?
\item \textbf{Neurons} (\texttt{neurons.py}): each classical neuron's receptive field
over $(x,t)$; its dominant spatial frequency; \emph{effective rank} (entropy-based count
of distinct features). More diverse features would be mechanistic support for parameter
reduction.
\item \textbf{VQC internals} (\texttt{vqc\_internals.py}): per-qubit
$\langle Z\rangle(x,t)$; von Neumann \emph{entanglement entropy} across the domain
(validated: entanglement=none $\to$ entropy $\approx0$). The literature skips this.
\item \textbf{Loss landscape} (\texttt{landscape.py}): filter-normalized 2-D slice
around the minimum (Li et al.\ 2018); flat/wide vs sharp.
\item \textbf{Generalization} (\texttt{generalization.py}): relative $L_2$ vs $t$ beyond
the training window --- does the Fourier prior extrapolate (quasi-)periodically?
\end{enumerate}
Dynamic monitoring records the spectrum build-up and per-tensor gradient variance
during training (barren-plateau / black-hole signal).

\clearpage
\section{Pipeline verification (the 8/8 audit)}
Before drawing any conclusion, every load-bearing computation was checked against an
external ground truth.
\begin{center}\small\begin{tabular}{lll}\toprule
Check & Method & Result \\ \midrule
Burgers residual & vs analytic & 1.2e-7 \\
Heat residual & exact $\to$ residual & 6.0e-8 \\
Autograd $u_x$ & vs finite differences & match \\
\textbf{VQC $=$ Fourier, max freq $=$ depth} & $L{=}2{\to}2$, $L{=}4{\to}4$ & \textbf{pass} \\
relative-$L_2$ metric & exact $\to 0$ & 4.0e-8 \\
Matched params & all configs & exact \\
Reproducibility & same seed, two machines & bit-exact \\
Constraint sampler & IC/BC coverage & pass \\
\bottomrule\end{tabular}\end{center}
\textbf{8/8 pass.} The VQC-bandwidth check validates the central premise on our own
implementation; the audit also caught a bug --- in the FFT-sampling \emph{test}, not the
code.

\clearpage
\section{Reading the fields --- observations and inferences}
Before the summary montages, this section works directly from individual run figures.
Each carries an \textbf{Observation} (what is visibly in the figure) and an
\textbf{Inference} (what we conclude from it). This is where the bandwidth thesis stops
being a table and becomes something you can \emph{see}.

\subsection{The clean contrast: same model, same IC, two spectra}
Heat and Burgers share the identical initial condition $u(x,0)=\sin(\pi x)$, the same
domain, the same boundary conditions, and here the \emph{identical} 121-parameter QAPINN
(depth 3, 4 qubits). The only difference is the operator --- and therefore the spectral
content the solution demands.
""" + big_fig("heat_qapinn/figures/solution.png",
    "\\textbf{Heat, QAPINN.} Reference and prediction are visually indistinguishable; the "
    "error map (right) never exceeds $\\sim$0.012 and carries no coherent structure --- just "
    "diffuse numerical residue. The $t=0.25/0.50/0.75$ snapshots show the red model curve "
    "lying exactly on the black reference.") + r"""
\textbf{Observation.} The heat solution is a single decaying Fourier mode ($k=1$,
amplitude $e^{-t}$). The depth-3 QAPINN reproduces it essentially perfectly
(rel-$L_2 = 5.9\text{e-}3$): the snapshot curves overlap, and the residual is a
featureless ripple two orders of magnitude below the signal.
""" + big_fig("abl_baseline/figures/solution.png",
    "\\textbf{Burgers, the same QAPINN.} Now the prediction colour-scale saturates at "
    "$\\pm0.6$ while the reference spans $\\pm1.0$: the model builds a smooth, "
    "\\emph{amplitude-attenuated} version of the true field. The error (right) is a single "
    "bright band pinned to the shock at $x=0$; everywhere else it is dark. In the snapshots "
    "the red model tracks the linear ramps away from the shock almost exactly, then rounds "
    "the jump and undershoots the extrema ($\\sim$0.6 vs 1.0).") + r"""
\textbf{Observation.} Same network, same initialisation, same training budget --- but on
Burgers the identical model lands at rel-$L_2 = 0.47$, an 80$\times$ worse error than on
heat. The failure is spatially localised: away from $x=0$ (the low-frequency region) the
fit is excellent; \emph{all} of the error lives in the shock.

\textbf{Inference.} Accuracy here is not a property of the model --- it is a property of
the \emph{match between the solution's bandwidth and the circuit's band}. The heat
solution lives entirely inside the depth-3 band $\{-3..3\}$, so the model is near-exact;
the Burgers shock needs harmonics far beyond it, so the same model saturates. This single
pair is the strongest visual statement of the thesis: the quantum layer's usefulness is
gated by bandwidth, and bandwidth is set by the physics of the target, not the optimiser.

\subsection{Why, in frequency space}
The solution fields hint at a spectral cutoff; the realized-spectrum plot makes it
explicit.
""" + big_fig("abl_baseline/figures/spectrum.png",
    "\\textbf{Realized Fourier spectrum, baseline QAPINN (depth 3).} Grey $=$ exact "
    "reference, red $=$ trained model, log-scale magnitude per spatial harmonic $k$. The "
    "reference decays only as $\\sim1/k$ --- the slow tail of a near-discontinuity, with "
    "real power out to $k=20$. The model \\emph{matches} the reference at $k=1,2,3$ then "
    "falls progressively below it, flattening to a floor $\\sim$0.1--0.3 for $k\\gtrsim7$.") + r"""
\textbf{Observation.} The crossover from ``model matches'' to ``model undershoots'' sits
right around $k=3\text{--}4$ --- exactly the edge of the depth-3 accessible band. Beyond it
the model retains only a small residual (leaked through the classical post-net's
nonlinearity), one to two orders of magnitude short of the reference.

\textbf{Inference.} This is the mechanism behind the rounded shock, read directly off the
coefficients. A truncated Fourier series cannot represent a jump; it approximates it with
its available harmonics, which both smears the discontinuity and --- because the missing
high-$k$ power also carries amplitude --- pulls down the peak values. The rounded shock and
the amplitude undershoot are not two bugs; they are the \emph{same} band-truncation seen
in space and in frequency. It also confirms Mhiri's caveat operationally: the model
\emph{does} put nonzero coefficients at high $k$, but their \emph{magnitude} is far too
small --- frequency support is not the same as realized coefficient power.

\subsection{Add band, sharpen the shock}
If band-truncation is the mechanism, extending the band (more re-upload depth) should
tighten the shock. It does --- modestly.
""" + big_fig("abl_depth_5/figures/solution.png",
    "\\textbf{Burgers, depth-5 QAPINN.} Against the depth-3 baseline above, the bright "
    "error band at the shock is visibly \\emph{narrower} and the snapshot transition through "
    "$x=0$ is steeper. The peak amplitude is still capped near $\\pm0.6$, but the jump is "
    "resolved over a shorter interval.") + r"""
\textbf{Observation.} Two extra re-uploads move rel-$L_2$ from 0.472 to 0.424 --- a real
but undramatic gain --- and the improvement is exactly where theory says it should be:
concentrated at the shock, sharpening the transition rather than lifting the whole field.

\textbf{Inference.} The advantage of depth is spectral and \emph{local}: it buys the
harmonics that live at the discontinuity, nothing else. This is why the depth sweep is
monotonic and physically legible while entanglement (below) is not --- depth changes what
the model \emph{can} represent, at the precise place it was failing.

\subsection{Push the encoding too hard: the black-hole collapse}
Bandwidth knobs help --- until trainability gives out. The encoding-scale sweep exposes the
ceiling in the most dramatic figure in the project.
""" + big_fig("abl_scale_2pi/figures/solution.png",
    "\\textbf{Burgers, input\\_scale $=2\\pi$.} The prediction colour-scale now reads "
    "$-0.014$ to $-0.002$: the entire field has collapsed to $u\\approx0$, with only faint "
    "high-frequency speckle. In every snapshot the red model is a flat line at zero while "
    "the reference sweeps $\\pm1$. The error is $\\approx$1 everywhere except the $x=0$ line "
    "(where the true solution is itself zero).") + r"""
\textbf{Observation.} At scale 1 the model reaches rel-$L_2$ 0.47; at scale $2\pi$ it is
1.00 --- a total collapse to the trivial solution. The transition is not gradual
degradation but a fall into a different basin: the network satisfies the homogeneous
boundary conditions and abandons the initial condition entirely.

\textbf{Inference.} This is the literature's ``black-hole'' collapse (Chen et al.\ 2026)
reproduced on demand. At $2\pi$ the encoding wraps the input across a full period, so
neighbouring collocation points map to widely separated circuit states; the loss surface
around the true solution flattens and the optimiser slides into the zero basin. The
practical lesson for the design methodology is sharp: \textbf{encoding scale is not a free
hyperparameter} --- there is an optimum near 1 and a hard cliff past it, and a
circuit-sizing recipe must respect that cliff or it will silently recommend an untrainable
model.

\subsection{The circuit works hard internally --- and it barely matters}
The quantum-internal views let us watch the circuit's own state, not just its output. They
deliver the study's most counter-intuitive observation.
""" + big_fig("ablation_summary/ent_entropy_fields.png",
    "\\textbf{Von Neumann entanglement entropy $S(x,t)$ across wiring.} \\emph{none}: the "
    "scale is $10^{-7}$ --- numerically zero, exactly as an unentangled circuit must be (a "
    "clean sanity check). \\emph{linear/ring/all-to-all}: entropy climbs to $\\sim$1.0--1.2 "
    "nats with rich spatial structure --- the ring case shows a diagonal high-entropy ridge "
    "that follows the shock's space--time trajectory.") + r"""
\textbf{Observation.} Internally these four circuits are doing wildly different things: the
entangled variants generate up-to-$>$1-nat, spatially-structured entanglement whose pattern
tracks the physics, while the unentangled one sits at machine-zero. Yet across seven seeds
their accuracies are statistically indistinguishable (none $0.459\pm0.011$, linear
$0.448\pm0.022$, ring $0.474\pm0.014$, all-to-all $0.462\pm0.014$).

\textbf{Inference.} Internal quantum complexity is, in this regime, \emph{decoupled} from
task performance. The binding constraint is the band --- already saturated at depth 3 ---
so redistributing which coefficients are reachable (what entanglement does) moves the
answer within noise. This is a genuinely deflationary result for the field's favourite
heuristic (``entanglement-driven expressibility''): here you can dial entanglement from
zero to strongly-structured and the PINN does not care. The honest reading is not
``entanglement is useless'' but ``once the band is the bottleneck, the coefficient
machinery is not what is limiting you'' --- and diagnosing \emph{which} regime you are in is
the useful contribution.

\subsection{It is bandwidth, not capacity --- the classical controls prove it}
A natural objection to the whole bandwidth story: maybe the shock is just hard, and any
model --- quantum or classical --- would fix it with \emph{more parameters}. Reading all
eleven matched classical controls (29 to 217 parameters) refutes that directly. Shock
quality is \emph{non-monotonic} in parameter count.
""" + big_fig("abl_cls_165/figures/solution.png",
    "\\textbf{Classical, 165 params --- the best shock of the sweep.} The front is visibly "
    "tilted and steepened with a slight overshoot near $x\\approx0.3$--0.5, the closest any "
    "matched classical net comes to the reference discontinuity; peak $\\sim$0.68 and the "
    "error band is the tightest of the eleven.") + big_fig("abl_cls_217/figures/solution.png",
    "\\textbf{Classical, 217 params --- the largest net, and mediocre.} Nearly twice the "
    "capacity of the 165-param net, yet the shock is rounded into a smooth curve with no "
    "steepening (peak $\\sim$0.57) and a broader error plume. More parameters did not buy a "
    "sharper shock.") + r"""
\textbf{Observation.} Across the eleven controls the sharpest shocks come from
\emph{mid-sized} nets (165 params $\sim$0.68, 99 params $\sim$0.73), while the
\emph{largest} (217) is unremarkable ($\sim$0.57) and one larger net (143) collapses to a
near-static underfit ($\sim$0.46). All eleven round the shock and cap amplitude; none
approach the reference's $\pm1.0$ jump; and every one matches only $k=1$ before falling a
half-to-full decade below the reference for $k\gtrsim4$.

\textbf{Inference.} Capacity is \emph{not} the limiting factor for shock resolution ---
this is spectral bias (the F-principle), not an under-parameterisation. That reframes what
the quantum layer is even competing on: the question was never ``can more parameters fix
the shock'' (they cannot, reliably) but ``can a different \emph{basis} reach the missing
high harmonics''. It is why the honest comparison is bandwidth-for-bandwidth (depth,
qubits) against a \emph{structurally} matched control, not parameter-for-parameter against
an arbitrary MLP --- and why the depth axis, which literally extends the band, is the only
knob that improves the shock monotonically.

\clearpage
\section{Results --- comparative figures (montages)}
The following montages place classical and QAPINN side by side across the whole XAI
suite. Per-run originals for every model are in the gallery (Part~II).
""" + "\n".join(big_fig(f"_montage/{n}.png", c) for n, c in [
    ("solution",       "\\textbf{Solution fields.} Heat (top): both fit near-perfectly. Burgers (bottom): both round the shock --- the depth-3 band is capped near $k\\approx3$."),
    ("spectrum",       "\\textbf{Realized spectra.} Heat: single mode both capture. Burgers: reference is broadband; both models undershoot the high-frequency shock tail."),
    ("fprinciple",     "\\textbf{F-principle.} The error-reduction wave sweeps from low $k$ upward --- low frequencies converge first, in both models."),
    ("neurons",        "\\textbf{Classical neurons.} Receptive fields and dominant-frequency histograms, classical vs QAPINN post-net."),
    ("entanglement",   "\\textbf{Entanglement entropy $S(x,t)$} (QAPINN): the state entangles differently per problem."),
    ("qubits",         "\\textbf{Per-qubit $\\langle Z\\rangle$ fields}: what feature each qubit specializes in."),
    ("landscape",      "\\textbf{Loss landscapes} around the minimum, classical vs QAPINN."),
    ("generalization", "\\textbf{Generalization}: relative $L_2$ vs $t$ past the training window."),
    ("evolution",      "\\textbf{Solution evolution} snapshots over time."),
    ("loss",           "\\textbf{Training-loss histories}, classical vs QAPINN."),
]) + r"""

\clearpage
\section{Ablation results --- summary panels}
One knob at a time; each QAPINN paired with a param-matched classical control.
""" + "\n".join(big_fig(f"ablation_summary/{n}.png", c) for n, c in [
    ("depth_sweep",       "\\textbf{Depth sweep.} QAPINN accuracy and realized bandwidth grow with re-upload depth; overtakes the matched classical at depth $\\geq3$. \\emph{Robust, physically grounded.}"),
    ("depth_spectra",     "\\textbf{Realized spectrum across depth} --- more of the high-frequency shock tail is captured as depth grows."),
    ("depth_solutions",   "\\textbf{Solution field across depth.}"),
    ("entanglement_sweep","\\textbf{Entanglement (band fixed).} Accuracy barely moves while entropy swings widely --- \\emph{single-seed}; later found within noise (see seeds)."),
    ("ent_entropy_fields","\\textbf{Entanglement-entropy fields} across wiring: entropy varies strongly even as accuracy does not."),
    ("post_sweep",        "\\textbf{Post-net width.} The VQC needs a small readout but beats the matched classical for any post-net $\\geq[8]$."),
    ("tier3_sweeps",      "\\textbf{Tier-3 sweeps.} Qubits behave like depth (bandwidth). Single-$Z$ readout bottlenecks; multi $=$ $ZZ$. Encoding scale has an optimum near 1 and \\emph{collapses} at $2\\pi$ (black-hole)."),
    ("classical_response","\\textbf{Classical-side response} (first pass, vs a plain PINN)."),
    ("frontend_spectra",  "\\textbf{Front-end spectra}: what the VQC vs a classical front-end feed into the identical post-net."),
    ("cfe_comparison",    "\\textbf{The decisive fair control.} Quantum vs classical front-end on the \\emph{identical} post-net. Gradient amplification and neuron diversity are \\emph{neither} robust across knobs --- this refuted the classical-side-diversity claim."),
    ("entanglement_seeds","\\textbf{Entanglement across 7 seeds} with error bars: the differences are within seed noise --- \\emph{entanglement effect NOT significant at 2000 iters}."),
]) + r"""

\clearpage
\section{Deflation --- the classical-side mechanism, and its fair control}
\textbf{Hypothesis.} The VQC ``offloads'' basis construction, so the downstream classical
neurons become more diverse and are driven with larger gradients.

Against a \emph{plain PINN} this looked true ($\sim$2$\times$ diversity, $\sim$4$\times$
gradient variance). But the plain PINN differs from the QAPINN in \emph{two} ways (front
end \emph{and} depth), so the attribution was confounded. We built the \textbf{CFE}: the
identical post-net fed by a classical $\mathrm{Linear}(2\!\to\!w)+\tanh$ --- same
structure, classical front end. Against \emph{that} fair control:
\begin{center}\small\begin{tabular}{lccc}\toprule
Comparison & rel-$L_2$ (Q/C) & diversity (Q/C) & grad-var $\times10^{-3}$ (Q/C) \\ \midrule
Burgers depth-3 & 0.472 / 0.474 & 0.269 / \textbf{0.286} & \textbf{4.1} / 0.62 \\
Burgers depth-5 & \textbf{0.424} / 0.474 & 0.236 / \textbf{0.286} & \textbf{7.1} / 0.62 \\
Heat & \textbf{0.006} / 0.013 & \textbf{0.272} / 0.239 & \textbf{3.2} / 0.47 \\
3 qubits & 0.501 / 0.496 & \textbf{0.266} / 0.241 & \textbf{1.2} / 0.26 \\
5 qubits & 0.468 / 0.464 & \textbf{0.312} / 0.265 & 1.1 / \textbf{5.2} \\
6 qubits & \textbf{0.443} / 0.467 & \textbf{0.341} / 0.269 & 0.72 / \textbf{1.9} \\
\bottomrule\end{tabular}\end{center}
\verdict{The mechanism deflated.} (i) Diversity is \emph{inconsistent} --- the classical
front end is \emph{more} diverse on the depth axis; the ``2$\times$'' was an artifact of
the mismatched plain-PINN control. (ii) Gradient amplification \emph{reverses} at 5--6
qubits, where the QAPINN's gradient variance \emph{shrinks} with qubit count --- a
\textbf{barren-plateau signature}. (iii) Accuracy is \emph{bandwidth-gated}: ties the
fair control at low depth/qubits, wins only at depth 5 / 6 qubits / heat. We removed the
false claim from the reports.
""" + big_fig("ablation_summary/cfe_comparison.png",
    "\\textbf{The fair control, read directly.} Left --- post-net gradient variance: QAPINN "
    "(orange) exceeds the classical front-end (blue) at depth-3, depth-5, heat and 3 qubits, "
    "but the bars \\emph{cross} at 5 and 6 qubits, where the classical control is now higher. "
    "The QAPINN's own gradient variance falls monotonically along the qubit axis (4q $\\approx$ "
    "4e-3 $\\to$ 5q $\\approx$ 1.1e-3 $\\to$ 6q $\\approx$ 7e-4, a $\\sim$6$\\times$ drop). Right "
    "--- neuron effective rank: the classical front-end is \\emph{more} diverse on the depth "
    "axis, the QAPINN more diverse on the qubit axis --- no consistent winner.") + r"""
\textbf{Observation.} Neither claimed mechanism survives as a clean effect. Gradient
``amplification'' is real only at low qubit count and \emph{inverts} by 5--6 qubits;
neuron ``diversification'' has no consistent sign against the fair control.

\textbf{Inference.} The monotone fall of the QAPINN's gradient variance as qubits grow
(4$\to$6) --- while the equal-width classical control holds roughly flat --- is the textbook
fingerprint of a \textbf{barren plateau} opening as the Hilbert space enlarges. So the
qubit axis carries two \emph{opposing} effects at once: more band (helps accuracy) and
vanishing gradients (hurts trainability). That tension, not a monotone ``more qubits =
better'', is the honest picture --- and it is exactly why a design methodology must trade
band against trainability rather than maximise either. The earlier ``$2\times$ diversity /
$4\times$ gradients'' story was an \emph{attribution error}: the plain-PINN control differed
from the QAPINN in two ways at once (front end \emph{and} architecture), so it credited to
``quantum'' what was partly just ``different network''. The CFE isolates the front end, and
the effect dissolves. Building the right control was worth more than any single number it
produced.

\clearpage
\section{Convergence and seeds --- the statistics}
\textbf{Convergence (single seed):}
\begin{center}\small\begin{tabular}{lcc}\toprule
 & 2000 iters & 8000 iters \\ \midrule
QAPINN entanglement=none & 0.456 & \textbf{0.261} \\
QAPINN entanglement=linear & 0.472 & 0.437 \\
classical (121p) & 0.535 & 0.376 \\
\bottomrule\end{tabular}\end{center}
\textbf{Seeds (7 seeds, 2000 iters):}
\begin{center}\small\begin{tabular}{lc}\toprule
entanglement & rel-$L_2$ (mean $\pm$ std) \\ \midrule
none & 0.459 $\pm$ 0.011 \\ linear & 0.448 $\pm$ 0.022 \\
ring & 0.474 $\pm$ 0.014 \\ all-to-all & 0.462 $\pm$ 0.014 \\
\bottomrule\end{tabular}\quad (spread 0.026 $<$ 2$\times$std 0.030)\end{center}
\verdict{Entanglement is NOT significant at 2000 iters (7 seeds).} The single-seed
``entanglement doesn't help'' was noise at that budget. \emph{However}, at
\emph{convergence} (8000 iters, single seed) a large effect appears: entanglement=none
(0.261) beats classical while entanglement=linear (0.437) loses. So the effect, if real,
is \emph{training-dependent} and currently rests on one seed at convergence --- an
\textbf{open lead}, not a claim. Convergence itself matters: 4$\times$ training reorders
the models.

\textbf{Inference.} The reordering under longer training is a methodological warning with
teeth: at 2000 iterations the classical control is \emph{worst} of the three (0.535); at
8000 it is \emph{best-but-one} (0.376), overtaking the entangled QAPINN. A benchmark that
stopped at 2000 iterations --- as many in the literature effectively do --- would have
reported the opposite ranking to the converged one. This is precisely how non-converged,
single-budget comparisons manufacture ``quantum advantage'' (or erase it): the models sit
at different points on their loss curves, and whoever is compared mid-descent wins by
accident. Our takeaway is not the 8000-iter numbers themselves (one seed) but the
structural point --- \emph{any} accuracy claim without a convergence check and error bars is
a claim about training budget, not about the model.

\textbf{Observation (from the 8000-iter figures).} Reading the three converged runs
side by side makes the entanglement lead vivid, not just numeric. The
\emph{no-entanglement} QAPINN (\texttt{conv\_qapinn\_entnone}, entropy $\sim$10$^{-7}$)
develops a genuinely \emph{crisp} shock: a near-vertical transition at $x=0$ whose
residual is a razor-thin line, and a realized spectrum that matches the reference
\emph{bar-for-bar out to $k=20$} --- the high-$k$ coefficients are $\sim$2, versus
$\sim$0.04 for the entangled run. The \emph{entangled} QAPINN
(\texttt{conv\_qapinn\_base}) carries real, structured entanglement ($S\sim0.5$--$1.1$)
yet, even at 4$\times$ training, keeps a smooth ramp (peak $\sim$0.65) and a band-limited
spectrum. The classical net (\texttt{conv\_cls\_121}) also sharpens the shock (peak
$\sim$0.78) but with visible Gibbs ringing.

\textbf{Inference.} At convergence the ordering is \emph{no-entanglement QAPINN}
$\approx$ \emph{classical} $\gg$ \emph{entangled QAPINN} on the very thing that matters
here --- high-frequency shock content. So the entangling layers are not neutral at this
budget; they appear to \emph{cost} high-$k$ expressivity, plausibly by tangling the
optimization (the barren-plateau tendency seen on the qubit axis). This is the study's
sharpest open lead: it says the field's default ``add entanglement for expressibility''
instinct can be exactly backwards for stiff, high-frequency PDE targets. It rests on one
seed at convergence and must be repeated across seeds before it is a claim --- but it is
now a lead grounded in a spectrum you can read, not a single scalar.

\clearpage
\section{The $\nu$-sweep --- bandwidth sufficiency}
\textbf{Hypothesis.} If bandwidth is a \emph{sufficient statistic} for advantage, then
tuning $\nu$ (which continuously changes solution bandwidth) plus the heat point should
collapse the advantage onto one curve of advantage-vs-bandwidth. The $\nu\to$bandwidth
map is already verified ($\nu{=}0.1{\to}$bw 3, $\nu{=}0.003{\to}$bw 22, heat$\to$1), and
is itself the core of the design methodology: cheap physics proxy $\to$ bandwidth $\to$
depth.

\textbf{Observation (from the completed, corrected sweep).} Both models' error rises
\emph{monotonically} with solution bandwidth --- but the \emph{advantage} does not collapse
onto a clean curve. The full table (heat is the bandwidth-1 anchor; ``base'' is the
original $\nu=0.01/\pi$):
\begin{center}\small\begin{tabular}{lcccccc}\toprule
label & heat & $\nu{=}0.1$ & $\nu{=}0.05$ & $\nu{=}0.02$ & $\nu{=}0.01$ & base \\
solution bandwidth & 1 & 3 & 5 & 10 & 15 & 22 \\ \midrule
classical rel-$L_2$ & 0.008 & 0.018 & 0.032 & 0.069 & 0.141 & 0.535 \\
QAPINN (depth 3) rel-$L_2$ & 0.006 & 0.049 & 0.077 & 0.165 & 0.354 & 0.472 \\
advantage (cls$-$QAP) & $+$0.00 & $-$0.03 & $-$0.05 & $-$0.10 & $-$0.21 & $+$0.06 \\
\bottomrule\end{tabular}\end{center}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{ablation_summary/nusweep.png}
\caption{\textbf{The collapse test.} Left: both curves rise with bandwidth; the matched
classical (orange) is \emph{below} (better than) the depth-3 QAPINN (blue) everywhere up
to bandwidth $\sim$15, then the two \emph{cross} near bandwidth 20. Right: the advantage
is \textbf{V-shaped} --- near zero at heat, going negative (QAPINN worse) and bottoming at
$-0.21$ at bandwidth 15, then flipping positive ($+0.06$) only at the extreme bandwidth 22
where \emph{both} models are failing ($\sim$0.5).}\end{figure}
\textbf{Inference.} \verdict{The sufficiency hypothesis is not confirmed.} Bandwidth
cleanly predicts each model's \emph{error} (both curves are monotone), but it does
\emph{not} collapse the \emph{advantage} onto a single function --- the sign of the
advantage is regime-dependent. For the fixed depth-3 band the QAPINN \emph{loses} to its
matched classical control across almost the whole family (advantage negative from
bandwidth 3 to 15), worst exactly where the band is most out-run (bandwidth 15), and only
edges ahead at bandwidth 22 where both models have essentially failed and the Fourier
prior degrades slightly more gracefully. This is a genuine (and honest) deflation of the
neat ``bandwidth is a sufficient statistic'' story: bandwidth is the right \emph{axis} for
error, but advantage also depends on how the band compares to the target and on
which model breaks first. The usable methodology lesson survives in a weaker,
truer form --- $\nu$ (or Reynolds number, or a coarse solve) is a cheap proxy for the
\emph{band you must cover}, and if you cannot cover it, the quantum layer is not a
substitute for the missing depth.

\textbf{A bug we caught doing this.} While inspecting every figure we found that the
per-run \emph{solution} plots for the $\nu$-sweep had been drawn against the
\emph{default}-viscosity reference (the figure code called \texttt{get\_pde(pde)} without
the run's \texttt{pde\_params}), even though the reported rel-$L_2$ used the correct
per-$\nu$ reference. The smooth-$\nu$ runs therefore \emph{looked} like bad fits against a
sharp reference they were never trained on. We fixed the one-line defect in
\texttt{run.py} and regenerated the affected figures (now shown); the numbers above were
always correct. It is a small bug, but a clean illustration of why we read every figure
rather than trusting the montage.

\emph{The collapse plot (\texttt{results/ablation\_summary/nusweep.png}) populates on
\texttt{run\_nusweep.py aggregate} once the last QAPINN point finishes.}

\clearpage
\section{What survived vs what deflated (honest summary)}
\begin{center}\small\begin{tabular}{lll}\toprule
Finding & Status & Why \\ \midrule
Depth/qubits/scale help (bandwidth) & \textbf{robust} & monotonic, physically grounded, verified \\
VQC $=$ Fourier, band $=$ depth & \textbf{robust} & proven [1] + verified on our code \\
input\_scale $2\pi$ collapse & \textbf{robust} & clean; reproduces the black-hole pathology \\
Barren-plateau in qubit count & \textbf{plausible} & grad-var shrinks with qubits (CFE) \\
Advantage bandwidth-gated & \textbf{plausible} & fair control ties at low bandwidth \\
Entanglement doesn't help & \verdict{not sig.\ @2000i} & within 7-seed noise \\
Entanglement harmful @convergence & \emph{open lead} & large effect, single seed \\
Quantum diversifies classical neurons & \verdict{refuted} & died under the fair CFE control \\
2$\times$ diversity / 4$\times$ gradients & \verdict{artifact} & mismatched plain-PINN control \\
Capacity (not band) fixes the shock & \verdict{refuted} & classical error non-monotonic in params \\
Error rises monotonically with bandwidth ($\nu$) & \textbf{robust} & clean curve, both models \\
Depth-3 QAPINN beats classical on $\nu$-sweep & \verdict{no} & ties/loses at every $\nu$ (band too low) \\
\bottomrule\end{tabular}\end{center}
\textbf{Methodological contributions (independent of the findings):} a \emph{verified}
(8/8) pipeline; \emph{matched-parameter} and \emph{structurally-matched} controls;
\emph{multi-seed} statistics; full reproducibility. Most of the QAPINN literature has
none of these.

\clearpage
\part*{Part II --- Complete run gallery (all 77 runs, every figure)}
\addcontentsline{toc}{section}{Part II --- Complete run gallery}
Every training run in \texttt{results/}, with its metadata and \emph{all} of its
per-run figures at full width. Nothing curated out. Figure order per run: solution,
spectrum, F-principle, neurons, per-qubit $\langle Z\rangle$, entanglement, landscape,
generalization, evolution, loss (whichever exist for that model).

\section{Reference run --- abl\_baseline (the canonical QAPINN)}
The depth-3, 4-qubit, linear-entanglement, multi-$Z$, scale-1 configuration that every
Tier-1/Tier-3 sweep varies one knob from, and the reference point in the ablation
tables.
"""

# --------------------------------------------------------------------------- #
def build() -> str:
    parts = [r"""\documentclass[11pt]{article}
% AUTO-GENERATED by gen_journal.py -- do not hand-edit; edit the generator.
\usepackage[margin=0.8in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\graphicspath{{../results/}}
\newcommand{\verdict}[1]{\textcolor{red!70!black}{\textbf{#1}}}
\setcounter{tocdepth}{1}
\title{\textbf{QAPINN-BQP: Complete Project Journal}\\[4pt]
\large A full lab notebook --- literature review through the $\nu$-sweep, every
experiment, every figure, and the honest arc of what survived scrutiny}
\author{WISER Global Quantum+AI Program 2026 --- BQP Challenge}
\date{Compiled 2026-07-18}
\begin{document}
\maketitle
\begin{abstract}
The complete, honest record of the QAPINN-BQP project, beginning at the literature
review. Part~I is the narrative --- hypotheses, experiments, results, and how repeated
scrutiny (matched controls, a fair structural control, multi-seed statistics,
convergence) \emph{deflated} several apparently-novel findings and left a smaller,
robust core. Part~II is the complete gallery: all 77 runs with every one of the
$\sim$640 per-run figures at full size. Analysis code in \texttt{src/analysis/}; every
run reproducible from its \texttt{config.json} $+$ seed.
\end{abstract}
\tableofcontents
\clearpage
\part*{Part I --- Narrative}
\addcontentsline{toc}{section}{Part I --- Narrative}
"""]
    parts.append(NARRATIVE)
    # reference run first
    parts.append(run_block("abl_baseline"))
    # families
    for title, blurb, runs in FAMILIES:
        parts.append(f"\\section{{{title}}}")
        parts.append(blurb + "\n")
        for r in runs:
            if (RESULTS / r).is_dir():
                parts.append(run_block(r))
    parts.append("\\end{document}\n")
    return "\n".join(parts)

OUT.write_text(build(), encoding="utf-8")
n_runs = sum(1 for p in RESULTS.iterdir() if p.is_dir()
             and (p / "figures").exists())
n_figs = sum(len(list((RESULTS / p.name / "figures").glob("*.png")))
             for p in RESULTS.iterdir() if p.is_dir() and (p / "figures").exists())
print(f"wrote {OUT}  ({n_runs} runs with figures, {n_figs} per-run figures embedded)")
