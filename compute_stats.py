"""Compute every statistic cited in the paper directly from results/*/summary.json.

Replaces the hand-typed p-values in make_master_table.py. Reports mean, sample
std (ddof=1), n, and two-sided Welch (unequal-variance) t-test p-values, with
Student and Mann-Whitney U alongside for robustness. Writes report/stats.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path("results")


def rel(name):
    p = R / name / "summary.json"
    return json.load(open(p))["rel_l2"] if p.exists() else None


def seeds(base, idx=range(9)):
    return [v for v in (rel(f"{base}_s{i}") for i in idx) if v is not None]


def describe(v):
    v = np.asarray(v)
    return dict(n=len(v), mean=float(v.mean()), std=float(v.std(ddof=1)),
                min=float(v.min()), max=float(v.max()), values=[float(x) for x in v])


def compare(a, b):
    return dict(welch_p=float(stats.ttest_ind(a, b, equal_var=False).pvalue),
                student_p=float(stats.ttest_ind(a, b, equal_var=True).pvalue),
                mannwhitney_p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))


GROUPS = {
    # small (121-param) converged core
    "cv_cls": seeds("cv_cls"),
    "cv_cfe": seeds("cv_cfe"),
    "cv_qd5_lin": seeds("cv_qd5_lin"),
    "cv_qd3_lin_n3": seeds("cv_qd3_lin", range(3)),
    "cv_qd3_none_n3": seeds("cv_qd3_none", range(3)),
    "cv_qd3_lin": seeds("cv_qd3_lin"),
    "cv_qd3_none": seeds("cv_qd3_none"),
    # larger (~1273-param) regime + fair controls
    "big_qd3_none": seeds("big_qd3_none"),
    "big_qd3_lin": seeds("big_qd3_lin"),
    "big_cls_deep": seeds("big_cls_deep"),
    "big_cfe": seeds("big_cfe"),
    "big_cls_bal": seeds("big_cls_bal"),
    "big_cls": seeds("big_cls"),
}

COMPARISONS = [
    ("cv_qd3_none_n3", "cv_qd3_lin_n3", "entanglement, small model, n=3"),
    ("cv_qd3_none", "cv_qd3_lin", "entanglement, small model, n=6"),
    ("cv_cls", "cv_qd3_lin", "small: classical vs reference QAPINN"),
    ("cv_cls", "cv_qd5_lin", "small: classical vs best QAPINN"),
    ("cv_cls", "cv_qd3_none", "small: classical vs QAPINN-none (n=6)"),
    ("cv_cfe", "cv_qd5_lin", "small: CFE control vs best QAPINN"),
    ("big_qd3_none", "big_qd3_lin", "large: entanglement"),
    ("big_qd3_none", "big_cls", "large: QAPINN vs shallow classical"),
    ("big_qd3_none", "big_cls_bal", "large: QAPINN vs balanced classical"),
    ("big_qd3_none", "big_cls_deep", "large: QAPINN vs depth-matched classical"),
    ("big_qd3_none", "big_cfe", "large: QAPINN vs structural CFE control"),
]


def main():
    desc = {k: describe(v) for k, v in GROUPS.items() if v}
    comp = []
    for a, b, label in COMPARISONS:
        c = compare(GROUPS[a], GROUPS[b])
        comp.append(dict(a=a, b=b, label=label, **c))
    # sensitivity: drop the anomalous big_qd3_none seed (largest value)
    none = sorted(GROUPS["big_qd3_none"])[:-1]
    comp.append(dict(a="big_qd3_none_drop_max", b="big_cls_deep",
                     label="large: depth-matched, anomalous QAPINN seed dropped",
                     **compare(none, GROUPS["big_cls_deep"])))

    print(f"{'group':18s} {'n':>2s} {'mean':>7s} {'sd':>7s}")
    for k, d in desc.items():
        print(f"{k:18s} {d['n']:2d} {d['mean']:7.4f} {d['std']:7.4f}")
    print(f"\n{'comparison':55s} {'Welch':>7s} {'Student':>8s} {'MWU':>7s}")
    for c in comp:
        print(f"{c['label']:55s} {c['welch_p']:7.4f} {c['student_p']:8.4f} {c['mannwhitney_p']:7.4f}")
    Path("report/stats.json").write_text(json.dumps(dict(groups=desc, comparisons=comp), indent=2))
    print("\nwrote report/stats.json")


if __name__ == "__main__":
    main()
