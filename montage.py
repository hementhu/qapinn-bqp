"""Assemble per-category contact sheets across runs for quick visual comparison.

    python montage.py                      # default 4-run 2x2 grid
    python montage.py run1 run2 run3 ...   # custom set

Layout per category: columns = classical | QAPINN, rows = heat | Burgers, so each
sheet shows how one figure type varies across model and PDE. Writes to
results/_montage/<category>.png. QAPINN-only figures (qubits, entanglement) get a
1-row sheet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SHARED = ["solution", "spectrum", "fprinciple", "neurons", "landscape",
          "generalization", "evolution", "loss"]
QAPINN_ONLY = ["qubits", "entanglement"]


def _sheet(cells, categories, out_dir, results_root):
    """cells: 2D list of run names (rows x cols)."""
    rows, cols = len(cells), len(cells[0])
    for cat in categories:
        fig, axes = plt.subplots(rows, cols, figsize=(9.5 * cols, 5.2 * rows), squeeze=False)
        any_img = False
        for i in range(rows):
            for j in range(cols):
                ax = axes[i][j]; ax.axis("off")
                run = cells[i][j]
                p = Path(results_root) / run / "figures" / f"{cat}.png"
                if p.exists():
                    ax.imshow(mpimg.imread(p)); ax.set_title(run, fontsize=13); any_img = True
        if any_img:
            fig.suptitle(f"{cat.upper()}", fontsize=17, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(Path(out_dir) / f"{cat}.png", dpi=85)
        plt.close(fig)


def main():
    runs = sys.argv[1:] or ["heat_classical", "heat_qapinn", "burgers_classical", "burgers_qapinn"]
    results_root = "results"
    out = Path(results_root) / "_montage"
    out.mkdir(parents=True, exist_ok=True)

    # default 4-run layout: rows heat/burgers, cols classical/qapinn
    if runs == ["heat_classical", "heat_qapinn", "burgers_classical", "burgers_qapinn"]:
        grid = [["heat_classical", "heat_qapinn"], ["burgers_classical", "burgers_qapinn"]]
        qgrid = [["heat_qapinn", "burgers_qapinn"]]
    else:
        grid = [runs]
        qgrid = [[r for r in runs if "qapinn" in r]]

    _sheet(grid, SHARED, out, results_root)
    if qgrid[0]:
        _sheet(qgrid, QAPINN_ONLY, out, results_root)
    print(f"montages -> {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
