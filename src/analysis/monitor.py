"""Monitor: collect probe time-series during training and save to .npz.

A probe returns a 1-D numpy array each time it fires (e.g. the spatial spectrum,
or per-layer gradient variances). The Monitor stacks each named series over
iterations so the dynamics (how the spectrum / gradients evolve during training)
can be analysed post-hoc — this is what powers the F-principle and barren-plateau
figures. Scalars are stored as length-1 arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Monitor:
    def __init__(self):
        self._iters: dict[str, list[int]] = {}
        self._vals: dict[str, list[np.ndarray]] = {}
        self._labels: dict[str, list[str]] = {}   # optional per-column labels

    def record(self, name: str, iteration: int, value, labels: list[str] | None = None) -> None:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self._iters.setdefault(name, []).append(iteration)
        self._vals.setdefault(name, []).append(arr)
        if labels is not None:
            self._labels[name] = labels

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (iters (T,), values (T, dim)) for one probe."""
        return np.array(self._iters[name]), np.stack(self._vals[name])

    def save(self, path: str | Path) -> None:
        """Save all series to one .npz: <name>_iters and <name>_values per probe."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, np.ndarray] = {}
        for name in self._iters:
            out[f"{name}_iters"] = np.array(self._iters[name])
            out[f"{name}_values"] = np.stack(self._vals[name])
            if name in self._labels:
                out[f"{name}_labels"] = np.array(self._labels[name])
        np.savez_compressed(path, **out)

    @property
    def names(self) -> list[str]:
        return list(self._iters.keys())
