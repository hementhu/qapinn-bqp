"""Lightweight, dependency-free run logging (CSV history + JSON summary).

No wandb / accounts — the repo must be reproducible from a clean checkout.
- ``RunLogger.log_step`` appends a row to ``history.csv`` (loss curves, etc.).
- ``RunLogger.write_summary`` dumps final metrics + config to ``summary.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes per-step history to CSV and a final summary to JSON."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.csv"
        self.summary_path = self.run_dir / "summary.json"
        self._fieldnames: list[str] | None = None

    def log_step(self, **fields: Any) -> None:
        """Append one row of scalars to history.csv (header written once).

        The first write truncates any pre-existing file (mode "w") so re-running
        an experiment starts a clean history instead of stacking onto the old
        run's rows; subsequent writes append.
        """
        write_header = self._fieldnames is None
        mode = "w" if write_header else "a"
        if write_header:
            self._fieldnames = list(fields.keys())
        with self.history_path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(fields)

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write the final metrics + config dict to summary.json."""
        with self.summary_path.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
