# Laptop-2 Runbook (parallel experiments)

Second machine runs a **different** slice of experiments so nothing overlaps.
Laptop 1 does: stability batch + nu-sweep. Laptop 2 does: **seed expansion**
(seeds 3-5 on the entanglement headline), and optionally more.

## 1. Setup (once)
```bash
# unzip the qapinn-bqp bundle, then from inside qapinn-bqp/:
python -m venv .venv
.venv\Scripts\activate                      # Windows;  source .venv/bin/activate on Linux/mac
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Verify the environment is correct (must pass 8/8)
```bash
python verify_pipeline.py
```
If it prints `8/8 checks passed`, the machine reproduces the pipeline exactly.

## 3. Run laptop-2's slice
```bash
python run_seeds.py preflight 3 4 5         # sanity: param counts, run list
python run_seeds.py 3 4 5                    # runs seeds 3,4,5 (entanglement + classical)
```
~12 QAPINN runs (~35 min each) + 3 classical (fast) => ~7 h. It skips anything
already done, so it is safe to stop/restart.

Watch progress: `tail results/<name>/history.csv`, or `ls results/seed*`.

## 4. (Optional) more work if laptop-2 is free
```bash
python run_seeds.py 6 7                       # even more seeds
python run_ablation.py run --tier 3           # if not already done here
```

## 5. Send results back to laptop 1
The only outputs are under `results/seed*/`. Copy those folders to laptop 1's
`results/` directory (they are uniquely named, no collisions). On laptop 1:
```bash
# after copying results/seed3_* etc. into laptop-1 results/
# re-run whatever aggregation uses the seeds (to be added), or just inspect summary.json
```

## Notes
- Same `seed` + same config => bit-identical result on any machine (verified).
- Do NOT run the same config name on both machines (wasted compute). The split
  above (laptop 1 = seeds 1-2, laptop 2 = seeds 3-5) guarantees no overlap.
- `default.qubit` (CPU) only; no GPU needed.
