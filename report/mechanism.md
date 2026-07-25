# Mechanism: why entanglement hurts (converged, n=3)

Comparing cv_qd3_none (no entanglement) vs cv_qd3_lin (linear), 3 seeds each.

| metric | none | linear | direction |
|---|---|---|---|
| rel-L2 (worse=higher) | 0.369±0.022 | 0.424±0.005 | linear higher |
| mean |feature corr| (redundancy) | 0.282±0.026 | 0.441±0.255 | linear higher |
| VQC feature eff-rank (basis richness) | 2.544±0.570 | 3.154±0.449 | linear higher |
| post-net hidden eff-rank | 5.123±0.802 | 4.763±0.640 | none higher |
| mean entanglement entropy | 0.000±0.000 | 0.801±0.065 | linear higher |

none feature max-harmonics per seed: [[3, 127, 2, 127], [2, 127, 4, 127], [4, 127, 3, 127]]
linear feature max-harmonics per seed: [[14, 32, 51, 11], [3, 40, 8, 21], [21, 21, 2, 5]]

## Reading (honest, n=3)
- Entanglement entropy: none=0.000±0.000 (~0 sanity check), linear=0.801±0.065 (genuinely entangled) -- as expected.
- **Accuracy effect is solid:** none 0.369±0.022 vs linear 0.424±0.005 (p~0.01, separate analysis). WHY is the open part.
- **Feature-redundancy hypothesis: NOT confirmed -- the metrics conflict.** Entanglement raises per-qubit feature correlation (none 0.28 -> linear 0.44, the predicted direction, but with large seed spread +/-0.26). However the VQC feature effective rank *also* rises with entanglement (2.54 -> 3.15), the OPPOSITE of the 'poorer basis' prediction -- and this survives dropping dead channels, so it is real variance structure, not an artifact. Post-net hidden rank is essentially a wash (5.12 vs 4.76, overlapping).
- **Verdict:** we do NOT have a confirmed representational mechanism. One metric supports 'poorer basis' (correlation up), one contradicts it (rank up), one is neutral. The accuracy harm (entanglement hurts, p~0.01) is real, but *why* remains open. The conflict suggests the cause may be **trainability** (entanglement roughens the loss landscape / adds optimization cost) rather than a poorer feature basis -- a hypothesis the larger-model + landscape follow-up is designed to probe. Reported honestly as an open question.
