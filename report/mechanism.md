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
- **Feature-redundancy hypothesis: only partially supported.** Entanglement raises the per-qubit feature correlation (none 0.28 -> linear 0.44), the direction predicted -- but linear's seed spread is large (+/-0.26) so the bars overlap. Post-net effective rank is slightly lower with entanglement (4.76 vs 5.12), also the predicted direction but overlapping. The raw VQC-feature rank goes the OTHER way (linear higher), confounded by dead/near-constant channels in the unentangled model.
- **Verdict:** the 'entanglement redundifies the readout's feature basis' story is *directionally consistent* (correlation up, post-net rank down) but NOT statistically clean at n=3 -- one of three metrics is confounded and all error bars overlap. The accuracy harm is real; a confirmed mechanism needs more seeds and a dead-channel-robust rank measure. We report this as a supported-but-unconfirmed hypothesis, not a result.
