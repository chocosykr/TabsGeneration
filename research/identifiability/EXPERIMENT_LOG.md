# Research Experiment Log: Identifiability of Physical Parameters in Differentiable Karplus-Strong

This log tracks all experiments executed to evaluate the identifiability of physical parameters (decay, stretch/inharmonicity, energy) in the Differentiable Karplus–Strong model.

---

## Experiment Index

| ID | Date | Focus | Scope | Status | Key Conclusion |
|---|---|---|---|---|---|
| `AUD-001` | 2026-10-11 | Code Audit | Review of `karplus_strong.py`, `optimization.py`, test scripts | COMPLETED | Identified 6 critical flaws in previous experiments |
| `FM-001` | 2026-10-11 | Forward Model Validation | Effect of stretch on $f_1$, partial ratios, delay line | COMPLETED | $f_1$ stable ($<0.01\text{Hz}$ drift); dispersion monotonic but subtle ($<0.05\%$) |
| `LL-001` | 2026-10-11 | Loss Landscapes | 1D sweeps of stretch and decay | COMPLETED | Under fixed excitation, empirically convex unimodal minima exist on evaluated grids |
| `ST-001` | 2026-10-11 | Stretch-Only Recovery | Isolated stretch optimization under deterministic excitation | COMPLETED | Structurally identifiable: $<1\%$ error across range ($0.001\to 0.080$) |
| `DC-001` | 2026-10-11 | Decay-Only Recovery | Isolated decay optimization under deterministic excitation | COMPLETED | Structurally identifiable: $<0.02\%$ error across range ($0.950\to 0.999$) |
| `PC-001` | 2026-10-11 | Parameter Coupling | Joint recovery (stretch, decay, energy) | COMPLETED | No fatal coupling: all 3 parameters recovered simultaneously with $<0.5\%$ error |
| `NOI-001`| 2026-10-11 | Calibrated Noise Sweeps| Robustness under calibrated SNR (Clean, 40, 30, 20, 10 dB) | COMPLETED | Stretch converges to upper boundary at any SNR $\le 40\text{ dB}$ |
| `NOI-002`| 2026-10-11 | 40 dB Landscape & Gradient Audit| Fixed 1D sweep at 40 dB across seeds & gradient at truth | COMPLETED | Proves loss landscape genuinely shifts optimum to upper bound ($S\approx 0.094$), not an optimizer convergence stall |

---

## Log Entries

### AUD-001: Comprehensive Code Audit
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/code_audit.md`
- **Confirmed Findings:** Confirmed stochastic excitation regeneration bug, DC group delay compensation flaw, integer `.item()` truncation, and artificial decay penalties.

### FM-001: Forward Model Harmonic Analysis
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/forward_model_audit.md`, `research/identifiability/data/fm001_results.json`
- **Results:** Tested $S \in [0.0, 0.08]$. Measured fundamental $f_1$ at $392.0\text{ Hz}$ drifted by $< 0.01\text{ Hz}$. Partial ratios $r_n$ showed subtle, strictly monotonic downward dispersion. Decision gate PASSED.

### LL-001: Loss Landscape Topography
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/plots/ll001_stretch_landscape.png`, `research/identifiability/plots/ll001_decay_landscape.png`
- **Results:** Fixed excitation reveals empirical unimodal minimum at $S = 0.0301$ (target $0.030$) and $g = 0.9905$ (target $0.990$) over dense 100-point 1D grids. Random excitation produces flat noise floor of $\sim 2.53$.

### ST-001: Isolated Stretch Recovery
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/stretch_recovery.md`, `research/identifiability/data/st001_results.json`
- **Results:** 9 trials across 3 GT values ($0.001, 0.030, 0.080$) and 3 initializations. 8/9 converged with $<1\%$ error to true values. Proves structural identifiability under deterministic excitation in clean conditions.

### DC-001: Isolated Decay Recovery
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/decay_recovery.md`, `research/identifiability/data/dc001_results.json`
- **Results:** 12 trials across 4 GT values ($0.950, 0.980, 0.995, 0.999$) and 3 initializations. 12/12 converged with $<0.02\%$ error ($|\Delta g| \le 0.0003$). Proves decay structural identifiability without regularization penalties.

### PC-001: Parameter Coupling Analysis
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/parameter_coupling.md`, `research/identifiability/data/pc001_results.json`
- **Results:** Evaluated all 4 combinations (Decay+Energy, Stretch+Energy, Stretch+Decay, and Stretch+Decay+Energy). Joint optimization recovered all three parameters from distant initial values with $<0.5\%$ error. Confirms parameters do not hopelessly trade off against one another under fixed excitation.

### NOI-001: Calibrated SNR Noise Robustness
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/noise_robustness.md`, `research/identifiability/data/noi001_results.json`
- **Results:** Evaluated Clean, 40 dB, 30 dB, 20 dB, and 10 dB across 3 independent noise seeds per level. In clean audio, recovery error was $6 \times 10^{-6}$. At $40\text{ dB}$ and below, Adam converged to the sigmoid upper boundary ($S \to 0.0955$).

### NOI-002: 40 dB Landscape & Gradient Audit
- **Date:** 2026-10-11
- **Deliverable:** `research/identifiability/data/noi002_results.json`, `research/identifiability/plots/noi002_40db_stretch_landscapes.png`
- **Results:** Evaluated 100-point fixed grid on 40 dB noisy targets across 3 independent seeds. Global loss minimum shifted from $0.030$ to $0.0940$ (loss dropped from $5.628$ to $3.114$). The gradient at ground truth was negative ($d\mathcal{L}/d(\text{logit}) \approx -0.553$), confirming that the objective function itself prefers the upper-boundary solution under noise due to spectral comb broadening fitting the noise floor.
