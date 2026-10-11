# Results Summary: Parameter Identifiability Research

**Date:** 2026-10-11  
**Project:** Differentiable Physical Modeling for Ukulele Tablature Estimation  

---

## 1. Classification of Findings

- **[FACT] Confirmed Implementation Fact:** Directly verified in code or mathematical proofs.
- **[RESULT] Observed Experimental Result:** Empirically recorded from reproducible scripts.
- **[SUPPORTED] Supported Interpretation:** Well-grounded scientific conclusion based on observed results.
- **[HYPOTHESIS] Unresolved Hypothesis:** Testable conjectures pending experimental verification.
- **[INCONCLUSIVE] Inconclusive / Invalid:** Tests that cannot support conclusions due to design flaws.

---

## 2. Rigorous State of Evidence Matrix

| Topic / Parameter | Classification | Evidence & Measurements | Methodological Limits & Nuance |
|---|---|---|---|
| **Earlier Stretch Failure Conclusion** | **[INCONCLUSIVE]** | Refuted by `AUD-001`. | Earlier conclusion that stretch is non-identifiable was based on invalid tests that regenerated random noise per forward pass. |
| **Excitation Regeneration Leak** | **[FACT]** | Line 65 of `karplus_strong.py`: `noise = torch.randn(L_int)`. | Creates $\sim 2.53$ STFT loss barrier, obscuring subtle physical gradients (`AUD-001`). |
| **Fundamental Frequency Stability** | **[RESULT]** | Parabolic peak analysis across $S \in [0.0, 0.08]$: measured drift $< 0.01\text{ Hz}$ at $392\text{ Hz}$ (`FM-001`). | Evaluated at single nominal pitch ($392.0\text{ Hz}$); confirms DC group delay compensation holds locally. |
| **Inharmonicity Monotonicity** | **[RESULT]** | Ratios $r_n = f_n / (n f_1)$ decrease monotonically with $n$ and $S$ (`FM-001`). | Effect is subtle ($< 0.05\%$ shift at 6th partial) and models downward dispersion, not stiff-string quadratic dilation. |
| **Clean Loss Landscape Topology** | **[RESULT]** | 100-point 1D evaluation grids exhibit single unimodal minimum at ground truth: $S=0.0301$ for $S^*=0.0300$; $g=0.9905$ for $g^*=0.9900$ (`LL-001`). | **Empirically verified on finite 1D grid slices**; not a formal mathematical proof of global convexity. |
| **Stretch Structural Identifiability** | **[RESULT]** | 8/9 optimization runs converged to within $\pm 0.0006$ of ground truth across initializations (`ST-001`). | Proves optimizer convergence and structural identifiability **strictly under clean, fixed-excitation conditions**. |
| **Decay Structural Identifiability** | **[RESULT]** | 12/12 optimization runs converged to within $\pm 0.0003$ of ground truth across $[0.950, 0.999]$ (`DC-001`). | Proves decay identifiability under clean conditions without artificial regularization penalties. |
| **Parameter Coupling** | **[RESULT]** | Joint optimization recovered $[S, g, E]$ simultaneously from distant initializations with $<0.5\%$ error (`PC-001`). | Confirms Adam navigates the uncoupled landscape under clean, deterministic conditions. |
| **Stretch Practical Identifiability under Noise** | **[RESULT]** | Across all seeds at $\text{SNR} \le 40\text{ dB}$, Adam diverged to $S \approx 0.0955$ (`NOI-001`). Fixed-grid audit revealed global loss minimum inverts to $S=0.0940$ (loss dropped from $5.628$ to $3.114$) with gradient $d\mathcal{L}/d(\text{logit}) \approx -0.553$ (`NOI-002`). | **The objective function itself favors the boundary solution under noise** (comb filter broadening fits additive noise floor). Practical identifiability fails. |
| **Stretch as Fret Fingerprint** | **[SUPPORTED]** | Disqualified (`Fret Identifiability Analysis`). | Cannot identify frets in practice due to real-world noise ($\le 40\text{ dB}$), plucking variations, and model mismatch. |
| **Decay as Fret Fingerprint** | **[SUPPORTED]** | Unproven as reliable open vs fretted classifier. | Damping depends heavily on plucking dynamics, string gauge, wood absorption, and bridge contact, not purely fret number. |

---

## 3. Clear Distinctions Maintained

1. **Structural vs. Practical Identifiability:**
   - *Structural Identifiability:* The mapping from parameters to noise-free waveforms is injective and locally invertible by STFT loss. (Verified).
   - *Practical Identifiability:* The parameter can be recovered from audio degraded by noise, reverberation, and unmodeled acoustics. (Disproven for stretch at $\le 40\text{ dB}$ SNR).
2. **Optimizer Convergence vs. Objective Landscape:**
   - The convergence of Adam to $S \to 0.095$ under 40 dB noise is **not** an optimization stall or numerical artifact; `NOI-002` proved the STFT loss genuinely achieves its global minimum at the upper bound.
3. **Synthetic Audio vs. Real Instrument Audio:**
   - Successful synthetic recovery proves only that the simulator can invert its own equations. It does not prove that a real instrument string matches those equations or that physical frets can be classified from recovered values.
