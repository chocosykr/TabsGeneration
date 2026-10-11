# Proposed Tablature Experiment: Spectral Multiplicity & Ergonomic Disambiguation

**Document ID:** `PROP-TAB-001`  
**Date:** 2026-10-11  
**Target Audio:** Three real ukulele recordings (`input/g maj.wav`, `input/g min.wav`, `input/e min.wav`)  
**Status:** PROPOSAL / SPECIFICATION (Not yet implemented)  

---

## 1. Context & Motivation

Phase 1–6 established that `stretch` cannot serve as a reliable fret fingerprint in noisy real-world audio.
However, chords on a 4-string ukulele frequently produce notes with **multiplicity** (e.g., in E minor $\{E_4, G_4, B_4\}$, four strings are played, meaning either $E_4$ or $G_4$ is doubled).

The core tablature inference challenge is:
> *Given a real acoustic chord recording containing polyphonic pitch mixtures, how accurately can candidate chord shapes be ranked and fingered without relying on inharmonicity fingerprints?*

---

## 2. Experimental Design

This benchmark compares three scoring methods against the known ground-truth fingerings across the three available chord recordings.

### 2.1 The Three Evaluated Approaches:

1. **Baseline 1: Current Spectral Candidate Ranking (`baseline_spectral`)**
   - The existing heuristic FFT spectral scorer from `fretboard_mapper.py`.
   - Evaluates whether predicted chord shapes match the dominant FFT peaks, without explicit harmonic template deconvolution or multiplicity constraints.

2. **Approach 2: Non-Negative Least Squares Harmonic Multiplicity (`nnls_multiplicity`)**
   - Synthesizes an idealized spectral template dictionary $T \in \mathbb{R}^{K \times 4}$ for the 4 candidate strings based on harmonic series models ($f_0, 2f_0, \dots, M f_0$).
   - Solves the constrained NNLS problem against the observed STFT magnitude vector $\mathbf{y}$:
     $$\min_{\mathbf{w} \ge 0} \|\mathbf{y} - T \mathbf{w}\|_2^2$$
   - Measures how well the predicted multiplicity of strings explains the observed harmonic comb intensities without penalizing unobserved spectral nulls.

3. **Approach 3: Harmonic Multiplicity with Ergonomic Constraints (`nnls_ergonomic`)**
   - Augments the NNLS spectral score with a biomechanical hand-feasibility prior:
     $$\text{Score} = \text{Score}_{\text{NNLS}} - \lambda_{\text{span}} \cdot \text{SpanPenalty} - \lambda_{\text{fingers}} \cdot \text{FingerCountPenalty}$$
   - **Span Penalty:** $\max(\text{fret}) - \min(\text{fret} > 0) > 4$ (excessive finger stretch).
   - **Finger Count Penalty:** Penalizes finger assignments exceeding 4 fingers.
   - Rewards open strings ($fret = 0$) and standard playable grip geometry.

---

## 3. Evaluation Dataset & Ground Truth

| File | Chord Name | Strummed Strings | True Fret Fingerings | Notes Produced | Note Multiset |
|---|---|---|---|---|---|
| `input/g maj.wav` | G Major | Strings 1, 2, 3, 4 | $(2, 3, 2, 0)$ | $B_4, G_4, D_4, G_4$ | $\{D_4, G_4, G_4, B_4\}$ |
| `input/g min.wav` | G Minor | Strings 1, 2, 3, 4 | $(1, 3, 2, 0)$ | $A^\sharp_4, G_4, D_4, G_4$ | $\{D_4, G_4, G_4, A^\sharp_4\}$ |
| `input/e min.wav` | E Minor | Strings 1, 2, 3, 4 | $(2, 3, 4, 0)$ or $(0, 4, 3, 2)$ | $B_4, G_4, E_4, G_4$ | $\{E_4, G_4, G_4, B_4\}$ |

---

## 4. Evaluation Metrics

1. **Ground Truth Candidate Rank (Primary Metric):**
   - The rank index ($1 \dots N$) of the true fingering among all generated candidate hypotheses (lower is better, Rank 1 is optimal).
2. **Top-1 Exact Fingering Accuracy:**
   - Percentage of chords where the Rank 1 prediction matches the true 4-tuple of frets exactly.
3. **Top-1 Note-Set Accuracy:**
   - Percentage of chords where the Rank 1 prediction matches the correct multiset of notes, even if fret assignments differ.
4. **Spectral Residual Error:**
   - The normalized residual $\|\mathbf{y} - \hat{\mathbf{y}}\|_2 / \|\mathbf{y}\|_2$ between observed audio spectrum and reconstructed candidate spectrum.

---

## 5. Explicit Limitations & Scientific Caveats

> [!WARNING]
> **Sample Size Limitation:**
> Three recordings are insufficient to establish general statistical validity across instruments, players, tempos, or dynamic ranges. 
> This proposed experiment serves strictly as a **controlled benchmark** to measure whether NNLS multiplicity and ergonomic constraints resolve known failure cases (such as the E-minor ranking collapse previously documented) without breaking previously solved chords (G major, G minor).
> No claims of general production readiness will be made without broader multi-instrument validation.
