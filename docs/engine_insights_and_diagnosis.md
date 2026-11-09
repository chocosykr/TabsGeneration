# Ukulele Transcription Engine: Deep Diagnosis & Acoustic Synthesis Analysis

*Documented: November 9, 2026*
*Repository Branch: `suyash`*

---

## 1. Executive Summary

During testing of real-world YouTube audio (specifically a solo ukulele cover of Radiohead's *Creep*), two critical failure modes were identified:
1. **Misclassification of Single Plucks**: Plucking a single note (**String 1 [A], Fret 2 = B4, ~493.88 Hz**) resulted in an erroneous chord detection with phantom notes (**A:2 + C:2**).
2. **Acoustic Realism Gap**: The synthesized Karplus-Strong audio does not sound like a genuine wooden ukulele, sounding instead like an electronic harpsichord or synthetic string.

This document records the underlying acoustic, mathematical, and algorithmic causes of these failures, and outlines the architectural dilemma of the *Analysis-by-Synthesis* paradigm.

---

## 2. Root Cause Analysis: The "Creep" Single Pluck Failure

When a single string is plucked, the system is expected to identify one note on one string. Instead, it hallucinated multiple notes. Three distinct phenomena occurred simultaneously:

```
[Real Acoustic Pluck: B4 (493.88 Hz)]
  │
  ├── 1. Forced Polyphony Constraint (UI / Call signature)
  │      └── UI sends `notes: 4`, strictly forcing the algorithm to fit 4 voices.
  │
  ├── 2. Wood Body & Helmholtz Air Cavity Resonances (~130–300 Hz)
  │      └── Energy from the wooden soundbox was mapped to low string notes (e.g. C:2 = D4, 293 Hz).
  │
  └── 3. Harmonic Overtone & Template Mismatch
         └── NNLS chord templates fitted overtones & room acoustics to a multi-note chord.
```

### A. Forced 4-Voice Polyphony Constraint
* The UI endpoint `/api/transcribe_segment` hardcoded `notes: 4`.
* The engine was never instructed to estimate the *number* of active voices. Forced to explain the audio with 4 simulated strings, it combed the spectrum for residual energy to assign to inactive strings rather than reporting silence.

### B. The "Wooden Box" Resonance (The Ghost C:2 Note)
* **What is C:2?** String 3 (C) fret 2 produces **D4 (293.66 Hz)**.
* When *any* note is plucked on an acoustic ukulele, the energy transfers through the bridge into the wooden body.
* The internal air cavity (Helmholtz resonance) vibrates strongly between **130 Hz and 160 Hz**, while the wooden soundboard plates resonate between **250 Hz and 320 Hz**.
* In a microphone recording, this wooden resonance creates a prominent low-frequency spectral peak around ~260–300 Hz. The NNLS chord template search mapped that physical body thump directly onto **D4 (C:2)**.

### C. Chord Search Pipeline on Monophonic Input
* The codebase contains two distinct paths:
  1. `num_notes = 1`: Uses **pYIN** (Probabilistic YIN), an autocorrelation-based pitch tracker that cleanly tracks single pitches.
  2. `num_notes = 4`: Uses **Non-Negative Least Squares (NNLS) Harmonic Template Matching**, which assumes a multi-note chord.
* Feeding a monophonic pluck into the chord template matcher resulted in fitting chord shapes against overtone spikes and room noise.

---

## 3. Why the Synthesized Sound Does Not Match Real Audio

The current Karplus-Strong implementation (`prototype/karplus_strong.py`) is an idealized mathematical string model. Four physical discrepancies prevent it from sounding like a physical ukulele:

| Component | Current Implementation | Real Acoustic Instrument | Acoustic Consequence |
| :--- | :--- | :--- | :--- |
| **Excitation** | Random white noise burst (`torch.randn`) | Nonlinear fingertip / plectrum release curve | Produces a sharp, clicky, synthetic "buzzy" attack rather than a warm, round pluck. |
| **Body Filtering** | None (vibrates in a vacuum) | Resonant wooden soundboard + soundhole cavity | Lacks woody thump, body warmth, and natural room coloration. |
| **Damping Profile** | Single global scalar decay multiplier | Frequency-dependent internal friction | High harmonics (3 kHz+) linger unnaturally long instead of damping in 20–30ms. |
| **Strum Spread** | Simultaneous excitation at $t=0$ | Hand sweeps across strings over 10–30 ms | Sounds like a digital block chord rather than a human strum. |

---

## 4. The Core Architectural Dilemma: Analysis-by-Synthesis Paradox

The foundational premise of TabsGeneration is:
> *Synthesize candidate tabs using Karplus-Strong, and choose the tab whose synthesized audio minimizes multi-scale STFT loss against the input recording.*

### The Paradox:
When evaluating STFT loss between a **dry mathematical synthesizer** and a **real microphone recording in a bedroom**:
1. **The STFT loss is dominated by acoustic artifacts**: Microphone frequency curve, room reverberation tails, ambient noise, and body resonance.
2. The distance between the real audio and the synthetic model is far larger than the subtle acoustic difference between Fret 2 and Fret 3.
3. Therefore, candidate rankings become noisy on real-world recordings, even while achieving 100% accuracy on synthetic tests.

---

## 5. Potential Solutions and Next Steps

1. **Dynamic Polyphony Detection**: Automatically detect whether the targeted segment contains 0, 1, 2, or 4 active notes before selecting the transcription pipeline.
2. **Body Impulse Response (IR) Convolution**: Convolve the Karplus-Strong string output with an acoustic ukulele body impulse response `.wav`.
3. **Harmonic Filtering**: Filter out non-musical frequencies (< 200 Hz and > 5 kHz) before template matching.
4. **Hybrid Detection**: Decouple pitch detection (using dedicated MIR/neural pitch detectors) from tab mapping (using physical inharmonicity and ergonomics).

