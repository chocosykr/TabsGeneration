# Fret Identifiability Analysis: Physical vs Model Parameters

**Author:** Antigravity Research Subsystem  
**Date:** 2026-10-11  
**Project:** Differentiable Physical Modeling for Ukulele Tablature Recovery  

---

## 1. Executive Summary

A critical question guiding this research is:
> *"Does the successful recovery of `stretch` from audio prove that `stretch` uniquely identifies which fret or string produced the note?"*

**Scientific Answer: NO.**  
A parameter can be mathematically identifiable in a digital waveguide model without being a reliable physical fingerprint for string/fret assignment on a real instrument.

This document establishes the rigorous physical and mathematical distinctions between:
1. **Model Parameter Recovery** (Level 1: Inverse simulation)
2. **Physical Parameter Estimation** (Level 2: Acoustic instrument measurement)
3. **Fret/String Disambiguation** (Level 3: Tablature inference)

---

## 2. The Three Levels of Identifiability

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Inverse Parameter Estimation (Mathematical)        │
│   Target: Waveform from DifferentiableKarplusStrong(θ*)    │
│   Question: Can Adam find θ* from STFT loss?                 │
│   Status: VERIFIED (Exp ST-001: <1% error on clean KS audio) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 2: Physical Parameter Measurement (Acoustical)         │
│   Target: Audio from real physical ukulele                  │
│   Question: Does a nylon/fluorocarbon string obey this 1st- │
│             order all-pass dispersion model?                 │
│   Status: UNVERIFIED / QUESTIONABLE                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 3: Fret / String Fingerprinting (Inference)           │
│   Target: Predict fret index f ∈ {0, 1, ..., 15}            │
│   Question: Is stretch(f) strictly monotonic and invariant   │
│             across plucking position, force, and string age?│
│   Status: UNJUSTIFIED WITHOUT CALIBRATION                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Physical String Mechanics vs. First-Order All-Pass Filter

### 3.1 The Physical Physics of Inharmonicity
In a real stiff string (Fletcher & Rossing, *The Physics of Musical Instruments*), the partial frequencies follow:
$$f_n = n f_0 \sqrt{1 + B n^2}$$
where the inharmonicity coefficient $B$ is governed by physical constants:
$$B = \frac{\pi^3 Q d^4}{64 T L^3}$$
- $Q$: Young's modulus of the string material
- $d$: String diameter
- $T$: String tension
- $L$: Speaking string length (which shortens with fret index $f$ as $L(f) = L_0 \cdot 2^{-f/12}$)

Notice that physical inharmonicity **increases quadratically with partial index $n$**:
$$\frac{f_n}{n f_1} \approx 1 + \frac{1}{2} B n^2$$
This causes upper partials to be stretched **upward** ($f_n > n f_0$).

### 3.2 What the First-Order All-Pass Filter Actually Does
In `prototype/karplus_strong.py`:
$$H_{\text{ap}}(z) = \frac{S + z^{-1}}{1 + S z^{-1}}, \quad S \in [0, 0.1]$$
As measured in `FM-001`, this filter introduces phase delay that causes a slight **downward** shift in upper partials relative to the fundamental ($r_n < 1.0$), not the upward quadratic dilation of a stiff string!

Therefore, the parameter named `stretch` in the model is **not physical string stiffness $B$**; it is an arbitrary dispersion filter parameter that alters loop phase.

---

## 4. Why Model Recovery Cannot Justify a "Stretch $\to$ Fret" Lookup Table

1. **Plucking Position Confounder:**
   Plucking a string near the bridge (ponticello) excites high harmonics heavily, while plucking near the fretboard (tasto) excites almost purely the fundamental. In a real recording, spectral centroid variations caused by plucking technique can easily dwarf the $< 0.05\%$ partial shift created by the all-pass filter.
2. **Instrument Variation & String Gauges:**
   Ukuleles typically use four different string gauges (e.g., standard GCEA tuning with a thin A-string and a thicker C-string). The inharmonicity of String 3 (C4, thick) at fret 0 may be higher than String 1 (A4, thin) at fret 3. Fret number alone does not determine inharmonicity.
3. **No Empirical Ukulele Calibration Data:**
   Because the ukulele is broken and only three chord mixtures are available without isolated single-string ground-truth calibration recordings, any hardcoded mapping of `stretch = 0.03 -> Fret 3` would be an ungrounded heuristic fitting.

---

## 5. Legitimate Future Role for the Physical Model

While `stretch` cannot currently be used as a heuristic fret classifier, the Karplus–Strong model serves two legitimate roles:
1. **Generative Consistency Validation:**
   Verifying whether a predicted chord shape is physically plausible as an ensemble of vibrating strings.
2. **Decay / Sustain Estimation:**
   As proven in `DC-001`, `decay` is extraordinarily identifiable ($|\Delta g| \le 0.0003$). Damping provides real physical information: fretted strings (where a finger dampens vibration) exhibit shorter sustain than open ringing strings.

### Conclusion:
- Do not map `stretch` directly to fret numbers.
- Base tab fingering disambiguation on **spectral template multiplicity** and **instrument biomechanics / fingering ergonomics**.
