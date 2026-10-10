# TabsGeneration — Problems, Bugs & Design Decisions Log

A chronological record of every major problem encountered during development,
its root cause, and how it was resolved or is currently being investigated.

---

## Problem 1 — Narrow Dynamic Range for Decay Parameter

**Status:** ✅ Resolved  
**Affected File:** `karplus_strong.py`

### Symptom
The Karplus-Strong optimizer was converging to nearly identical decay values for
all strings, regardless of the input audio. The decay parameter had no useful
discriminatory power between fretted and open strings.

### Root Cause
The `logit_decay` parameter was mapped through a `sigmoid` into the range `[0, 1]`,
but the practical range of meaningful decay values for a ukulele string is very
narrow (e.g. `0.990` to `0.9999`). The optimizer was saturating near the top of
the sigmoid, effectively collapsing the dynamic range.

### Fix
Clamped and re-scaled the sigmoid output to the range `[0.90, 0.9999]` to give
the optimizer useful gradients throughout the physically plausible range.

---

## Problem 2 — No Inharmonicity (Stretch) Model

**Status:** ✅ Resolved  
**Affected File:** `karplus_strong.py`

### Symptom
The differentiable string model had no way to simulate or measure the slight
sharpening of overtones (inharmonicity / "stretch") that characterises a real
plucked string. This meant the optimizer could only fit decay and energy, not
the physical stiffness of the string.

### Root Cause
The original Karplus-Strong delay line used a pure first-order low-pass filter.
A real string has bending stiffness, which causes overtones to be slightly sharp
relative to a perfect harmonic series. This is modelled by an All-Pass filter in
the delay loop.

### Fix
Added a parameterized All-Pass filter stage (`logit_stretch`) in the Karplus-Strong
delay loop. The `stretch` parameter now captures inharmonicity. Open strings
calibrate to `stretch ≈ 0.002`; fretted strings saturate quickly (`stretch ≈ 0.10`
by fret 3).

---

## Problem 3 — Fretboard Mapper Relied on Inharmonicity to Fix Spectral Mistakes

**Status:** ✅ Resolved (Root Cause Fixed via Architecture Change)  
**Affected Files:** `fretboard_mapper.py`, `pitch_estimation.py`

### Symptom
The pipeline worked for G major and G minor chords but completely failed for
E minor. The E minor chord was being transcribed as a wrong chord.

### Root Cause (Deep)
The pipeline had a fundamental information-loss problem:

```
BEFORE (broken):
  audio → spectral scorer picks single best note SET → optimizer → fretboard mapper
```

The spectral scorer was returning, for example, `{E4, E4, G4, B4}` for E minor
instead of the correct `{E4, G4, G4, B4}`. Once the pitch SET was collapsed to
a single guess, the later stages had no way to recover the missing G4.

The true E minor shape is `g:0, c:4, e:3, a:2`:
- The open G string (g:0) produces G4.
- The c-string fret 4 produces E4.
- The e-string fret 3 produces G4 (unison with g-string).
- The a-string fret 2 produces B4.

Because the pipeline expected a distinct note SET (no duplicates), the two G4s
were collapsed, and the optimizer was given the wrong pitches to fit.

### Fix
Changed `estimate_chord_shape` in `pitch_estimation.py` to return the top-N
distinct note **multisets** (sorted tuples allowing duplicates, e.g. `(E4, G4, G4, B4)`)
instead of a single note set. This preserves unison-doubling information and
passes the correct multiplicities to the optimizer.

---

## Problem 4 — Single-Candidate Pipeline (No Multi-Hypothesis Evaluation)

**Status:** ✅ Resolved  
**Affected File:** `main.py`

### Symptom
The `main.py` pipeline took the single best output from the spectral scorer and
passed it straight to the optimizer. If the spectral scorer ranked the true chord
poorly, there was no recovery path.

### Root Cause
The architecture was a strict linear pipeline with no branching. The spectral
scorer's single choice was treated as ground truth for all downstream stages.

### Fix
Implemented a **Top-N multi-hypothesis evaluation loop** in `main.py`:
1. The spectral scorer now returns the top-N distinct multisets (configurable `TOP_N`, default 8).
2. The optimizer runs **independently** for each candidate multiset, fitting the
   Karplus-Strong physics model to that specific set of pitches.
3. Each candidate receives a combined score: `α × spectral_norm + β × physics_norm`.
4. The combined-score winner is selected as the final output.

---

## Problem 5 — Diagnostic Script Had Wrong String-to-Fret Mapping

**Status:** ✅ Resolved  
**Affected File:** `diagnose_ranking.py`

### Symptom
Running `diagnose_ranking.py` showed the true G major shape (`g:0, c:2, e:3, a:2`)
ranked `#48,515 / 65,536` — nearly dead last. This made us believe the spectral
scorer was fundamentally broken for G major and G minor.

### Root Cause
The diagnostic script was interpreting fret order as `(A, E, C, G)` instead of
the correct ukulele string order `(G, C, E, A)`. The true shape frets were being
passed in the wrong order, comparing apples to oranges.

### Fix
Corrected the string order in the `--true-frets` argument and in the diagnostic
display. After fixing, the true G major and G minor shapes were confirmed to rank
**#1** by the spectral scorer — the scorer was correct all along.

---

## Problem 6 — Optimizer Overfitting: Muting Wrong Strings to Cheat the Loss

**Status:** 🔬 Fix Applied — Results Pending  
**Affected File:** `optimization.py`

### Symptom
After implementing the Top-N multi-hypothesis evaluation loop, the combined
physics+spectral score was consistently awarding the **wrong** candidate the
highest score. For example, for G major, the winner was `g:7, c:2, e:3, a:2`
(a high D5 on the g-string at fret 7) rather than the true `g:0, c:2, e:3, a:2`.

### Root Cause
The Karplus-Strong optimizer had complete freedom over the `decay` and
`excitation_energy` parameters of every voice. When given a wrong candidate (e.g.,
one that contains D5 instead of G4), the optimizer would simply **drop the energy
or decay of the wrong string to near-zero**, effectively muting it. Because the
muted string made no sound, the STFT loss wasn't penalized at all. The `stretch`
parameter for the muted string would then float to a random value, which would
sometimes coincidentally score high in the fretboard mapper.

```
The cheat cycle:
  Wrong pitch given → optimizer mutes that string → STFT loss unaffected
  → stretch floats randomly → fretboard mapper scores randomly → sometimes "wins"
```

### Fix Applied
Added a `decay_penalty` term to the optimizer loss function in `optimization.py`:

```python
# Penalise any active string whose decay drops below 0.95
decay_penalty = sum(relu(0.95 - decay_i) for each active voice)
loss = stft_loss + (10.0 * decay_penalty)
```

This forces every active (strummed) voice to sustain at full volume. A wrong note
can no longer be hidden — it must ring out, directly contributing to the STFT
loss and exposing the spectral mismatch between the wrong synthetic note and the
real recording.

---

## Problem 7 — E Minor True Shape Ranked #7 by Spectral Scorer

**Status:** 🔬 Under Investigation  
**Affected File:** `pitch_estimation.py`

### Symptom
After fixing the string mapping bug (Problem 5), G major and G minor true shapes
rank #1 spectrally. However, E minor's true shape (`g:0, c:4, e:3, a:2`) still
only ranks #7 out of 65,536 shapes.

### Root Cause (Hypothesis)
The harmonic template scorer awards energy proportional to `mag[bin] / harmonic_index`.
A candidate containing B5 can "claim" the very bright spectral peak at the B5
frequency as its **fundamental** (earning `mag / 1`). The true shape correctly
attributes that same energy to the 2nd harmonic of the B4 string (earning only
`mag / 2`). The fundamental claim always wins, rewarding candidates that
hallucinate real strings at overtone frequencies.

### Potential Fixes (Not Yet Applied)
1. **Ergonomic penalty:** Increase `_ERGO_WEIGHT` from `0.001` to something
   larger (e.g. `0.5`) to strongly prefer open/low-fret shapes, making it harder
   for high-fret hallucinations to win.
2. **Overtone discount:** If a candidate shape contains note X where X is a
   near-integer harmonic of another note Y in the same shape, discount the score
   contribution of X's fundamental, since that spectral energy is already explained
   by Y's harmonic series.

---

## Open Questions / Next Experiments

| Question | Experiment |
|---|---|
| Does the decay constraint (Problem 6 fix) fully solve optimizer overfitting? | Compare rankings before/after — in progress |
| Does spectral rank-7 for E minor affect final output after constraint fix? | If physics correctly scores true shape #1, spectral rank doesn't matter |
| Is `_ALPHA=0.4 / _BETA=0.6` the right weight split? | Inspect score distributions once constraint fix results are in |
| Is an overtone discount needed for E minor? | Run E minor end-to-end with constraint, check if true shape wins |
