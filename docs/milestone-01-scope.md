# Milestone 0.1 — "One note, one string"

**Project:** Differentiable Karplus-Strong synthesis for guitar/ukulele tab recovery from audio.

**Status:** Scoped, not yet started. Last updated: 2026-08-18.

**Team:** 4 people. **Stack:** Python/PyTorch, PostgreSQL, Next.js + Tailwind, local Ubuntu workstation.

---

## Prior art to build on, not re-invent

The closest prior work — **"Differentiable Karplus-Strong"** (Tablas de Paula, Marttila & Reiss, DMRN+20, Dec 2025; public code) — already validates the core idea: a differentiable, time-domain extended Karplus-Strong algorithm (KSA) optimized end-to-end by gradient descent, reconstructing **real recorded acoustic guitar notes** using a **Lagrange-interpolated fractional delay** and a **multi-scale spectral (MSS) loss**. It beats a genetic-algorithm baseline.

**Our differentiation:** (1) output is a *discrete (string, fret) tab position* rather than a timbre match, (2) ukulele as well as guitar, (3) a usable product (upload audio → see tab), (4) eventually pluck technique. M1 reuses DiffKS's proven parameterization; we do not re-derive it.

Other work to differentiate from (not replicate):
- Lee et al., *Differentiable Modal Synthesis for Physical Modeling of Planar String Sound and Motion Simulation* (NeurIPS 2024, arXiv:2407.05516) — differentiable synthesis for strings via modal/FEM, not Karplus-Strong.
- Jin et al., *DiffSound* (SIGGRAPH 2024, arXiv:2409.13486) — differentiable synthesis for inverse problems, applied to general object sounds, not strings/tabs.

---

## Goal

A Python/PyTorch pipeline that takes a single short recording of one plucked ukulele note (string **known**), inverts a differentiable KS model via gradient descent, and prints the recovered **fret** plus decay/brightness/amplitude parameters.

## Definition of done

- Correct fret on **≥90%** of held-out real notes (single string, frets 0–10), median error 0, max error ≤ 1 fret
- Inversion **≤ 30 s per note** on the workstation CPU (no GPU needed)
- A **smoke test** proves the pipeline recovers a *synthetic* KS note exactly before real audio is attempted
- CLI works end-to-end; web UI is optional/stretch

## Key framing

This is **per-note optimization (an inverse problem), not supervised learning** — we fit the model to *each recording* with gradient descent. No training set is required. The ~40 real recordings are purely a **validation** set. A synthetic dataset only matters later, if we add a trained model for speed.

---

## 1. Forward model (differentiable KS) — the only ML-critical piece

| Param | Meaning | Notes |
|---|---|---|
| `D` | Delay length (samples), **continuous** | f0 = fs/D. Render with **Lagrange-interpolated fractional delay** (as in DiffKS) so gradients flow smoothly — a raw integer delay has zero gradient. Init from a fret candidate; round at the end |
| `g` | Loop gain | Decouples decay time from pitch: T60 ≈ −3·D / (fs·log₁₀ ḡ) |
| `α` | One-pole averaging/lowpass coefficient | Controls brightness; couples to decay in classic KS, so keep `g` for independent decay |
| `x_exc` | Excitation burst (~64–256 samples), **learnable** | Init = white noise or a fixed burst. A free excitation absorbs pluck position, nail-vs-flesh, and mic color — this is what lets real notes fit without exploding parameter count |
| `A` | Output gain | Needed for the loss to converge |

Keep the loop stable: constrain `g` ∈ (0.98, 1.0) via sigmoid parameterization. Pluck *position* as an explicit parameter (spectral-zero control) is M1.5, not M1.

**Implementation note:** a sample-by-sample Python loop over ~44k samples is too slow for 300+ optimization steps. Instead: (a) **batch the multi-start** — all candidate frets as separate channels in one tensor, one forward pass per iteration; (b) unroll the delay line in **chunks of 256–1024 samples**. Then 300–500 Adam steps × ~13 candidates is trivial on CPU.

## 2. Loss

Multi-scale spectral loss (MSS), as in DDSP/DiffKS: sum of L1 on log-magnitude STFTs at 3–4 window sizes (e.g., 512/1024/2048/4096 @ 22.05 kHz). Optionally a small time-domain term for the attack transient. Skip phase and perceptual terms in M1.

## 3. Optimization strategy

1. **Init:** cheap pitch estimate (YIN or autocorrelation via `librosa.pyin`) → nearest fret → init `D` from that fret's f0.
2. **Multi-start:** also init at fret ±2 semitones around the YIN guess (or all frets 0–10 batched), run Adam (lr ~1e–2, cosine schedule, 300–500 iters), keep the lowest-loss run. The loss gap between best and second-best run = confidence score.
3. **Round:** `fret = round(D → f0 → fret)`; report the confidence.

## 4. Data & validation (parallel track)

- Record **~30–40 notes on one string** (ukulele string 3, gCEA tuning): frets 0–10 × 3 takes, varying pluck intensity/position, consistent mic distance, 1–1.5 s per note.
- Recording script (sounddevice) enforces naming (`s3_f05_take2.wav`) and writes a metadata CSV — prevents annotation drift.
- Preprocess: mono, resample 44.1→22.05 kHz, detect onset, trim to onset + 1.5 s, peak-normalize.
- Eval: hold out one take per fret; report accuracy + median error + per-note runtime + a listening check.

## 5. Work breakdown

Work items (no assignments — pick up in any order):

- **Forward model + inversion loop + synthetic smoke test** — Wk 1–2
- **Eval harness + tuning** — Wk 3
- **Recording/annotation script + dataset** (~40 notes on string 3) — Wk 1 (quick), re-record as needed
- **CLI + thin FastAPI endpoint** wrapping inversion — Wk 2–3
- **Postgres schema** (recordings, results, params) **+ Next.js upload page** ("String C · Fret 5 · decay/brightness" + before/after spectrogram) — Wk 3–4 (stretch, only if core is green by wk 3)

## 6. Explicitly deferred (M2+)

- Unknown-string disambiguation (run per-string, compare residuals + a brightness prior — the same f0 on different strings is the interesting research bit)
- Multi-note phrases (onset segmentation → per-note inversion)
- Technique (hammer-on/pull-off/slide) via excitation/articulation params
- Synthetic dataset at scale (only if we later train a model for speed)

## 7. Risks & mitigations

- **Octave/local-minima errors** → multi-start + YIN init + round-to-nearest-fret
- **Integer-delay gradient dead zone** → Lagrange fractional delay (proven in DiffKS)
- **Real note won't fit** (room, body resonance) → learnable excitation absorbs it; accept higher loss; don't chase perfection in M1
- **Slow Python loops** → batched multi-start + chunked unroll
- **Wrong annotations** → script-enforced naming + spot checks

---

## Decisions to lock before building

1. **Single M1 string:** ukulele string 3 (C string in gCEA tuning) — matches the physical hardware, fewer candidates.
2. **Web UI is stretch:** core CLI green first; the inverse problem is the risky part.
