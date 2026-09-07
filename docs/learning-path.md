# Learning Path — Differentiable Karplus-Strong Tab Recovery

For the whole team, regardless of background. Goal: in ~4 weeks of part-time study, everyone can explain the pipeline, read the core code, and contribute. This is the same path for everyone — work through it top to bottom.

---

## The big picture (read this first)

We record a plucked note. We want to know *which string and fret* produced it. Instead of training a classifier, we **build a small synthesizer** that generates plucked-string audio, make every knob in it differentiable, then **tune the knobs with gradient descent until the synthesized audio matches the recording**. The knobs that fit best ARE the answer (delay length → pitch → fret).

So the project has two halves:
1. **Forward:** a differentiable Karplus-Strong (KS) synthesizer — takes (delay, gain, brightness, noise burst) → audio.
2. **Inverse:** gradient descent over those knobs to minimize a spectral loss vs. a real recording.

The single most important sentence in this whole project: **the same f0 can be played on different strings** — recovering *string* is an ambiguity problem, which is why M1 fixes the string and recovers only the fret.

---

## What you'll need to learn (concept checklist)

### Layer 0 — How audio works (no math beyond high school)
- Sampling, sample rate, Nyquist; mono vs stereo; amplitude vs frequency
- Time domain vs frequency domain; why a "note" is a fundamental + harmonics (overtones)
- Timbre = the *pattern* of harmonics — this is what makes a ukulele sound like a ukulele
- Envelope: attack / decay (a pluck is a sharp attack + exponential decay)
- **Do:** plot the waveform and spectrogram of a ukulele note in Python (librosa)

### Layer 1 — Karplus-Strong (the algorithm we make differentiable)
- A delay line (a buffer); pitch = sample rate / delay length
- The feedback loop: delay → average with previous sample → back into the delay
- Noise burst excitation = the "pluck"; the loop's averaging = decay + brightness
- **Do:** implement KS in numpy, listen to it, change the delay length, hear the pitch change

### Layer 2 — Gradient descent & autograd (the "differentiable" part)
- What a loss function is; what gradient descent does (roll the ball downhill)
- What a *derivative* is at the level of "how much does the output change if I nudge this knob"
- Autograd/backprop: the framework computes all derivatives for you automatically
- Adam: the standard optimizer (a smarter version of plain gradient descent)
- **Do:** PyTorch 60-minute blitz; then re-implement your numpy KS in PyTorch with delay-length as a `nn.Parameter` and nudge it to match a target pitch

### Layer 3 — Differentiable DSP (the paradigm this project belongs to)
- DDSP (Engel et al. 2020): make DSP blocks differentiable, then optimize them to match audio
- The multi-scale spectrogram loss (MSS) — compare synthesized vs. real audio in the frequency domain at several window sizes
- Why *fractional* delay is needed: an integer delay has no gradient, so we interpolate (Lagrange) to make pitch smoothly differentiable
- **Do:** skim the DDSP review; then read the DiffKS paper (next section) with the code open

---

## Papers to read, in order

Read them in this order; each builds on the last. The first two are short and classic; #5 is the one that matters most for us.

1. **Karplus & Strong, "Digital Synthesis of Plucked-String and Drum Timbres" (Computer Music Journal, 1983)** — 5 pages, the original algorithm. Understand: delay line, noise excitation, averaging filter, why pitch = fs/N.
2. **JOS (Julius O. Smith), "Physical Audio Signal Processing", Chapter on Karplus-Strong** — free online book chapter (ccrma.stanford.edu/~jos/pasp/). The definitive explanation, including fractional-delay variants. Skim; use as reference.
3. **Engel et al., "DDSP: Differentiable Digital Signal Processing" (ICLR 2020)** — the paradigm paper: how you make a synthesizer differentiable and optimize it against real audio. Introduces the spectral loss we'll use.
4. **Hayes et al., "A Review of Differentiable Digital Signal Processing for Music & Speech Synthesis" (2023)** — a map of the whole field. Read the intro and the sections on synthesis/inverse problems; skim the rest. This gives you the landscape to describe the project in one paragraph.
5. **Tablas de Paula, Marttila & Reiss, "Differentiable Karplus-Strong" (DMRN+20, Dec 2025)** — **read fully, with the code open.** Our closest prior art, with public code. This is effectively our M1 blueprint: extended KS, Lagrange fractional delay, gain stages, MSS loss, gradient-descent inversion of real guitar notes.
6. **Hayes et al., "Sinusoidal Frequency Estimation by Gradient Descent" (2022/23)** — short; the trick of recovering a continuous frequency via gradient descent on a spectral loss; directly relevant to recovering fret.
7. **Cwitkowitz et al., "FretNet" (ICASSP 2023)** — end-to-end *black-box* guitar tab transcription from audio. This is what our approach is *not* — read the intro to understand the problem we're solving differently.
8. **Dahan et al., "SynthTab" (ISMER/ISMIR 2023)** — generating synthetic tab training data with KS synthesis. Relevant later if we scale with a synthetic dataset.

Optional context to skim when you want the full picture: Wiggins & Kim *TabCNN* (ISMIR 2019, black-box tab), Lee et al. *DMSP* (NeurIPS 2024, differentiable *modal* synthesis — the thing we're differentiating from), Kehling et al. *Automatic Tablature Transcription* (DAFx 2014, classical guitar-tab baseline).

---

## Free resources that cover Layers 0–2 better than any textbook

- **3Blue1Brown — "But what is the Fourier Transform?"** (YouTube): the single best intuition for time↔frequency. Watch before anything else.
- **3Blue1Brown — Neural Networks series, esp. "Gradient descent" episode**: why "roll downhill" works.
- **Monty Montgomery, "Digital Show & Tell"** (Xiph, YouTube): why digital audio works, sample rates, aliasing — fun and visual.
- **PyTorch 60-minute blitz** (pytorch.org): tensors + autograd in one afternoon.
- **librosa tutorial** (librosa.org): loading audio, STFT, spectrograms, `pyin` for pitch.
- **sounddevice** (Python): 20 lines of code to record from the mic — your recording script.

---

## Suggested 4-week study plan (everyone, part-time)

| Week | Study | Build |
|---|---|---|
| 1 | 3B1B Fourier; KS paper; plot spectrograms of a ukulele in librosa | A recording/annotation script (sounddevice → WAV + metadata CSV) |
| 2 | PyTorch blitz; re-implement KS in numpy, then in torch | Record the M1 dataset (~40 notes, string 3) |
| 3 | DDSP (skim) + **DiffKS paper + code** | A CLI that runs the inversion on a recorded note |
| 4 | DiffKS code walkthrough; full-pipeline demo | A simple FastAPI endpoint + upload page around the CLI |

Checkpoint: **end of week 4 you can demo the pipeline and explain every block in the diagram.**

---

## Glossary (jargon you'll meet)

- **f0 / fundamental:** the lowest, dominant frequency of a note; the "pitch"
- **Harmonics (partials/overtones):** integer multiples of f0; their relative strengths = timbre
- **Timbre:** the "color" of a sound — what makes a ukulele vs. a piano play the same note differently
- **Envelope:** how loudness changes over time (attack → decay)
- **Sample rate (fs):** number of samples per second (44.1 kHz = CD quality)
- **STFT / spectrogram:** a time-frequency picture of audio; our loss compares these
- **Window / hop:** the slices and steps used to compute the STFT
- **Magnitude vs. phase:** how big vs. where-in-the-cycle each frequency is; we mostly compare magnitude
- **Delay line:** a buffer that plays back audio from N samples ago; the heart of KS
- **Fractional delay:** a delay of a non-integer number of samples, via interpolation — required for gradients
- **Loop gain:** how much signal survives each pass through the feedback loop → controls decay time (T60)
- **Excitation:** the initial noise burst that starts the string vibrating (the "pluck")
- **Lowpass / averaging filter:** smooths the signal; in KS it kills high harmonics → brightness and decay
- **Onset:** the instant a note starts (the attack transient)
- **Semitone / MIDI note:** the standard unit of pitch (12 semitones = 1 octave)
- **Tuning / gCEA:** the pitch of each open string; ukulele = G C E A (string 4→1). Guitar = E A D G B E
- **Fret:** the position on the neck that shortens the string and raises pitch
- **Differentiable:** a function whose output changes smoothly with its inputs, so we can compute gradients
- **Autograd / backprop:** the framework (PyTorch) computing all gradients automatically
- **Gradient descent:** iteratively nudge parameters in the direction that reduces the loss
- **Adam:** the standard, robust optimizer
- **Loss function:** a number measuring how far synthesized audio is from the recording; we minimize it
- **MSS (multi-scale spectral loss):** spectral error summed over several STFT window sizes
- **Inversion / inverse problem:** given audio, recover the parameters that produced it (our whole project)
- **Multi-start:** run optimization from several starting guesses, keep the best — avoids getting stuck
- **Transcription:** turning audio into notation (here: tab = string + fret); the end goal
- **Synthesis:** the opposite — turning parameters into audio (the forward model)

---

## Questions to be able to answer after week 2

1. Why does pitch = sample rate / delay length in KS?
2. Why can't we just use an integer delay length in the optimization?
3. What does the averaging filter in the KS loop control: decay, brightness, or both?
4. Why is a single note ambiguous for (string, fret) recovery?
5. What is the loss comparing, exactly, and why the frequency domain?
6. What does "learning the excitation" buy us when fitting a real recording?
