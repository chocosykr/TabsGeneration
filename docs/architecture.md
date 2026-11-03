# TabsGeneration Architecture Document

This document provides a comprehensive overview of the architecture of the TabsGeneration project. It details the reasoning behind each component, how they interact, and why specific design choices were made.

## 1. High-Level Pipeline

The pipeline is designed to transcribe monophonic, dyadic, and polyphonic ukulele recordings into standard tablature (string and fret combinations). It operates in three main stages:

1.  **Pitch Estimation & Candidate Generation (`pitch_estimation.py`, `spectral_template.py`)**
2.  **Karplus-Strong Optimization & Candidate Audio Comparison (`karplus_strong.py`, `optimization.py`)**
3.  **Fretboard Mapping & Candidate Selection (`fretboard_mapper.py`, `main.py`)**

---

## 2. Component Details & Rationale

### 2.1 Pitch Estimation (`pitch_estimation.py`)

This component is responsible for analyzing the raw `.wav` audio to determine which notes are being played.

*   **Why an Onset Window (`_extract_peak_energy_window`)?**
    Real audio has noise, transients (the pick/finger hitting the string), and decay. Pitch tracking is most stable in the "sustain" portion immediately after the initial transient. This function extracts ~0.5s of audio just after the loudest peak to ensure cleaner pitch detection.
*   **Mono Path (`estimate_initial_pitch`):**
    For single notes, we use **pYIN (probabilistic YIN)** from `librosa`. pYIN is standard and highly robust for monophonic signals. We convert the detected frequencies to a probability-weighted MIDI mode to snap the raw Hz to exact musical semitones.
*   **Polyphonic Path (`estimate_chord_shape`):**
    Standard pitch trackers fail on chords because harmonics overlap. Instead of iteratively subtracting frequencies (which is fragile), the polyphonic path uses a **brute-force template matching** approach:
    *   **How it works:** It evaluates all possible fret combinations (up to fret 15) for the strummed strings (up to 65,536 combinations for 4 strings).
    *   **Scoring (Union of Bins & NNLS):** Evaluates candidate harmonic combs against the observed FFT magnitude spectrum. It produces top-N distinct multiset candidate hypotheses for downstream physical and acoustic verification.

### 2.2 Synthesis Model (`karplus_strong.py`)

The Karplus-Strong (KS) string synthesis algorithm is a physically-inspired model of a plucked string. We use a **differentiable** implementation written in PyTorch.

*   **Why Karplus-Strong?** Instead of just returning pitches, we simulate the acoustic physics of the ukulele. KS uses a burst of noise fed into a delay line with a low-pass filter (decay) and all-pass filter (stretch / inharmonicity) to model a string's timbre.
*   **Why Differentiable?** By writing it in PyTorch, we can compute gradients. This allows us to start with physical priors and use gradient descent (Adam optimizer) to adjust our synthetic string parameters until its audio output matches the real recording.
*   **Reproducible Excitation:** Supports deterministic excitation buffers (`forward(excitation=...)`) to ensure fair, reproducible audio comparisons between candidate hypotheses and the target recording.

### 2.3 Optimization & Candidate Audio Comparison (`optimization.py`)

This module fits the differentiable KS model to the observed audio and directly compares the candidate tabs' acoustic output against the input tabs audio.

*   **Multi-Scale STFT Loss & Audio Comparison (`compare_candidate_audio`):**
    The comparison function evaluates the candidate tab's synthesized output audio against the input audio across multiple FFT window sizes (512, 1024, 2048). It captures both fine temporal transients and harmonic overtones, calculating:
    - Multi-scale STFT loss ($SC + L_{mag}$)
    - Normalized spectral convergence
    - Log-magnitude STFT loss
    - Waveform correlation (cosine similarity)
    - Monotonic audio similarity score in $(0, 1]$
*   **Candidate Selection Scoring:**
    Candidates are scored by a weighted combination of their **direct audio comparison match** with the input tabs audio and their spectral fit:
    $$\text{Combined Score} = w_{\text{audio}} \cdot \text{AudioScore}_{\text{norm}} + w_{\text{spectral}} \cdot \text{SpectralScore}_{\text{norm}} + w_{\text{physics}} \cdot \text{PhysicsScore}_{\text{norm}}$$
    Candidates whose synthesized sound diverges from the input recording are penalized, while tabs whose acoustic output matches the input audio rank highest.

### 2.4 Fretboard Mapping (`fretboard_mapper.py`)

This module translates theoretical frequencies (Hz) into physical playing positions.

*   **Ukulele Tuning:** Hardcoded to standard re-entrant tuning (High G: G4, C4, E4, A4).
*   **Audio-Informed Fingering Disambiguation:** When multiple candidate fingerings or shapes produce the same notes (e.g. open vs fretted unisons), candidate tab shapes synthesize their expected audio and compare directly to the input audio to pick the fingering that matches the recorded timbre.

---

## 3. Data Flow Example (Full Chord)

1.  **Input:** `test_four_C_Major.wav`, `--notes 4`, `--strummed 1,2,3,4`
2.  `pitch_estimation` computes the FFT of the onset window and generates top-N candidate tab multisets.
3.  For each candidate tab:
    - Initialize physical parameters based on the tab's fretboard positions.
    - Run the differentiable optimizer and synthesize the candidate tab's output audio.
    - Compare the candidate tab's synthesized output audio with the input tabs audio (`compare_candidate_audio`).
4.  Rank all candidate tabs by combined audio match + spectral score. The candidate whose output audio best matches the input tabs audio wins.
5.  **Output:** The CLI prints the candidate rankings and final tablature.
