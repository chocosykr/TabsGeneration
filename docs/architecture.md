# TabsGeneration Architecture Document

This document provides a comprehensive overview of the architecture of the TabsGeneration project. It details the reasoning behind each component, how they interact, and why specific design choices were made.

## 1. High-Level Pipeline

The pipeline is designed to transcribe monophonic, dyadic, and polyphonic ukulele recordings into standard tablature (string and fret combinations). It operates in three main stages:

1.  **Pitch Estimation & Chord Search (`pitch_estimation.py`)**
2.  **Karplus-Strong Optimization (`karplus_strong.py`, `optimization.py`)**
3.  **Fretboard Mapping & Output (`fretboard_mapper.py`)**

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
    *   **Scoring (Union of Bins):** It predicts the harmonic series (fundamental + 4 harmonics) for each string in a candidate shape and sees how much energy that shape explains in the actual audio's FFT. 
    *   **Why "Union of Bins"?** If two strings share a harmonic (e.g., C4 and C5), a simple sum would double-count that energy. By using a union, overlapping harmonics are only counted once, preventing the system from falsely favoring shapes that just happen to share harmonics.

### 2.2 Synthesis Model (`karplus_strong.py`)

The Karplus-Strong (KS) string synthesis algorithm is a physically-inspired model of a plucked string. We use a **differentiable** implementation written in PyTorch.

*   **Why Karplus-Strong?** Instead of just returning pitches, we want to simulate the physics of the ukulele. KS uses a short burst of noise fed into a delay line with a low-pass filter (decay) to model a string's timbre.
*   **Why Differentiable?** By writing it in PyTorch, we can compute gradients. This allows us to start with a guess for the parameters (decay, energy) and use gradient descent (Adam optimizer) to adjust our synthetic string until its audio output matches the real recording.
*   **Log-Pitch Parameterization:** Frequency parameters are strictly positive. By optimizing `log(pitch)` instead of raw `pitch`, we prevent the optimizer from ever pushing the frequency into negative, invalid values.

### 2.3 Optimization Loop (`optimization.py`)

This module fits the differentiable KS model to the observed audio.

*   **What it does:** It takes the pitches found by `pitch_estimation.py`, freezes them, and optimizes the `decay` and `excitation_energy` parameters of the `MultiVoiceKarplusStrong` model.
*   **Why freeze pitch?** Our brute-force pitch estimator is highly accurate for a discrete semitone grid. Letting the optimizer freely tune pitch often causes it to get stuck in local minima (e.g., tuning an octave down). 
*   **Multi-Scale STFT Loss:** The loss function compares the real audio to the synthesized audio across multiple FFT window sizes (512, 1024, 2048). This captures both fine time details and fine frequency details simultaneously, ensuring the synthesized timbre closely matches the real timbre.

### 2.4 Fretboard Mapping (`fretboard_mapper.py`)

This module translates theoretical frequencies (Hz) into physical playing positions.

*   **Ukulele Tuning:** Hardcoded to standard re-entrant tuning (High G: G4, C4, E4, A4).
*   **Equivalent Fingerings:** The core philosophy is that *the output tab only needs to sound like the input*. The mapper will find any combination of string/fret that produces the target frequency. For monophonic inputs, it heuristically selects the lowest fret (easiest to play).
*   *Note:* In polyphonic mode, `estimate_chord_shape` directly returns the (string, fret) pairs, effectively bypassing the need for a separate mapping step.

---

## 3. Data Flow Example (Full Chord)

1.  **Input:** `test_four_C_Major.wav`, `--notes 4`, `--strummed 1,2,3,4`
2.  `pitch_estimation` computes the FFT of the onset window.
3.  It evaluates 65,536 fret combinations. The shape `0003` (C-Major) explains the most spectral energy without double-counting shared harmonics.
4.  `optimization` initializes 4 Karplus-Strong voices at the frequencies for `0003`. It uses gradient descent to adjust the decay and energy of those 4 voices to best match the timbre of the input `.wav`.
5.  **Output:** The CLI prints the detected shape: `String 4: 0, String 3: 0, String 2: 0, String 1: 3`.

## 4. Known Limitations

*   **Subset-Strumming:** If you play 4 strings but tell the CLI you only strummed 3, the search algorithm will drop the note with the least energy.
*   **Monophonic Limitations:** Currently, the pipeline evaluates a single block of audio. It cannot detect strumming patterns over time (rhythm/envelopes) or sequential melodies.
