# Technical Report: Neural Network Mode
## Automatic Music Transcription Architecture & Fretboard Mapping

*TabsGeneration System Documentation | Engine: Spotify Basic Pitch | Target Instruments: Ukulele & Guitar*  
**PDF Document**: [neural_network_mode_report.pdf](file:///c:/Users/Tejas%20Naik/Desktop/final%20yr%20project/tabs-app/TabsGeneration/docs/neural_network_mode_report.pdf)

---

### 1. Overview
The TabsGeneration system provides two transcription pathways in its Web UI and CLI:
1. **Physical Modeling Engine**: Based on pYIN, CQT harmonic templates, and Karplus-Strong analysis-by-synthesis.
2. **Neural Network Engine (BP)**: Based on deep convolutional inference.

The neural mode integrates **Spotify Basic Pitch**, an open-source, lightweight polyphonic neural network for music transcription. It automatically detects concurrent notes across continuous audio recordings without requiring fixed voice-count parameters, allocates them to physical fretboard fingerings, and validates the result via acoustic resynthesis.

---

### 2. Neural Network Architecture

#### System Specifications
- **Model**: Spotify Basic Pitch (Rachel Bittner et al., ICASSP 2022)
- **Inference Engine**: ONNX / PyTorch via `basic_pitch.inference.predict`
- **Audio Input**: 16,000 Hz Mono WAV audio
- **Inference Latency**: ~150 ms per analyzed audio window (CPU / GPU)

#### 2.1 Harmonic Constant-Q Transform (HCQT)
Standard spectrograms and Short-Time Fourier Transforms (STFT) suffer from a fundamental trade-off between time resolution at low frequencies and frequency resolution at high frequencies. Basic Pitch addresses this by using a Harmonic Constant-Q Transform (HCQT). The HCQT evaluates logarithmic frequency bins across multiple harmonic multiples of the fundamental frequency ($f_0, 2f_0, 3f_0, 4f_0, 5f_0$). This allows the network to distinguish true fundamental pitches from overtone spectra produced by vibrating strings.

#### 2.2 Multi-Task Output Layers
The convolutional network predicts three synchronized time-series matrix representations:
1. **Note Onsets**: Transient peaks indicating the exact moment a string is plucked or strummed.
2. **Pitch Salience**: Continuous time-frequency energy contours across all 88 MIDI pitches.
3. **Note Activations**: Probability masks representing sustained note durations and release points.

#### 2.3 Dynamic Polyphony
Unlike monophonic pitch estimators (pYIN) or fixed multi-pitch trackers, Basic Pitch dynamically identifies any number of concurrent notes. It naturally accommodates single-note melodic lines, two-note dyads, triads, and full polyphonic chords without forcing inactive strings to sound.

---

### 3. End-to-End Processing Pipeline

```
Raw Audio Input (.wav)
  │
  ▼
Harmonic Constant-Q Transform (HCQT)
  │
  ▼
Basic Pitch Convolutional Neural Inference
  │
  ▼
Polyphonic Note Events (MIDI Pitches, Onsets, Offsets)
  │
  ▼
Fretboard Allocation Optimization
  ├── Ukulele Engine: map_pitches_to_ukulele_tab()  [g4-C4-E4-A4]
  └── Guitar Engine:  map_pitches_to_guitar_tab()   [E2-A2-D3-G3-B3-E4]
  │
  ▼
Physical Acoustic Verification (Karplus-Strong & STFT Error Match)
  │
  ▼
Tablature Output (ASCII Score, Interactive SVG Fretboard)
```

#### Step-by-Step Execution:
1. **Audio Segmentation**: The targeted time segment $[t_{start}, t_{end}]$ selected in the UI is extracted and resampled to 16 kHz.
2. **Neural Inference**: The `predict()` function generates candidate note intervals with estimated MIDI values and confidence metrics.
3. **Chord Event Clustering**: Notes with onsets occurring within an 80 ms window are aggregated into a single chord event.
4. **Fretboard Allocation**: The candidate MIDI pitches are mapped onto the target instrument's fretboard:
   - **Ukulele**: 4 strings ($gCEA$), frets 0 to 15, re-entrant tuning.
   - **Guitar**: 6 strings ($EADGBE$), frets 0 to 19, standard tuning.
   - **Optimization Criteria**: Minimizes overall fret hand span, eliminates string collisions, prefers lower fret positions, and accommodates muted strings ($-1$ / `x`).
5. **Acoustic Verification**: The determined tab shape is passed to a Karplus-Strong string synthesis model. The synthesized waveform is compared to the original audio recording using Short-Time Fourier Transform (STFT) error loss to compute an objective Acoustic Match percentage.

---

### 4. Technical Comparison: Neural vs. Physical Mode

| Characteristic | Neural Mode (Basic Pitch) | Physical Mode (Analysis-by-Synthesis) |
| :--- | :--- | :--- |
| **Core Method** | Pre-trained deep CNN on HCQT representations | CQT harmonic template matching + pYIN |
| **Polyphony** | Dynamic (automatically detects 1 to 6 notes) | Fixed voice setting (1, 2, 4, or 6 strings) |
| **Execution Time** | Fast feedforward inference (~150 ms) | Iterative Adam optimization (1–3 seconds) |
| **Fretboard Mapping** | Ergonomic post-hoc pitch-to-string allocation | Physical string waveguide constraints |
| **Optimal Use** | Melodic lines, fingerpicking, mixed plucks | Rhythmic chord strums, acoustic inharmonicity calibration |
| **Hardware Requirements** | Standard CPU or GPU via ONNX/PyTorch | Standard CPU vector operations |

---

### 5. User Interface Integration
Within the interactive Web UI (`prototype/ui_server.py`):
- **Engine Selection**: Toggle between `Neural (BP)` and `Physical (KS)`.
- **Adaptive Controls**: Voice count selectors are automatically hidden when Neural mode is selected, as voice count is inferred by the model.
- **Visual Presentation**: Results are displayed on an interactive 2D SVG fretboard, alongside detected note names, acoustic match confidence, and standard ASCII tablature scores.
