# Comparative Analysis: Physical vs. Neural Network Modes
## Architectural Principles, Working Mechanisms, and System Differences

*TabsGeneration System Documentation | Engines: Karplus-Strong DKS vs. Spotify Basic Pitch | Target: Ukulele & Guitar*  
**PDF Document**: [transcription_modes_comparison_report.pdf](file:///c:/Users/Tejas%20Naik/Desktop/final%20yr%20project/tabs-app/TabsGeneration/docs/transcription_modes_comparison_report.pdf)

---

### 1. Introduction & Core Philosophy

The TabsGeneration framework implements two distinct paradigms for automatic music transcription (AMT) and tablature extraction:

1. **Physical Mode (Analysis-by-Synthesis & Classical DSP)**: Rooted in physical acoustics, music theory, and differentiable digital waveguide modeling. It formulates transcription as an inverse acoustic problem: finding physical string parameters and finger positions whose synthesized sound best matches the target audio recording.
2. **Neural Network Mode (Data-Driven Deep Learning)**: Rooted in statistical pattern recognition. It utilizes a pre-trained deep convolutional neural network (CNN) operating on multi-harmonic representations to detect notes directly from complex spectrogram features in a single forward inference pass.

---

### 2. Mode 1: Physical Mode (Analysis-by-Synthesis)

#### System Specifications
- **DSP Front-End**: Probabilistic YIN (pYIN) for monophonic tracking; Constant-Q Transform (CQT) & STFT harmonic profiles for chords.
- **Acoustic Model**: Differentiable Extended Karplus-Strong (DKS) string synthesis with non-ideal string inharmonicity.
- **Optimization**: Candidate template correlation + PyTorch Adam gradient descent on STFT error loss.
- **Dataset Dependency**: None (pure mathematical, acoustic, and music-theoretic formulation).

#### 2.1 How Physical Mode Works

```
Input Audio (.wav)
  │
  ├─► [1 Voice / Monophonic]: pYIN Pitch Tracking ──► Fundamental Frequency (f0)
  │
  └─► [Multi-Voice / Chord]:
        │
        ├─► Spectral Peak Extraction & Candidate Generation
        │     • Ukulele: CQT chroma cross-correlation against 48 theoretical chord templates.
        │     • Guitar: Hybrid search combining 48 chord templates + salience-guided note sets.
        │
        ├─► Differentiable Karplus-Strong (DKS) Synthesis
        │     • Simulates plucked string delay lines with string stiffness (inharmonicity B).
        │     • Computes STFT spectrogram loss between synthesized audio and recording.
        │
        └─► Optimization & Candidate Selection
              • Evaluates top candidate chord fingerings using composite loss:
                Score = (w_audio * Acoustic_Match) + (w_spectral * Spectral_Correlation)
              • Selects winning fretboard configuration & generates tabs.
```

#### 2.2 Core Components of the Physical Engine
- **Extended Karplus-Strong Physical Model**: Models plucked strings using digital feedback delay lines. Non-ideal string stiffness causes higher partials to ring sharp according to $f_n = n f_0 \sqrt{1 + B n^2}$, where inharmonicity $B = B_0 (1 + f/12)^2$. The engine calibrates distinct $B_0$ constants for wound strings ($E_2, A_2, D_3 \approx 0.00015$) vs. plain steel strings ($G_3, B_3, E_4 \approx 0.00060$).
- **Theory-Guided Pruning**: For 6-string guitar, an exhaustive search across 19 frets yields $20^6 = 64,000,000$ permutations. Physical mode solves this combinatorial bottleneck by evaluating valid chord shapes from standard harmonic libraries combined with spectral peak salience, testing $< 300$ high-probability shapes in $< 15$ ms.
- **Analysis-by-Synthesis Loss**: Rather than predicting probabilities, the model synthesizes candidate waveforms in real time and computes multiscale STFT spectral loss:
  $$\mathcal{L}_{\text{STFT}} = \frac{\| |\text{STFT}(y_{\text{real}})| - |\text{STFT}(y_{\text{synth}})| \|_F}{\| |\text{STFT}(y_{\text{real}})| \|_F}$$

---

### 3. Mode 2: Neural Network Mode (Spotify Basic Pitch)

#### System Specifications
- **Model**: Spotify Basic Pitch (Rachel Bittner et al., ICASSP 2022).
- **Feature Representation**: Harmonic Constant-Q Transform (HCQT) across 5 harmonic multiples.
- **Inference Engine**: Pre-trained deep convolutional neural network running via ONNX Runtime / PyTorch.
- **Post-Processing**: Dynamic polyphonic clustering + ergonomic fretboard span optimizer.

#### 3.1 How Neural Mode Works

```
Input Audio (.wav)
  │
  ▼
Harmonic Constant-Q Transform (HCQT) Feature Representation
  │  [Evaluates f0, 2f0, 3f0, 4f0, 5f0 across logarithmic bins]
  ▼
Deep Convolutional Neural Network (Basic Pitch)
  │  [Joint inference across Note Onsets, Pitch Contours, and Note Activations]
  ▼
Discrete Musical Note Events (MIDI Pitches, Onset Times, Durations)
  │
  ▼
Ergonomic Fretboard Allocation
  │  • Ukulele: map_pitches_to_ukulele_tab()  [g4-C4-E4-A4, frets 0-15]
  │  • Guitar:  map_pitches_to_guitar_tab()   [E2-A2-D3-G3-B3-E4, frets 0-19]
  │  • Objective: Minimizes fret span, avoids string collisions, favors open positions.
  ▼
Acoustic Verification & Tablature Output
  └── Synthesizes winning tab via Karplus-Strong to compute acoustic match & display tab.
```

#### 3.2 Core Components of the Neural Engine
- **Harmonic Constant-Q Transform (HCQT)**: Unlike standard STFTs, HCQT stacks multiple constant-Q transforms tuned to harmonic ratios ($1\times, 2\times, 3\times, 4\times, 5\times$). This gives the CNN direct visual representation of harmonic overtone structures, allowing it to distinguish note fundamentals from string harmonics.
- **Multi-Task Joint Predictions**: A single CNN produces note onset maps (transients), pitch salience contours (continuous frequency energy across all 88 MIDI keys), and note duration masks simultaneously.
- **Dynamic Polyphony**: The neural model does not enforce a voice count. It automatically detects whether the musician played a single plucked note, a 2-note harmony, a 3-note triad, or a full 6-string chord.
- **Ergonomic Tab Allocation**: The detected MIDI pitches are allocated to instrument strings using a combinatorial search that minimizes hand stretching (fret span $\le 4$), avoids assigning two simultaneous notes to the same physical string, and incorporates muted strings ($-1$ / `x`).

---

### 4. Technical Comparison & Structural Differences

| Dimension | ⚙️ Physical Mode (Analysis-by-Synthesis) | 🧠 Neural Network Mode (Basic Pitch) |
| :--- | :--- | :--- |
| **Underlying Paradigm** | Physics-based acoustic simulation & theory optimization | Data-driven deep convolutional pattern recognition |
| **Input Feature** | STFT magnitude, CQT chroma, time-domain autocorrelation | Harmonic Constant-Q Transform (HCQT) multi-channel stack |
| **Voice Specification** | Manual or configured voice count (1, 2, 3, 4, 6 strings) | Dynamic auto-detection (infers 1 to 6 voices automatically) |
| **Inharmonicity Modeling** | Explicit physical modeling ($B_0$ stiffness for plain vs. wound) | Implicitly learned from real-world instrument training audio |
| **Execution Time** | Iterative search & optimization (1.0 – 3.0 s per segment) | Fast single-pass feedforward (~100 – 150 ms per segment) |
| **Fretboard Constraints** | Native to candidate selection & waveguide delay lines | Applied post-hoc via ergonomic distance minimization |
| **Hardware / Dependencies** | Lightweight CPU math (PyTorch autograd / NumPy / SciPy) | ONNX Runtime / PyTorch neural weights (~20 MB) |
| **Explainability** | High: Transparent physical equations, delay lines, and losses | Low/Medium: Deep neural network feature embeddings |
| **Best Suited For** | Structured chord strumming, physical sound matching, studio audio | Solo melodies, fingerpicking, rapid arpeggios, mixed plucks |

---

### 5. When to Choose Which Mode

#### 5.1 When to Use Physical Mode
1. **Rhythmic Chord Progressions**: When analyzing rhythmic strums of standard major, minor, or seventh chords, Physical mode uses music theory templates to enforce accurate open and barre fingerings.
2. **Physical Accuracy & Inharmonicity**: When dealing with low guitar strings ($E_2, A_2$) where overtone stiffness shifts harmonics sharp, Physical mode's explicit $B$ parameter prevents mistaking stiff overtones for separate notes.
3. **Verification & Explainability**: When you need objective, mathematically explainable STFT match metrics showing how closely the physical string model reproduces the recording.

#### 5.2 When to Use Neural Network Mode
1. **Melodic Plucks & Fingerpicking**: When transcription involves complex melodies, alternating fingerpicking, or arpeggiated patterns where the number of active notes changes from millisecond to millisecond.
2. **Interactive Real-Time Performance**: With ~150 ms latency, Neural mode provides rapid interactive transcription without waiting for iterative gradient descent.
3. **Non-Standard & Partial Chords**: When the musician plays unconventional voicings, two-note dyads, or uncataloged chord shapes that are not in fixed chord dictionaries.

---

### 6. Integration in the Interactive Web UI

The interactive Web interface (`prototype/ui_server.py`) exposes both engines under unified controls:
- **Engine Toggle**: Easily switch between `⚙️ Physical (KS)` and `🧠 Neural (BP)`.
- **Voice Count Visibility**: Selecting Physical mode exposes voice buttons (1 to 6 strings) so the user can restrict the search space. Selecting Neural mode hides voice buttons because polyphony is automatically resolved by the CNN.
- **Unified Visualization**: Both modes pipe their final tab configurations into the same interactive SVG fretboard renderer (supporting both Ukulele and Guitar), STFT spectrogram guides, and ASCII tablature export files.
