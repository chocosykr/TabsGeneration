# TabsGeneration: Differentiable Ukulele Tab Transcription

A differentiable physical-acoustics signal processing pipeline that listens to a ukulele recording (`.wav` file) and outputs the **string numbers** and **fret positions** needed to reproduce it — i.e., automatic ukulele tablature transcription. 

Supports single plucked notes, two-note dyads, and full 4-string chords, combining **Non-Negative Least Squares (NNLS)** harmonic template matching, **physical string inharmonicity**, **Differentiable Karplus-Strong (DKS)** synthesis, and a **direct Candidate Audio Comparison engine** with dedicated GPU acceleration.

---

## Key Features

- **Analysis-by-Synthesis Candidate Audio Comparison:** Directly synthesizes candidate tab options and compares their acoustic waveforms against the input recording using multi-scale STFT loss, spectral convergence, and waveform envelope correlation.
- **Physical String Inharmonicity Modeling:** Differentiates between identical pitches on different strings (e.g. string 1 fret 0 vs string 4 fret 2) by analyzing overtone stiffness and dispersion.
- **High-Performance Dedicated GPU Engine:** Seamlessly detects and leverages dedicated NVIDIA GPUs (e.g. RTX 4050/4060/3060). Uses JIT-compiled native recurrence (`DifferentiableKSFunction`) with CUDA-accelerated STFT loss, delivering over 45+ optimization iterations per second (up to 900× faster).
- **Easy-to-Read Terminal Statistics:** Tablature rankings, acoustic similarity percentages ($0\% - 100\%$), diagnostics, and real-time execution timing are formatted cleanly in ASCII tables.
- **Synthesized Audio Export:** Can save the acoustic audio of the winning transcribed tab as a `.wav` file (`--save-synth`) to verify the transcribed tab by ear.

---

## How It Works

The transcription pipeline operates in five sequential stages:

```text
[Input Audio Recording (.wav)]
             │
             ▼
 1. Audio Preprocessing & Onset Extraction
    Extracts the peak-energy attack window (~0.5s) where pluck dynamics and string physics are clearest.
             │
             ▼
 2. Note Detection & Candidate Hypotheses (NNLS Harmonic Matching)
    Single note: pYIN pitch tracking on onset window.
    Chords (2-4 notes): Brute-force Non-Negative Least Squares (NNLS) search over all fretboard shapes,
    filtering for the Top-N candidate multisets that best explain recorded harmonic peaks.
             │
             ▼
 3. Differentiable Karplus-Strong (DKS) Physical Simulation
    Simulates string vibrations with physical parameters:
      • Fundamental Frequency (Pitch)
      • Damping / Sustain (Decay factor)
      • String Stiffness / Dispersion (Inharmonicity)
    The Adam optimizer fine-tunes decay and timbre parameters using Multi-Scale STFT loss.
             │
             ▼
 4. Candidate Output Audio Comparison Engine
    Directly synthesizes candidate tabs and compares each candidate waveform against the input audio:
      • Multi-Scale STFT Loss (timbre & spectral envelope match)
      • Spectral Convergence (energy distribution match)
      • Waveform Envelope Correlation
             │
             ▼
 5. String Physics Fit & Final Tablature Selection
    Combines: [Audio Match %] + [Spectral Fit %] + [Physics Fit %]
    Selects the winning candidate tab and outputs the final tablature and execution time.
```

---

## Project Structure

```text
TabsGeneration/
├── README.md
├── .gitignore
├── docs/
│   ├── architecture.md          # Detailed system architecture and mathematical formulations
│   ├── problems_and_solutions.md# Engineering challenges, GPU latency resolution, and fixes
│   ├── milestone-01-scope.md    # Initial milestone goals
│   └── learning-path.md         # Reference literature and audio DSP guides
└── prototype/
    ├── cli.py                   # Unified CLI entry-point (transcribe, synthesize)
    ├── main.py                  # Pipeline orchestrator and formatting engine
    ├── device_config.py         # Dedicated GPU & CUDA device configuration
    ├── pitch_estimation.py      # pYIN and NNLS harmonic template chord search
    ├── optimization.py          # Adam optimizer, audio comparison metrics, multi-scale STFT loss
    ├── karplus_strong.py        # High-performance Differentiable Karplus-Strong model
    ├── fretboard_mapper.py      # Ukulele tuning, physical inharmonicity priors, tab mapping
    ├── test_synthetic.py        # Synthetic test audio generator
    ├── test_audio_comparison_candidate_selection.py # Verification tests for audio comparison
    └── *.wav                    # Sample and benchmark audio files
```

---

## Installation

### Prerequisites

- Python 3.9+ (Python 3.10 – 3.14 supported)
- A working `pip` environment
- *(Optional but recommended)* NVIDIA dedicated GPU with CUDA support

### Steps

```bash
# 1. Clone the repository and enter the directory
git clone <repo-url>
cd TabsGeneration

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r prototype/requirements.txt
```

### Dedicated GPU Setup (NVIDIA RTX / CUDA)

To enable GPU acceleration on Windows:
```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```
When running, `prototype/device_config.py` will automatically detect your dedicated GPU:
```text
[device_config] Dedicated GPU active: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) on cuda:0
```

---

## Usage

You can run the pipeline either via the unified CLI (`prototype/cli.py`) or directly through `prototype/main.py`.

### 1. Transcribe a Full 4-String Chord
Transcribes a chord audio file, compares candidate shapes against the input audio, and saves the synthesized output audio of the winning tab:

```powershell
python prototype/cli.py transcribe test_four_C_Major.wav --notes 4 --strummed 1,2,3,4 --top-n 3 --iter 5 --save-synth
```

### 2. Fast Evaluation Mode (`--iter 0`)
Runs candidate generation and instant candidate audio comparison without parameter optimization iterations (finishes in seconds):

```powershell
python prototype/cli.py transcribe test_four_C_Major.wav --notes 4 --top-n 3 --iter 0
```

### 3. Transcribe a Single Plucked Note
Transcribes a single note and resolves ambiguous string positions:

```powershell
python prototype/cli.py transcribe prototype/test_B4_s1f2.wav --notes 1 --iter 0 --save-synth
```

---

## Example Terminal Output

### Chord Transcription Table
```text
==================================================================================================
          CANDIDATE CHORD RANKINGS (Acoustic Audio Comparison vs Input Recording)
==================================================================================================
 Rank   Candidate Notes         Tab (4-3-2-1)    Audio Match  Spectral Fit    Combined   Verdict   
 ----   ----------------------  ---------------  -----------  ------------  ----------   ----------
  # 1   [C4, E4, G4, D5]        0 - 0 - 0 - 5         100.0%         99.8%       99.9%   [WINNER]
  # 2   [C4, E4, G4, B4]        0 - 0 - 0 - 2          98.8%        100.0%       99.4%   Rejected
  # 3   [C4, E4, G4, A4]        0 - 0 - 0 - 0          92.6%         95.5%       94.1%   Rejected
--------------------------------------------------------------------------------------------------
 * Audio Match: Synthesized candidate tab audio compared directly with input recording (STFT loss).
 * Spectral Fit: Harmonic template alignment via NNLS (higher is better).
 * Combined: Weighted multi-criteria score ranking acoustic match, spectrum, and physics.
==================================================================================================

==================================================================================================
                 WINNER ACOUSTIC DIAGNOSTICS (Synthesized Audio vs Input Audio)
==================================================================================================
   Metric                                 Value   Rating           Interpretation
   --------------------------------  ----------   --------------   -----------------------------------
   Multi-Scale STFT Error                3.9679   Best in search   Timbre & harmonic frequency error (lower is better)
   Spectral Convergence                  0.7365   Optimal          Normalized energy distribution deviation (lower is better)
   Log-Magnitude STFT Loss               3.2314   Strong           Balance across low/high harmonics (lower is better)
   Waveform Envelope Corr.               0.1762   Moderate         Time-domain envelope alignment (-1.0 to +1.0)
   Relative Audio Match                  100.0%   #1 Rank          Audio match score relative to runner-up candidates
==================================================================================================

==================================================================================================
                                   FINAL TRANSCRIBED TABLATURE
==================================================================================================
   Fretboard Layout (Ukulele Strings 4 to 1):
      String 4 (open G4):  Fret 0 (open)   -->  G4  (392.00 Hz)
      String 3 (open C4):  Fret 0 (open)   -->  C4  (261.63 Hz)
      String 2 (open E4):  Fret 0 (open)   -->  E4  (329.63 Hz)
      String 1 (open A4):  Fret 5          -->  D5  (587.33 Hz)
--------------------------------------------------------------------------------------------------
   Summary Tab:     [ G: 0 | C: 0 | E: 0 | A: 5 ]
   Chord Notes:     [C4, E4, G4, D5]
   Acoustic Match:  100.0% (STFT Error: 3.9679) | Overall Score: 99.9%
   Execution Time:  11.03s (Device: NVIDIA GeForce RTX 4050 Laptop GPU)
==================================================================================================
```

---

## Chord Sequence Identification & Progression Transcription

For audio recordings containing a sequence or progression of multiple different chords (e.g. `F,C,G.wav` or `F,C.wav`), use the sequence pipeline:

```powershell
# Default: Harmonic progression view (merges repeated strums, shows strum count)
python prototype/cli.py sequence F,C,G.wav

# Strum-by-strum view: Splits repeated strums (e.g. C -> C -> C) into distinct events
python prototype/cli.py sequence F,C.wav --split-strums --save-synth
```

### Sequence Output Example
```text
==================================================================================================
                    UKULELE CHORD SEQUENCE IDENTIFICATION & TRANSCRIPTION
==================================================================================================
   Audio File:     F,C,G.wav
   Duration:       8.70s  |  Total Chords Detected: 3
   Hardware:       NVIDIA GeForce RTX 4050 Laptop GPU  |  Processing Time: 6.71s
==================================================================================================

   CHORD PROGRESSION SEQUENCE:
   [ F Major ]  ==>  [ C Major ]  ==>  [ G Major ]

--------------------------------------------------------------------------------------------------
   CHORD SEQUENCE TIMELINE & TABLATURE:
==================================================================================================
  #   Time Interval      Duration  Chord       Tab (4-3-2-1)    Notes Played            Match %
 --   ----------------  ---------  ----------  ---------------  --------------------  ---------
 #1   1.50s - 2.69s         1.18s  F (Major)   2 - 0 - 1 - 0    [A4, C4, F4, A4]          80.9%
 #2   3.78s - 5.22s         1.44s  C (Major)   0 - 0 - 0 - 3    [G4, C4, E4, C5]          88.9%
 #3   6.30s - 7.49s         1.18s  G (Major)   0 - 2 - 3 - 2    [G4, D4, G4, B4]          78.4%
--------------------------------------------------------------------------------------------------

   ASCII UKULELE TABLATURE SCORE (Strings 1 to 4):
   --------------------------------------------------------------------------------
   String 1 (A4) |--0 --------|--3 --------|--2 --------|
   String 2 (E4) |--1 --------|--0 --------|--3 --------|
   String 3 (C4) |--0 --------|--0 --------|--2 --------|
   String 4 (G4) |--2 --------|--0 --------|--0 --------|
   --------------------------------------------------------------------------------
                 |  [F]      |  [C]      |  [G]      |
==================================================================================================
```

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `audio_file` | *(required)* | Path to the ukulele `.wav` file |
| `--notes N` | `4` | Number of simulated voices/strings (1–4). `1` = mono pYIN path; `2-4` = chord search |
| `--strummed S` | `1,2,3,4` | Comma-separated list of strings actually strummed (e.g. `1,2,3` or `1,2,3,4`) |
| `--iter N` | `50` | Number of Adam optimization iterations (`0` for instant evaluation) |
| `--lr F` | `0.5` | Learning rate for the Adam optimizer |
| `--top-n N` | `8` | Number of candidate chord shapes to evaluate with audio comparison |
| `--audio-weight W` | `0.5` | Weight for candidate output audio comparison score in candidate selection |
| `--spectral-weight W`| `0.5` | Weight for spectral template score in candidate selection |
| `--sequence` | `False` | Identify and transcribe a sequence of multiple chords across time |
| `--min-duration D` | `auto` | Minimum chord duration in seconds, or `auto` for adaptive physical onset segmentation |
| `--split-strums` | `False` | Split repeated strums of the same chord into distinct events |
| `--save-synth` | `False` | Save the winning candidate tab's synthesized audio to a `.wav` file |
| `--synth-out PATH` | `output_tab_synth.wav` | Output filename for the synthesized audio file |
| `--tab-out PATH` | `output_tabs.txt` | Output text file path where formatted tabs are written (overwritten every run) |

---

## Generating Synthetic Test Audio

You can synthesize test audio files directly using the physical Karplus-Strong model:

```powershell
python prototype/cli.py synthesize --frets "4:0,3:0,2:0,1:3" --out "test_c_major.wav"
```

Or run the synthetic batch generator:
```powershell
python prototype/test_synthetic.py
```

---

## Verification & Testing

To run the automated verification test suite for the candidate audio comparison engine:
```powershell
python prototype/test_audio_comparison_candidate_selection.py
```

Expected output:
```text
[device_config] Dedicated GPU active: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) on cuda:0
[OK] compare_candidate_audio successfully distinguishes matching vs wrong audio.
[OK] Candidate tab output audio comparison correctly identifies the true chord tab.
[OK] Single-note candidate tabs include audio comparison with input audio.
ALL AUDIO COMPARISON CANDIDATE SELECTION TESTS PASSED!
```

---

## Ukulele Tuning Reference

This system models a standard **re-entrant (High G) concert/soprano ukulele**:

| String | Open Note | Fundamental Frequency |
|:---:|:---:|:---:|
| **4** | G4 | 392.00 Hz |
| **3** | C4 | 261.63 Hz |
| **2** | E4 | 329.63 Hz |
| **1** | A4 | 440.00 Hz |

* Maximum supported fret: **15** frets.
* String 1 is the highest-pitch open string (closest to the floor).
