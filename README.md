# TabsGeneration

A differentiable signal-processing pipeline that listens to a ukulele recording (a `.wav` file) and outputs the **string numbers** and **fret positions** needed to reproduce it — i.e., automatic ukulele tablature transcription. Supports single notes, two-note dyads, and full 4-string chords.

---

## How It Works

The pipeline has three sequential stages:

```
Audio (.wav)
    │
    ▼
1. Pitch Estimation (pYIN / Ukulele-Constrained Chord Search)
    │   Single note: pYIN on the onset window.
    │   Polyphony (2-4 notes): brute-force search over all possible chord
    │   shapes on the strummed strings, scoring each by how much spectral
    │   energy its harmonic series explains (each FFT bin counted once).
    ▼
2. Differentiable Karplus-Strong Optimisation
    │   Fits a physically-inspired string model (1-4 voices, one per
    │   strummed string; extra simulated strings stay silent) to the audio
    │   to refine decay and timbre parameters via gradient descent.
    ▼
3. Tablature Output
        The winning chord shape, as (string, fret) pairs on a
        standard re-entrant ukulele (G4–C4–E4–A4 tuning).
```

### Module Overview

| File | Role |
|---|---|
| `main.py` | CLI entry-point; orchestrates all three stages |
| `pitch_estimation.py` | pYIN for mono; ukulele-constrained chord-shape search for polyphony (2-4 notes) |
| `optimization.py` | Adam optimisation loop using a multi-scale STFT loss |
| `karplus_strong.py` | Differentiable Karplus-Strong synthesiser (PyTorch), 1-4 voices |
| `fretboard_mapper.py` | Maps Hz → (string, fret) using standard ukulele tuning |
| `test_synthetic.py` | Generates synthetic `.wav` test files for validation |

---

## Installation

### Prerequisites

- Python 3.9+
- A working `pip` environment (a virtual environment is recommended)

### Steps

```bash
# 1. Clone the repo and enter the project directory
git clone <repo-url>
cd "ukulele tabs"

# 2. (Recommended) Create and activate a virtual environment
python -m venv prototype/venv
source prototype/venv/bin/activate

# 3. Install dependencies
pip install -r prototype/requirements.txt
```

### Dependencies

```
torch
torchaudio
librosa
numpy
scipy
tqdm
```

+------------------------------------------------------------------------------------------------------------+
|                                                                                                            |
|                                                                   ## Usage                                 |
|                                                                                                            |
+------------------------------------------------------------------------------------------------------------+
All commands are run from inside the `prototype/` directory.

```bash
cd prototype
```

### Basic Usage

```bash
# Full 4-string chord (default): strum all strings
python main.py my_chord.wav

# Single note
python main.py my_note.wav --notes 1
```

**Example (Single Note):**

```bash
python main.py test_E4.wav --notes 1
```

**Example output:**

```
Loading audio from test_E4.wav...
Loaded 32000 samples at 16000 Hz (2.00s).

1. Estimating Initial Pitch (pYIN on onset window)...
   Top MIDI candidates: MIDI 64 (329.6 Hz, w=4.21), MIDI 65 (349.2 Hz, w=0.10)
   Estimated F0: 329.63 Hz

2. Running Differentiable Karplus-Strong Optimization...
Starting optimisation  (pitch fixed at 329.63 Hz, optimising decay/energy) ...
   iter  10  loss=0.3241  pitch=329.63 Hz  decay=0.9801
   ...
   Optimized Pitch: 329.63 Hz
   Optimized Decay: 0.9803

3. Mapping Physical Parameters to Tablature...
   -> Plucked String: 2 (open E4 string)
   -> Fret Number:    0
   -> Note:           MIDI 64

   Alternative Fingering Candidates:
      String 2 (open E4), Fret 0
      String 3 (open C4), Fret 4
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `audio_file` | *(required)* | Path to a ukulele `.wav` file |
| `--notes N` | `4` | Number of strings being **simulated** (1-4). `1` = monophonic pYIN path; `2-4` = chord-shape search |
| `--strummed S` | `1,2,3,4` | Comma-separated list of strings actually **strummed** (subset of 1-4). Must be ≤ `--notes` |
| `--iter N` | `50` | Number of Adam optimisation iterations |
| `--lr F` | `0.5` | Learning rate for the Adam optimiser |

**Example with custom options:**

```bash
python3 main.py test_B4_s1f2.wav --notes 1 --iter 100 --lr 0.2
```

**Example (Two Notes / Dyad):**

```bash
# G-string fret 2 + E-string fret 1 (an F chord fragment)
python3 main.py test_two_F_chord_G2_E1.wav --notes 2 --strummed 4,2
```

**Example (Full 4-String Chord):**

```bash
# C-Major: 0003
python3 main.py test_four_C_Major.wav --notes 4 --strummed 1,2,3,4
```

> **How `--notes` and `--strummed` interact:** you simulate `--notes` strings in the synthesis model, and `--strummed` picks which of them actually sound in the recording. Strumming fewer strings than simulated is fine (the unused simulated strings stay silent). Strumming more than simulated is rejected with an error. When you strum a chord, unfretted strings ring open — the chord search assigns a fret (possibly 0 = open) to every strummed string.

---

## Input Audio Requirements

| Property | Requirement |
|---|---|
| Format | WAV (16-bit PCM recommended) |
| Channels | Mono or stereo (stereo is auto-mixed to mono) |
| Content | 1-4 simultaneous notes (mono, dyads, or full chords) |
| Pitch range | G2 (~98 Hz) to A5 (~1050 Hz) |
| Duration | At least ~0.3 s of sustain after the attack |

---

## Generating Synthetic Test Audio

You can synthesise test `.wav` files directly using the Karplus-Strong model, which is useful for verifying the pipeline end-to-end.

```bash
python test_synthetic.py
```

This creates single-note, two-note, and full-chord files in the `prototype/` directory:

| File | Content |
|---|---|
| `test_E4_s2f0.wav` | E4 (string 2, open) |
| `test_B4_s1f2.wav` | B4 (string 1, fret 2) |
| `test_two_C4_E4.wav` | Dyad: C4 + E4 |
| `test_two_E4_A4.wav` | Dyad: E4 + A4 |
| `test_two_G4_C5.wav` | Dyad: G4 + C5 |
| `test_four_C_Major.wav` | C-Major chord `0003` (all 4 strings) |

You can add your own test cases by editing the `__main__` block in `test_synthetic.py`:

```python
# Format: string_fret_to_hz(string_number, fret_number)
generate_test_audio(string_fret_to_hz(4, 3), "test_Bb4_s4f3.wav")   # Bb4
generate_test_audio(string_fret_to_hz(3, 7), "test_G4_s3f7.wav")    # G4
generate_multi_note_audio([string_fret_to_hz(1, 2), string_fret_to_hz(2, 3)], "test_dyad.wav")
```

---

## Ukulele Tuning Reference

This tool uses **standard re-entrant (High G) tuning**:

| String | Open Note | Frequency |
|---|---|---|
| 4 | G4 | 392.00 Hz |
| 3 | C4 | 261.63 Hz |
| 2 | E4 | 329.63 Hz |
| 1 | A4 | 440.00 Hz |

> **Note:** String 1 is the string closest to the floor when the ukulele is held in playing position.  
> Maximum supported fret: **15**.

---

## Project Structure

```
ukulele tabs/
├── README.md
├── .gitignore
├── filestructure.txt
├── docs/
│   ├── learning-path.md
│   ├── milestone-01-scope.md
│   └── handoff.md             # Status + handoff notes for AI agents
└── prototype/
    ├── main.py               # CLI entry-point
    ├── pitch_estimation.py   # pYIN pitch detector
    ├── optimization.py       # Karplus-Strong Adam optimizer
    ├── karplus_strong.py     # Differentiable KS synthesiser
    ├── fretboard_mapper.py   # Hz → (string, fret) mapper
    ├── test_synthetic.py     # Synthetic WAV generator
    ├── requirements.txt      # Python dependencies
    └── *.wav                 # Sample/test audio files
```

---

## Known Limitations

- **Monophonic note / dyad / chord only** — simultaneous notes only; the system cannot yet transcribe melodies or strumming patterns.
- **Dry recordings preferred** — heavy reverb or noise may degrade pitch accuracy.
- **String disambiguation is heuristic** — when the same pitch is playable on multiple strings, the chord-shape search picks the shape that best explains the recorded spectrum (equivalent-sounding fingerings are accepted, per the project's design goal).
- **Subset-strumming is ambiguous by nature** — if the audio contains notes on strings you did *not* list in `--strummed`, the search will silently drop the least-energetic note.

---

## Roadmap

See `docs/milestone-01-scope.md` and `docs/learning-path.md` for the planned development milestones.
