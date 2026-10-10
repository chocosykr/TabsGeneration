# TabsGeneration CLI Documentation

The `cli.py` script provides a unified interface for interacting with the core functionalities of the TabsGeneration pipeline. It allows you to both transcribe real/synthetic recordings and generate new synthetic test files.

## Installation

To use this as a global command (like `npm` or `pip`), you need to install the package using `pip` from the root directory of the project. This will register the `utab` command globally in your environment.

```bash
# From the root of the "ukulele tabs" directory
pip install -e .
```

*Note: The `-e` flag installs it in "editable" mode, meaning any changes you make to the code will immediately be reflected in the CLI tool without needing to reinstall.*

## Available Commands

The CLI supports two main subcommands:
1. `transcribe`: Converts an audio file into tablature.
2. `synthesize`: Generates synthetic audio files using the built-in Karplus-Strong synthesizer.

---

### 1. Transcribe (`transcribe`)

Reads a `.wav` file and outputs the detected (string, fret) combinations.

**Usage:**
```bash
utab transcribe <audio_file> [options]
```

**Options:**
*   `--notes N`: Number of strings being simulated (1-4). Use `1` for monophonic inputs (uses pYIN). Use `2-4` for chords (uses polyphonic search). Default is `4`.
*   `--strummed S`: Comma-separated list of strings actually strummed. Standard ukulele strings are 4 (G), 3 (C), 2 (E), 1 (A). Default is `1,2,3,4`.
*   `--iter N`: Number of iterations for the Adam optimizer (decay/energy fitting). Default is `50`.
*   `--lr F`: Learning rate for the optimizer. Default is `0.5`.

**Examples:**
```bash
# Transcribe a full 4-string chord (C-Major)
utab transcribe test_four_C_Major.wav --notes 4 --strummed 1,2,3,4

# Transcribe a 2-note dyad played on the G and E strings (strings 4 and 2)
utab transcribe test_two_F_chord_G2_E1.wav --notes 2 --strummed 4,2

# Transcribe a single monophonic note
utab transcribe test_E4.wav --notes 1
```

---

### 2. Synthesize (`synthesize`)

Generates a perfectly controlled synthetic `.wav` file for testing the transcription engine. You can specify the notes to generate either by physical string/fret combinations or by raw frequencies.

**Usage:**
```bash
utab synthesize [options]
```

**Options:**
*   `--out PATH`: The name/path of the output file. Default is `synthetic_output.wav`.
*   `--frets PAIRS`: Comma-separated list of `string:fret` values to synthesize.
*   `--pitches HZ`: Comma-separated list of raw frequencies (Hz) to synthesize.

*Note: You must provide either `--frets` or `--pitches`.*

**Examples:**
```bash
# Generate a C-Major chord by specifying string:fret pairs
# (String 1: Fret 3, String 2: Fret 0, String 3: Fret 0, String 4: Fret 0)
utab synthesize --out my_c_major.wav --frets 1:3,2:0,3:0,4:0

# Generate a single note on String 4, Fret 5 (C5)
utab synthesize --out single_c5.wav --frets 4:5

# Generate a dyad using exact raw frequencies (Hz)
utab synthesize --out my_dyad.wav --pitches 440.0,329.63
```
