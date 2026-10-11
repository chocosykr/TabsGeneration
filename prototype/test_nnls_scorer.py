"""
test_nnls_scorer.py — Validate the NNLS spectral template scorer.

Tests specifically:
 1. E minor: [E4,G4,G4,B4] vs [E4,E4,G4,B4] vs [E4,E4,G4,A4] (last run's winner)
 2. G minor: [D4,G4,G4,A#4] — should score well
 3. G major: [D4,G4,G4,B4]  — should score well

Compares the NNLS score to the current spectral scorer for each chord.
"""

import librosa
import numpy as np
from pitch_estimation import _extract_peak_energy_window
from spectral_template import score_candidate_nnls
from fretboard_mapper import UKULELE_TUNING, freq_to_midi

_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_to_name(m): return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"

def hz(string_idx, fret):
    """Frequency of ukulele string (1=a,2=e,3=c,4=g) at given fret."""
    return UKULELE_TUNING[string_idx]["freq"] * (2 ** (fret / 12.0))

def run_chord(audio_file, candidates, label):
    print(f"\n{'='*64}")
    print(f"  {label}  —  {audio_file}")
    print(f"{'='*64}")

    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    y_win = _extract_peak_energy_window(y, sr)

    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_win * np.hanning(len(y_win)), n=n_fft))

    best_score = -1.0
    results = []
    for name, freqs in candidates:
        res = score_candidate_nnls(mag, freqs, n_fft, sr)
        results.append((name, freqs, res['score'], res['amplitudes']))
        best_score = max(best_score, res['score'])

    # Normalize
    print(f"\n  {'Candidate':<30} {'NNLS':>6}  {'Norm':>6}  Amplitudes")
    print(f"  {'─'*30} {'─'*6}  {'─'*6}  {'─'*36}")
    for name, freqs, score, amps in sorted(results, key=lambda x: -x[2]):
        norm = score / best_score if best_score > 0 else 0.0
        amp_str = "  ".join(f"{a:.2f}" for a in amps)
        marker = "  ◄── TRUE" if "TRUE" in name else ""
        print(f"  {name:<30} {score:>6.4f}  {norm:>6.3f}  [{amp_str}]{marker}")

if __name__ == "__main__":
    # ── E minor ────────────────────────────────────────────────────────────
    # True shape: a:2 e:3 c:4 g:0 → B4, G4, E4, G4
    run_chord(
        "input/e min.wav",
        candidates=[
            ("TRUE [E4,G4,G4,B4]", [hz(1,2), hz(2,3), hz(3,4), hz(4,0)]),
            ("[E4,E4,G4,B4]",       [hz(1,2), hz(2,0), hz(3,4), hz(4,0)]),
            ("[E4,E4,G4,A4]",       [hz(1,2), hz(2,0), hz(3,4), hz(4,2)]),  # last run's winner
            ("[E4,G4,B4,G5]",       [hz(1,2), hz(2,3), hz(3,4), hz(4,7)]),
        ],
        label="E minor"
    )

    # ── G minor ────────────────────────────────────────────────────────────
    # True shape: a:1 e:3 c:2 g:0 → A#4, G4, D4, G4
    run_chord(
        "input/g min.wav",
        candidates=[
            ("TRUE [A#4,G4,D4,G4]", [hz(1,1), hz(2,3), hz(3,2), hz(4,0)]),
            ("[A#4,D4,G4,D5]",       [hz(1,1), hz(2,10), hz(3,2), hz(4,0)]),
        ],
        label="G minor"
    )

    # ── G major ────────────────────────────────────────────────────────────
    # True shape: a:2 e:3 c:2 g:0 → B4, G4, D4, G4
    run_chord(
        "input/g maj.wav",
        candidates=[
            ("TRUE [B4,G4,D4,G4]", [hz(1,2), hz(2,3), hz(3,2), hz(4,0)]),
            ("[B4,D5,D4,G4]",       [hz(1,2), hz(2,10), hz(3,2), hz(4,0)]),
        ],
        label="G major"
    )
