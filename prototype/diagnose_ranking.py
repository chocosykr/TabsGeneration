#!/usr/bin/env python3
"""
Diagnostic: Where does the TRUE chord shape rank in the spectral scorer?

This script runs the full brute-force harmonic template search from
estimate_chord_shape, but instead of returning only the winner, it
ranks ALL 65,536 shapes and reports the position of known ground-truth
shapes.

Usage:
    python diagnose_ranking.py input/e\ min.wav --true-frets 0,4,3,2
    python diagnose_ranking.py input/g\ min.wav --true-frets 0,3,3,1
    python diagnose_ranking.py input/g\ maj.wav --true-frets 0,0,3,2
"""

import argparse
import heapq
import itertools

import librosa
import numpy as np
import torch

from fretboard_mapper import UKULELE_TUNING, freq_to_midi
from pitch_estimation import _extract_peak_energy_window, _hz_to_bin


def rank_all_shapes(audio_path: str,
                    strummed_strings: list[int],
                    true_frets: tuple[int, ...] | None = None,
                    max_fret: int = 15,
                    num_harmonics: int = 5,
                    report_top_n: int = 20,
                    max_fret_span: int = 4):
    """
    Run the spectral scorer on ALL possible chord shapes and report
    the full ranking.  Optionally highlight where `true_frets` lands.
    """

    # --- Load audio ---
    y_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    y_window = _extract_peak_energy_window(y_np, sr)

    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_window * np.hanning(len(y_window)), n=n_fft))

    # --- Pre-compute open frequencies ---
    open_freqs = [UKULELE_TUNING[s]["freq"] for s in strummed_strings]
    frets_range = range(max_fret + 1)
    num_strings = len(strummed_strings)
    search_radius = 3

    # --- Pre-compute harmonic bins (same logic as estimate_chord_shape) ---
    harm_bins = []
    for open_freq in open_freqs:
        per_fret = []
        for fret in frets_range:
            f0 = open_freq * (2 ** (fret / 12.0))
            bins = {}
            for h in range(1, num_harmonics + 1):
                freq = h * f0
                if freq > sr / 2:
                    break
                center_bin = _hz_to_bin(freq, n_fft, sr)
                low_bin = max(0, center_bin - search_radius)
                high_bin = min(len(mag) - 1, center_bin + search_radius)
                for b in range(low_bin, high_bin + 1):
                    prev = bins.get(b, 0)
                    if h > prev:
                        bins[b] = h
            per_fret.append(bins)
        harm_bins.append(per_fret)

    # --- Score every shape (mirrors pitch_estimation.py scoring exactly) ---
    _UNISON_BONUS            = 0.1
    _ERGO_WEIGHT             = 0.5    # strong ergonomic prior: prefer low frets
    _OVERTONE_FACTOR         = 0.25   # discount when note is a harmonic of another
    _OVERTONE_RATIO_TOLERANCE = 0.03  # Hz tolerance for harmonic ratio check

    open_freqs_all = [UKULELE_TUNING[s]["freq"] for s in strummed_strings]

    all_shapes = []   # list of (score, frets)

    for frets in itertools.product(frets_range, repeat=num_strings):
        # Fret-span constraint: skip physically unplayable shapes
        fretted = [f for f in frets if f > 0]
        if fretted and (max(fretted) - min(fretted)) > max_fret_span:
            continue

        # Overtone discount map
        candidate_freqs = [
            open_freqs_all[i] * (2 ** (fret / 12.0))
            for i, fret in enumerate(frets)
        ]
        overtone_discount = [1.0] * num_strings
        for i, fi in enumerate(candidate_freqs):
            for j, fj in enumerate(candidate_freqs):
                if i == j or fj == 0:
                    continue
                ratio = fi / fj
                nearest_int = round(ratio)
                if nearest_int >= 2:
                    if abs(ratio - nearest_int) / nearest_int < _OVERTONE_RATIO_TOLERANCE:
                        overtone_discount[i] = min(overtone_discount[i], _OVERTONE_FACTOR)
                        break

        covered = {}
        score = 0.0

        for string_i, fret in enumerate(frets):
            discount = overtone_discount[string_i]
            for b, h in harm_bins[string_i][fret].items():
                effective_weight = discount if h == 1 else 1.0
                if b not in covered:
                    covered[b] = (h, 1)
                    score += effective_weight * mag[b] / h
                else:
                    prev_h, prev_count = covered[b]
                    if h == prev_h:
                        covered[b] = (h, prev_count + 1)
                        score += _UNISON_BONUS * effective_weight * mag[b] / h
                    elif h < prev_h:
                        score -= mag[b] / prev_h
                        score += effective_weight * mag[b] / h
                        covered[b] = (h, 1)

        score -= _ERGO_WEIGHT * sum(frets)
        all_shapes.append((score, frets))

    # --- Sort descending by score ---
    all_shapes.sort(key=lambda x: -x[0])

    total = len(all_shapes)
    best_score = all_shapes[0][0]

    # --- Report top N ---
    print(f"\n{'='*70}")
    print(f"  Spectral Shape Ranking  —  {audio_path}")
    print(f"  {total:,} shapes evaluated")
    print(f"{'='*70}\n")

    print(f"  Top {report_top_n} shapes:\n")
    print(f"  {'Rank':>5}  {'Frets':>15}  {'Score':>10}  {'Rel%':>6}  Notes")
    print(f"  {'─'*5}  {'─'*15}  {'─'*10}  {'─'*6}  {'─'*30}")

    for rank, (score, frets) in enumerate(all_shapes[:report_top_n], 1):
        notes = []
        for i, s in enumerate(strummed_strings):
            midi = int(round(freq_to_midi(UKULELE_TUNING[s]["freq"]))) + frets[i]
            note_name = _midi_to_note_name(midi)
            notes.append(f"{note_name}")

        rel = score / best_score * 100
        marker = ""
        if true_frets is not None and frets == true_frets:
            marker = "  ◄── TRUE"
        print(f"  {rank:>5}  {str(frets):>15}  {score:>10.3f}  {rel:>5.1f}%  "
              f"{', '.join(notes)}{marker}")

    # --- Find the true shape's rank ---
    if true_frets is not None:
        true_rank = None
        true_score = None
        for rank, (score, frets) in enumerate(all_shapes, 1):
            if frets == true_frets:
                true_rank = rank
                true_score = score
                break

        print(f"\n  {'─'*60}")
        if true_rank is not None:
            rel = true_score / best_score * 100
            print(f"  TRUE SHAPE {true_frets}:")
            print(f"    Rank:  {true_rank:,} / {total:,}")
            print(f"    Score: {true_score:.3f}  ({rel:.1f}% of best)")

            # What N would you need?
            for n in [1, 2, 4, 8, 16, 32, 64]:
                included = "✅" if true_rank <= n else "❌"
                print(f"    N={n:>3}  →  {included}")
        else:
            print(f"  TRUE SHAPE {true_frets}: NOT FOUND (shouldn't happen)")

    # --- Distinct note-set analysis ---
    # How many distinct note-sets are in the top N?
    print(f"\n  {'─'*60}")
    print(f"  Distinct note-sets in top shapes:\n")
    seen_sets = []
    for rank, (score, frets) in enumerate(all_shapes[:32], 1):
        midi_set = frozenset(
            int(round(freq_to_midi(UKULELE_TUNING[s]["freq"]))) + frets[i]
            for i, s in enumerate(strummed_strings)
        )
        midi_list = sorted(
            int(round(freq_to_midi(UKULELE_TUNING[s]["freq"]))) + frets[i]
            for i, s in enumerate(strummed_strings)
        )
        if midi_set not in [s for s, _, _ in seen_sets]:
            notes = [_midi_to_note_name(m) for m in midi_list]
            seen_sets.append((midi_set, rank, score))
            print(f"    First at rank {rank:>3}: {{{', '.join(notes)}}}  "
                  f"(score {score:.3f})")

    print()
    return all_shapes


_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _midi_to_note_name(midi: int) -> str:
    octave = (midi // 12) - 1
    note = _NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose spectral ranking of chord shapes")
    parser.add_argument("audio_file", type=str)
    parser.add_argument("--true-frets", type=str, default=None,
                        help="Ground-truth frets, comma-separated (e.g. 0,4,3,2)")
    parser.add_argument("--strummed", type=str, default="1,2,3,4",
                        help="Strummed strings (default: 1,2,3,4)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="How many top shapes to show (default: 20)")
    args = parser.parse_args()

    strummed = [int(s.strip()) for s in args.strummed.split(",")]

    true_frets = None
    if args.true_frets:
        true_frets = tuple(int(f.strip()) for f in args.true_frets.split(","))

    rank_all_shapes(args.audio_file, strummed,
                    true_frets=true_frets, report_top_n=args.top_n)
