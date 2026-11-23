"""
test_audio_comparison_candidate_selection.py

Verifies that in the candidate selection process:
1. Candidate tab output audio is synthesized.
2. The synthesized candidate audio is directly compared against the input tabs audio.
3. Audio comparison metrics (STFT loss, audio similarity, spectral convergence, waveform correlation)
   are computed and used to score and rank candidates.
4. Correct candidate tabs whose acoustic output matches the input audio score higher
   than candidates with mismatched notes.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
from fretboard_mapper import UKULELE_TUNING, map_parameters_to_tab, refine_chord_with_inharmonicity
from optimization import (
    compare_candidate_audio,
    synthesize_candidate_tab_audio,
    optimize_multi_parameters
)
from pitch_estimation import estimate_chord_shape

def test_audio_comparison_metrics():
    print("Testing compare_candidate_audio function...")
    sr = 16000
    duration = 0.5
    n_samples = int(sr * duration)

    # Reference target: 440 Hz (A4)
    torch.manual_seed(0)
    target = synthesize_candidate_tab_audio([440.0], sample_rate=sr, duration=duration)

    # Candidate 1: Identical pitch (440 Hz)
    torch.manual_seed(0)
    cand_match = synthesize_candidate_tab_audio([440.0], sample_rate=sr, duration=duration)
    comp_match = compare_candidate_audio(cand_match, target, sample_rate=sr)

    # Candidate 2: Wrong pitch (523.25 Hz - C5)
    torch.manual_seed(0)
    cand_wrong = synthesize_candidate_tab_audio([523.25], sample_rate=sr, duration=duration)
    comp_wrong = compare_candidate_audio(cand_wrong, target, sample_rate=sr)

    print(f"  Matching pitch STFT loss: {comp_match['stft_loss']:.4f}, similarity: {comp_match['similarity']:.4f}")
    print(f"  Mismatched pitch STFT loss: {comp_wrong['stft_loss']:.4f}, similarity: {comp_wrong['similarity']:.4f}")

    assert comp_match['stft_loss'] < comp_wrong['stft_loss'], (
        f"Expected matching candidate to have lower audio loss than wrong candidate: "
        f"{comp_match['stft_loss']} vs {comp_wrong['stft_loss']}"
    )
    assert comp_match['similarity'] > comp_wrong['similarity'], (
        f"Expected matching candidate to have higher audio similarity than wrong candidate: "
        f"{comp_match['similarity']} vs {comp_wrong['similarity']}"
    )
    print("[OK] compare_candidate_audio successfully distinguishes matching vs wrong audio.")


def test_chord_candidate_selection_with_audio_comparison():
    print("\nTesting chord candidate selection comparing output audio with input audio...")
    sr = 16000
    duration = 0.5

    # Target chord: C-Major [392.00 (G4), 261.63 (C4), 329.63 (E4), 523.25 (C5)]
    # Corresponding to tabs: G string open (0), C string open (0), E string open (0), A string fret 3 (3)
    target_pitches = [523.25, 329.63, 261.63, 392.00]  # strings 1, 2, 3, 4
    torch.manual_seed(42)
    target_audio = synthesize_candidate_tab_audio(target_pitches, sample_rate=sr, duration=duration)

    # Candidate A: True C-Major tab [C5, E4, C4, G4]
    cand_a_pitches = [523.25, 329.63, 261.63, 392.00]
    torch.manual_seed(42)
    synth_a = synthesize_candidate_tab_audio(cand_a_pitches, sample_rate=sr, duration=duration)
    comp_a = compare_candidate_audio(synth_a, target_audio, sample_rate=sr)

    # Candidate B: Wrong candidate [B5, E4, C4, G4] (987.77 Hz instead of 523.25 Hz)
    cand_b_pitches = [987.77, 329.63, 261.63, 392.00]
    torch.manual_seed(42)
    synth_b = synthesize_candidate_tab_audio(cand_b_pitches, sample_rate=sr, duration=duration)
    comp_b = compare_candidate_audio(synth_b, target_audio, sample_rate=sr)

    print(f"  Candidate A (True C-Maj) Output Audio vs Input Audio:")
    print(f"    STFT Loss = {comp_a['stft_loss']:.4f}, Match Similarity = {comp_a['similarity']:.4f}")
    print(f"  Candidate B (Wrong note B5) Output Audio vs Input Audio:")
    print(f"    STFT Loss = {comp_b['stft_loss']:.4f}, Match Similarity = {comp_b['similarity']:.4f}")

    assert comp_a['stft_loss'] < comp_b['stft_loss'], (
        f"Candidate A should have lower audio loss than Candidate B against input tabs audio"
    )
    print("[OK] Candidate tab output audio comparison correctly identifies the true chord tab.")


def test_single_note_candidate_selection_with_audio():
    print("\nTesting single note candidate tab selection with audio comparison...")
    sr = 16000
    duration = 0.5
    target_audio = synthesize_candidate_tab_audio([440.0], sample_rate=sr, duration=duration)

    res = map_parameters_to_tab(
        optimized_pitch=440.0,
        optimized_decay=0.98,
        optimized_inharm=0.002,
        target_audio=target_audio.unsqueeze(0),
        sample_rate=sr
    )

    assert "all_candidates" in res
    print(f"  Single note candidates evaluated: {len(res['all_candidates'])}")
    for cand in res['all_candidates']:
        print(f"    String {cand['string']} Fret {cand['fret']}: audio_loss={cand.get('audio_loss', 0):.4f}, total_score={cand['total_score']:.4f}")
        assert "audio_loss" in cand, "Candidate tab should contain audio_loss comparison"
        assert "audio_similarity" in cand, "Candidate tab should contain audio_similarity comparison"
    print("[OK] Single-note candidate tabs include audio comparison with input audio.")


if __name__ == "__main__":
    test_audio_comparison_metrics()
    test_chord_candidate_selection_with_audio_comparison()
    test_single_note_candidate_selection_with_audio()
    print("\nALL AUDIO COMPARISON CANDIDATE SELECTION TESTS PASSED!")
