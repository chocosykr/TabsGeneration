import argparse
import scipy.io.wavfile as wavfile
import numpy as np
import torch

from pitch_estimation import (estimate_initial_pitch, estimate_chord_shape,
                              _extract_peak_energy_window)
from optimization import optimize_multi_parameters
from fretboard_mapper import map_parameters_to_tab, refine_chord_with_inharmonicity

# Max audio length fed to the optimizer (seconds).
_MAX_OPT_SEC = 0.5

def process_audio_multi_notes(file_path: str,
                              num_notes: int = 4,
                              strummed_strings: list[int] = None,
                              num_iterations: int = 50, 
                              lr: float = 0.5):
    """Full pipeline for N simultaneously played notes."""
    if strummed_strings is None:
        strummed_strings = [1, 2, 3, 4]

    # --- Validate string selections ---
    valid_strings = {1, 2, 3, 4}
    if len(strummed_strings) == 0 or not set(strummed_strings) <= valid_strings:
        raise ValueError(
            f"Invalid --strummed {strummed_strings}: must be a non-empty "
            f"subset of {sorted(valid_strings)} (ukulele strings 1-4)."
        )
    if len(set(strummed_strings)) != len(strummed_strings):
        raise ValueError(f"--strummed contains duplicate strings: {strummed_strings}")
    if len(strummed_strings) > num_notes:
        raise ValueError(
            f"Cannot strum {len(strummed_strings)} strings with only "
            f"{num_notes} simulated strings (--notes). "
            f"Strummed strings must be <= simulated strings. "
            f"e.g. --notes {len(strummed_strings)} or --strummed fewer strings."
        )

    import librosa
    print(f"Loading audio from {file_path}...")
    # Load and force resample to 16kHz mono
    waveform_np, sample_rate = librosa.load(file_path, sr=16000, mono=True)

    print(f"Loaded {len(waveform_np)} samples at {sample_rate} Hz "
          f"({len(waveform_np)/sample_rate:.2f}s).")

    waveform_full = torch.from_numpy(waveform_np).unsqueeze(0)

    if num_notes == 1:
        # --- Single note: pYIN → Optimizer → Fretboard Mapper ---
        print("\n1. Estimating Initial Pitch (pYIN on onset window)...")
        pitch = estimate_initial_pitch(waveform_full, sample_rate)
        pitches = [pitch]
        print(f"   Detected: {pitch:.2f} Hz")

        # Truncate to onset window for optimization
        y_opt = _extract_peak_energy_window(waveform_np, sample_rate,
                                            window_sec=_MAX_OPT_SEC)
        max_samples = int(_MAX_OPT_SEC * sample_rate)
        y_opt = y_opt[:max_samples]
        waveform_opt = torch.from_numpy(y_opt).unsqueeze(0)
        print(f"   (Optimizer will use {len(y_opt)} samples = "
              f"{len(y_opt)/sample_rate:.2f}s onset window)")

        print("\n2. Running Differentiable Karplus-Strong Optimization...")
        voice_params = optimize_multi_parameters(
            target_audio=waveform_opt,
            pitches=pitches,
            sample_rate=sample_rate,
            num_iterations=num_iterations,
            lr=lr
        )

        print("\n3. Mapping to Tablature...")
        vp = voice_params[0]
        tab_result = map_parameters_to_tab(vp['pitch'], vp['decay'], vp['inharmonicity'])
        if "error" in tab_result:
            print(f"   ERROR: {tab_result['error']}")
        else:
            best = tab_result['all_candidates'][0]
            print(f"   -> Plucked String: {tab_result['string']} (open {best['open_name']} string)")
            print(f"   -> Fret Number:    {tab_result['fret']}")
            print(f"   -> Note:           MIDI {tab_result['midi_note']}")
            print("\n   Alternative Fingering Candidates:")
            for cand in tab_result['all_candidates']:
                print(f"      String {cand['string']} (open {cand['open_name']}), Fret {cand['fret']}")
    else:
        # --- Multi-note (chord): Top-N multiset search + physics re-ranking ---
        print(f"\n1. Searching chord shapes on strings {strummed_strings} "
              f"(brute-force harmonic template matching)...")
        chord_result = estimate_chord_shape(
            waveform_full, sample_rate, strummed_strings,
            top_n_multisets=16
        )

        candidates = chord_result["candidates"]

        # Truncate to onset window for optimization (done once)
        y_opt = _extract_peak_energy_window(waveform_np, sample_rate,
                                            window_sec=_MAX_OPT_SEC)
        max_samples = int(_MAX_OPT_SEC * sample_rate)
        y_opt = y_opt[:max_samples]
        waveform_opt = torch.from_numpy(y_opt).unsqueeze(0)
        print(f"   (Optimizer will use {len(y_opt)} samples = "
              f"{len(y_opt)/sample_rate:.2f}s onset window)")

        # --- Evaluate each candidate multiset ---
        # For each candidate, run the optimizer with that candidate's pitches,
        # then score it via physics (inharmonicity match).
        #
        # Combined score = α × spectral_norm + β × physics_norm
        # Both components are normalized to [0, 1] so weights are interpretable.
        _ALPHA = 1.0   # spectral weight
        _BETA  = 0.0   # physics weight (disabled due to identifiability failure)

        evaluated = []  # list of (combined_score, candidate_dict)

        for rank, candidate in enumerate(candidates, 1):
            pitches = candidate["pitches"]
            _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                           'F#', 'G', 'G#', 'A', 'A#', 'B']
            notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"
                     for m in candidate["multiset_midis"]]

            print(f"\n2.{rank}. Evaluating candidate #{rank}: "
                  f"[{', '.join(notes)}]  "
                  f"(spectral={candidate['spectral_score_norm']:.3f})")

            # If the model simulates more strings than were strummed, pad with
            # silent voices (strummed strings must be <= simulated strings).
            cand_pitches = list(pitches)
            active_mask = [True] * len(strummed_strings)
            while len(cand_pitches) < num_notes:
                cand_pitches.append(440.0)  # dummy pitch — voice stays silent
                active_mask.append(False)

            n_active = sum(active_mask)
            if n_active < num_notes:
                print(f"   ({num_notes - n_active} simulated string(s) kept silent; "
                      f"only {n_active} strummed)")

            voice_params = optimize_multi_parameters(
                target_audio=waveform_opt,
                pitches=cand_pitches,
                sample_rate=sample_rate,
                num_iterations=num_iterations,
                lr=lr,
                active_mask=active_mask
            )

            # Only pass the active (strummed) voices to the refinement function
            active_voice_params = [
                voice_params[i] for i, active in enumerate(active_mask) if active
            ]
            refined = refine_chord_with_inharmonicity(
                candidate, active_voice_params, strummed_strings
            )

            # Extract the physics score:
            # inharm_score is the sum of per-string inharmonicity Gaussian
            # similarities. Maximum possible = len(strummed_strings) (1.0 per
            # string). Normalize to [0, 1].
            max_physics = float(len(strummed_strings))
            raw_physics = refined.get("inharm_score", 0.0)
            physics_norm = raw_physics / max_physics if max_physics > 0 else 0.0

            combined = (_ALPHA * candidate["spectral_score_norm"]
                        + _BETA * physics_norm)

            # Attach all scores to the candidate for traceability
            candidate["voice_params"] = active_voice_params
            candidate["refined"] = refined
            candidate["physics_score"] = raw_physics
            candidate["physics_score_norm"] = physics_norm
            candidate["combined_score"] = combined

            evaluated.append(candidate)

            print(f"   → spectral={candidate['spectral_score_norm']:.3f}  "
                  f"physics={physics_norm:.3f}  "
                  f"combined={combined:.3f}")

        # Sort by combined score (descending)
        evaluated.sort(key=lambda c: -c["combined_score"])
        winner = evaluated[0]

        # --- Print rankings ---
        print(f"\n{'─'*60}")
        print("   Candidate Rankings (spectral + physics):\n")
        for rank, c in enumerate(evaluated, 1):
            notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"
                     for m in c["multiset_midis"]]
            marker = "  ◄── WINNER" if rank == 1 else ""
            print(f"      #{rank}: [{', '.join(notes)}]  "
                  f"spectral={c['spectral_score_norm']:.3f}  "
                  f"physics={c['physics_score_norm']:.3f}  "
                  f"combined={c['combined_score']:.3f}{marker}")

        # --- Print final tab from the winner ---
        refined = winner["refined"]
        mapping = refined["mapping"]
        print(f"\n3. Final Chord Tablature (top-N physics-refined):")
        for string_idx in sorted(strummed_strings, reverse=True):
            fret = mapping[string_idx]
            fret_str = str(fret) if fret > 0 else "0 (open)"
            print(f"      String {string_idx}: Fret {fret_str}")

        # Print per-string diagnostic info if available
        if "inharmonicity_scores" in refined:
            print("\n   Inharmonicity diagnostics:")
            for s_idx in sorted(strummed_strings, reverse=True):
                d = refined["inharmonicity_scores"][s_idx]
                print(f"      String {s_idx}: fret={d['fret']}  "
                      f"obs={d['observed_stretch']:.4f}  "
                      f"exp={d['expected_stretch']:.4f}  "
                      f"score={d['score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentiable Ukulele Tab Transcription")
    parser.add_argument("audio_file", type=str,
                        help="Path to the ukulele WAV file")
    parser.add_argument("--notes", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Number of simulated voices/pitches to extract (default: 4)")
    parser.add_argument("--strummed", type=str, default="1,2,3,4",
                        help="Comma-separated list of strummed strings (default: 1,2,3,4)")
    parser.add_argument("--iter", type=int, default=50,
                        help="Number of optimization iterations (default: 50)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate for Adam (default: 0.5)")
    args = parser.parse_args()

    strummed_strings = [int(s.strip()) for s in args.strummed.split(",") if s.strip().isdigit()]
    
    process_audio_multi_notes(
        args.audio_file, 
        num_notes=args.notes, 
        strummed_strings=strummed_strings, 
        num_iterations=args.iter, 
        lr=args.lr
    )
