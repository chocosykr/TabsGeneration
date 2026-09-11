import argparse
import scipy.io.wavfile as wavfile
import numpy as np
import torch

from pitch_estimation import (estimate_initial_pitch, estimate_chord_shape,
                              _extract_peak_energy_window)
from optimization import optimize_multi_parameters
from fretboard_mapper import map_parameters_to_tab

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

    print(f"Loading audio from {file_path}...")
    sample_rate, waveform_np = wavfile.read(file_path)

    # Convert to float32 and normalise
    waveform_np = waveform_np.astype(np.float32) / 32767.0

    # Mixdown stereo → mono
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=1)

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
        tab_result = map_parameters_to_tab(vp['pitch'], vp['decay'])
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
        # --- Multi-note (chord): Direct chord shape search ---
        print(f"\n1. Searching chord shapes on strings {strummed_strings} "
              f"(brute-force harmonic template matching)...")
        chord_result = estimate_chord_shape(
            waveform_full, sample_rate, strummed_strings
        )
        
        pitches = chord_result["pitches"]

        # If the model simulates more strings than were strummed, pad with
        # silent voices (strummed strings must be <= simulated strings).
        active_mask = [True] * len(strummed_strings)
        while len(pitches) < num_notes:
            pitches.append(440.0)      # dummy pitch — voice stays silent
            active_mask.append(False)

        # Truncate to onset window for optimization
        y_opt = _extract_peak_energy_window(waveform_np, sample_rate,
                                            window_sec=_MAX_OPT_SEC)
        max_samples = int(_MAX_OPT_SEC * sample_rate)
        y_opt = y_opt[:max_samples]
        waveform_opt = torch.from_numpy(y_opt).unsqueeze(0)
        print(f"   (Optimizer will use {len(y_opt)} samples = "
              f"{len(y_opt)/sample_rate:.2f}s onset window)")

        n_active = sum(active_mask)
        if n_active < num_notes:
            print(f"   ({num_notes - n_active} simulated string(s) kept silent; "
                  f"only {n_active} strummed)")
        print(f"\n2. Running {num_notes}-Voice Karplus-Strong Optimization "
              f"({n_active} active)...")
        voice_params = optimize_multi_parameters(
            target_audio=waveform_opt,
            pitches=pitches,
            sample_rate=sample_rate,
            num_iterations=num_iterations,
            lr=lr,
            active_mask=active_mask
        )

        print("\n3. Final Chord Tablature:")
        mapping = chord_result["mapping"]
        for string_idx in sorted(strummed_strings, reverse=True):
            fret = mapping[string_idx]
            fret_str = str(fret) if fret > 0 else "0 (open)"
            print(f"      String {string_idx}: Fret {fret_str}")


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
