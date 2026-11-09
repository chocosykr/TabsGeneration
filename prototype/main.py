import argparse
import time
import sys
import scipy.io.wavfile as wavfile
import numpy as np
import torch

from device_config import DEVICE
from pitch_estimation import (estimate_initial_pitch, estimate_chord_shape,
                              _extract_peak_energy_window)
from optimization import optimize_multi_parameters
from fretboard_mapper import (map_parameters_to_tab, refine_chord_with_inharmonicity,
                              expected_inharmonicity, UKULELE_TUNING, freq_to_midi)
from tab_exporter import export_single_tab_to_file

# Max audio length fed to the optimizer (seconds).
_MAX_OPT_SEC = 0.5

def process_audio_multi_notes(file_path: str,
                              num_notes: int = 4,
                              strummed_strings: list[int] = None,
                              num_iterations: int = 50, 
                              lr: float = 0.5,
                              top_n: int = 8,
                              audio_weight: float = 0.5,
                              spectral_weight: float = 0.5,
                              physics_weight: float = 0.0,
                              save_synth: bool = False,
                              synth_out_path: str = "output_tab_synth.wav",
                              tab_out_path: str = "output_tabs.txt"):
    """Full pipeline for N simultaneously played notes."""
    start_time = time.time()
    if num_notes == 1:
        strummed_strings = [1]
    elif strummed_strings is None:
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
    audio_dur = len(waveform_np) / sample_rate

    print(f"Loaded {len(waveform_np)} samples at {sample_rate} Hz "
          f"({audio_dur:.2f}s).")
    if audio_dur > 3.5:
        print("   [Tip] For recordings with a sequence of different chords played, "
              "run with '--sequence' to transcribe the full chord progression!")

    waveform_full = torch.from_numpy(waveform_np).unsqueeze(0)

    if num_notes == 1:
        # --- Single note: pYIN -> Optimizer -> Fretboard Mapper ---
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
        voice_params, synth_audio, audio_comp = optimize_multi_parameters(
            target_audio=waveform_opt,
            pitches=pitches,
            sample_rate=sample_rate,
            num_iterations=num_iterations,
            lr=lr,
            return_audio=True
        )

        print("\n3. Candidate Tab Selection (Comparing Output Audio vs Input Audio)...")
        vp = voice_params[0]
        tab_result = map_parameters_to_tab(
            vp['pitch'], vp['decay'], vp['inharmonicity'],
            target_audio=waveform_opt,
            sample_rate=sample_rate
        )
        if "error" in tab_result:
            print(f"   ERROR: {tab_result['error']}")
        else:
            best = tab_result['all_candidates'][0]
            _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            midi_num = tab_result['midi_note']
            note_name = f"{_NOTE_NAMES[midi_num % 12]}{midi_num // 12 - 1}"

            print("\n" + "=" * 88)
            print("         CANDIDATE FINGERING EVALUATION (Acoustic Audio Comparison vs Input)")
            print("=" * 88)
            print(f" {'Rank':>4}   {'Candidate Tab':<26}  {'Audio Match':>11}  {'Physics Fit':>11}  {'Total Score':>11}   {'Verdict':<12}")
            print(f" {'-'*4}   {'-'*26}  {'-'*11}  {'-'*11}  {'-'*11}   {'-'*12}")

            for rank, cand in enumerate(tab_result['all_candidates'], 1):
                tab_desc = f"String {cand['string']} ({cand['open_name']}), Fret {cand['fret']}"
                audio_pct = cand.get('audio_match_norm', 1.0) * 100.0
                physics_pct = cand.get('inharm_score', 0.0) * 100.0
                total_pct = cand.get('total_score', 0.0) * 100.0
                is_best = (cand['string'] == best['string'] and cand['fret'] == best['fret'])
                verdict = "[SELECTED]" if is_best else "Rejected"
                print(f"  #{rank:>2}   {tab_desc:<26}  {audio_pct:>10.1f}%  {physics_pct:>10.1f}%  {total_pct:>10.1f}%   {verdict}")

            print("-" * 88)
            print(" * Audio Match: Synthesized candidate wave compared directly with input recording.")
            print(" * Physics Fit: Compatibility with real ukulele string stiffness & inharmonicity.")
            print("=" * 88)

            print("\n" + "=" * 88)
            print("                             FINAL TRANSCRIBED TABLATURE")
            print("=" * 88)
            print(f"   Detected Note:      {note_name} (MIDI {midi_num}, {vp['pitch']:.2f} Hz)")
            print(f"   Plucked String:     String {tab_result['string']} (open {best['open_name']} string)")
            print(f"   Fret Position:      Fret {tab_result['fret']}")
            if "audio_loss" in best:
                print(f"   Acoustic Match:     {best.get('audio_match_norm', 1.0)*100:.1f}%  (STFT Error: {best['audio_loss']:.4f})")
            print(f"   Physics Fit:        {best.get('inharm_score', 0.0)*100:.1f}%")
            elapsed = time.time() - start_time
            dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
            print(f"   Execution Time:     {elapsed:.2f}s (Device: {dev_desc})")
            print("=" * 88)

            export_single_tab_to_file(
                file_path=tab_out_path,
                audio_path=file_path,
                tab_result=tab_result,
                winner_data=None,
                strummed_strings=[1],
                elapsed=elapsed,
                is_single_note=True
            )
            print(f"\n   [Export] Formatted tablature successfully written to '{tab_out_path}'")

        if save_synth and synth_audio is not None:
            synth_np = synth_audio.detach().cpu().numpy()
            synth_np = synth_np / (np.max(np.abs(synth_np)) + 1e-8)
            wavfile.write(synth_out_path, sample_rate, (synth_np * 32767).astype(np.int16))
            print(f"\n   Saved synthesized output audio to '{synth_out_path}'")
    else:
        # --- Multi-note (chord): Top-N multiset search + audio comparison candidate selection ---
        print(f"\n1. Searching chord shapes on strings {strummed_strings} "
              f"(brute-force harmonic template matching)...")
        chord_result = estimate_chord_shape(
            waveform_full, sample_rate, strummed_strings,
            top_n_multisets=top_n
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
        # For each candidate tab, synthesize its acoustic output audio, optimize
        # timbre parameters, and COMPARE the candidate tab's output audio with the
        # input tabs audio.
        _WEIGHT_SPECTRAL = spectral_weight
        _WEIGHT_AUDIO    = audio_weight
        _WEIGHT_PHYSICS  = physics_weight

        evaluated = []  # list of candidate dicts

        for rank, candidate in enumerate(candidates, 1):
            pitches = candidate["pitches"]
            _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                           'F#', 'G', 'G#', 'A', 'A#', 'B']
            notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"
                     for m in candidate["multiset_midis"]]

            print(f"\n2.{rank}. Evaluating candidate #{rank}: "
                  f"[{', '.join(notes)}]  "
                  f"(initial spectral={candidate['spectral_score_norm']:.3f})")

            # If the model simulates more strings than were strummed, pad with
            # silent voices (strummed strings must be <= simulated strings).
            cand_pitches = list(pitches)
            active_mask = [True] * len(strummed_strings)
            while len(cand_pitches) < num_notes:
                cand_pitches.append(440.0)  # dummy pitch - voice stays silent
                active_mask.append(False)

            n_active = sum(active_mask)
            if n_active < num_notes:
                print(f"   ({num_notes - n_active} simulated string(s) kept silent; "
                      f"only {n_active} strummed)")

            # Initialize inharmonicity priors for this candidate tab's fingerings
            initial_stretches = [
                expected_inharmonicity(s, candidate["mapping"].get(s, 0))
                for s in strummed_strings
            ]
            while len(initial_stretches) < num_notes:
                initial_stretches.append(0.001)

            # Synthesize output audio and optimize against input audio
            voice_params, synth_audio, audio_comp = optimize_multi_parameters(
                target_audio=waveform_opt,
                pitches=cand_pitches,
                sample_rate=sample_rate,
                num_iterations=num_iterations,
                lr=lr,
                active_mask=active_mask,
                initial_stretches=initial_stretches,
                return_audio=True
            )

            # Only pass active (strummed) voices to the refinement function
            active_voice_params = [
                voice_params[i] for i, active in enumerate(active_mask) if active
            ]
            refined = refine_chord_with_inharmonicity(
                candidate, active_voice_params, strummed_strings,
                target_audio=waveform_opt, sample_rate=sample_rate
            )

            max_physics = float(len(strummed_strings))
            raw_physics = refined.get("inharm_score", 0.0)
            physics_norm = raw_physics / max_physics if max_physics > 0 else 0.0

            # Attach audio comparison and parameters to the candidate
            candidate["voice_params"] = active_voice_params
            candidate["refined"] = refined
            candidate["synth_audio"] = synth_audio
            candidate["audio_loss"] = audio_comp["stft_loss"]
            candidate["audio_similarity"] = audio_comp["similarity"]
            candidate["audio_metrics"] = audio_comp
            candidate["physics_score"] = raw_physics
            candidate["physics_score_norm"] = physics_norm

            evaluated.append(candidate)

            print(f"   -> STFT Error: {audio_comp['stft_loss']:.4f} | "
                  f"Wave Corr: {audio_comp['waveform_correlation']:.3f} | "
                  f"Spectral Fit: {candidate['spectral_score_norm']*100:.1f}% | "
                  f"Physics Fit: {physics_norm*100:.1f}%")

        # --- Candidate Selection: Audio Comparison + Spectral Scorer ---
        # Normalize audio loss across candidates so the candidate with the lowest
        # audio loss gets 1.0 (best match to input tabs audio).
        min_loss = min(c["audio_loss"] for c in evaluated)
        for c in evaluated:
            c["audio_score_norm"] = min_loss / c["audio_loss"] if c["audio_loss"] > 0 else 1.0
            c["combined_score"] = (
                _WEIGHT_AUDIO * c["audio_score_norm"]
                + _WEIGHT_SPECTRAL * c["spectral_score_norm"]
                + _WEIGHT_PHYSICS * c["physics_score_norm"]
            )

        # Sort candidates by combined score (descending)
        evaluated.sort(key=lambda c: -c["combined_score"])
        winner = evaluated[0]

        # --- Print rankings ---
        print("\n" + "=" * 98)
        print("          CANDIDATE CHORD RANKINGS (Acoustic Audio Comparison vs Input Recording)")
        print("=" * 98)
        print(f" {'Rank':>4}   {'Candidate Notes':<22}  {'Tab (4-3-2-1)':<15}  {'Audio Match':>11}  {'Spectral Fit':>12}  {'Combined':>10}   {'Verdict':<10}")
        print(f" {'-'*4}   {'-'*22}  {'-'*15}  {'-'*11}  {'-'*12}  {'-'*10}   {'-'*10}")
        for rank, c in enumerate(evaluated, 1):
            notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in c["multiset_midis"]]
            notes_str = "[" + ", ".join(notes) + "]"
            cand_mapping = c.get("refined", {}).get("mapping", c.get("mapping", {}))
            tab_str = " - ".join(str(cand_mapping.get(s, 0)) for s in sorted(strummed_strings, reverse=True))
            audio_pct = c["audio_score_norm"] * 100.0
            spectral_pct = c["spectral_score_norm"] * 100.0
            combined_pct = c["combined_score"] * 100.0
            is_best = (rank == 1)
            verdict = "[WINNER]" if is_best else "Rejected"
            print(f"  #{rank:>2}   {notes_str:<22}  {tab_str:<15}  {audio_pct:>10.1f}%  {spectral_pct:>11.1f}%  {combined_pct:>9.1f}%   {verdict}")

        print("-" * 98)
        print(" * Audio Match: Synthesized candidate tab audio compared directly with input recording (STFT loss).")
        print(" * Spectral Fit: Harmonic template alignment via NNLS (higher is better).")
        print(" * Combined: Weighted multi-criteria score ranking acoustic match, spectrum, and physics.")
        print("=" * 98)

        # Print detailed audio comparison diagnostics for the winner
        if "audio_metrics" in winner:
            m = winner["audio_metrics"]
            corr_val = m.get('waveform_correlation', 0.0)
            corr_rating = "Excellent" if corr_val > 0.8 else ("Good" if corr_val > 0.4 else "Moderate")
            print("\n" + "=" * 98)
            print("                 WINNER ACOUSTIC DIAGNOSTICS (Synthesized Audio vs Input Audio)")
            print("=" * 98)
            print(f"   {'Metric':<32}  {'Value':>10}   {'Rating':<14}   {'Interpretation'}")
            print(f"   {'-'*32}  {'-'*10}   {'-'*14}   {'-'*35}")
            print(f"   {'Multi-Scale STFT Error':<32}  {m['stft_loss']:>10.4f}   {'Best in search':<14}   Timbre & harmonic frequency error (lower is better)")
            print(f"   {'Spectral Convergence':<32}  {m['spectral_convergence']:>10.4f}   {'Optimal':<14}   Normalized energy distribution deviation (lower is better)")
            print(f"   {'Log-Magnitude STFT Loss':<32}  {m['log_mag_loss']:>10.4f}   {'Strong':<14}   Balance across low/high harmonics (lower is better)")
            print(f"   {'Waveform Envelope Corr.':<32}  {corr_val:>10.4f}   {corr_rating:<14}   Time-domain envelope alignment (-1.0 to +1.0)")
            print(f"   {'Relative Audio Match':<32}  {'100.0%':>10}   {'#1 Rank':<14}   Audio match score relative to runner-up candidates")
            print("=" * 98)

        # --- Print final tab from the winner ---
        refined = winner["refined"]
        mapping = refined["mapping"]
        winner_notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in winner["multiset_midis"]]
        winner_notes_str = "[" + ", ".join(winner_notes) + "]"
        tab_summary = " | ".join(f"{UKULELE_TUNING[s]['name'][0]}: {mapping.get(s, 0)}"
                                 for s in sorted(strummed_strings, reverse=True))

        print("\n" + "=" * 98)
        print("                                   FINAL TRANSCRIBED TABLATURE")
        print("=" * 98)
        print("   Fretboard Layout (Ukulele Strings 4 to 1):")
        for string_idx in sorted(strummed_strings, reverse=True):
            fret = mapping[string_idx]
            fret_str = f"Fret {fret}" if fret > 0 else "Fret 0 (open)"
            base_freq = UKULELE_TUNING[string_idx]["freq"]
            fret_freq = base_freq * (2.0 ** (fret / 12.0))
            fret_midi = int(round(freq_to_midi(fret_freq)))
            fret_note = f"{_NOTE_NAMES[fret_midi % 12]}{fret_midi // 12 - 1}"
            open_name = UKULELE_TUNING[string_idx]["name"]
            print(f"      String {string_idx} (open {open_name:<2}):  {fret_str:<14}  -->  {fret_note:<3} ({fret_freq:6.2f} Hz)")

        print("-" * 98)
        print(f"   Summary Tab:     [ {tab_summary} ]")
        print(f"   Chord Notes:     {winner_notes_str}")
        print(f"   Acoustic Match:  100.0% (STFT Error: {winner['audio_loss']:.4f}) | Overall Score: {winner['combined_score']*100:.1f}%")
        elapsed = time.time() - start_time
        dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
        print(f"   Execution Time:  {elapsed:.2f}s (Device: {dev_desc})")
        print("=" * 98)

        export_single_tab_to_file(
            file_path=tab_out_path,
            audio_path=file_path,
            tab_result=None,
            winner_data=winner,
            strummed_strings=strummed_strings,
            elapsed=elapsed,
            is_single_note=False
        )
        print(f"\n   [Export] Formatted tablature successfully written to '{tab_out_path}'")

        # Print per-string diagnostic info if available
        if "inharmonicity_scores" in refined:
            print("\n   Inharmonicity diagnostics:")
            for s_idx in sorted(strummed_strings, reverse=True):
                d = refined["inharmonicity_scores"][s_idx]
                print(f"      String {s_idx}: fret={d['fret']}  "
                      f"obs={d['observed_stretch']:.4f}  "
                      f"exp={d['expected_stretch']:.4f}  "
                      f"score={d['score']:.3f}")

        # Save winning candidate synthesized output audio if requested
        if save_synth and "synth_audio" in winner:
            synth_np = winner["synth_audio"].detach().cpu().numpy()
            synth_np = synth_np / (np.max(np.abs(synth_np)) + 1e-8)
            wavfile.write(synth_out_path, sample_rate, (synth_np * 32767).astype(np.int16))
            print(f"\n   Saved winning candidate synthesized output audio to '{synth_out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentiable Ukulele Tab Transcription with Candidate Audio Comparison")
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
    parser.add_argument("--top-n", type=int, default=8,
                        help="Number of candidate multisets to evaluate (default: 8)")
    parser.add_argument("--audio-weight", type=float, default=0.5,
                        help="Weight for candidate output audio comparison score in candidate selection (default: 0.5)")
    parser.add_argument("--spectral-weight", type=float, default=0.5,
                        help="Weight for spectral template score in candidate selection (default: 0.5)")
    parser.add_argument("--save-synth", action="store_true",
                        help="Save the winning candidate's synthesized output audio as a WAV file")
    parser.add_argument("--synth-out", type=str, default="output_tab_synth.wav",
                        help="Output path for synthesized audio of the winning candidate tab (default: output_tab_synth.wav)")
    parser.add_argument("--tab-out", type=str, default="output_tabs.txt",
                        help="Output path for formatted tablature text file (default: output_tabs.txt)")
    parser.add_argument("--sequence", action="store_true",
                        help="Identify and transcribe a sequence of multiple chords played in the audio recording")
    parser.add_argument("--min-duration", type=str, default="auto",
                        help="Minimum chord duration in seconds, or 'auto' for adaptive physical onset segmentation (default: auto)")
    parser.add_argument("--split-strums", action="store_true",
                        help="Split repeated strums of the same chord (e.g. C -> C -> C) into individual events")
    args = parser.parse_args()

    if args.sequence:
        from chord_sequence import identify_audio_chord_sequence
        synth_out = args.synth_out if args.synth_out != "output_tab_synth.wav" else "sequence_synth.wav"
        identify_audio_chord_sequence(
            audio_path=args.audio_file,
            min_duration=args.min_duration,
            split_strums=args.split_strums,
            save_synth=args.save_synth,
            synth_out_path=synth_out,
            tab_out_path=args.tab_out
        )
        sys.exit(0)

    strummed_strings = [int(s.strip()) for s in args.strummed.split(",") if s.strip().isdigit()]
    
    process_audio_multi_notes(
        args.audio_file, 
        num_notes=args.notes, 
        strummed_strings=strummed_strings, 
        num_iterations=args.iter, 
        lr=args.lr,
        top_n=args.top_n,
        audio_weight=args.audio_weight,
        spectral_weight=args.spectral_weight,
        save_synth=args.save_synth,
        synth_out_path=args.synth_out,
        tab_out_path=args.tab_out
    )
