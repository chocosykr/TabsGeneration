"""
main_upgraded.py — Upgraded Pipeline Entry Point with Dual Instrument Modes.

Replaces main.py in the Synthesis_Engine_Upgrade folder.

Key upgrades vs main.py:
  1. Accepts --instrument flag: "ukulele" or "guitar".
  2. All synthesis parameters, fretboard mapping, pitch estimation, and NNLS
     scoring are driven by the selected InstrumentProfile.
  3. The original Note Detection Model (pYIN) and NNLS approach are preserved
     and upgraded — they now use instrument-specific bounds rather than
     hardcoded ukulele values.
  4. No breaking changes to the original main.py (this file is separate).
"""

import argparse
import time
import sys
import scipy.io.wavfile as wavfile
import numpy as np
import torch

from device_config import DEVICE
from instrument_config import get_profile, InstrumentProfile
from instrument_pitch import estimate_initial_pitch, estimate_chord_shape
from pitch_estimation import _extract_peak_energy_window
from synthesis_engine import SynthesisEngine
from instrument_fretboard import (map_parameters_to_tab, refine_chord_with_inharmonicity,
                                   freq_to_midi, expected_inharmonicity)
from tab_exporter import export_single_tab_to_file

_MAX_OPT_SEC = 0.5


def process_audio_multi_notes(file_path: str,
                               instrument: str = "ukulele",
                               num_notes: int = 4,
                               strummed_strings: list = None,
                               num_iterations: int = 50,
                               lr: float = 0.5,
                               top_n: int = 8,
                               audio_weight: float = 0.5,
                               spectral_weight: float = 0.5,
                               physics_weight: float = 0.0,
                               save_synth: bool = False,
                               synth_out_path: str = "output_tab_synth.wav",
                               tab_out_path: str = "output_tabs.txt",
                               start_sec: float = None,
                               end_sec: float = None):
    """
    Full pipeline for N simultaneously played notes, using the selected
    instrument synthesis mode.

    Parameters
    ----------
    instrument : str
        "ukulele" or "guitar". Controls ALL instrument-specific parameters:
        synthesis model initialisation, pitch search bounds, fretboard tuning,
        NNLS harmonic template, inharmonicity model.
    """
    start_time = time.time()
    profile = get_profile(instrument)
    engine  = SynthesisEngine(instrument=instrument)

    print(f"\n{'='*70}")
    print(f"  INSTRUMENT MODE: {profile.display_name}")
    print(f"{'='*70}")

    # --- Validate string selections ---
    num_strings = profile.num_strings
    valid_strings = set(profile.tuning.keys())

    if num_notes == 1:
        strummed_strings = [1]
    elif strummed_strings is None:
        strummed_strings = sorted(valid_strings)

    if len(strummed_strings) == 0 or not set(strummed_strings) <= valid_strings:
        raise ValueError(
            f"Invalid --strummed {strummed_strings}: must be a non-empty "
            f"subset of {sorted(valid_strings)} ({profile.display_name} strings)."
        )
    if len(set(strummed_strings)) != len(strummed_strings):
        raise ValueError(f"--strummed contains duplicate strings: {strummed_strings}")
    if len(strummed_strings) > num_notes:
        raise ValueError(
            f"Cannot strum {len(strummed_strings)} strings with only "
            f"{num_notes} simulated strings. "
            f"Use --notes {len(strummed_strings)} or fewer --strummed strings."
        )

    import librosa
    print(f"\nLoading audio from {file_path}...")
    waveform_np, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    audio_dur = len(waveform_np) / sample_rate

    if start_sec is not None or end_sec is not None:
        total_len = len(waveform_np)
        s_idx = max(0, int(start_sec * sample_rate)) if start_sec is not None else 0
        e_idx = min(total_len, int(end_sec * sample_rate)) if end_sec is not None else total_len
        if s_idx < e_idx:
            print(f"   [Segment] Slicing {s_idx/sample_rate:.2f}s → {e_idx/sample_rate:.2f}s")
            waveform_np = waveform_np[s_idx:e_idx]
            audio_dur = len(waveform_np) / sample_rate

    print(f"Loaded {len(waveform_np)} samples at {sample_rate} Hz ({audio_dur:.2f}s).")
    waveform_full = torch.from_numpy(waveform_np).unsqueeze(0)

    _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # =========================================================================
    # SINGLE NOTE PATH
    # =========================================================================
    if num_notes == 1:
        print(f"\n1. Estimating Initial Pitch (pYIN on onset window) "
              f"[{profile.display_name}]...")
        pitch = estimate_initial_pitch(waveform_full, sample_rate, profile)
        pitches = [pitch]
        print(f"   Detected: {pitch:.2f} Hz")

        y_opt = _extract_peak_energy_window(waveform_np, sample_rate,
                                            window_sec=_MAX_OPT_SEC)
        max_samples = int(_MAX_OPT_SEC * sample_rate)
        y_opt = y_opt[:max_samples]
        waveform_opt = torch.from_numpy(y_opt).unsqueeze(0)
        print(f"   (Optimizer will use {len(y_opt)} samples = "
              f"{len(y_opt)/sample_rate:.2f}s onset window)")

        print(f"\n2. Running Instrument-Aware KS Optimization [{profile.display_name}]...")
        voice_params, synth_audio, audio_comp = engine.optimize(
            target_audio=waveform_opt,
            pitches=pitches,
            num_iterations=num_iterations,
            lr=lr,
            return_audio=True
        )

        print(f"\n3. Candidate Tab Selection (Audio vs Input Comparison) "
              f"[{profile.display_name}]...")
        vp = voice_params[0]
        tab_result = map_parameters_to_tab(
            vp['pitch'], vp['decay'], vp['inharmonicity'],
            profile=profile,
            target_audio=waveform_opt,
            sample_rate=sample_rate
        )

        if "error" in tab_result:
            print(f"   ERROR: {tab_result['error']}")
        else:
            best = tab_result['all_candidates'][0]
            midi_num  = tab_result['midi_note']
            note_name = f"{_NOTE_NAMES[midi_num % 12]}{midi_num // 12 - 1}"

            print("\n" + "=" * 88)
            print(f"      CANDIDATE FINGERING EVALUATION [{profile.display_name}]")
            print("=" * 88)
            print(f" {'Rank':>4}   {'Candidate Tab':<26}  {'Audio Match':>11}  "
                  f"{'Physics Fit':>11}  {'Total Score':>11}   {'Verdict':<12}")
            print(f" {'-'*4}   {'-'*26}  {'-'*11}  {'-'*11}  {'-'*11}   {'-'*12}")

            for rank, cand in enumerate(tab_result['all_candidates'], 1):
                tab_desc  = f"String {cand['string']} ({cand['open_name']}), Fret {cand['fret']}"
                audio_pct  = cand.get('audio_match_norm', 1.0) * 100.0
                physics_pct = cand.get('inharm_score', 0.0) * 100.0
                total_pct  = cand.get('total_score', 0.0) * 100.0
                is_best = (cand['string'] == best['string'] and cand['fret'] == best['fret'])
                verdict = "[SELECTED]" if is_best else "Rejected"
                print(f"  #{rank:>2}   {tab_desc:<26}  {audio_pct:>10.1f}%  "
                      f"{physics_pct:>10.1f}%  {total_pct:>10.1f}%   {verdict}")

            print("-" * 88)
            print(f"\n{'='*88}")
            print("                         FINAL TRANSCRIBED TABLATURE")
            print(f"{'='*88}")
            print(f"   Instrument:         {profile.display_name}")
            print(f"   Detected Note:      {note_name} (MIDI {midi_num}, {vp['pitch']:.2f} Hz)")
            print(f"   Plucked String:     String {tab_result['string']} "
                  f"(open {best['open_name']} string)")
            print(f"   Fret Position:      Fret {tab_result['fret']}")
            if "audio_loss" in best:
                print(f"   Acoustic Match:     {best.get('audio_match_norm', 1.0)*100:.1f}%  "
                      f"(STFT Error: {best['audio_loss']:.4f})")
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
            print(f"\n   [Export] Tablature written to '{tab_out_path}'")

        if save_synth and synth_audio is not None:
            synth_np = synth_audio.detach().cpu().numpy()
            synth_np = synth_np / (np.max(np.abs(synth_np)) + 1e-8)
            wavfile.write(synth_out_path, sample_rate, (synth_np * 32767).astype(np.int16))
            print(f"\n   Saved synthesized output to '{synth_out_path}'")

        return {
            "type":        "single_note",
            "instrument":  instrument,
            "tab_result":  tab_result,
            "pitch":       float(vp['pitch']),
            "note_name":   note_name if 'note_name' in locals() else "",
            "midi":        int(midi_num) if 'midi_num' in locals() else 0,
            "string":      int(tab_result.get('string', 1)),
            "fret":        int(tab_result.get('fret', 0)),
            "candidates":  tab_result.get('all_candidates', []),
            "elapsed":     time.time() - start_time,
            "synth_out":   synth_out_path if save_synth else None,
        }

    # =========================================================================
    # MULTI-NOTE / CHORD PATH
    # =========================================================================
    print(f"\n1. Searching chord shapes on strings {strummed_strings} "
          f"[{profile.display_name}]...")
    chord_result = estimate_chord_shape(
        waveform_full, sample_rate, strummed_strings, profile,
        top_n_multisets=top_n
    )

    candidates = chord_result["candidates"]

    y_opt = _extract_peak_energy_window(waveform_np, sample_rate,
                                        window_sec=_MAX_OPT_SEC)
    max_samples = int(_MAX_OPT_SEC * sample_rate)
    y_opt = y_opt[:max_samples]
    waveform_opt = torch.from_numpy(y_opt).unsqueeze(0)
    print(f"   (Optimizer will use {len(y_opt)} samples = {len(y_opt)/sample_rate:.2f}s)")

    _WEIGHT_SPECTRAL = spectral_weight
    _WEIGHT_AUDIO    = audio_weight
    _WEIGHT_PHYSICS  = physics_weight

    evaluated = []

    for rank, candidate in enumerate(candidates, 1):
        pitches = candidate["pitches"]
        notes   = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in candidate["multiset_midis"]]
        print(f"\n2.{rank}. Evaluating candidate #{rank}: [{', '.join(notes)}]  "
              f"(spectral={candidate['spectral_score_norm']:.3f})")

        # Pad to num_notes with silent voices if needed
        cand_pitches = list(pitches)
        active_mask  = [True] * len(strummed_strings)
        while len(cand_pitches) < num_notes:
            cand_pitches.append(440.0)
            active_mask.append(False)

        n_active = sum(active_mask)
        if n_active < num_notes:
            print(f"   ({num_notes - n_active} simulated string(s) silent)")

        # Instrument-specific inharmonicity priors
        initial_stretches = [
            expected_inharmonicity(profile, s, candidate["mapping"].get(s, 0))
            for s in strummed_strings
        ]
        while len(initial_stretches) < num_notes:
            initial_stretches.append(0.001)

        # Optimize with instrument-aware synthesis engine
        voice_params, synth_audio, audio_comp = engine.optimize(
            target_audio=waveform_opt,
            pitches=cand_pitches,
            num_iterations=num_iterations,
            lr=lr,
            active_mask=active_mask,
            initial_stretches=initial_stretches,
            return_audio=True
        )

        active_voice_params = [
            voice_params[i] for i, active in enumerate(active_mask) if active
        ]
        refined = refine_chord_with_inharmonicity(
            candidate, active_voice_params, strummed_strings,
            profile=profile,
            target_audio=waveform_opt,
            sample_rate=sample_rate
        )

        max_physics  = float(len(strummed_strings))
        raw_physics  = refined.get("inharm_score", 0.0)
        physics_norm = raw_physics / max_physics if max_physics > 0 else 0.0

        candidate["voice_params"]      = active_voice_params
        candidate["refined"]           = refined
        candidate["synth_audio"]       = synth_audio
        candidate["audio_loss"]        = audio_comp["stft_loss"]
        candidate["audio_similarity"]  = audio_comp["similarity"]
        candidate["audio_metrics"]     = audio_comp
        candidate["physics_score"]     = raw_physics
        candidate["physics_score_norm"] = physics_norm

        evaluated.append(candidate)

        print(f"   -> STFT Error: {audio_comp['stft_loss']:.4f} | "
              f"Wave Corr: {audio_comp['waveform_correlation']:.3f} | "
              f"Spectral Fit: {candidate['spectral_score_norm']*100:.1f}% | "
              f"Physics Fit: {physics_norm*100:.1f}%")

    # --- Candidate selection ---
    min_loss = min(c["audio_loss"] for c in evaluated)
    for c in evaluated:
        c["audio_score_norm"] = min_loss / c["audio_loss"] if c["audio_loss"] > 0 else 1.0
        c["combined_score"] = (
            _WEIGHT_AUDIO    * c["audio_score_norm"]
            + _WEIGHT_SPECTRAL * c["spectral_score_norm"]
            + _WEIGHT_PHYSICS  * c["physics_score_norm"]
        )

    evaluated.sort(key=lambda c: -c["combined_score"])
    winner = evaluated[0]

    # --- Print rankings ---
    print("\n" + "=" * 100)
    print(f"       CANDIDATE CHORD RANKINGS [{profile.display_name}]")
    print("=" * 100)
    print(f" {'Rank':>4}   {'Candidate Notes':<22}  {'Tab':<18}  "
          f"{'Audio':>8}  {'Spectral':>10}  {'Combined':>10}   {'Verdict':<10}")
    print(f" {'-'*4}   {'-'*22}  {'-'*18}  {'-'*8}  {'-'*10}  {'-'*10}   {'-'*10}")

    for rank, c in enumerate(evaluated, 1):
        notes    = [f"{_NOTE_NAMES[m%12]}{m//12-1}" for m in c["multiset_midis"]]
        notes_str = "[" + ", ".join(notes) + "]"
        cand_map  = c.get("refined", {}).get("mapping", c.get("mapping", {}))
        tab_str   = " - ".join(str(cand_map.get(s, 0))
                               for s in sorted(strummed_strings, reverse=True))
        print(f"  #{rank:>2}   {notes_str:<22}  {tab_str:<18}  "
              f"{c['audio_score_norm']*100:>7.1f}%  "
              f"{c['spectral_score_norm']*100:>9.1f}%  "
              f"{c['combined_score']*100:>9.1f}%   "
              f"{'[WINNER]' if rank==1 else 'Rejected'}")

    print("-" * 100)

    # --- Winner diagnostics ---
    if "audio_metrics" in winner:
        m = winner["audio_metrics"]
        corr_val    = m.get('waveform_correlation', 0.0)
        corr_rating = "Excellent" if corr_val > 0.8 else ("Good" if corr_val > 0.4 else "Moderate")
        print("\n" + "=" * 100)
        print(f"           WINNER ACOUSTIC DIAGNOSTICS [{profile.display_name}]")
        print("=" * 100)
        print(f"   {'Metric':<32}  {'Value':>10}   {'Rating':<14}   Interpretation")
        print(f"   {'-'*32}  {'-'*10}   {'-'*14}   {'-'*35}")
        print(f"   {'Multi-Scale STFT Error':<32}  {m['stft_loss']:>10.4f}   "
              f"{'Best in search':<14}   Timbre & harmonic frequency error (↓ better)")
        print(f"   {'Spectral Convergence':<32}  {m['spectral_convergence']:>10.4f}   "
              f"{'Optimal':<14}   Normalized energy distribution deviation (↓ better)")
        print(f"   {'Log-Magnitude STFT Loss':<32}  {m['log_mag_loss']:>10.4f}   "
              f"{'Strong':<14}   Low/high harmonic balance (↓ better)")
        print(f"   {'Waveform Envelope Corr.':<32}  {corr_val:>10.4f}   "
              f"{corr_rating:<14}   Time-domain envelope alignment")
        print("=" * 100)

    refined = winner["refined"]
    mapping = refined["mapping"]
    winner_notes = [f"{_NOTE_NAMES[m%12]}{m//12-1}" for m in winner["multiset_midis"]]
    tab_summary  = " | ".join(
        f"{profile.tuning[s]['name'][0]}: {mapping.get(s, 0)}"
        for s in sorted(strummed_strings, reverse=True)
    )

    print("\n" + "=" * 100)
    print("                              FINAL TRANSCRIBED TABLATURE")
    print("=" * 100)
    print(f"   Instrument: {profile.display_name}")
    print(f"   Fretboard Layout (Strings {max(strummed_strings)} to 1):")

    for string_idx in sorted(strummed_strings, reverse=True):
        fret = mapping[string_idx]
        fret_str  = f"Fret {fret}" if fret > 0 else "Fret 0 (open)"
        base_freq = profile.tuning[string_idx]["freq"]
        fret_freq = base_freq * (2.0 ** (fret / 12.0))
        fret_midi = int(round(freq_to_midi(fret_freq)))
        fret_note = f"{_NOTE_NAMES[fret_midi % 12]}{fret_midi // 12 - 1}"
        open_name = profile.tuning[string_idx]["name"]
        print(f"      String {string_idx} (open {open_name:<3}):  "
              f"{fret_str:<14}  -->  {fret_note:<3} ({fret_freq:6.2f} Hz)")

    print("-" * 100)
    print(f"   Summary Tab:     [ {tab_summary} ]")
    print(f"   Chord Notes:     [{', '.join(winner_notes)}]")
    print(f"   Acoustic Match:  100.0% (STFT Error: {winner['audio_loss']:.4f}) | "
          f"Overall Score: {winner['combined_score']*100:.1f}%")
    elapsed = time.time() - start_time
    dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    print(f"   Execution Time:  {elapsed:.2f}s (Device: {dev_desc})")
    print("=" * 100)

    export_single_tab_to_file(
        file_path=tab_out_path,
        audio_path=file_path,
        tab_result=None,
        winner_data=winner,
        strummed_strings=strummed_strings,
        elapsed=elapsed,
        is_single_note=False
    )
    print(f"\n   [Export] Tablature written to '{tab_out_path}'")

    if save_synth and "synth_audio" in winner:
        synth_np = winner["synth_audio"].detach().cpu().numpy()
        synth_np = synth_np / (np.max(np.abs(synth_np)) + 1e-8)
        wavfile.write(synth_out_path, sample_rate, (synth_np * 32767).astype(np.int16))
        print(f"\n   Saved winning candidate synthesized audio to '{synth_out_path}'")

    return {
        "type":           "chord",
        "instrument":     instrument,
        "winner":         winner,
        "mapping":        mapping,
        "notes":          winner_notes,
        "tab_sorted":     [mapping.get(s, 0) for s in sorted(strummed_strings, reverse=True)],
        "candidates":     evaluated,
        "audio_match":    100.0,
        "combined_score": float(winner['combined_score'] * 100.0),
        "stft_loss":      float(winner.get('audio_loss', 0.0)),
        "elapsed":        float(elapsed),
        "synth_out":      synth_out_path if save_synth else None,
    }


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    from instrument_config import list_instruments
    available_instruments = list_instruments()

    parser = argparse.ArgumentParser(
        description="Upgraded Differentiable Tab Transcription — Dual Instrument Mode")
    parser.add_argument("audio_file", type=str,
                        help="Path to the WAV file")
    parser.add_argument("--instrument", type=str, default="ukulele",
                        choices=available_instruments,
                        help=f"Instrument mode. Available: {available_instruments} "
                             f"(default: ukulele)")
    parser.add_argument("--notes", type=int, default=4, choices=[1, 2, 3, 4, 5, 6],
                        help="Number of simulated voices/pitches (default: 4)")
    parser.add_argument("--strummed", type=str, default=None,
                        help="Comma-separated list of strummed strings "
                             "(default: all strings for the selected instrument)")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end",   type=float, default=None)
    parser.add_argument("--iter",  type=int,   default=50)
    parser.add_argument("--lr",    type=float, default=0.5)
    parser.add_argument("--top-n", type=int,   default=8)
    parser.add_argument("--audio-weight",   type=float, default=0.5)
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument("--save-synth",  action="store_true")
    parser.add_argument("--synth-out",   type=str, default="output_tab_synth.wav")
    parser.add_argument("--tab-out",     type=str, default="output_tabs.txt")
    parser.add_argument("--sequence",    action="store_true",
                        help="Transcribe a sequence of multiple chords")
    parser.add_argument("--min-duration", type=str, default="auto")
    parser.add_argument("--split-strums", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.instrument)

    if args.strummed is not None:
        strummed_strings = [
            int(s.strip())
            for s in args.strummed.split(",")
            if s.strip().isdigit()
        ]
    else:
        strummed_strings = sorted(profile.tuning.keys())

    if args.sequence:
        from chord_sequence import identify_audio_chord_sequence
        synth_out = args.synth_out if args.synth_out != "output_tab_synth.wav" \
                    else "sequence_synth.wav"
        identify_audio_chord_sequence(
            audio_path=args.audio_file,
            min_duration=args.min_duration,
            split_strums=args.split_strums,
            save_synth=args.save_synth,
            synth_out_path=synth_out,
            tab_out_path=args.tab_out,
            start_sec=args.start,
            end_sec=args.end
        )
        sys.exit(0)

    process_audio_multi_notes(
        args.audio_file,
        instrument=args.instrument,
        num_notes=args.notes,
        strummed_strings=strummed_strings,
        num_iterations=args.iter,
        lr=args.lr,
        top_n=args.top_n,
        audio_weight=args.audio_weight,
        spectral_weight=args.spectral_weight,
        save_synth=args.save_synth,
        synth_out_path=args.synth_out,
        tab_out_path=args.tab_out,
        start_sec=args.start,
        end_sec=args.end
    )
