import time
import math
import argparse
import sys
import os
import numpy as np
import scipy.ndimage
import scipy.io.wavfile as wavfile
import librosa
import torch
from typing import List, Dict, Optional, Tuple, Union

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from device_config import DEVICE
from fretboard_mapper import UKULELE_TUNING, freq_to_midi, expected_inharmonicity
from optimization import synthesize_candidate_tab_audio, compare_candidate_audio
from tab_exporter import export_sequence_tabs_to_file

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Standard Ukulele Chord Voicings (Strings 4-3-2-1: G - C - E - A)
UKULELE_CHORD_DEFINITIONS = {
    # Major Chords
    "C":   ((0, 0, 0, 3), "Major", "C"),
    "C#":  ((1, 1, 1, 4), "Major", "C#"),
    "Db":  ((1, 1, 1, 4), "Major", "C#"),
    "D":   ((2, 2, 2, 0), "Major", "D"),
    "D#":  ((3, 3, 3, 1), "Major", "D#"),
    "Eb":  ((0, 3, 3, 1), "Major", "D#"),
    "E":   ((1, 4, 0, 2), "Major", "E"),
    "F":   ((2, 0, 1, 0), "Major", "F"),
    "F#":  ((3, 1, 2, 1), "Major", "F#"),
    "Gb":  ((3, 1, 2, 1), "Major", "F#"),
    "G":   ((0, 2, 3, 2), "Major", "G"),
    "G#":  ((5, 3, 4, 3), "Major", "G#"),
    "Ab":  ((5, 3, 4, 3), "Major", "G#"),
    "A":   ((2, 1, 0, 0), "Major", "A"),
    "A#":  ((3, 2, 1, 1), "Major", "A#"),
    "Bb":  ((3, 2, 1, 1), "Major", "A#"),
    "B":   ((4, 3, 2, 2), "Major", "B"),

    # Minor Chords
    "Cm":  ((0, 3, 3, 3), "Minor", "C"),
    "C#m": ((1, 4, 4, 4), "Minor", "C#"),
    "Dbm": ((1, 4, 4, 4), "Minor", "C#"),
    "Dm":  ((2, 2, 1, 0), "Minor", "D"),
    "D#m": ((3, 3, 2, 1), "Minor", "D#"),
    "Ebm": ((3, 3, 2, 1), "Minor", "D#"),
    "Em":  ((0, 4, 3, 2), "Minor", "E"),
    "Fm":  ((1, 0, 1, 3), "Minor", "F"),
    "F#m": ((2, 1, 2, 0), "Minor", "F#"),
    "Gbm": ((2, 1, 2, 0), "Minor", "F#"),
    "Gm":  ((0, 2, 3, 1), "Minor", "G"),
    "G#m": ((4, 3, 4, 2), "Minor", "G#"),
    "Abm": ((4, 3, 4, 2), "Minor", "G#"),
    "Am":  ((2, 0, 0, 0), "Minor", "A"),
    "A#m": ((3, 1, 1, 1), "Minor", "A#"),
    "Bbm": ((3, 1, 1, 1), "Minor", "A#"),
    "Bm":  ((4, 2, 2, 2), "Minor", "B"),

    # Dominant 7th Chords
    "C7":  ((0, 0, 0, 1), "7th", "C"),
    "D7":  ((2, 0, 2, 0), "7th", "D"),
    "E7":  ((1, 2, 0, 2), "7th", "E"),
    "F7":  ((2, 3, 1, 0), "7th", "F"),
    "G7":  ((0, 2, 1, 2), "7th", "G"),
    "A7":  ((0, 1, 0, 0), "7th", "A"),
    "B7":  ((2, 3, 2, 2), "7th", "B"),
    "Bb7": ((1, 2, 1, 1), "7th", "A#"),

    # Major 7th & Minor 7th Chords
    "Cmaj7": ((0, 0, 0, 2), "Maj7", "C"),
    "Fmaj7": ((2, 4, 1, 0), "Maj7", "F"),
    "Gmaj7": ((0, 2, 2, 2), "Maj7", "G"),
    "Am7":   ((0, 0, 0, 0), "m7", "A"),
    "Dm7":   ((2, 2, 1, 3), "m7", "D"),
    "Em7":   ((0, 2, 0, 2), "m7", "E"),
}


def _build_chord_templates() -> Dict[str, dict]:
    chord_lib = {}
    for name, (tab, quality, root) in UKULELE_CHORD_DEFINITIONS.items():
        s4, s3, s2, s1 = tab
        f4 = UKULELE_TUNING[4]["freq"] * (2.0 ** (s4 / 12.0))
        f3 = UKULELE_TUNING[3]["freq"] * (2.0 ** (s3 / 12.0))
        f2 = UKULELE_TUNING[2]["freq"] * (2.0 ** (s2 / 12.0))
        f1 = UKULELE_TUNING[1]["freq"] * (2.0 ** (s1 / 12.0))

        def _m2n(freq):
            m = int(round(freq_to_midi(freq)))
            return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

        notes_4to1 = [_m2n(f4), _m2n(f3), _m2n(f2), _m2n(f1)]

        r_idx = NOTE_NAMES.index(root)
        chroma = np.zeros(12, dtype=float)
        chroma[r_idx] = 1.0  # Root

        if quality == "Major":
            chroma[(r_idx + 4) % 12] = 0.8
            chroma[(r_idx + 7) % 12] = 0.9
        elif quality == "Minor":
            chroma[(r_idx + 3) % 12] = 0.8
            chroma[(r_idx + 7) % 12] = 0.9
        elif quality == "7th":
            chroma[(r_idx + 4) % 12] = 0.8
            chroma[(r_idx + 7) % 12] = 0.8
            chroma[(r_idx + 10) % 12] = 0.7
        elif quality == "Maj7":
            chroma[(r_idx + 4) % 12] = 0.8
            chroma[(r_idx + 7) % 12] = 0.8
            chroma[(r_idx + 11) % 12] = 0.7
        elif quality == "m7":
            chroma[(r_idx + 3) % 12] = 0.8
            chroma[(r_idx + 7) % 12] = 0.8
            chroma[(r_idx + 10) % 12] = 0.7

        norm = np.linalg.norm(chroma)
        chroma_norm = chroma / norm if norm > 0 else chroma

        chord_lib[name] = {
            "name": name,
            "quality": quality,
            "root": root,
            "tab_4321": tab,
            "tab_1234": [s1, s2, s3, s4],
            "pitches_1234": [f1, f2, f3, f4],
            "pitches_4321": [f4, f3, f2, f1],
            "notes_4321": notes_4to1,
            "chroma": chroma_norm
        }
    return chord_lib

CHORD_LIB = _build_chord_templates()


def segment_and_identify_chords(y: np.ndarray,
                                sr: int = 16000,
                                min_duration: Union[float, str] = "auto",
                                energy_threshold: float = 0.005,
                                hop_length: int = 512,
                                split_strums: bool = False) -> List[dict]:
    """
    Identifies a sequence of chords played in an audio recording using
    Adaptive Onset-to-Onset (O2O) Physical Segmentation.

    Why Adaptive Segmentation?
      Songs have variable strum and pluck durations: fast intro riffs / fingerstyle
      plucks can last 0.18s - 0.35s, while slow ringing chords last > 2.0s.
      A fixed min_duration either drops fast notes or causes spurious over-segmentation.
      Adaptive segmentation uses physical string attack onsets as dynamic boundaries,
      weights chroma by instantaneous RMS energy, gates by harmonic confidence,
      and merges sustained chords.

    Parameters:
      min_duration: 'auto' (default) adapts dynamically to fast plucks (~0.18s) and
                    long sustained chords. If set to a positive float (e.g. 0.65),
                    it enforces an explicit minimum duration floor.
      split_strums: If True, splits consecutive strums of the same chord into
                    individual events. If False (default), merges repeated strums
                    into sustained chord events and records strum counts.
    """
    fps = sr / hop_length
    total_duration = len(y) / sr

    # 1. Detect physical string attack onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    min_wait_frames = max(2, int(0.18 * fps))
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=6, post_max=6,
        pre_avg=6, post_avg=6,
        delta=0.35,
        wait=min_wait_frames
    )
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # 2. Build adaptive boundary intervals
    boundaries = [0.0]
    for t in peak_times:
        if (t - boundaries[-1]) >= 0.18:
            boundaries.append(float(t))
    if (total_duration - boundaries[-1]) > 0.18:
        boundaries.append(float(total_duration))

    # 3. Extract RMS and Chroma CQT
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    raw_segments = []
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        dur = t1 - t0
        f0 = int(t0 * fps)
        f1 = int(t1 * fps)
        if f1 <= f0:
            continue

        interval_rms = rms[f0:f1]
        if len(interval_rms) == 0 or np.max(interval_rms) < energy_threshold:
            continue

        # Energy-weighted chroma pooling (attack transient dominates decaying tail)
        weights = interval_rms / (np.sum(interval_rms) + 1e-8)
        chroma_pooled = np.sum(chroma[:, f0:f1] * weights, axis=1)
        norm = np.linalg.norm(chroma_pooled)
        if norm < 1e-4:
            continue
        chroma_norm = chroma_pooled / norm

        # Template correlation matching against ukulele chord dictionary
        best_name = None
        best_sim = -1.0
        for name, data in CHORD_LIB.items():
            sim = float(np.dot(chroma_norm, data["chroma"]))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # Harmonic confidence gating (0.72 confidence floor)
        if best_sim >= 0.72:
            raw_segments.append({
                "chord": best_name,
                "start": t0,
                "end": t1,
                "duration": dur,
                "confidence": best_sim,
                "chord_data": CHORD_LIB[best_name]
            })

    # 4. Continuation and Root Merging
    if not split_strums:
        merged = []
        for s in raw_segments:
            if not merged:
                merged.append(s)
                continue
            prev = merged[-1]
            same_chord = (prev["chord"] == s["chord"])
            same_root = (prev["chord_data"]["root"] == s["chord_data"]["root"])
            gap = s["start"] - prev["end"]
            if gap < 0.5 and (same_chord or same_root):
                prev["end"] = s["end"]
                prev["duration"] = prev["end"] - prev["start"]
                prev["confidence"] = max(prev["confidence"], s["confidence"])
            else:
                merged.append(s)
        final_segments = merged
    else:
        final_segments = raw_segments

    # Count physical strums in each segment
    for seg in final_segments:
        s_count = sum(1 for t in peak_times if (seg["start"] - 0.15) <= t <= (seg["end"] + 0.1))
        seg["strum_count"] = max(1, s_count)

    # 5. Optional explicit duration floor filter
    is_auto = (str(min_duration).strip().lower() in ["auto", "none", "0"])
    if not is_auto:
        try:
            dur_floor = float(min_duration)
            if dur_floor > 0:
                final_segments = [s for s in final_segments if s["duration"] >= dur_floor]
        except (ValueError, TypeError):
            pass

    return final_segments


def verify_segment_acoustics(seg: dict, 
                             y: np.ndarray, 
                             sr: int) -> dict:
    chord_data = seg["chord_data"]
    start_sample = int(seg["start"] * sr)
    end_sample = min(len(y), int(seg["end"] * sr))
    seg_audio = y[start_sample:end_sample]

    comp_samples = min(len(seg_audio), int(0.4 * sr))
    if comp_samples < int(0.15 * sr):
        seg["acoustic_match"] = seg["confidence"]
        seg["stft_loss"] = 0.0
        return seg

    target_tensor = torch.from_numpy(seg_audio[:comp_samples]).float().unsqueeze(0).to(DEVICE)
    dur_sec = comp_samples / sr

    pitches = chord_data["pitches_1234"]
    frets = chord_data["tab_1234"]
    stretches = [expected_inharmonicity(s, f) for s, f in zip([1, 2, 3, 4], frets)]

    try:
        with torch.no_grad():
            synth_tensor = synthesize_candidate_tab_audio(
                pitches=pitches,
                sample_rate=sr,
                duration=dur_sec,
                stretches=stretches
            )
            comp = compare_candidate_audio(synth_tensor, target_tensor, sample_rate=sr)
            seg["acoustic_metrics"] = comp
            seg["stft_loss"] = comp["stft_loss"]
            seg["acoustic_match"] = comp["similarity"]
    except Exception:
        seg["acoustic_match"] = seg["confidence"]
        seg["stft_loss"] = 0.0

    return seg


def generate_synthesized_sequence_audio(segments: List[dict], 
                                        total_duration: float, 
                                        sr: int = 16000) -> np.ndarray:
    total_samples = int(total_duration * sr) + int(0.5 * sr)
    full_audio = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        chord_data = seg["chord_data"]
        start_idx = int(seg["start"] * sr)
        dur = max(0.5, seg["duration"])

        pitches = chord_data["pitches_1234"]
        frets = chord_data["tab_1234"]
        stretches = [expected_inharmonicity(s, f) for s, f in zip([1, 2, 3, 4], frets)]

        with torch.no_grad():
            synth = synthesize_candidate_tab_audio(
                pitches=pitches,
                sample_rate=sr,
                duration=dur,
                stretches=stretches
            )
        synth_np = synth.detach().cpu().squeeze().numpy()
        end_idx = min(total_samples, start_idx + len(synth_np))
        full_audio[start_idx:end_idx] += synth_np[:end_idx - start_idx]

    max_val = np.max(np.abs(full_audio))
    if max_val > 1e-6:
        full_audio = full_audio / max_val * 0.95

    return full_audio


def print_sequence_results(audio_path: str, 
                           segments: List[dict], 
                           total_duration: float, 
                           elapsed: float,
                           split_strums: bool = False):
    dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    mode_str = "Individual Strum Mode (--split-strums)" if split_strums else "Harmonic Progression Mode"

    print("\n" + "=" * 98)
    print("                    UKULELE CHORD SEQUENCE IDENTIFICATION & TRANSCRIPTION")
    print("=" * 98)
    print(f"   Audio File:     {audio_path}")
    print(f"   Duration:       {total_duration:.2f}s  |  Total Events: {len(segments)}  ({mode_str})")
    print(f"   Hardware:       {dev_desc}  |  Processing Time: {elapsed:.2f}s")
    print("=" * 98)

    if not segments:
        print("\n   [!] No sustained chords detected in the provided audio file.")
        print("=" * 98)
        return

    # 1. Progression Sequence Flow
    prog_str = "  ==>  ".join(f"[ {s['chord']} {s['chord_data']['quality']} ]" for s in segments)
    print("\n   CHORD PROGRESSION SEQUENCE:")
    print(f"   {prog_str}")
    print("\n" + "-" * 98)

    # 2. Timeline Table
    print("   CHORD SEQUENCE TIMELINE & TABLATURE:")
    print("=" * 98)
    print(f" {'#':>2}   {'Time Interval':<16}  {'Duration':>9}  {'Chord':<16}  {'Tab (4-3-2-1)':<15}  {'Notes Played':<20}  {'Match %':>9}")
    print(f" {'-'*2}   {'-'*16}  {'-'*9}  {'-'*16}  {'-'*15}  {'-'*20}  {'-'*9}")

    for idx, s in enumerate(segments, 1):
        time_str = f"{s['start']:4.2f}s - {s['end']:4.2f}s"
        dur_str = f"{s['duration']:4.2f}s"
        strum_note = f" ({s['strum_count']} strums)" if not split_strums and s.get("strum_count", 1) > 1 else ""
        chord_label = f"{s['chord']} {s['chord_data']['quality']}{strum_note}"
        tab_str = " - ".join(str(f) for f in s["chord_data"]["tab_4321"])
        notes_str = "[" + ", ".join(s["chord_data"]["notes_4321"]) + "]"
        match_pct = s.get("confidence", 0.0) * 100.0

        print(f" #{idx:<2}  {time_str:<16}  {dur_str:>9}  {chord_label:<16}  {tab_str:<15}  {notes_str:<20}  {match_pct:>8.1f}%")

    print("-" * 98)

    # 3. Multi-measure ASCII Ukulele Tablature Staff
    print("\n   ASCII UKULELE TABLATURE SCORE (Strings 1 to 4):")
    print("   " + "-" * 80)

    staff_lines = {
        1: "   String 1 (A4) |",
        2: "   String 2 (E4) |",
        3: "   String 3 (C4) |",
        4: "   String 4 (G4) |"
    }
    chord_markers = "                 |"

    for s in segments:
        tab_4321 = s["chord_data"]["tab_4321"]
        s4, s3, s2, s1 = tab_4321
        w = 12
        staff_lines[1] += f"--{s1:<2}" + "-" * (w - 4) + "|"
        staff_lines[2] += f"--{s2:<2}" + "-" * (w - 4) + "|"
        staff_lines[3] += f"--{s3:<2}" + "-" * (w - 4) + "|"
        staff_lines[4] += f"--{s4:<2}" + "-" * (w - 4) + "|"

        chord_str = f"[{s['chord']}]"
        chord_markers += f"  {chord_str:<9}|"

    for str_idx in [1, 2, 3, 4]:
        print(staff_lines[str_idx])
    print("   " + "-" * 80)
    print(chord_markers)
    print("=" * 98)


def identify_audio_chord_sequence(audio_path: str,
                                  min_duration: Union[float, str] = "auto",
                                  energy_threshold: float = 0.005,
                                  split_strums: bool = False,
                                  save_synth: bool = False,
                                  synth_out_path: str = "sequence_synth.wav",
                                  tab_out_path: str = "output_tabs.txt",
                                  start_sec: Optional[float] = None,
                                  end_sec: Optional[float] = None) -> List[dict]:
    start_time = time.time()
    print(f"\n[TabsGeneration] Loading audio from '{audio_path}' for chord sequence analysis...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    dur = len(y) / sr

    # Slice targeted segment if specified
    time_offset = 0.0
    if start_sec is not None or end_sec is not None:
        total_len = len(y)
        s_idx = max(0, int(start_sec * sr)) if start_sec is not None else 0
        e_idx = min(total_len, int(end_sec * sr)) if end_sec is not None else total_len
        if s_idx < e_idx:
            time_offset = s_idx / sr
            print(f"   [Segment Targeting] Slicing {time_offset:.2f}s to {e_idx/sr:.2f}s "
                  f"({(e_idx - s_idx)/sr:.2f}s duration)")
            y = y[s_idx:e_idx]
            dur = len(y) / sr

    print(f"Loaded {len(y)} samples at {sr} Hz ({dur:.2f}s duration).")

    print("\n1. Performing Harmonic Segmentation & Multi-Chord Detection...")
    segments = segment_and_identify_chords(
        y, sr, 
        min_duration=min_duration, 
        energy_threshold=energy_threshold,
        split_strums=split_strums
    )

    print(f"   Found {len(segments)} distinct chord segments.")

    # Verify acoustics BEFORE shifting timestamps (y is still the local slice)
    print("\n2. Performing Acoustic Synthesis & Waveform Verification for each chord...")
    for idx, seg in enumerate(segments, 1):
        verify_segment_acoustics(seg, y, sr)

    # Shift timestamps by time_offset AFTER acoustic verification
    if time_offset > 0:
        for seg in segments:
            seg["start"] += time_offset
            seg["end"] += time_offset

    for idx, seg in enumerate(segments, 1):
        stft_info = f" (STFT Loss: {seg['stft_loss']:.4f})" if "stft_loss" in seg and seg['stft_loss'] > 0 else ""
        print(f"   Chord #{idx}: {seg['chord']:<4} [{seg['start']:4.2f}s - {seg['end']:4.2f}s]  ->  Tab: {seg['chord_data']['tab_4321']}  Match: {seg['confidence']*100:.1f}%{stft_info}")

    elapsed = time.time() - start_time
    print_sequence_results(audio_path, segments, dur, elapsed, split_strums=split_strums)

    # Export formatted tablature to text file (overwritten every run)
    export_sequence_tabs_to_file(
        file_path=tab_out_path,
        audio_path=audio_path,
        segments=segments,
        total_duration=dur,
        elapsed=elapsed,
        split_strums=split_strums
    )
    print(f"\n   [Export] Formatted tablature successfully written to '{tab_out_path}'")

    if save_synth and segments:
        print(f"\n3. Synthesizing full chord progression audio track to '{synth_out_path}'...")
        synth_track = generate_synthesized_sequence_audio(segments, dur, sr=sr)
        wavfile.write(synth_out_path, sr, (synth_track * 32767).astype(np.int16))
        print(f"   Successfully exported synthesized audio sequence to '{synth_out_path}'!")

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ukulele Chord Sequence Identification & Tablature Transcription Pipeline")
    parser.add_argument("audio_file", type=str,
                        help="Path to the ukulele WAV file with a sequence of chords")
    parser.add_argument("--min-duration", type=float, default=0.65,
                        help="Minimum chord duration in seconds (default: 0.65)")
    parser.add_argument("--energy-thresh", type=float, default=0.005,
                        help="RMS energy threshold for silence/noise gating (default: 0.005)")
    parser.add_argument("--split-strums", action="store_true",
                        help="Split repeated strums of the same chord into individual chord events")
    parser.add_argument("--save-synth", action="store_true",
                        help="Save the synthesized audio of the chord progression")
    parser.add_argument("--synth-out", type=str, default="sequence_synth.wav",
                        help="Output path for synthesized audio (default: sequence_synth.wav)")
    parser.add_argument("--tab-out", type=str, default="output_tabs.txt",
                        help="Output path for formatted tablature text file (default: output_tabs.txt)")
    parser.add_argument("--start", type=float, default=None,
                        help="Start time in seconds for targeted segment execution")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds for targeted segment execution")
    args = parser.parse_args()

    identify_audio_chord_sequence(
        audio_path=args.audio_file,
        min_duration=args.min_duration,
        energy_threshold=args.energy_thresh,
        split_strums=args.split_strums,
        save_synth=args.save_synth,
        synth_out_path=args.synth_out,
        tab_out_path=args.tab_out,
        start_sec=args.start,
        end_sec=args.end
    )
