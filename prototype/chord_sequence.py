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
from fretboard_mapper import UKULELE_TUNING, GUITAR_TUNING, get_instrument_tuning, freq_to_midi, expected_inharmonicity
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

# Standard Guitar Chord Voicings (Strings 6-5-4-3-2-1: E2 - A2 - D3 - G3 - B3 - E4)
# -1 indicates muted or unplayed string
GUITAR_CHORD_DEFINITIONS = {
    # Major Chords
    "C":   ((-1, 3, 2, 0, 1, 0), "Major", "C"),
    "C#":  ((-1, 4, 3, 1, 2, 1), "Major", "C#"),
    "Db":  ((-1, 4, 3, 1, 2, 1), "Major", "C#"),
    "D":   ((-1, -1, 0, 2, 3, 2), "Major", "D"),
    "D#":  ((-1, -1, 1, 3, 4, 3), "Major", "D#"),
    "Eb":  ((-1, -1, 1, 3, 4, 3), "Major", "D#"),
    "E":   ((0, 2, 2, 1, 0, 0), "Major", "E"),
    "F":   ((1, 3, 3, 2, 1, 1), "Major", "F"),
    "F#":  ((2, 4, 4, 3, 2, 2), "Major", "F#"),
    "Gb":  ((2, 4, 4, 3, 2, 2), "Major", "F#"),
    "G":   ((3, 2, 0, 0, 0, 3), "Major", "G"),
    "G#":  ((4, 6, 6, 5, 4, 4), "Major", "G#"),
    "Ab":  ((4, 6, 6, 5, 4, 4), "Major", "G#"),
    "A":   ((-1, 0, 2, 2, 2, 0), "Major", "A"),
    "A#":  ((-1, 1, 3, 3, 3, 1), "Major", "A#"),
    "Bb":  ((-1, 1, 3, 3, 3, 1), "Major", "A#"),
    "B":   ((-1, 2, 4, 4, 4, 2), "Major", "B"),

    # Minor Chords
    "Cm":  ((-1, 3, 5, 5, 4, 3), "Minor", "C"),
    "C#m": ((-1, 4, 6, 6, 5, 4), "Minor", "C#"),
    "Dbm": ((-1, 4, 6, 6, 5, 4), "Minor", "C#"),
    "Dm":  ((-1, -1, 0, 2, 3, 1), "Minor", "D"),
    "D#m": ((-1, -1, 1, 3, 4, 2), "Minor", "D#"),
    "Ebm": ((-1, -1, 1, 3, 4, 2), "Minor", "D#"),
    "Em":  ((0, 2, 2, 0, 0, 0), "Minor", "E"),
    "Fm":  ((1, 3, 3, 1, 1, 1), "Minor", "F"),
    "F#m": ((2, 4, 4, 2, 2, 2), "Minor", "F#"),
    "Gbm": ((2, 4, 4, 2, 2, 2), "Minor", "F#"),
    "Gm":  ((3, 5, 5, 3, 3, 3), "Minor", "G"),
    "G#m": ((4, 6, 6, 4, 4, 4), "Minor", "G#"),
    "Abm": ((4, 6, 6, 4, 4, 4), "Minor", "G#"),
    "Am":  ((-1, 0, 2, 2, 1, 0), "Minor", "A"),
    "A#m": ((-1, 1, 3, 3, 2, 1), "Minor", "A#"),
    "Bbm": ((-1, 1, 3, 3, 2, 1), "Minor", "A#"),
    "Bm":  ((-1, 2, 4, 4, 3, 2), "Minor", "B"),

    # Dominant 7th Chords
    "C7":  ((-1, 3, 2, 3, 1, 0), "7th", "C"),
    "D7":  ((-1, -1, 0, 2, 1, 2), "7th", "D"),
    "E7":  ((0, 2, 0, 1, 0, 0), "7th", "E"),
    "F7":  ((1, 3, 1, 2, 1, 1), "7th", "F"),
    "G7":  ((3, 2, 0, 0, 0, 1), "7th", "G"),
    "A7":  ((-1, 0, 2, 0, 2, 0), "7th", "A"),
    "B7":  ((-1, 2, 1, 2, 0, 2), "7th", "B"),
    "Bb7": ((-1, 1, 3, 1, 3, 1), "7th", "A#"),

    # Major 7th & Minor 7th Chords
    "Cmaj7": ((-1, 3, 2, 0, 0, 0), "Maj7", "C"),
    "Fmaj7": ((-1, -1, 3, 2, 1, 0), "Maj7", "F"),
    "Gmaj7": ((3, 2, 0, 0, 0, 2), "Maj7", "G"),
    "Am7":   ((-1, 0, 2, 0, 1, 0), "m7", "A"),
    "Dm7":   ((-1, -1, 0, 2, 1, 1), "m7", "D"),
    "Em7":   ((0, 2, 0, 0, 0, 0), "m7", "E"),
}


def _build_chord_templates(instrument: str = "ukulele") -> Dict[str, dict]:
    is_guitar = str(instrument).lower() in ("guitar", "acoustic_guitar")
    definitions = GUITAR_CHORD_DEFINITIONS if is_guitar else UKULELE_CHORD_DEFINITIONS
    tuning = GUITAR_TUNING if is_guitar else UKULELE_TUNING
    string_order_desc = [6, 5, 4, 3, 2, 1] if is_guitar else [4, 3, 2, 1]
    string_order_asc  = [1, 2, 3, 4, 5, 6] if is_guitar else [1, 2, 3, 4]

    chord_lib = {}
    for name, (tab_tuple, quality, root) in definitions.items():
        pitches_desc = []
        pitches_asc = []
        notes_desc = []

        def _m2n(freq):
            m = int(round(freq_to_midi(freq)))
            return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

        for s_idx, fret in zip(string_order_desc, tab_tuple):
            if fret >= 0:
                f = tuning[s_idx]["freq"] * (2.0 ** (fret / 12.0))
                pitches_desc.append(f)
                notes_desc.append(_m2n(f))
            else:
                pitches_desc.append(0.0)  # muted
                notes_desc.append("x")

        # Ascending (String 1 upwards)
        rev_tab = list(reversed(tab_tuple))
        for s_idx, fret in zip(string_order_asc, rev_tab):
            if fret >= 0:
                pitches_asc.append(tuning[s_idx]["freq"] * (2.0 ** (fret / 12.0)))
            else:
                pitches_asc.append(0.0)

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
            "instrument": instrument,
            "tab_raw": tab_tuple,
            "tab_4321": list(tab_tuple) if not is_guitar else list(tab_tuple[2:]), # backward compat for 4 strings
            "tab_desc": list(tab_tuple),
            "pitches_desc": pitches_desc,
            "pitches_asc": pitches_asc,
            "notes_desc": notes_desc,
            "notes_4321": notes_desc if not is_guitar else notes_desc,
            "chroma": chroma_norm
        }
    return chord_lib

CHORD_LIB_UKE = _build_chord_templates("ukulele")
CHORD_LIB_GUITAR = _build_chord_templates("guitar")
CHORD_LIB = CHORD_LIB_UKE


def segment_and_identify_chords(y: np.ndarray,
                                sr: int = 16000,
                                min_duration: Union[float, str] = "auto",
                                energy_threshold: float = 0.005,
                                hop_length: int = 512,
                                split_strums: bool = False,
                                instrument: str = "ukulele") -> List[dict]:
    """
    Identifies a sequence of chords played in an audio recording using
    Adaptive Onset-to-Onset (O2O) Physical Segmentation.
    """
    fps = sr / hop_length
    total_duration = len(y) / sr
    is_guitar = str(instrument).lower() in ("guitar", "acoustic_guitar")
    chord_lib = CHORD_LIB_GUITAR if is_guitar else CHORD_LIB_UKE

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
    # For guitar, lower fmin on CQT down to E2 (~82 Hz), with n_octaves bounded by Nyquist
    fmin_cqt = 75.0 if is_guitar else librosa.note_to_hz('C3')
    n_octaves = min(7, int(np.floor(np.log2((sr / 2.0) / fmin_cqt))))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin_cqt, n_octaves=n_octaves)

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

        # Template correlation matching against chord dictionary
        best_name = None
        best_sim = -1.0
        for name, data in chord_lib.items():
            sim = float(np.dot(chroma_norm, data["chroma"]))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # Harmonic confidence gating (0.72 confidence floor)
        if best_sim >= 0.70:
            raw_segments.append({
                "chord": best_name,
                "start": t0,
                "end": t1,
                "duration": dur,
                "confidence": best_sim,
                "chord_data": chord_lib[best_name]
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
                             sr: int,
                             instrument: str = "ukulele") -> dict:
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

    is_guitar = str(instrument).lower() in ("guitar", "acoustic_guitar")
    string_order = [6, 5, 4, 3, 2, 1] if is_guitar else [4, 3, 2, 1]
    tab_raw = chord_data.get("tab_raw", chord_data.get("tab_4321", []))
    tuning = GUITAR_TUNING if is_guitar else UKULELE_TUNING

    pitches = []
    stretches = []
    for s_idx, f in zip(string_order, tab_raw):
        if f >= 0:
            pitches.append(tuning[s_idx]["freq"] * (2.0 ** (f / 12.0)))
            stretches.append(expected_inharmonicity(s_idx, f, instrument=instrument))

    if not pitches:
        seg["acoustic_match"] = seg["confidence"]
        seg["stft_loss"] = 0.0
        return seg

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
                                        sr: int = 16000,
                                        time_offset: float = 0.0,
                                        instrument: str = "ukulele") -> np.ndarray:
    total_samples = int(total_duration * sr) + int(0.5 * sr)
    full_audio = np.zeros(total_samples, dtype=np.float32)
    is_guitar = str(instrument).lower() in ("guitar", "acoustic_guitar")
    string_order = [6, 5, 4, 3, 2, 1] if is_guitar else [4, 3, 2, 1]
    tuning = GUITAR_TUNING if is_guitar else UKULELE_TUNING

    for seg in segments:
        chord_data = seg.get("chord_data", seg)
        start_sec_local = max(0.0, seg["start"] - time_offset)
        start_idx = int(start_sec_local * sr)
        dur = max(0.5, seg["duration"])

        tab_raw = chord_data.get("tab_raw", chord_data.get("tab_4321", seg.get("tab_raw", [])))
        pitches = []
        stretches = []
        for s_idx, f in zip(string_order, tab_raw):
            if f >= 0:
                pitches.append(tuning[s_idx]["freq"] * (2.0 ** (f / 12.0)))
                stretches.append(expected_inharmonicity(s_idx, f, instrument=instrument))

        if not pitches:
            continue

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
                           split_strums: bool = False,
                           instrument: str = "ukulele"):
    dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    mode_str = "Individual Strum Mode (--split-strums)" if split_strums else "Harmonic Progression Mode"
    is_guitar = str(instrument).lower() in ("guitar", "acoustic_guitar")
    inst_name = "GUITAR" if is_guitar else "UKULELE"

    print("\n" + "=" * 98)
    print(f"                    {inst_name} CHORD SEQUENCE IDENTIFICATION & TRANSCRIPTION")
    print("=" * 98)
    print(f"   Audio File:     {audio_path}")
    print(f"   Instrument:     {instrument.capitalize()}")
    print(f"   Duration:       {total_duration:.2f}s  |  Total Events: {len(segments)}  ({mode_str})")
    print(f"   Hardware:       {dev_desc}  |  Processing Time: {elapsed:.2f}s")
    print("=" * 98)

    if not segments:
        print("\n   [!] No sustained chords detected in the provided audio file.")
        print("=" * 98)
        return

    # 1. Progression Sequence Flow
    prog_str = "  ==>  ".join(f"[ {s['chord']} {s['chord_data']['quality']} ]" for s in segments)
    print(f"\n   CHORD PROGRESSION SEQUENCE:")
    print(f"   {prog_str}")
    print("\n" + "-" * 98)

    # 2. Timeline Table
    tab_hdr = "Tab (6 to 1)" if is_guitar else "Tab (4-3-2-1)"
    print("   CHORD SEQUENCE TIMELINE & TABLATURE:")
    print("=" * 98)
    print(f" {'#':>2}   {'Time Interval':<16}  {'Duration':>9}  {'Chord':<16}  {tab_hdr:<17}  {'Notes Played':<22}  {'Match %':>8}")
    print(f" {'-'*2}   {'-'*16}  {'-'*9}  {'-'*16}  {'-'*17}  {'-'*22}  {'-'*8}")

    for idx, s in enumerate(segments, 1):
        time_str = f"{s['start']:4.2f}s - {s['end']:4.2f}s"
        dur_str = f"{s['duration']:4.2f}s"
        strum_note = f" ({s['strum_count']} strums)" if not split_strums and s.get("strum_count", 1) > 1 else ""
        chord_label = f"{s['chord']} {s['chord_data']['quality']}{strum_note}"
        tab_raw = s["chord_data"].get("tab_raw", s["chord_data"].get("tab_4321", []))
        tab_str = " - ".join("x" if f < 0 else str(f) for f in tab_raw)
        notes_str = "[" + ", ".join(s["chord_data"].get("notes_desc", [])) + "]"
        match_pct = s.get("confidence", 0.0) * 100.0

        print(f" #{idx:<2}  {time_str:<16}  {dur_str:>9}  {chord_label:<16}  {tab_str:<17}  {notes_str:<22}  {match_pct:>7.1f}%")

    print("-" * 98)

    # 3. Multi-measure ASCII Tablature Staff
    print(f"\n   ASCII {inst_name} TABLATURE SCORE:")
    print("   " + "-" * 88)

    if is_guitar:
        staff_lines = {
            1: "   String 1 (E4) |",
            2: "   String 2 (B3) |",
            3: "   String 3 (G3) |",
            4: "   String 4 (D3) |",
            5: "   String 5 (A2) |",
            6: "   String 6 (E2) |"
        }
        chord_markers = "                 |"
        for s in segments:
            tab_raw = s["chord_data"].get("tab_raw", [0, 0, 0, 0, 0, 0])
            w = 12
            # Strings 6 to 1 in tab_raw
            for str_num in range(1, 7):
                f = tab_raw[6 - str_num]
                f_str = "x" if f < 0 else str(f)
                staff_lines[str_num] += f"--{f_str:<2}" + "-" * (w - 4) + "|"

            chord_str = f"[{s['chord']}]"
            chord_markers += f"  {chord_str:<9}|"

        for str_idx in [1, 2, 3, 4, 5, 6]:
            print(staff_lines[str_idx])
        print("   " + "-" * 88)
        print(chord_markers)
    else:
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
        print("   " + "-" * 88)
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
                                  end_sec: Optional[float] = None,
                                  instrument: str = "ukulele") -> List[dict]:
    start_time = time.time()
    print(f"\n[TabsGeneration] Loading audio from '{audio_path}' for {instrument} chord sequence analysis...")
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

    print(f"\n1. Performing Harmonic Segmentation & Multi-Chord Detection ({instrument.capitalize()})...")
    segments = segment_and_identify_chords(
        y, sr, 
        min_duration=min_duration, 
        energy_threshold=energy_threshold,
        split_strums=split_strums,
        instrument=instrument
    )

    print(f"   Found {len(segments)} distinct chord segments.")

    # Verify acoustics BEFORE shifting timestamps (y is still the local slice)
    print("\n2. Performing Acoustic Synthesis & Waveform Verification for each chord...")
    for idx, seg in enumerate(segments, 1):
        verify_segment_acoustics(seg, y, sr, instrument=instrument)

    # Shift timestamps by time_offset AFTER acoustic verification
    if time_offset > 0:
        for seg in segments:
            seg["start"] += time_offset
            seg["end"] += time_offset

    for idx, seg in enumerate(segments, 1):
        stft_info = f" (STFT Loss: {seg['stft_loss']:.4f})" if "stft_loss" in seg and seg['stft_loss'] > 0 else ""
        tab_raw = seg['chord_data'].get('tab_raw', seg['chord_data'].get('tab_4321', []))
        tab_disp = " - ".join("x" if f < 0 else str(f) for f in tab_raw)
        print(f"   Chord #{idx}: {seg['chord']:<4} [{seg['start']:4.2f}s - {seg['end']:4.2f}s]  ->  Tab: {tab_disp}  Match: {seg['confidence']*100:.1f}%{stft_info}")

    elapsed = time.time() - start_time
    print_sequence_results(audio_path, segments, dur, elapsed, split_strums=split_strums, instrument=instrument)

    # Export formatted tablature to text file (overwritten every run)
    export_sequence_tabs_to_file(
        file_path=tab_out_path,
        audio_path=audio_path,
        segments=segments,
        total_duration=dur,
        elapsed=elapsed,
        split_strums=split_strums,
        instrument=instrument
    )
    print(f"\n   [Export] Formatted tablature successfully written to '{tab_out_path}'")

    if save_synth and segments:
        print(f"\n3. Synthesizing full chord progression audio track to '{synth_out_path}'...")
        synth_track = generate_synthesized_sequence_audio(segments, dur, sr=sr, time_offset=time_offset, instrument=instrument)
        wavfile.write(synth_out_path, sr, (synth_track * 32767).astype(np.int16))
        print(f"   Successfully exported synthesized audio sequence to '{synth_out_path}'!")

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chord Sequence Identification & Tablature Transcription Pipeline")
    parser.add_argument("audio_file", type=str,
                        help="Path to the WAV file with a sequence of chords")
    parser.add_argument("--instrument", type=str, default="ukulele", choices=["ukulele", "guitar"],
                        help="Instrument profile (ukulele or guitar)")
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
        end_sec=args.end,
        instrument=args.instrument
    )
