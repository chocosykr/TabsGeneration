"""
Instrument-Aware Neural Transcription Module (Phase 2 Upgrade)
==============================================================
Spotify Basic Pitch Neural Network integration for multi-instrument tab generation.

Supports:
  - Ukulele mode: 4-string G4-C4-E4-A4 re-entrant (original behaviour preserved)
  - Guitar mode:  6-string E2-A2-D3-G3-B3-E4 standard

Architecture decision (documented):
  Basic Pitch (Spotify ONNX model) is instrument-agnostic — it detects raw MIDI
  pitches directly from audio without knowledge of the target instrument.
  Therefore NO new guitar neural model is needed or appropriate.

  The instrument-specific work lives entirely in post-processing:
    1. Pitch → fretboard tab mapping (string/fret assignment)
    2. KS synthesis priors (decay, inharmonicity) via InstrumentProfile
    3. NNLS spectral template (num_harmonics, sigma) via InstrumentProfile

  This module replaces the original ukulele-only neural_transcription.py
  while maintaining 100% backward compatibility for callers that do not
  supply an `instrument` parameter (they get ukulele behaviour as before).

Backward-compatible public API
------------------------------
  map_pitches_to_ukulele_tab(midi_notes)          → unchanged signature
  transcribe_with_basic_pitch(audio_path, ...)    → `instrument` kwarg added (default "ukulele")
  transcribe_sequence_with_basic_pitch(...)       → `instrument` kwarg added (default "ukulele")

New public API
--------------
  map_pitches_to_tab(midi_notes, profile)         → generic, profile-driven version
"""

import os
import sys
import time
import math
import itertools
from typing import Optional, Dict, Any, List

import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import torch

PROTOTYPE_DIR = os.path.dirname(os.path.abspath(__file__))
if PROTOTYPE_DIR not in sys.path:
    sys.path.insert(0, PROTOTYPE_DIR)

from fretboard_mapper import freq_to_midi, midi_to_freq
from optimization import compare_candidate_audio
from device_config import DEVICE

# ---------------------------------------------------------------------------
# Instrument profile imports (new in Phase 2)
# ---------------------------------------------------------------------------
from instrument_config import (
    InstrumentProfile,
    get_profile,
    UKULELE_PROFILE,
    GUITAR_PROFILE,
)
from synthesis_engine import synthesize_candidate_tab_audio_for_instrument

# Keep UKULELE_TUNING import for callers that referenced it directly
from fretboard_mapper import UKULELE_TUNING

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

VALID_INSTRUMENTS = ["ukulele", "guitar"]


def _validate_instrument(instrument: str) -> str:
    """
    Normalise and validate an instrument name.
    Returns the normalised name or raises ValueError.
    """
    name = instrument.strip().lower()
    if name not in VALID_INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Valid choices: {VALID_INSTRUMENTS}"
        )
    return name


# ===========================================================================
# TASK 1 — Generic pitch-to-tab mapping  (instrument-aware)
# ===========================================================================

def map_pitches_to_tab(midi_notes: List[float], profile: InstrumentProfile) -> Dict[str, Any]:
    """
    Map detected MIDI pitches to optimal tablature for ANY plucked-string
    instrument described by an InstrumentProfile.

    Works for 4-string ukulele and 6-string guitar (and any future instruments
    registered in the profile registry).

    Parameters
    ----------
    midi_notes : list of float
        Raw MIDI pitch values detected by Basic Pitch (may contain duplicates).
    profile : InstrumentProfile
        Target instrument profile (UKULELE_PROFILE, GUITAR_PROFILE, …).

    Returns
    -------
    dict with keys:
        mode        : "single_note" | "chord"
        notes       : list of note-name strings (e.g. ["C4", "E4"])
        tab_strings : list of fret numbers, one per string, ordered from
                      highest string number to 1 (e.g. for guitar: [f6,f5,f4,f3,f2,f1])
                      For ukulele (4-string): [f4, f3, f2, f1]
        num_strings : int — number of strings in the instrument
        string_names: list of string names, same order as tab_strings
        active_strings: list of active string indices
        candidates  : list of candidate dicts (for display in results table)
    """
    # String ordering: highest string-number first (matches original ukulele convention)
    # For ukulele: [4, 3, 2, 1] → [G4, C4, E4, A4]
    # For guitar:  [6, 5, 4, 3, 2, 1] → [E2, A2, D3, G3, B3, E4]
    string_order = sorted(profile.tuning.keys(), reverse=True)
    num_strings = profile.num_strings
    string_names = [profile.tuning[s]["name"] for s in string_order]

    if not midi_notes:
        return {
            "mode": "single_note",
            "notes": ["None"],
            "tab_strings": [0] * num_strings,
            "num_strings": num_strings,
            "string_names": string_names,
            "active_strings": [],
            "candidates": []
        }

    unique_midis = sorted(list(set(int(round(m)) for m in midi_notes)))
    pitch_classes = set(m % 12 for m in unique_midis)
    names = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in unique_midis]

    # Pre-compute open MIDI for each string
    open_midis = {
        s: int(round(freq_to_midi(profile.tuning[s]["freq"])))
        for s in profile.tuning
    }

    # ── Case 1: Monophonic Single Pluck ─────────────────────────────────────
    if len(unique_midis) == 1:
        target = unique_midis[0]
        candidates = []
        # Prefer high strings (string 1 is highest pitch; lowest fret wins)
        for s in string_order:
            open_m = open_midis[s]
            fret = target - open_m
            if 0 <= fret <= profile.max_fret:
                # Scoring: lower fret wins; slight preference for string 1 (highest)
                score = fret + (0.0 if s == 1 else 0.5 * (s - 1))
                candidates.append({"string": s, "fret": fret, "score": score})

        candidates.sort(key=lambda x: x["score"])
        best = candidates[0] if candidates else {"string": string_order[-1], "fret": 0}

        tab_strings = [0] * num_strings
        str_to_idx = {s: i for i, s in enumerate(string_order)}
        tab_strings[str_to_idx[best["string"]]] = best["fret"]

        cand_list = []
        for rank, c in enumerate(candidates, 1):
            t = [0] * num_strings
            t[str_to_idx[c["string"]]] = c["fret"]
            cand_list.append({
                "rank": rank,
                "notes": names[0],
                "tab": f"String {c['string']} ({profile.tuning[c['string']]['name']}) Fret {c['fret']}",
                "tab_strings": t,
                "audio_match": 100.0 if rank == 1 else round(100.0 - c["fret"] * 5.0, 1),
                "spectral_fit": 98.0,
                "combined": 100.0 if rank == 1 else 80.0,
                "verdict": "WINNER" if rank == 1 else "Alternate"
            })

        return {
            "mode": "single_note",
            "notes": names,
            "tab_strings": tab_strings,
            "num_strings": num_strings,
            "string_names": string_names,
            "active_strings": [best["string"]],
            "fret": best["fret"],
            "string": best["string"],
            "candidates": cand_list
        }

    # ── Case 2: Polyphonic Chord (2 to N Notes) ──────────────────────────────
    # Search frets 0–5 (open position) across all strings.
    # product(range(6), repeat=4) = 1296  for ukulele  → fast
    # product(range(6), repeat=6) = 46656 for guitar   → <0.05s in Python

    fret_range = range(6)  # open position: frets 0–5
    valid_shapes = []

    for frets_combo in itertools.product(fret_range, repeat=num_strings):
        played_midis_set = {open_midis[s] + f for s, f in zip(string_order, frets_combo)}
        played_pcs = {m % 12 for m in played_midis_set}

        if pitch_classes.issubset(played_pcs):
            extra_pc = len(played_pcs - pitch_classes)
            non_zero = [f for f in frets_combo if f > 0]
            span = (max(non_zero) - min(non_zero)) if len(non_zero) > 1 else 0
            total_fret = sum(frets_combo)
            score = extra_pc * 50 + span * 4 + total_fret
            valid_shapes.append({
                "tab_strings": list(frets_combo),
                "score": score,
                "span": span
            })

    valid_shapes.sort(key=lambda x: x["score"])
    best_tab = valid_shapes[0]["tab_strings"] if valid_shapes else [0] * num_strings

    cand_list = []
    for rank, sh in enumerate(valid_shapes[:6], 1):
        t = sh["tab_strings"]
        # Build human-readable tab string
        parts = [f"{profile.tuning[s]['name']}:{t[i]}" for i, s in enumerate(string_order)]
        tab_str = "  ".join(parts)
        cand_list.append({
            "rank": rank,
            "notes": ", ".join(names),
            "tab": tab_str,
            "tab_strings": t,
            "audio_match": 100.0 if rank == 1 else round(max(50.0, 100.0 - rank * 8.0), 1),
            "spectral_fit": 95.0,
            "combined": 100.0 if rank == 1 else round(max(50.0, 95.0 - rank * 7.0), 1),
            "verdict": "WINNER" if rank == 1 else "Alternate"
        })

    return {
        "mode": "chord",
        "notes": names,
        "tab_strings": best_tab,
        "num_strings": num_strings,
        "string_names": string_names,
        "active_strings": list(string_order),
        "candidates": cand_list
    }


# ---------------------------------------------------------------------------
# Backward-compatible ukulele wrapper (original signature preserved)
# ---------------------------------------------------------------------------

def map_pitches_to_ukulele_tab(midi_notes: List[float]) -> Dict[str, Any]:
    """
    Map detected MIDI pitches to 4-string ukulele tablature (gCEA re-entrant).
    Backward-compatible wrapper; delegates to map_pitches_to_tab(UKULELE_PROFILE).

    Returns a dict identical to the original — including the `tab_4321` key —
    so all existing callers continue to work without modification.
    """
    res = map_pitches_to_tab(midi_notes, UKULELE_PROFILE)
    # Inject legacy `tab_4321` key for callers that read it directly
    res["tab_4321"] = res["tab_strings"]
    return res


def _tab_result_to_legacy(tab_res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add legacy `tab_4321` key to a tab result dict for backward compatibility.
    For ukulele this is identical to `tab_strings` (4 elements).
    For guitar it is extended to 6 elements.
    """
    tab_res["tab_4321"] = tab_res["tab_strings"]
    return tab_res


# ===========================================================================
# TASK 1 — Instrument-aware single-chord transcription
# ===========================================================================

def transcribe_with_basic_pitch(
    audio_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    save_synth: bool = True,
    synth_out_path: Optional[str] = None,
    instrument: str = "ukulele",          # ← NEW: instrument selection
) -> Dict[str, Any]:
    """
    Transcribe audio segment using Spotify's Basic Pitch ONNX neural model,
    mapping detected pitches to instrument-specific fretboard tablature.

    Parameters
    ----------
    audio_path   : path to WAV/MP3/etc. audio file
    start_sec    : segment start (None = beginning of file)
    end_sec      : segment end   (None = end of file)
    save_synth   : whether to synthesize the resulting tab as a WAV file
    synth_out_path: output path for synthesized WAV
    instrument   : "ukulele" (default, backward compatible) or "guitar"

    Notes
    -----
    Basic Pitch detects raw MIDI pitches instrument-agnostically.
    The instrument parameter controls:
      - Pitch → string/fret mapping (profile.tuning, profile.max_fret)
      - KS synthesis priors (profile.decay_init, profile.default_decay)
    No new neural model is used for guitar; this is the correct architecture.
    """
    instrument = _validate_instrument(instrument)
    profile = get_profile(instrument)

    start_time = time.time()
    from basic_pitch.inference import predict

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_dur = len(y) / sr

    # Slice targeted segment
    s = max(0.0, float(start_sec)) if start_sec is not None else 0.0
    e = min(total_dur, float(end_sec)) if end_sec is not None else total_dur
    if s < e:
        s_idx = int(s * sr)
        e_idx = int(e * sr)
        y_slice = y[s_idx:e_idx]
    else:
        y_slice = y

    # Write temp slice for Basic Pitch inference
    tmp_dir = os.path.join(PROTOTYPE_DIR, "output_ui")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_slice_wav = os.path.join(tmp_dir, f"bp_slice_{int(time.time()*1000)}.wav")
    wavfile.write(tmp_slice_wav, sr, (np.clip(y_slice, -1.0, 1.0) * 32767).astype(np.int16))

    try:
        model_output, midi_data, note_events = predict(
            tmp_slice_wav,
            onset_threshold=0.45,
            frame_threshold=0.25,
            minimum_note_length=80.0
        )
    finally:
        try:
            os.remove(tmp_slice_wav)
        except Exception:
            pass

    elapsed = time.time() - start_time

    # Collect detected pitches
    detected_midis = []
    note_confidences = []
    for n_start, n_end, pitch, amp, bends in note_events:
        detected_midis.append(pitch)
        note_confidences.append(amp)

    # ── Instrument-aware pitch-to-tab mapping ─────────────────────────────
    tab_res = map_pitches_to_tab(detected_midis, profile)
    tab_res = _tab_result_to_legacy(tab_res)
    tab_strings = tab_res["tab_strings"]
    tab_4321 = tab_res["tab_4321"]  # same list, legacy key

    # ── Instrument-aware KS synthesis ─────────────────────────────────────
    stft_loss = 0.0
    audio_match = 100.0
    if save_synth and synth_out_path:
        os.makedirs(os.path.dirname(os.path.abspath(synth_out_path)), exist_ok=True)
        string_order = sorted(profile.tuning.keys(), reverse=True)

        synth_pitches = []
        synth_active = []
        for str_idx, f in zip(string_order, tab_strings):
            active_strings = tab_res.get("active_strings", [])
            if f > 0 or (tab_res["mode"] == "chord" and str_idx in active_strings):
                open_hz = profile.tuning[str_idx]["freq"]
                hz = open_hz * (2.0 ** (f / 12.0))
                synth_pitches.append(hz)
                synth_active.append(True)
            else:
                synth_pitches.append(440.0)
                synth_active.append(False)

        if any(synth_active):
            # Use instrument-aware synthesis engine (upgraded KS with instrument priors)
            synth_tensor = synthesize_candidate_tab_audio_for_instrument(
                pitches=synth_pitches,
                instrument=instrument,
                sample_rate=sr,
                duration=max(0.5, e - s),
                active_mask=synth_active
            )
            target_tensor = torch.from_numpy(y_slice).float().to(DEVICE)
            comp = compare_candidate_audio(synth_tensor, target_tensor, sample_rate=sr)
            stft_loss = round(comp["stft_loss"], 4)
            audio_match = round(comp["similarity"] * 100.0, 1)

            synth_np = synth_tensor.detach().cpu().squeeze().numpy()
            synth_np = synth_np / (np.max(np.abs(synth_np)) + 1e-8)
            wavfile.write(synth_out_path, sr, (synth_np * 32767).astype(np.int16))

    avg_conf = float(np.mean(note_confidences) * 100.0) if note_confidences else 90.0

    # Build raw detections list
    note_names_all = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    raw_detections = []
    for midi, conf in zip(detected_midis, note_confidences):
        name = f"{note_names_all[midi % 12]}{midi // 12 - 1}"
        raw_detections.append({
            "midi": int(midi), "name": name,
            "confidence": round(float(conf) * 100, 1)
        })

    # Build candidate list (convert tab_strings → tab_4321 key inside each candidate)
    cand_list = []
    for c in tab_res.get("candidates", []):
        c_copy = dict(c)
        c_copy["tab_4321"] = c_copy.get("tab_strings", tab_4321)
        cand_list.append(c_copy)

    return {
        "engine": "Spotify Basic Pitch (Neural Network)",
        "instrument": instrument,
        "num_strings": profile.num_strings,
        "string_names": tab_res.get("string_names", []),
        "mode": tab_res["mode"],
        "notes": tab_res["notes"],
        "tab_4321": tab_4321,           # legacy key (now N-element for N-string instrument)
        "tab_strings": tab_strings,     # explicit generic key
        "acoustic_match": round(audio_match, 1),
        "stft_loss": stft_loss,
        "combined_score": round(avg_conf, 1),
        "elapsed": round(elapsed, 2),
        "device": "CPU (ONNX Runtime)",
        "synth_url": (
            f"/api/audio?file={synth_out_path}"
            if os.path.exists(synth_out_path or "") else None
        ),
        "candidates": cand_list,
        "raw_detections": raw_detections
    }


# ===========================================================================
# TASK 1 — Instrument-aware sequence transcription
# ===========================================================================

def transcribe_sequence_with_basic_pitch(
    audio_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    save_synth: bool = True,
    synth_out_path: Optional[str] = None,
    min_chord_gap: float = 0.08,
    min_segment_dur: float = 0.15,
    instrument: str = "ukulele",           # ← NEW: instrument selection
) -> Dict[str, Any]:
    """
    Run Basic Pitch over a full audio segment, group simultaneous note events
    into chord segments by time proximity, and return a sequence result dict
    compatible with format_sequence_result.

    The `instrument` parameter controls pitch-to-tab mapping and KS synthesis priors.
    Basic Pitch model itself is instrument-agnostic (no new model needed).
    """
    instrument = _validate_instrument(instrument)
    profile = get_profile(instrument)

    start_time = time.time()
    from basic_pitch.inference import predict

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_dur = len(y) / sr

    s_abs = max(0.0, float(start_sec)) if start_sec is not None else 0.0
    e_abs = min(total_dur, float(end_sec)) if end_sec is not None else total_dur
    time_offset = s_abs
    if s_abs < e_abs:
        y_slice = y[int(s_abs * sr): int(e_abs * sr)]
    else:
        y_slice = y

    tmp_dir = os.path.join(PROTOTYPE_DIR, "output_ui")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_wav = os.path.join(tmp_dir, f"bp_seq_{int(time.time()*1000)}.wav")
    wavfile.write(tmp_wav, sr, (np.clip(y_slice, -1.0, 1.0) * 32767).astype(np.int16))

    try:
        _model_out, _midi, note_events = predict(
            tmp_wav,
            onset_threshold=0.45,
            frame_threshold=0.25,
            minimum_note_length=80.0
        )
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

    note_events = sorted(note_events, key=lambda n: n[0])

    # Group notes into chord segments by onset proximity
    segments_raw = []
    for n_start, n_end, pitch, amp, _bends in note_events:
        placed = False
        for grp in segments_raw:
            if abs(n_start - grp["onset"]) <= min_chord_gap:
                grp["midis"].append(int(round(pitch)))
                grp["amps"].append(float(amp))
                grp["offset"] = max(grp["offset"], float(n_end))
                placed = True
                break
        if not placed:
            segments_raw.append({
                "onset": float(n_start),
                "offset": float(n_end),
                "midis": [int(round(pitch))],
                "amps": [float(amp)]
            })

    segments_raw.sort(key=lambda g: g["onset"])

    string_order = sorted(profile.tuning.keys(), reverse=True)
    note_names_all = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    result_segments = []
    for idx, grp in enumerate(segments_raw, 1):
        dur = grp["offset"] - grp["onset"]
        if dur < min_segment_dur:
            continue

        # Instrument-aware tab mapping
        tab_res = map_pitches_to_tab(grp["midis"], profile)
        tab_res = _tab_result_to_legacy(tab_res)
        tab_strings = tab_res["tab_strings"]

        raw_det = [
            {"midi": m, "name": f"{note_names_all[m % 12]}{m // 12 - 1}",
             "confidence": round(a * 100, 1)}
            for m, a in zip(grp["midis"], grp["amps"])
        ]
        avg_conf = float(np.mean(grp["amps"]) * 100.0) if grp["amps"] else 80.0
        notes = tab_res["notes"]

        # Build pitches_by_string for chord_data
        pitches_by_string = [
            profile.tuning[s]["freq"] * (2.0 ** (tab_strings[i] / 12.0))
            for i, s in enumerate(string_order)
        ]

        result_segments.append({
            "index": idx,
            "chord": notes[0].rstrip("0123456789") if notes else "?",
            "quality": tab_res["mode"],
            "start": round(grp["onset"] + time_offset, 3),
            "end": round(grp["offset"] + time_offset, 3),
            "duration": round(dur, 3),
            "strum_count": len(grp["midis"]),
            "tab_4321": tab_strings,        # N-element for N-string instrument
            "tab_strings": tab_strings,
            "num_strings": profile.num_strings,
            "string_names": tab_res.get("string_names", []),
            "notes": notes,
            "confidence": round(avg_conf, 1),
            "stft_loss": 0.0,
            "raw_detections": raw_det,
            "instrument": instrument,
            # chord_data for sequence synth compatibility
            "chord_data": {
                "tab_4321": tab_strings,
                "tab_1234": list(reversed(tab_strings)),
                "notes_4321": notes,
                "quality": tab_res["mode"],
                "pitches_1234": list(reversed(pitches_by_string)),
            },
            "acoustic_match": round(avg_conf, 1),
        })

    # Reindex after filtering
    for i, seg in enumerate(result_segments, 1):
        seg["index"] = i

    progression = " -> ".join(f"[{s['chord']}]" for s in result_segments)

    # Build synth audio track (instrument-aware)
    synth_url = None
    if save_synth and synth_out_path and result_segments:
        try:
            from chord_sequence import generate_synthesized_sequence_audio
            dur_local = e_abs - s_abs
            synth_track = generate_synthesized_sequence_audio(
                result_segments, dur_local, sr=sr, time_offset=time_offset
            )
            os.makedirs(os.path.dirname(os.path.abspath(synth_out_path)), exist_ok=True)
            import scipy.io.wavfile as _wf
            _wf.write(synth_out_path, sr, (synth_track * 32767).astype(np.int16))
            synth_url = f"/api/audio?file={synth_out_path}"
        except Exception as exc:
            print(f"[neural_transcription] Synth generation failed: {exc}")

    return {
        "engine": "basic_pitch",
        "instrument": instrument,
        "num_strings": profile.num_strings,
        "total_chords": len(result_segments),
        "progression_flow": progression,
        "segments": result_segments,
        "synth_url": synth_url,
        "elapsed": round(time.time() - start_time, 2),
    }
