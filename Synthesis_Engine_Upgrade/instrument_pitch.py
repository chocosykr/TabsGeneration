"""
instrument_pitch.py — Instrument-Aware Pitch Estimation (Upgrade Layer).

Extends pitch_estimation.py with instrument-profile-driven bounds and
HSS search range. The original pitch_estimation.py is NOT modified.

Key changes from pitch_estimation.py:
  - pYIN fmin/fmax come from InstrumentProfile.fmin_hz / fmax_hz.
  - HSS constrained search range comes from InstrumentProfile.midi_range.
  - poly_fmin_hz (sub-harmonic ghost rejection) comes from the profile.
  - Analysis window length comes from InstrumentProfile.analysis_window_sec.
  - Post-attack transient skip comes from InstrumentProfile.post_attack_skip_ms.
"""
import numpy as np

def freq_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)

import numpy as np
import librosa
import torch
from typing import List

from instrument_config import InstrumentProfile
# Re-use the shared FFT-utility from pitch_estimation (no circular import)
from pitch_estimation import _hz_to_bin, _suppress_harmonics, _extract_peak_energy_window


# ---------------------------------------------------------------------------
# Single-note pitch estimation (pYIN)
# ---------------------------------------------------------------------------

def estimate_initial_pitch(audio_tensor: torch.Tensor,
                           sample_rate: int,
                           profile: InstrumentProfile) -> float:
    """
    Estimate fundamental frequency using probabilistic YIN (pYIN), constrained
    to the pitch range of the given instrument profile.

    Uses the analysis window length and post-attack skip duration from the profile.
    """
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    y_np = audio_tensor.detach().cpu().numpy()

    # --- 1. Extract the peak-energy onset window (profile-aware duration) ---
    window_sec = profile.analysis_window_sec
    post_attack_skip_sec = profile.post_attack_skip_ms / 1000.0
    y_window = _extract_peak_energy_window_profiled(y_np, sample_rate,
                                                     window_sec, post_attack_skip_sec)

    # --- 2. Run pYIN with instrument-specific pitch bounds ---
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_window,
        fmin=profile.fmin_hz,
        fmax=profile.fmax_hz,
        sr=sample_rate,
    )

    # --- 3. Snap to modal MIDI note (probability-weighted) ---
    voiced_mask = voiced_flag & ~np.isnan(f0)
    valid_f0    = f0[voiced_mask]
    valid_probs = voiced_probs[voiced_mask]

    if len(valid_f0) == 0:
        print(f"  [{profile.display_name}] pYIN found no voiced frames; "
              "falling back to YIN on onset window.")
        f0_yin = librosa.yin(y_window, fmin=profile.fmin_hz,
                             fmax=profile.fmax_hz, sr=sample_rate)
        valid_f0_yin = f0_yin[~np.isnan(f0_yin)]
        if len(valid_f0_yin) == 0:
            print(f"  [{profile.display_name}] Warning: No pitch detected; "
                  "returning default 440.0 Hz.")
            return 440.0
        median_hz  = float(np.median(valid_f0_yin))
        modal_midi = int(round(69 + 12 * np.log2(median_hz / 440.0)))
        return float(440.0 * (2 ** ((modal_midi - 69) / 12.0)))

    midi_notes = np.round(69 + 12 * np.log2(valid_f0 / 440.0)).astype(int)
    note_weights: dict = {}
    for note, prob in zip(midi_notes, valid_probs):
        note_weights[note] = note_weights.get(note, 0.0) + float(prob)

    modal_midi = max(note_weights, key=note_weights.get)

    top = sorted(note_weights.items(), key=lambda x: -x[1])[:3]
    top_str = ", ".join(
        f"MIDI {n} ({440.0*(2**((n-69)/12.0)):.1f} Hz, w={w:.2f})"
        for n, w in top
    )
    print(f"   [{profile.display_name}] Top MIDI candidates: {top_str}")

    return float(440.0 * (2 ** ((modal_midi - 69) / 12.0)))


# ---------------------------------------------------------------------------
# Polyphonic pitch search (constrained HSS)
# ---------------------------------------------------------------------------

def _instrument_salience_search(mag: np.ndarray,
                                n_fft: int,
                                sr: int,
                                profile: InstrumentProfile,
                                num_harmonics: int = 5) -> int:
    """
    Finds the dominant fundamental frequency constrained to the MIDI range
    specified in the instrument profile.

    Returns the best MIDI note, or -1 if no confident match found.
    """
    lo_midi, hi_midi = profile.midi_range
    best_midi  = -1
    best_score = -1.0
    search_radius = 3

    for midi in range(lo_midi, hi_midi + 1):
        f0 = 440.0 * (2 ** ((midi - 69) / 12.0))
        # Reject pitches below the poly_fmin bound (sub-harmonic ghost rejection)
        if f0 < profile.poly_fmin_hz:
            continue
        score = 0.0
        for h in range(1, num_harmonics + 1):
            freq = h * f0
            if freq > sr / 2:
                break
            center_bin = _hz_to_bin(freq, n_fft, sr)
            lo_bin = max(0, center_bin - search_radius)
            hi_bin = min(len(mag) - 1, center_bin + search_radius)
            score += mag[lo_bin: hi_bin + 1].max()

        if score > best_score:
            best_score = score
            best_midi  = midi

    if best_score < 1e-9:
        return -1
    return best_midi


def estimate_multi_pitches(audio_tensor: torch.Tensor,
                           sample_rate: int,
                           profile: InstrumentProfile,
                           num_notes: int = 4) -> List[float]:
    """
    Estimate `num_notes` dominant fundamental frequencies from a polyphonic
    signal using an iterative constrained HSS search, bounded to the
    given instrument profile's playable range.
    """
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    y_np = audio_tensor.detach().cpu().numpy()

    y_window = _extract_peak_energy_window_profiled(
        y_np, sample_rate,
        profile.analysis_window_sec,
        profile.post_attack_skip_ms / 1000.0
    )

    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_window * np.hanning(len(y_window)), n=n_fft))

    estimated_pitches = []

    for i in range(num_notes):
        midi = _instrument_salience_search(mag, n_fft, sample_rate,
                                           profile, num_harmonics=5)

        if midi < 0:
            print(f"  [{profile.display_name}] Warning: could not detect "
                  f"pitch {i+1}; duplicating previous or falling back to 440 Hz.")
            f0 = estimated_pitches[-1] if estimated_pitches else 440.0
        else:
            f0 = 440.0 * (2 ** ((midi - 69) / 12.0))
            print(f"   [{profile.display_name}] Pitch {i+1} (HSS) → "
                  f"MIDI {midi} ({f0:.2f} Hz)")

        estimated_pitches.append(f0)
        mag = _suppress_harmonics(mag, f0, n_fft, sample_rate)

    return estimated_pitches


# ---------------------------------------------------------------------------
# Profile-aware onset window extraction
# ---------------------------------------------------------------------------

def _extract_peak_energy_window_profiled(y: np.ndarray,
                                          sr: int,
                                          window_sec: float,
                                          post_attack_skip_sec: float) -> np.ndarray:
    """
    Returns a slice of `y` starting just after the attack transient.
    Uses profile-specific window length and transient skip duration.
    """
    frame_len = int(sr * 0.02)   # 20ms frames
    hop_len   = frame_len // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    peak_frame  = int(np.argmax(rms))
    peak_sample = librosa.frames_to_samples(peak_frame, hop_length=hop_len)

    skip_samples = int(post_attack_skip_sec * sr)
    start = min(peak_sample + skip_samples, len(y) - 1)
    end   = min(len(y), start + int(window_sec * sr))

    if end - start < int(0.1 * sr):
        start = max(0, peak_sample)
        end   = min(len(y), start + int(window_sec * sr))

    return y[start:end]


# ---------------------------------------------------------------------------
# Instrument-aware chord shape search (spectral template scoring)
# ---------------------------------------------------------------------------

def estimate_chord_shape(audio_tensor: torch.Tensor,
                         sample_rate: int,
                         strummed_strings: List[int],
                         profile: InstrumentProfile,
                         max_fret: int = None,
                         num_harmonics: int = None,
                         top_n_multisets: int = 8,
                         max_fret_span: int = 4) -> dict:
    """
    Instrument-aware version of pitch_estimation.estimate_chord_shape.

    Uses profile.tuning instead of UKULELE_TUNING, and profile-specific
    num_harmonics, ergo_weight, unison_bonus, and overtone_factor for scoring.
    """
    if max_fret is None:
        max_fret = profile.max_fret
    if num_harmonics is None:
        num_harmonics = profile.num_harmonics

    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    y_np = audio_tensor.detach().cpu().numpy()

    y_window = _extract_peak_energy_window_profiled(
        y_np, sample_rate,
        profile.analysis_window_sec,
        profile.post_attack_skip_ms / 1000.0
    )

    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_window * np.hanning(len(y_window)), n=n_fft))

    open_freqs = [profile.tuning[s]["freq"] for s in strummed_strings]
    open_midis = [int(round(freq_to_midi(f))) for f in open_freqs]

    frets_range   = range(max_fret + 1)
    num_strings   = len(strummed_strings)
    search_radius = 3

    _UNISON_BONUS    = profile.unison_bonus
    _ERGO_WEIGHT     = profile.ergo_weight
    _OVERTONE_FACTOR = profile.overtone_factor
    _OVERTONE_TOL    = profile.overtone_ratio_tol

    # Pre-compute harmonic bin maps
    harm_bins = []
    for open_freq in open_freqs:
        per_fret = []
        for fret in frets_range:
            f0 = open_freq * (2 ** (fret / 12.0))
            bins = {}
            for h in range(1, num_harmonics + 1):
                freq = h * f0
                if freq > sample_rate / 2:
                    break
                center_bin = _hz_to_bin(freq, n_fft, sample_rate)
                lo_bin = max(0, center_bin - search_radius)
                hi_bin = min(len(mag) - 1, center_bin + search_radius)
                for b in range(lo_bin, hi_bin + 1):
                    prev = bins.get(b, 0)
                    if h > prev:
                        bins[b] = h
            per_fret.append(bins)
        harm_bins.append(per_fret)

    best_per_multiset = {}

    for frets in __import__('itertools').product(frets_range, repeat=num_strings):
        fretted = [f for f in frets if f > 0]
        if fretted and (max(fretted) - min(fretted)) > max_fret_span:
            continue

        candidate_freqs = [
            open_freqs[i] * (2 ** (fret / 12.0))
            for i, fret in enumerate(frets)
        ]

        overtone_discount = [1.0] * num_strings
        for i, fi in enumerate(candidate_freqs):
            for j, fj in enumerate(candidate_freqs):
                if i == j:
                    continue
                ratio = fi / fj if fj > 0 else 0.0
                nearest_int = round(ratio)
                if nearest_int >= 2:
                    if abs(ratio - nearest_int) / nearest_int < _OVERTONE_TOL:
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

        multiset_key = tuple(sorted(open_midis[i] + frets[i] for i in range(num_strings)))
        prev = best_per_multiset.get(multiset_key)
        if prev is None or score > prev[0]:
            best_per_multiset[multiset_key] = (score, frets)

    ranked_multisets = sorted(best_per_multiset.items(), key=lambda item: -item[1][0])
    top_multisets    = ranked_multisets[:top_n_multisets]
    best_spectral    = top_multisets[0][1][0] if top_multisets else 1.0

    candidates = []
    for multiset_key, (spectral_score, frets) in top_multisets:
        mapping = {}
        pitches = []
        for i, s in enumerate(strummed_strings):
            mapping[s] = frets[i]
            pitches.append(open_freqs[i] * (2 ** (frets[i] / 12.0)))

        candidates.append({
            "mapping":            mapping,
            "pitches":            pitches,
            "frets":              frets,
            "strummed_strings":   strummed_strings,
            "spectral_score":     spectral_score,
            "spectral_score_norm": spectral_score / best_spectral if best_spectral > 0 else 0.0,
            "multiset_midis":     multiset_key,
            "instrument":         profile.name,
        })

    # NNLS re-ranking (uses profile.num_harmonics)
    try:
        from instrument_spectral import score_candidate_nnls_profiled
        _NNLS_AVAILABLE = True
    except ImportError:
        _NNLS_AVAILABLE = False

    if _NNLS_AVAILABLE:
        _PRESENCE_THRESHOLD = profile.nnls_presence_threshold
        _MUTE_PENALTY       = profile.nnls_mute_penalty

        nnls_results = []
        for candidate in candidates:
            res = score_candidate_nnls_profiled(
                mag, candidate["pitches"], n_fft, sample_rate, profile
            )
            amps = res["amplitudes"]
            max_amp = max(amps) if amps else 1.0
            n_muted = sum(1 for a in amps if a < _PRESENCE_THRESHOLD * max_amp)
            nnls_score = res["score"] - n_muted * _MUTE_PENALTY

            candidate["nnls_score"]      = nnls_score
            candidate["nnls_score_raw"]  = res["score"]
            candidate["nnls_amplitudes"] = amps
            candidate["n_muted_voices"]  = n_muted
            nnls_results.append(nnls_score)

        best_nnls = max(nnls_results) if nnls_results else 1.0
        for candidate in candidates:
            candidate["spectral_score_norm"] = (
                candidate["nnls_score"] / best_nnls if best_nnls > 0 else 0.0
            )
        candidates.sort(key=lambda c: -c["nnls_score"])

    _NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    print(f"   [{profile.display_name}] Top {len(candidates)} distinct multisets:")
    for rank, c in enumerate(candidates, 1):
        notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in c["multiset_midis"]]
        nnls_str = (f"  nnls={c['nnls_score_raw']:.4f}  muted={c['n_muted_voices']}"
                    if "nnls_score_raw" in c else "")
        print(f"      #{rank}: frets={c['frets']}  notes=[{', '.join(notes)}]  "
              f"norm={c['spectral_score_norm']:.3f}{nnls_str}")

    return {"candidates": candidates, "best": candidates[0] if candidates else None}
