import torch
import librosa
import numpy as np
from typing import List


# Ukulele frequency bounds: low G2 (~98 Hz) to high A5 (~880 Hz)
_FMIN = 98.0
_FMAX = 1050.0

# Lower bound for polyphonic pitch search.
# Re-entrant (high-G) ukulele's lowest playable note is C4 (open C string,
# 261.63 Hz), so searching below ~240 Hz only finds ghost sub-harmonics.
_POLY_FMIN = 240.0

# How long a window to analyse for pitch (seconds)
_ANALYSIS_WINDOW_SEC = 0.5


def _extract_peak_energy_window(y: np.ndarray, sr: int,
                                 window_sec: float = _ANALYSIS_WINDOW_SEC) -> np.ndarray:
    """
    Returns a slice of `y` starting just AFTER the attack transient — i.e.
    the stable sustain region where the pitch is cleanest.

    Strategy:
      1. Find the loudest frame (onset / attack peak).
      2. Skip a short post-attack hold (~60 ms) to let the inharmonic
         transient die away.
      3. Extract `window_sec` seconds of the resulting sustain region.
    """
    frame_len = int(sr * 0.02)   # 20ms frames
    hop_len   = frame_len // 2

    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]

    peak_frame  = int(np.argmax(rms))
    peak_sample = librosa.frames_to_samples(peak_frame, hop_length=hop_len)

    # Skip ~60 ms after the attack peak to land in the sustain region
    attack_skip_samples = int(0.06 * sr)
    start = min(peak_sample + attack_skip_samples, len(y) - 1)
    end   = min(len(y), start + int(window_sec * sr))

    # Safety: if the remaining audio is very short, fall back to the peak window
    if end - start < int(0.1 * sr):
        start = max(0, peak_sample)
        end   = min(len(y), start + int(window_sec * sr))

    return y[start:end]


def estimate_initial_pitch(audio_tensor: torch.Tensor, sample_rate: int) -> float:
    """
    Estimates the fundamental frequency (F0) using probabilistic YIN (pYIN)
    applied only to the peak-energy onset window of the recording.

    Improvements over the old approach:
      * pYIN is more robust than YIN on real (compressed) recordings.
      * Analysing only the loudest 0.5s avoids codec silence and fade-out
        frames that bias the median toward wrong values.
      * Voiced-probability weighted median gives higher weight to frames
        where the algorithm is confident.
    """
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()

    y_np = audio_tensor.detach().cpu().numpy()

    # --- 1. Extract the peak-energy onset window ---
    y_window = _extract_peak_energy_window(y_np, sample_rate)

    # --- 2. Run pYIN on the window ---
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_window,
        fmin=_FMIN,
        fmax=_FMAX,
        sr=sample_rate,
    )

    # --- 3. Snap to modal MIDI note (probability-weighted) ---
    # Music is always on a discrete semitone grid.
    # Instead of a continuous weighted median (which can land between notes),
    # we round each pYIN frame to the nearest integer MIDI note, accumulate
    # voiced-probability weights per note, then pick the mode.
    voiced_mask = voiced_flag & ~np.isnan(f0)
    valid_f0    = f0[voiced_mask]
    valid_probs = voiced_probs[voiced_mask]

    if len(valid_f0) == 0:
        # Fallback: plain YIN on the same window, snapped to nearest semitone
        print("  pYIN found no voiced frames; falling back to YIN on onset window.")
        f0_yin = librosa.yin(y_window, fmin=_FMIN, fmax=_FMAX, sr=sample_rate)
        valid_f0_yin = f0_yin[~np.isnan(f0_yin)]
        if len(valid_f0_yin) == 0:
            print("  Warning: No pitch detected at all, returning default 440.0 Hz.")
            return 440.0
        # Snap to nearest semitone
        median_hz   = float(np.median(valid_f0_yin))
        modal_midi  = int(round(69 + 12 * np.log2(median_hz / 440.0)))
        return float(440.0 * (2 ** ((modal_midi - 69) / 12.0)))

    # Convert continuous Hz → nearest integer MIDI note per frame
    midi_notes = np.round(69 + 12 * np.log2(valid_f0 / 440.0)).astype(int)

    # Accumulate voiced-probability weights per MIDI note → pick the mode
    note_weights: dict[int, float] = {}
    for note, prob in zip(midi_notes, valid_probs):
        note_weights[note] = note_weights.get(note, 0.0) + float(prob)

    modal_midi = max(note_weights, key=note_weights.get)

    # Log the top candidates so the user can verify
    top = sorted(note_weights.items(), key=lambda x: -x[1])[:3]
    top_str = ", ".join(
        f"MIDI {n} ({440.0*(2**((n-69)/12.0)):.1f} Hz, w={w:.2f})"
        for n, w in top
    )
    print(f"   Top MIDI candidates: {top_str}")

    # Convert modal MIDI back to exact Hz
    estimated_pitch = float(440.0 * (2 ** ((modal_midi - 69) / 12.0)))
    return estimated_pitch


# ---------------------------------------------------------------------------
# Two-pitch estimation
# ---------------------------------------------------------------------------

def _hz_to_bin(freq_hz: float, n_fft: int, sr: int) -> int:
    """Convert a frequency in Hz to the nearest FFT bin index."""
    return int(round(freq_hz * n_fft / sr))


def _suppress_harmonics(mag: np.ndarray, f0_hz: float, n_fft: int, sr: int,
                        num_harmonics: int = 12,
                        semitone_radius: float = 1.5) -> np.ndarray:
    """
    Zero out bins within `semitone_radius` semitones of every harmonic
    k * f0_hz (k = 1 … num_harmonics) in the magnitude spectrum `mag`.
    Returns a copy of `mag` with those bins zeroed.
    """
    mag_out = mag.copy()
    for k in range(1, num_harmonics + 1):
        center_hz = k * f0_hz
        if center_hz > sr / 2:
            break
        center_bin = _hz_to_bin(center_hz, n_fft, sr)
        # Width of the suppression window in bins
        ratio = 2 ** (semitone_radius / 12.0)
        low_hz  = center_hz / ratio
        high_hz = center_hz * ratio
        low_bin  = max(0, _hz_to_bin(low_hz,  n_fft, sr))
        high_bin = min(len(mag_out) - 1, _hz_to_bin(high_hz, n_fft, sr))
        mag_out[low_bin : high_bin + 1] = 0.0
    return mag_out


def _ukulele_salience_search(mag: np.ndarray, n_fft: int, sr: int,
                             num_harmonics: int = 5) -> int:
    """
    Finds the dominant fundamental frequency, explicitly constrained to the
    25 possible MIDI notes on a standard re-entrant ukulele (C4=60 to C6=84).
    
    Scores each valid MIDI note by taking the sum of the magnitudes of 
    its expected harmonics (Harmonic Sum Spectrum). This is more resilient
    than a product when searching for a second pitch where some harmonics 
    may have been zeroed out by the first pitch's suppression.
    """
    best_midi = -1
    best_score = -1.0
    
    # Valid ukulele range: C4 (open C string) to C6 (A string fret 15)
    for midi in range(60, 85):
        f0 = 440.0 * (2 ** ((midi - 69) / 12.0))
        score = 0.0
        
        for h in range(1, num_harmonics + 1):
            freq = h * f0
            if freq > sr / 2:
                break
                
            center_bin = _hz_to_bin(freq, n_fft, sr)
            search_radius = 3  # bins (approx ~6-10 Hz at 8192 NFFT)
            
            low_bin = max(0, center_bin - search_radius)
            high_bin = min(len(mag) - 1, center_bin + search_radius)
            
            peak_mag = mag[low_bin : high_bin + 1].max()
            score += peak_mag
            
        if score > best_score:
            best_score = score
            best_midi = midi
            
    if best_score < 1e-9:
        return -1
        
    return best_midi


def estimate_multi_pitches(audio_tensor: torch.Tensor,
                           sample_rate: int, 
                           num_notes: int = 4) -> List[float]:
    """
    Estimate `num_notes` dominant fundamental frequencies from a polyphonic
    signal using an iterative constrained ukulele HSS search.
    """
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    y_np = audio_tensor.detach().cpu().numpy()

    # Use the same onset window as the mono path
    y_window = _extract_peak_energy_window(y_np, sample_rate)

    # --- Magnitude spectrum (large FFT for fine frequency resolution) ---
    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_window * np.hanning(len(y_window)), n=n_fft))
    
    estimated_pitches = []

    for i in range(num_notes):
        # --- Find F0: Constrained HSS search on current spectrum ---
        midi = _ukulele_salience_search(mag, n_fft, sample_rate)
        
        if midi < 0:
            print(f"  Warning: could not detect pitch {i+1}; duplicating previous pitch or falling back to 440 Hz.")
            f0 = estimated_pitches[-1] if estimated_pitches else 440.0
        else:
            f0 = 440.0 * (2 ** ((midi - 69) / 12.0))
            print(f"   Pitch {i+1} (HSS) → MIDI {midi} ({f0:.2f} Hz)")

        estimated_pitches.append(f0)

        # --- Suppress harmonics of F0 in the spectrum for the next iteration ---
        mag = _suppress_harmonics(mag, f0, n_fft, sample_rate)

    return estimated_pitches


def estimate_chord_shape(audio_tensor: torch.Tensor,
                         sample_rate: int,
                         strummed_strings: List[int],
                         max_fret: int = 15,
                         num_harmonics: int = 5,
                         top_n_multisets: int = 8,
                         max_fret_span: int = 4) -> dict:
    """
    Search over all possible chord shapes (fret combinations) on the
    specified strummed strings and score each against the observed spectrum.

    Returns the best-scoring representative shape for each of the top
    `top_n_multisets` distinct note *multisets* (sorted tuples of MIDI values).
    Two shapes belong to the same multiset if they produce exactly the same
    notes in the same quantities (e.g. G4-E4-G4-B4), even if those notes
    are assigned to different strings.

    This is critical for unison handling: the multiset {E4, G4, G4, B4}
    is different from {E4, E4, G4, B4} — the former requires two G4 voices
    from the optimizer, the latter requires two E4 voices.

    `max_fret_span`: maximum allowed fret span across all fretted (non-open)
    strings. A human hand can comfortably span at most 4 frets. Open strings
    (fret 0) are excluded from the span calculation since they need no finger.
    Set to a very large value to disable this constraint.

    Returns a dict with:
      - 'candidates': list of candidate dicts, one per distinct multiset,
            each with 'mapping', 'pitches', 'frets', 'spectral_score',
            'strummed_strings', and 'multiset_midis'.
      - 'best': shortcut to candidates[0] (highest spectral score).
    """
    from fretboard_mapper import UKULELE_TUNING, freq_to_midi
    import itertools

    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    y_np = audio_tensor.detach().cpu().numpy()

    y_window = _extract_peak_energy_window(y_np, sample_rate)

    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_window * np.hanning(len(y_window)), n=n_fft))

    # Pre-compute open-string MIDI notes and frequencies
    open_freqs = [UKULELE_TUNING[s]["freq"] for s in strummed_strings]
    open_midis = [int(round(freq_to_midi(f))) for f in open_freqs]

    frets_range = range(max_fret + 1)
    num_strings = len(strummed_strings)
    search_radius = 3

    # Pre-compute, for each strummed string and each fret, a dict mapping each
    # covered FFT bin -> the HIGHEST harmonic index that claims it.
    #
    # Why highest, not lowest? If a bin is claimed by string A's fundamental
    # AND string B's 2nd harmonic (e.g. an octave: C5's fundamental lands on
    # C4's 2nd harmonic), crediting it as a fundamental would let an octave-
    # doubling shape steal the lower note's harmonic energy for free. Using
    # the highest (weakest) claim discounts shared bins, so a shape is only
    # rewarded for bins it explains as its own fundamentals/harmonics.
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
                low_bin = max(0, center_bin - search_radius)
                high_bin = min(len(mag) - 1, center_bin + search_radius)
                for b in range(low_bin, high_bin + 1):
                    prev = bins.get(b, 0)
                    if h > prev:
                        bins[b] = h
            per_fret.append(bins)
        harm_bins.append(per_fret)

    # Score each candidate shape by the TOTAL spectral energy it explains.
    #
    # Key rules:
    #   1. Each FFT bin is credited at most once per harmonic claim, using the
    #      LOWEST harmonic index that claims it (strongest physical claim).
    #   2. Unison doubling (two strings producing the same pitch) is REWARDED:
    #      when a second string claims a bin at the SAME harmonic index, a
    #      unison bonus is added. We keep this small (0.1) so the model doesn't
    #      sacrifice real quiet notes just to double a loud note.
    #   3. Ergonomic penalty: favours lower-fret shapes. This encodes the
    #      physical prior that a player will use the lowest viable fingering
    #      unless forced higher by the harmony. Tuned to overcome the raw
    #      spectral reward of hallucinating a high-fret string.
    #   4. Overtone discount: if a note in the candidate is a near-integer
    #      harmonic of another note already in the candidate, its fundamental
    #      spectral reward is multiplied by a discount factor. This is a
    #      general acoustic law — if string A is already producing frequency
    #      f, then 2f/3f/4f energy in the spectrum is already explained by A's
    #      harmonic series, so a separate string B claiming 2f as its
    #      fundamental should receive less credit.
    _UNISON_BONUS    = 0.1   # fraction of original credit added for each unison claim
    _ERGO_WEIGHT     = 0.5   # penalty per fret — strong enough to beat high-fret hallucinations
    _OVERTONE_FACTOR = 0.25  # fundamental credit multiplier when a note is an overtone
                             # of another note in the same candidate (0 = full suppression,
                             # 1 = no discount); 0.25 retains partial credit since a real
                             # string at 2f does add independent decay/energy even if
                             # the frequency is shared with another string's overtone.
    _OVERTONE_RATIO_TOLERANCE = 0.03  # fractional Hz tolerance for harmonic ratio check

    # Track the best shape per distinct multiset.
    # Key: sorted tuple of MIDI notes (the multiset).
    # Value: (spectral_score, frets_tuple).
    best_per_multiset: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {}

    for frets in itertools.product(frets_range, repeat=num_strings):
        # --- Fret-span constraint ---
        # A human hand can only comfortably fret notes within a ~4-fret window.
        # Open strings (fret 0) don't require a finger, so they're excluded.
        fretted = [f for f in frets if f > 0]
        if fretted and (max(fretted) - min(fretted)) > max_fret_span:
            continue


        # --- Compute the fundamental frequencies for this candidate ---
        candidate_freqs = [
            open_freqs[i] * (2 ** (fret / 12.0))
            for i, fret in enumerate(frets)
        ]

        # --- Overtone discount map ---
        # For each string, check if its fundamental is a near-integer multiple
        # of any OTHER string's fundamental in this candidate. If so, flag it
        # for a discount — its energy is already partially explained.
        overtone_discount = [1.0] * num_strings
        for i, fi in enumerate(candidate_freqs):
            for j, fj in enumerate(candidate_freqs):
                if i == j:
                    continue
                # Is fi an integer harmonic of fj? i.e. fi ≈ n * fj for n=2,3,4...
                ratio = fi / fj if fj > 0 else 0.0
                nearest_int = round(ratio)
                if nearest_int >= 2:
                    if abs(ratio - nearest_int) / nearest_int < _OVERTONE_RATIO_TOLERANCE:
                        # fi is approximately the nearest_int-th harmonic of fj.
                        # Apply discount to string i's fundamental contribution.
                        overtone_discount[i] = min(overtone_discount[i], _OVERTONE_FACTOR)
                        break

        covered = {}   # bin -> (harmonic_index, claim_count)
        score = 0.0

        for string_i, fret in enumerate(frets):
            discount = overtone_discount[string_i]
            for b, h in harm_bins[string_i][fret].items():
                # The fundamental bin (h==1) gets the overtone discount applied.
                # Harmonic bins (h>1) are not discounted — they are independent
                # acoustic evidence regardless of whether another string shares
                # the fundamental frequency.
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
                        # Upgrade the claim to a lower harmonic index (stronger)
                        score -= mag[b] / prev_h
                        score += effective_weight * mag[b] / h
                        covered[b] = (h, 1)

        score -= _ERGO_WEIGHT * sum(frets)

        # Compute the multiset key: sorted tuple of MIDI notes
        multiset_key = tuple(sorted(
            open_midis[i] + frets[i] for i in range(num_strings)
        ))

        prev = best_per_multiset.get(multiset_key)
        if prev is None or score > prev[0]:
            best_per_multiset[multiset_key] = (score, frets)

    # Sort multisets by their best spectral score, descending
    ranked_multisets = sorted(
        best_per_multiset.items(),
        key=lambda item: -item[1][0]
    )

    # Take the top N distinct multisets
    top_multisets = ranked_multisets[:top_n_multisets]

    # Normalize spectral scores to [0, 1] relative to the best
    best_spectral = top_multisets[0][1][0] if top_multisets else 1.0

    # Build candidate list
    candidates = []
    for multiset_key, (spectral_score, frets) in top_multisets:
        mapping = {}
        pitches = []
        for i, s in enumerate(strummed_strings):
            mapping[s] = frets[i]
            pitches.append(open_freqs[i] * (2 ** (frets[i] / 12.0)))

        candidates.append({
            "mapping": mapping,
            "pitches": pitches,
            "frets": frets,
            "strummed_strings": strummed_strings,
            "spectral_score": spectral_score,
            "spectral_score_norm": spectral_score / best_spectral if best_spectral > 0 else 0.0,
            "multiset_midis": multiset_key,
        })

    # ── NNLS re-ranking ─────────────────────────────────────────────────────
    # Replace the heuristic spectral score with a principled non-negative
    # least-squares amplitude fit.  For each candidate we build harmonic
    # templates and solve for the best non-negative amplitudes that explain
    # the observed spectrum.
    #
    # Voice-presence penalty (Experiment 3):
    # A candidate should not be allowed to explain a 4-string chord by
    # silently setting one voice's amplitude to ≈0.  If NNLS returns
    # amplitude < PRESENCE_THRESHOLD for a voice, we subtract a penalty
    # per silenced voice.  This prevents "candidate cheating" by voice
    # deletion.
    #
    # Note: we do NOT penalise by energy magnitude (plucking strength varies)
    # but by the *relative* amplitude within the candidate — if NNLS needs
    # a voice's amplitude to be essentially 0 to fit the recording, that
    # voice's fundamental frequency is not present in the audio.
    try:
        from spectral_template import score_candidate_nnls
        _NNLS_AVAILABLE = True
    except ImportError:
        _NNLS_AVAILABLE = False

    if _NNLS_AVAILABLE:
        _PRESENCE_THRESHOLD = 0.05  # fraction of max amplitude below which a voice is "absent"
        _MUTE_PENALTY       = 0.03  # subtracted per silenced voice from the NNLS score

        nnls_results = []
        for candidate in candidates:
            res = score_candidate_nnls(mag, candidate["pitches"], n_fft, sample_rate)
            amps = res["amplitudes"]

            # Voice presence: check relative amplitudes
            max_amp = max(amps) if amps else 1.0
            n_muted = sum(1 for a in amps if a < _PRESENCE_THRESHOLD * max_amp)

            nnls_score = res["score"] - n_muted * _MUTE_PENALTY

            candidate["nnls_score"]      = nnls_score
            candidate["nnls_score_raw"]  = res["score"]
            candidate["nnls_amplitudes"] = amps
            candidate["n_muted_voices"]  = n_muted
            nnls_results.append(nnls_score)

        # Re-normalise NNLS scores to [0, 1]
        best_nnls = max(nnls_results) if nnls_results else 1.0
        for candidate in candidates:
            candidate["spectral_score_norm"] = (
                candidate["nnls_score"] / best_nnls if best_nnls > 0 else 0.0
            )

        # Re-sort by NNLS score (descending)
        candidates.sort(key=lambda c: -c["nnls_score"])

    # Log the top candidates
    print(f"   Top {len(candidates)} distinct multisets (of "
          f"{len(best_per_multiset)} total):")
    _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                   'F#', 'G', 'G#', 'A', 'A#', 'B']
    for rank, c in enumerate(candidates, 1):
        notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in c["multiset_midis"]]
        nnls_str = (f"  nnls={c['nnls_score_raw']:.4f}  muted={c['n_muted_voices']}"
                    if "nnls_score_raw" in c else "")
        print(f"      #{rank}: frets={c['frets']}  notes=[{', '.join(notes)}]  "
              f"spectral={c['spectral_score']:.1f}  "
              f"norm={c['spectral_score_norm']:.3f}{nnls_str}")

    return {
        "candidates": candidates,
        "best": candidates[0] if candidates else None,
    }
