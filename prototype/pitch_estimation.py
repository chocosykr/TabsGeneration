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
                         num_harmonics: int = 5) -> dict:
    """
    Directly search over all possible chord shapes (fret combinations) on the
    specified strummed strings and score each against the observed spectrum.

    This avoids iterative harmonic suppression entirely. Instead, for each
    candidate chord shape we compute the total spectral energy it explains —
    the sum of FFT magnitudes across the union of every expected harmonic
    bin of every string's note (each bin counted once).

    For 4 strings × 16 frets this is 65,536 candidates, which runs in ~1 s.

    Returns a dict with:
      - 'mapping': {string_idx: fret, ...}
      - 'pitches': [Hz, ...] for each strummed string
      - 'score': total harmonic salience of the winning shape
    """
    from fretboard_mapper import UKULELE_TUNING
    import itertools

    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    y_np = audio_tensor.detach().cpu().numpy()

    y_window = _extract_peak_energy_window(y_np, sample_rate)

    n_fft = 8192
    mag = np.abs(np.fft.rfft(y_window * np.hanning(len(y_window)), n=n_fft))

    # Pre-compute open-string frequencies for the strummed strings
    open_freqs = []
    for s in strummed_strings:
        open_freqs.append(UKULELE_TUNING[s]["freq"])

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
                    # A bin claimed by a higher harmonic stays a harmonic
                    # (weaker claim); only upgrade it if an even higher
                    # harmonic index claims it.
                    prev = bins.get(b, 0)
                    if h > prev:
                        bins[b] = h
            per_fret.append(bins)
        harm_bins.append(per_fret)

    # Score each candidate shape by the TOTAL spectral energy it explains,
    # counting each FFT bin only ONCE (union across all strings/harmonics)
    # and discounting bins shared with higher harmonics (1 / harmonic index).
    #
    # Union dedup: without it, shared harmonics get double-counted and a wrong
    # shape can win — e.g. in C-major, G4's 2nd harmonic (784 Hz) lands on C4's
    # 3rd harmonic (784.9 Hz), so a candidate G4 picks up C4's energy for free.
    #
    # Harmonic-index discount: with it, an octave-doubling shape (e.g. C5
    # instead of a unison A4) can no longer steal the lower note's upper
    # harmonics, because every bin it would claim is already claimed by the
    # lower note's harmonic series.
    best_score = -1.0
    best_frets = None

    for frets in itertools.product(frets_range, repeat=num_strings):
        covered = {}   # bin -> highest harmonic index claiming it
        score = 0.0

        for string_i, fret in enumerate(frets):
            for b, h in harm_bins[string_i][fret].items():
                if b not in covered:
                    covered[b] = h
                    score += mag[b] / h
                elif h > covered[b]:
                    # A higher harmonic claims this bin too; the discount
                    # gets weaker, so adjust the score accordingly.
                    score -= mag[b] / covered[b]
                    covered[b] = h
                    score += mag[b] / h

        if score > best_score:
            best_score = score
            best_frets = frets

    # Build result
    mapping = {}
    pitches = []
    for i, s in enumerate(strummed_strings):
        mapping[s] = best_frets[i]
        pitches.append(open_freqs[i] * (2 ** (best_frets[i] / 12.0)))

    # Also find a few runner-up shapes for logging
    print(f"   Best chord shape score: {best_score:.2f}")
    for i, s in enumerate(strummed_strings):
        f = best_frets[i]
        hz = pitches[i]
        fret_str = str(f) if f > 0 else "0 (open)"
        print(f"      String {s}: Fret {fret_str}  ({hz:.2f} Hz)")

    return {
        "mapping": mapping,
        "pitches": pitches,
        "strummed_strings": strummed_strings,
        "score": best_score
    }
