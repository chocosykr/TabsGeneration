import torch
import librosa
import numpy as np


# Ukulele frequency bounds: low G2 (~98 Hz) to high A5 (~880 Hz)
_FMIN = 98.0
_FMAX = 1050.0

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

