"""
spectral_template.py — Non-negative amplitude fitting for polyphonic candidates.

For every candidate chord shape (a set of fundamental frequencies), this module:

  1. Builds a harmonic template matrix H  [num_bins × num_voices]
     Each column h_i contains the predicted spectral envelope of voice i:
     a unit-amplitude pulse at f_i, f_i*2, f_i*3 … (with 1/k roll-off).

  2. Solves a non-negative least-squares (NNLS) problem:
        â = argmin_{a ≥ 0} || S_obs - H @ a ||²
     where S_obs is the observed magnitude spectrum.

  3. Returns a reconstruction score measuring how well the fitted mixture
     explains the observed spectrum:
        explained energy / total observed energy  ∈ [0, 1]

This replaces the previous "max-claim-per-bin" heuristic with a principled
spectral decomposition, and correctly models unisons/doublings as amplified
contributions to the same harmonic bins rather than discounting them.

Design decisions:
 - Use log-compressed magnitudes to prevent a loud fundamental from completely
   dominating the loss; this also naturally downweights noise floors.
 - Roll-off: h[k] = 1/k mimics the ideal plucked-string harmonic series.
 - NNLS solver: scipy.optimize.nnls (fast, exact, no regularization needed).
 - Candidate score = explained_energy_ratio, normalized across candidates.
"""

import numpy as np
from scipy.optimize import nnls
from pitch_estimation import _hz_to_bin

# Number of harmonics to include in each template column
_NUM_HARMONICS  = 8

# Log-compression constant to avoid log(0); applied as log(mag + EPS)
_LOG_EPS        = 1e-6

# Gaussian bin-spread (σ) in bins around each harmonic center — accounts for
# slight pitch imprecision and FFT smearing
_GAUSSIAN_SIGMA = 1.5


def _build_harmonic_template(
        f0_hz: float,
        n_fft: int,
        sr: int,
        n_bins: int,
        num_harmonics: int = _NUM_HARMONICS,
        sigma: float = _GAUSSIAN_SIGMA) -> np.ndarray:
    """
    Build a single harmonic template vector for a note with fundamental f0_hz.

    Returns a 1-D array of shape (n_bins,) representing the expected spectral
    envelope — a sum of Gaussian bumps at k*f0 with amplitude 1/k.
    """
    template = np.zeros(n_bins, dtype=np.float32)
    for k in range(1, num_harmonics + 1):
        freq = k * f0_hz
        if freq > sr / 2:
            break
        center = _hz_to_bin(freq, n_fft, sr)
        amplitude = 1.0 / k   # harmonic roll-off

        # Spread the energy with a Gaussian around the center bin
        lo = max(0, int(center - 3 * sigma))
        hi = min(n_bins - 1, int(center + 3 * sigma))
        for b in range(lo, hi + 1):
            template[b] += amplitude * np.exp(-0.5 * ((b - center) / sigma) ** 2)

    # L2-normalise so each voice template has unit norm — this means the NNLS
    # amplitude â_i represents how much of the observed spectrum is attributable
    # to voice i, in units of "template norms."
    norm = np.linalg.norm(template)
    if norm > 1e-10:
        template /= norm

    return template


def score_candidate_nnls(
        observed_mag: np.ndarray,
        candidate_freqs: list[float],
        n_fft: int,
        sr: int,
        num_harmonics: int = _NUM_HARMONICS,
        log_compress: bool = True) -> dict:
    """
    Score a candidate chord shape by fitting non-negative amplitudes to the
    observed spectrum and measuring the explained energy fraction.

    Parameters
    ----------
    observed_mag : 1-D array, shape (n_bins,)
        Magnitude spectrum of the observed audio (output of np.abs(np.fft.rfft(...))).
    candidate_freqs : list of float
        Fundamental frequencies (Hz) of the candidate voices.
    n_fft : int
        FFT size used to produce observed_mag.
    sr : int
        Sample rate of the audio.
    log_compress : bool
        If True, apply log(1 + mag) before fitting. This down-weights very large
        peaks relative to mid-level harmonics, improving sensitivity to quieter
        voices and making the fit less dominated by the loudest string.

    Returns
    -------
    dict with keys:
        'score'           : float in [0, 1] — fraction of total spectral energy explained
        'amplitudes'      : fitted a_i for each voice
        'reconstruction'  : fitted spectrum array (H @ a)
        'residual_energy' : unexplained energy
        'total_energy'    : total observed energy (in compressed domain)
    """
    n_bins = len(observed_mag)

    # Optionally log-compress the spectrum
    if log_compress:
        S = np.log1p(observed_mag.astype(np.float32))
    else:
        S = observed_mag.astype(np.float32)

    total_energy = float(np.dot(S, S))
    if total_energy < 1e-12:
        return {
            'score': 0.0,
            'amplitudes': [0.0] * len(candidate_freqs),
            'reconstruction': np.zeros(n_bins, dtype=np.float32),
            'residual_energy': 0.0,
            'total_energy': 0.0,
        }

    # Build template matrix H: shape (n_bins, num_voices)
    templates = []
    for f0 in candidate_freqs:
        if f0 <= 0:
            templates.append(np.zeros(n_bins, dtype=np.float32))
        else:
            templates.append(_build_harmonic_template(f0, n_fft, sr, n_bins, num_harmonics))

    H = np.column_stack(templates)  # (n_bins, num_voices)

    # Solve NNLS:  min ||S - H @ a||²  subject to a ≥ 0
    a, residual_norm = nnls(H, S)

    # Reconstruction
    reconstruction = H @ a
    residual = S - reconstruction
    residual_energy = float(np.dot(residual, residual))
    explained_energy = total_energy - residual_energy

    score = max(0.0, explained_energy / total_energy)

    return {
        'score': score,
        'amplitudes': a.tolist(),
        'reconstruction': reconstruction,
        'residual_energy': residual_energy,
        'total_energy': total_energy,
    }
