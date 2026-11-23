"""
instrument_spectral.py — Instrument-Aware NNLS Spectral Scorer (Upgrade Layer).

Extends spectral_template.py with profile-driven harmonic template parameters.
The original spectral_template.py is NOT modified.

Key difference: _build_harmonic_template and score_candidate_nnls now read
num_harmonics and gaussian_sigma from the InstrumentProfile rather than
module-level constants.

This matters because:
  - Ukulele (nylon strings): 8 harmonics, sigma=1.5 — softer, faster harmonic roll-off
  - Guitar (steel strings): 12 harmonics, sigma=2.0 — richer upper partials, wider bins

Using the same template for both instruments would produce a systematically
biased spectral score.
"""

import numpy as np
from scipy.optimize import nnls
from typing import List

from instrument_config import InstrumentProfile
from pitch_estimation import _hz_to_bin


# ---------------------------------------------------------------------------
# Instrument-aware harmonic template builder
# ---------------------------------------------------------------------------

def _build_harmonic_template_profiled(f0_hz: float,
                                       n_fft: int,
                                       sr: int,
                                       n_bins: int,
                                       profile: InstrumentProfile) -> np.ndarray:
    """
    Build a single harmonic template for a note with fundamental f0_hz,
    using the num_harmonics and gaussian_sigma from the instrument profile.

    Returns a 1-D L2-normalised array of shape (n_bins,).

    Physical basis for profile-specific parameters:
      - Nylon-string ukulele: harmonics decay rapidly; 8 harmonics captures
        the perceptually important content without introducing noise-matching artefacts.
      - Steel-string guitar: harmonics sustain much longer; 12 harmonics captures
        the characteristic bright timbre and improves candidate discrimination.
      - Gaussian sigma controls bin-spread around each harmonic center.
        Guitar's lower fundamental frequencies have tighter harmonic spacing in
        FFT bins, requiring a slightly wider sigma (2.0 vs 1.5) to account for
        the same absolute Hz uncertainty at lower pitches.
    """
    num_harmonics = profile.num_harmonics
    sigma         = profile.gaussian_sigma

    template = np.zeros(n_bins, dtype=np.float32)
    for k in range(1, num_harmonics + 1):
        freq = k * f0_hz
        if freq > sr / 2:
            break
        center    = _hz_to_bin(freq, n_fft, sr)
        amplitude = 1.0 / k          # 1/k harmonic roll-off

        lo = max(0, int(center - 3 * sigma))
        hi = min(n_bins - 1, int(center + 3 * sigma))
        for b in range(lo, hi + 1):
            template[b] += amplitude * np.exp(-0.5 * ((b - center) / sigma) ** 2)

    norm = np.linalg.norm(template)
    if norm > 1e-10:
        template /= norm
    return template


# ---------------------------------------------------------------------------
# Instrument-aware NNLS candidate scorer
# ---------------------------------------------------------------------------

def score_candidate_nnls_profiled(observed_mag: np.ndarray,
                                   candidate_freqs: List[float],
                                   n_fft: int,
                                   sr: int,
                                   profile: InstrumentProfile,
                                   log_compress: bool = True) -> dict:
    """
    Score a candidate chord shape by fitting non-negative amplitudes to the
    observed spectrum using harmonic templates calibrated for the given instrument.

    Uses profile.num_harmonics and profile.gaussian_sigma for template construction,
    ensuring that the NNLS fit reflects the actual spectral character of the
    instrument rather than a generic template.

    Parameters
    ----------
    observed_mag    : magnitude spectrum (output of np.abs(np.fft.rfft(...)))
    candidate_freqs : fundamental frequencies (Hz) of the candidate voices
    n_fft           : FFT size used to produce observed_mag
    sr              : sample rate
    profile         : instrument profile (num_harmonics, gaussian_sigma)
    log_compress    : if True apply log(1 + mag) before fitting

    Returns
    -------
    dict with:
      'score'            : fraction of total spectral energy explained ∈ [0, 1]
      'amplitudes'       : fitted non-negative amplitude per voice
      'reconstruction'   : fitted spectrum H @ a
      'residual_energy'  : unexplained energy
      'total_energy'     : total observed energy
      'instrument'       : profile name (for traceability)
    """
    n_bins = len(observed_mag)

    if log_compress:
        S = np.log1p(observed_mag.astype(np.float32))
    else:
        S = observed_mag.astype(np.float32)

    total_energy = float(np.dot(S, S))
    if total_energy < 1e-12:
        return {
            'score':           0.0,
            'amplitudes':      [0.0] * len(candidate_freqs),
            'reconstruction':  np.zeros(n_bins, dtype=np.float32),
            'residual_energy': 0.0,
            'total_energy':    0.0,
            'instrument':      profile.name,
        }

    templates = []
    for f0 in candidate_freqs:
        if f0 <= 0:
            templates.append(np.zeros(n_bins, dtype=np.float32))
        else:
            templates.append(
                _build_harmonic_template_profiled(f0, n_fft, sr, n_bins, profile)
            )

    H = np.column_stack(templates)      # (n_bins, num_voices)
    a, _residual_norm = nnls(H, S)      # solve min ||S - H @ a||² s.t. a ≥ 0

    reconstruction   = H @ a
    residual         = S - reconstruction
    residual_energy  = float(np.dot(residual, residual))
    explained_energy = total_energy - residual_energy
    score            = max(0.0, explained_energy / total_energy)

    return {
        'score':           score,
        'amplitudes':      a.tolist(),
        'reconstruction':  reconstruction,
        'residual_energy': residual_energy,
        'total_energy':    total_energy,
        'instrument':      profile.name,
    }
