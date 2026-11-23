"""
instrument_config.py — Instrument Profile Registry for the Synthesis Engine Upgrade.

Defines physically calibrated instrument profiles for the dual-mode synthesis engine.
Each profile encapsulates all instrument-specific parameters required by the
Karplus-Strong synthesis model, the pitch estimation module, the NNLS spectral
scorer, and the fretboard mapper.

Design goals:
  - Zero hardcoded assumptions inside the synthesis / scoring logic.
  - New instruments can be added by inserting a new InstrumentProfile dataclass
    instance into the registry — no changes to core DSP code required.
  - Mode switching is clean: loading a profile replaces ALL instrument-specific
    state without contaminating other modules.

Instrument Profile parameters are grounded in acoustics:

  Karplus-Strong string model (karplus_strong.py):
    decay_init       : initial logit-decay value → sigmoid → physical decay ∈ (0.5, 0.9999)
    stretch_init     : initial logit-stretch value → sigmoid → all-pass coefficient ∈ [0, 0.1)
    decay_penalty_th : threshold below which a decay is penalised in multi-voice optimisation

  Harmonic template (spectral_template.py):
    num_harmonics    : how many overtones to include in NNLS template columns
    gaussian_sigma   : Gaussian bin-spread around each harmonic center

  Pitch search (pitch_estimation.py):
    fmin_hz          : lower bound for pYIN / HSS pitch search
    fmax_hz          : upper bound for pYIN / HSS pitch search
    poly_fmin_hz     : lower bound for polyphonic HSS search (above ghost sub-harmonics)
    midi_range       : (low_midi, high_midi) for the constrained HSS search

  Fretboard mapper (fretboard_mapper.py):
    tuning           : {string_idx: {"name": str, "freq": float}} open-string frequencies
    num_strings      : number of strings on the instrument
    max_fret         : highest playable fret (for candidate generation)
    stretch_open     : {string_idx: float} empirical open-string inharmonicity coefficients
    stretch_ratio    : per-fret growth ratio for the inharmonicity model
    stretch_max      : sigmoid ceiling for the stretch parameter

  Ergonomic / scoring weights:
    ergo_weight      : fret-height penalty in the spectral scorer (larger → prefers lower frets)
    unison_bonus     : fractional credit for unison doublings in the spectral scorer
    overtone_factor  : fundamental credit multiplier when a note is a harmonic of another voice

  Synthesis (optimization.py / synthesize_candidate_tab_audio):
    default_decay    : default physical decay used when no prior is available
    default_energy   : default log-energy initialisation
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import math


# ---------------------------------------------------------------------------
# InstrumentProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class InstrumentProfile:
    """
    Complete, self-contained specification of a plucked-string instrument.

    All parameters are physically grounded; none are arbitrary labels.
    The synthesis engine reads ONLY from this profile — no instrument-specific
    logic lives elsewhere in the upgraded codebase.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    name: str                        # e.g. "ukulele", "guitar"
    display_name: str                # e.g. "Ukulele (G-C-E-A)", "Guitar (E-A-D-G-B-E)"
    num_strings: int

    # ── Tuning  ───────────────────────────────────────────────────────────────
    # {string_idx: {"name": str, "freq": float}}
    # String indices are 1-based, highest pitch = 1 (ukulele convention)
    # or lowest pitch = 1 (guitar convention, as commonly labelled 1–6).
    tuning: Dict[int, Dict]

    # ── Fretboard ─────────────────────────────────────────────────────────────
    max_fret: int                    # highest playable fret

    # ── Inharmonicity model: stretch(fret) = stretch_open[s] * stretch_ratio^fret ──
    stretch_open: Dict[int, float]   # per-string open-string all-pass coefficient
    stretch_ratio: float             # per-fret growth ratio
    stretch_max: float               # sigmoid ceiling (must match KS model ceiling)

    # ── Karplus-Strong initial parameter values ────────────────────────────────
    # decay_init is logit of ((d - 0.5) / 0.4999) where d is the target initial decay.
    # Stored as a float rather than the raw decay so it can be fed directly to
    # voice.logit_decay.data.fill_().
    decay_init: float                # initial logit_decay value
    stretch_init: float              # initial logit_stretch value  (-5.0 → near-zero stretch)
    decay_penalty_threshold: float   # relu penalty activates below this physical decay

    # ── Synthesis defaults (used in synthesize_candidate_tab_audio) ────────────
    default_decay: float             # physical decay ∈ (0.5, 0.9999)
    default_energy: float            # log_energy initialisation

    # ── Pitch estimation bounds ────────────────────────────────────────────────
    fmin_hz: float
    fmax_hz: float
    poly_fmin_hz: float              # lower bound for polyphonic pitch search
    midi_range: Tuple[int, int]      # (lo_midi, hi_midi) for constrained HSS search

    # ── NNLS spectral template ────────────────────────────────────────────────
    num_harmonics: int
    gaussian_sigma: float

    # ── Scoring weights ───────────────────────────────────────────────────────
    ergo_weight: float               # fret-height penalty per fret in spectral search
    unison_bonus: float              # bonus for a unison doubling in HSS
    overtone_factor: float           # fundamental discount when note is harmonic of another
    overtone_ratio_tol: float        # fractional Hz tolerance for the harmonic-ratio check

    # ── Analysis window ───────────────────────────────────────────────────────
    analysis_window_sec: float       # sustain window extracted for pitch estimation
    post_attack_skip_ms: float       # ms to skip after the loudest frame (transient skip)

    # ── NNLS voice presence penalty ───────────────────────────────────────────
    nnls_presence_threshold: float   # relative amplitude below which a voice is "absent"
    nnls_mute_penalty: float         # subtracted per silenced voice from NNLS score


# ---------------------------------------------------------------------------
# Utility: convert a physical decay to a logit for initialisation
# ---------------------------------------------------------------------------

def _decay_to_logit(decay: float) -> float:
    """
    Convert a physical decay value ∈ (0.5, 0.9999) to the corresponding logit
    used by DifferentiableKarplusStrong.logit_decay.

    decay = sigmoid(logit) * 0.4999 + 0.5
    → sig = (decay - 0.5) / 0.4999
    → logit = log(sig / (1 - sig))
    """
    decay = max(0.5001, min(0.9998, decay))
    sig = (decay - 0.5) / 0.4999
    sig = max(1e-7, min(1 - 1e-7, sig))
    return math.log(sig / (1.0 - sig))


def _stretch_to_logit(stretch: float) -> float:
    """
    Convert a physical stretch coefficient ∈ [0, 0.1) to the corresponding logit
    used by DifferentiableKarplusStrong.logit_stretch.

    stretch = sigmoid(logit) * 0.1
    → sig = stretch / 0.1
    → logit = log(sig / (1 - sig))
    """
    stretch = max(1e-6, min(0.1 - 1e-6, stretch))
    sig = stretch / 0.1
    sig = max(1e-7, min(1 - 1e-7, sig))
    return math.log(sig / (1.0 - sig))


# ===========================================================================
# UKULELE PROFILE  (Standard Re-entrant High-G Tuning: G4-C4-E4-A4)
# ===========================================================================
#
# Physical basis for key parameters:
#
# Tuning: Standard soprano/concert/tenor ukulele re-entrant tuning.
#   String 4 (G4, 392.00 Hz) — high G (re-entrant)
#   String 3 (C4, 261.63 Hz) — lowest pitch string
#   String 2 (E4, 329.63 Hz)
#   String 1 (A4, 440.00 Hz) — highest string, closest to the nut
#
# Inharmonicity:
#   Open-string stretch ~0.001–0.004 (empirically measured from recordings)
#   Fret 3 stretch ~0.095–0.100 → growth ratio = (0.095/0.002)^(1/3) ≈ 3.60–4.56
#   We use 4.56 from previous calibration so even the lowest open (0.001)
#   saturates near 0.095 by fret 3.
#
# Decay:
#   Ukulele strings decay faster than guitar (shorter scale length, nylon strings).
#   Empirical observation: clean plucks sit around 0.985–0.995 decay per sample
#   at 16 kHz. We initialise at 0.99 and let the optimizer fine-tune.
#
# Harmonics:
#   Nylon strings have a softer attack and roll off harmonics faster than steel.
#   8 harmonics captures the perceptually important content without overfitting
#   noise in the NNLS spectral scorer.
#
# Pitch range:
#   Lowest playable note: C4 (open string 3, MIDI 60, 261.63 Hz)
#   Highest playable note: G#5/Ab5 on fret 15 of string 1 (A4), or
#     equivalently C6 on string 1 fret 15 → depends on convention.
#     Conservative upper bound: A5 (880 Hz / MIDI 81) covers all practical positions.
#   For pYIN: fmin=98 Hz (G2, below any real ukulele note — safety margin),
#              fmax=1050 Hz (above highest practical note with margin).
#   For HSS constrained search: MIDI 60 (C4) to MIDI 84 (C6).

UKULELE_PROFILE = InstrumentProfile(
    name="ukulele",
    display_name="Ukulele (G4-C4-E4-A4 Re-entrant)",
    num_strings=4,

    tuning={
        4: {"name": "G4", "freq": 392.00},
        3: {"name": "C4", "freq": 261.63},
        2: {"name": "E4", "freq": 329.63},
        1: {"name": "A4", "freq": 440.00},
    },

    max_fret=15,

    # Inharmonicity model (empirically calibrated from recorded ukulele samples)
    stretch_open={
        4: 0.002,   # G4: observed 0.001–0.004
        3: 0.001,   # C4: observed 0.000–0.002
        2: 0.001,   # E4: observed 0.000–0.002
        1: 0.002,   # A4: observed 0.001–0.004
    },
    stretch_ratio=4.56,     # calibrated: (0.095/0.001)^(1/3) ≈ 4.56
                            # String 3 (s0=0.001): 0.001 * 4.56^3 = 0.095 → at saturation limit at fret 3.
                            # String 4 (s0=0.002): 0.002 * 4.56^3 = 0.190 → clamped to 0.1 by fret 2.
                            # This matches empirical observation: fret 2-3 notes are already near saturation.
                            # The model is correct: stretch grows with fret but saturates for wound strings.
    stretch_max=0.1,

    # KS initial values
    # Ukulele: shorter scale, nylon strings → faster decay than guitar
    # Target initial physical decay: 0.990
    decay_init=_decay_to_logit(0.990),
    stretch_init=-5.0,      # near-zero initial stretch (all-pass coefficient ≈ 0)
    decay_penalty_threshold=0.95,

    # Synthesis defaults for deterministic candidate synthesis
    # Ukulele: moderate sustain, medium energy
    default_decay=0.990,
    default_energy=0.0,     # log_energy = 0 → softplus(0) + 1e-4 ≈ 0.693 (unit-ish energy)

    # Pitch estimation
    # Ukulele lowest note: C4 = 261.63 Hz (open string 3); safety margin: 98 Hz (G2)
    # Ukulele highest practical note: A5 = 880 Hz; ceiling: 1050 Hz
    fmin_hz=98.0,
    fmax_hz=1050.0,
    poly_fmin_hz=240.0,     # below C4 (261.63) only ghost sub-harmonics appear
    midi_range=(60, 84),    # C4 to C6 — full playable ukulele range

    # NNLS spectral template
    # Nylon strings: harmonics fall off quickly. 8 harmonics is sufficient
    # and avoids false matches from noise in the upper spectrum.
    num_harmonics=8,
    gaussian_sigma=1.5,     # bin-spread: ≈ 2–3 Hz at 16 kHz / 8192-point FFT

    # Scoring weights
    # Ukulele: 4 strings, re-entrant tuning. Ergonomic penalty kept at 0.5
    # (same as original prototype, well-calibrated for 4-string instruments).
    ergo_weight=0.5,
    unison_bonus=0.1,
    overtone_factor=0.25,
    overtone_ratio_tol=0.03,

    # Analysis windows
    # Ukulele: shorter sustain → 0.5s window captures the peak cleanly
    analysis_window_sec=0.5,
    post_attack_skip_ms=60.0,

    # NNLS voice-presence penalty
    nnls_presence_threshold=0.05,
    nnls_mute_penalty=0.03,
)


# ===========================================================================
# GUITAR PROFILE  (Standard Six-String Tuning: E2-A2-D3-G3-B3-E4)
# ===========================================================================
#
# Physical basis for key parameters:
#
# Tuning: Standard EADGBE guitar tuning (commonly used string numbering:
#   string 1 = highest E4, string 6 = lowest E2).
#   We use the same 1 = highest pitch convention as the ukulele so the
#   fretboard mapper's arithmetic is consistent across instruments.
#
#   String 6 (E2,  82.41 Hz) — lowest pitch
#   String 5 (A2, 110.00 Hz)
#   String 4 (D3, 146.83 Hz)
#   String 3 (G3, 196.00 Hz)
#   String 2 (B3, 246.94 Hz)
#   String 1 (E4, 329.63 Hz) — highest pitch
#
# Inharmonicity:
#   Steel strings are stiffer than nylon → higher inharmonicity coefficients.
#   Wound bass strings (E2, A2, D3) have higher bending stiffness per unit
#   length and therefore more pronounced inharmonicity than plain treble strings.
#   Approximate empirical values:
#     Open stretch:  wound strings ~0.004–0.008, plain strings ~0.001–0.003
#     Growth ratio:  wound ~3.0 (slower growth; higher base stretch),
#                    plain ~4.0 (faster growth; lower base stretch)
#   For this profile we use a single global ratio of 3.5 (geometric mean)
#   with per-string open stretch tuned to the wound/plain split.
#
# Decay:
#   Steel-string acoustic guitar: longer scale length (~650 mm vs ~380 mm ukulele)
#   and steel strings → longer sustain. Physical decay per sample at 16 kHz
#   empirically around 0.995–0.9985 (varies by string and plucking position).
#   We initialise at 0.996 and let the optimizer converge.
#
# Harmonics:
#   Steel strings sustain overtones much longer than nylon. Using 12 harmonics
#   in the NNLS template captures the perceptually rich upper partials that
#   are characteristic of steel-string guitar tone.
#
# Pitch range:
#   Lowest: E2 = 82.41 Hz (open string 6), MIDI 40
#   Highest: fret 15 on string 1 (E4 open → D5 at fret 10, G5 at fret 15)
#     E4 open = MIDI 64; fret 15 → MIDI 79 = G5 = 783.99 Hz
#   pYIN: fmin=70 Hz (below E2 for safety), fmax=1200 Hz (above G5 with margin)
#   HSS constrained search: MIDI 40 (E2) to MIDI 79 (G5)

GUITAR_PROFILE = InstrumentProfile(
    name="guitar",
    display_name="Guitar (E2-A2-D3-G3-B3-E4 Standard)",
    num_strings=6,

    tuning={
        6: {"name": "E2", "freq":  82.41},   # lowest bass string
        5: {"name": "A2", "freq": 110.00},
        4: {"name": "D3", "freq": 146.83},
        3: {"name": "G3", "freq": 196.00},
        2: {"name": "B3", "freq": 246.94},
        1: {"name": "E4", "freq": 329.63},   # highest treble string
    },

    max_fret=15,

    # Inharmonicity model for steel strings
    # Wound strings (6, 5, 4): higher open stretch (~0.005) due to core+winding stiffness
    # Plain strings (3, 2, 1): lower open stretch (~0.001–0.002) — lighter, less stiff
    stretch_open={
        6: 0.006,   # E2 wound: higher inharmonicity due to thick wound construction
        5: 0.005,   # A2 wound
        4: 0.004,   # D3 wound (lighter winding)
        3: 0.002,   # G3 plain
        2: 0.001,   # B3 plain
        1: 0.001,   # E4 plain (thinnest string, lowest inharmonicity)
    },
    stretch_ratio=3.5,      # steel strings grow inharmonicity more slowly per fret
                            # than nylon (higher base → lower ratio to avoid oversaturation)
    stretch_max=0.1,

    # KS initial values
    # Guitar: longer scale, steel strings → slower decay than ukulele
    # Target initial physical decay: 0.996
    decay_init=_decay_to_logit(0.996),
    stretch_init=-5.0,
    decay_penalty_threshold=0.95,

    # Synthesis defaults
    # Guitar: longer sustain than ukulele → higher initial decay
    default_decay=0.996,
    default_energy=0.0,

    # Pitch estimation
    # Guitar lowest: E2 = 82.41 Hz, safety fmin = 70 Hz
    # Guitar highest practical (fret 15 string 1 E4): G5 = 783.99 Hz; fmax = 1200 Hz
    fmin_hz=70.0,
    fmax_hz=1200.0,
    poly_fmin_hz=75.0,      # slightly below E2 open to capture it reliably
    midi_range=(40, 79),    # E2 (MIDI 40) to G5 (MIDI 79) — full standard guitar range

    # NNLS spectral template
    # Steel strings sustain more harmonics → use 12 to capture the richer timbre
    num_harmonics=12,
    gaussian_sigma=2.0,     # slightly wider bin-spread for lower-pitched guitar fundamentals
                            # (lower notes have tighter harmonic spacing in bin space)

    # Scoring weights
    # Guitar: 6 strings with wider pitch range.
    # ergo_weight needs to be slightly larger to prevent high-fret hallucinations
    # across 6 strings (more possible fret positions to search).
    ergo_weight=0.6,
    unison_bonus=0.1,
    overtone_factor=0.25,
    overtone_ratio_tol=0.03,

    # Analysis windows
    # Guitar: longer sustain → can afford a slightly longer analysis window
    analysis_window_sec=0.6,
    post_attack_skip_ms=50.0,  # steel strings have faster transient decay than nylon

    # NNLS voice-presence penalty
    nnls_presence_threshold=0.05,
    nnls_mute_penalty=0.03,
)


# ===========================================================================
# Profile Registry
# ===========================================================================

_REGISTRY: Dict[str, InstrumentProfile] = {
    "ukulele": UKULELE_PROFILE,
    "guitar":  GUITAR_PROFILE,
}


def get_profile(instrument: str) -> InstrumentProfile:
    """
    Look up an instrument profile by name (case-insensitive).

    Parameters
    ----------
    instrument : str
        Instrument name, e.g. "ukulele" or "guitar".

    Returns
    -------
    InstrumentProfile

    Raises
    ------
    ValueError if the instrument is not registered.
    """
    key = instrument.strip().lower()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Available profiles: {available}"
        )
    return _REGISTRY[key]


def register_profile(profile: InstrumentProfile) -> None:
    """
    Register a custom instrument profile.
    Allows third-party extensions without modifying this file.
    """
    _REGISTRY[profile.name.lower()] = profile


def list_instruments() -> list:
    """Return a list of all registered instrument names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Convenience: expected inharmonicity for a given profile + string + fret
# ---------------------------------------------------------------------------

def expected_inharmonicity(profile: InstrumentProfile, string_idx: int, fret: int) -> float:
    """
    Compute the expected all-pass stretch coefficient for a given
    instrument profile, string, and fret position.

    Model: stretch(fret) = stretch_open[string_idx] * stretch_ratio^fret
    clamped at stretch_max.
    """
    s0 = profile.stretch_open.get(string_idx, 0.002)
    return min(s0 * (profile.stretch_ratio ** fret), profile.stretch_max)
