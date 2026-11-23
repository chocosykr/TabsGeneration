"""
instrument_fretboard.py — Instrument-Aware Fretboard Mapper (Upgrade Layer).

Extends fretboard_mapper.py with instrument-profile-driven logic.
The original fretboard_mapper.py is NOT modified; this module provides
upgraded equivalents that accept an InstrumentProfile.

Key differences from fretboard_mapper.py:
  - expected_inharmonicity() reads from the profile's stretch_open / stretch_ratio
    instead of module-level constants.
  - map_parameters_to_tab() uses profile.tuning instead of UKULELE_TUNING.
  - refine_chord_with_inharmonicity() is instrument-agnostic (uses profile).
  - All functions accept an InstrumentProfile rather than assuming ukulele.
"""

import math
import numpy as np
import itertools
import torch
from typing import List, Dict, Optional, Tuple

from instrument_config import InstrumentProfile, expected_inharmonicity


# ---------------------------------------------------------------------------
# MIDI / frequency utilities (unchanged from fretboard_mapper.py)
# ---------------------------------------------------------------------------

def freq_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_freq(midi: float) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12.0))


# ---------------------------------------------------------------------------
# Inharmonicity scoring
# ---------------------------------------------------------------------------

def inharmonicity_score(observed_inharm: float,
                        string_idx: int,
                        fret: int,
                        profile: InstrumentProfile) -> float:
    """
    Gaussian-similarity score between observed and physically expected
    inharmonicity for a given instrument, string, and fret.

    Score ∈ [0, 1]: 1.0 = perfect match.
    Sigma = 0.04: wide enough to bridge open vs fretted clusters while
    still differentiating adjacent frets.
    """
    expected = expected_inharmonicity(profile, string_idx, fret)
    sigma = 0.04
    diff = observed_inharm - expected
    return float(math.exp(-0.5 * (diff / sigma) ** 2))


# ---------------------------------------------------------------------------
# Single-note tab mapping
# ---------------------------------------------------------------------------

def map_parameters_to_tab(optimized_pitch: float,
                           optimized_decay: float,
                           optimized_inharm: float,
                           profile: InstrumentProfile,
                           target_audio: Optional[torch.Tensor] = None,
                           sample_rate: int = 16000) -> dict:
    """
    Maps a single continuous optimized pitch to discrete (string, fret) coordinates
    for the given instrument profile.

    When target_audio is provided, synthesizes each candidate tab's audio and
    compares it directly against the input audio.
    """
    target_midi = freq_to_midi(optimized_pitch)
    target_midi_rounded = round(target_midi)
    candidates = []

    for string_idx, data in profile.tuning.items():
        open_midi = freq_to_midi(data["freq"])
        fret = int(round(target_midi_rounded - open_midi))
        if 0 <= fret <= profile.max_fret:
            candidates.append({
                "string":    string_idx,
                "fret":      fret,
                "open_name": data["name"],
            })

    if not candidates:
        return {
            "error": f"Pitch {optimized_pitch:.2f} Hz is out of bounds "
                     f"for {profile.display_name}."
        }

    INHARMONICITY_WEIGHT = 0.5
    AUDIO_WEIGHT = 0.5 if target_audio is not None else 0.0

    for candidate in candidates:
        s_inharm = inharmonicity_score(
            optimized_inharm, candidate["string"], candidate["fret"], profile
        )
        ergonomic_penalty = -candidate["fret"] * 0.001

        # Synthesize candidate audio and compare
        audio_sim = 0.0
        audio_loss_val = 0.0
        if target_audio is not None:
            try:
                from synthesis_engine import SynthesisEngine
                engine = SynthesisEngine(instrument=profile.name, sample_rate=sample_rate)
                exp_stretch = expected_inharmonicity(profile, candidate["string"], candidate["fret"])
                duration = target_audio.shape[-1] / sample_rate
                cand_synth = engine.synthesize_candidate(
                    pitches=[optimized_pitch],
                    duration=duration,
                    decays=[optimized_decay],
                    stretches=[exp_stretch],
                )
                comp = engine.compare_audio(cand_synth, target_audio)
                audio_loss_val = comp["stft_loss"]
                audio_sim = comp["similarity"]
                candidate["audio_loss"] = audio_loss_val
                candidate["audio_similarity"] = audio_sim
                candidate["audio_metrics"] = comp
            except Exception:
                pass

        candidate["inharm_score"] = s_inharm
        candidate["ergonomic_penalty"] = ergonomic_penalty

    # Normalise audio comparison scores
    cand_losses = [c["audio_loss"] for c in candidates
                   if "audio_loss" in c and c["audio_loss"] > 0]
    min_loss = min(cand_losses) if cand_losses else 1.0

    for candidate in candidates:
        if "audio_loss" in candidate and candidate["audio_loss"] > 0:
            candidate["audio_match_norm"] = min_loss / candidate["audio_loss"]
        else:
            candidate["audio_match_norm"] = candidate.get("audio_similarity", 0.0)

        candidate["total_score"] = (
            INHARMONICITY_WEIGHT * candidate["inharm_score"]
            + AUDIO_WEIGHT * candidate["audio_match_norm"]
            + candidate["ergonomic_penalty"]
        )

    candidates = sorted(candidates, key=lambda x: x["total_score"], reverse=True)
    best_candidate = candidates[0]

    return {
        "pitch_hz":   optimized_pitch,
        "midi_note":  target_midi_rounded,
        "string":     best_candidate["string"],
        "fret":       best_candidate["fret"],
        "instrument": profile.name,
        "all_candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Chord refinement with inharmonicity
# ---------------------------------------------------------------------------

def refine_chord_with_inharmonicity(chord_result: dict,
                                    voice_params: list,
                                    strummed_strings: list,
                                    profile: InstrumentProfile,
                                    target_audio: Optional[torch.Tensor] = None,
                                    sample_rate: int = 16000) -> dict:
    """
    Post-optimization chord refinement for the given instrument.

    Identical logic to fretboard_mapper.refine_chord_with_inharmonicity but
    parameterised from an InstrumentProfile rather than module-level UKULELE_TUNING.
    """
    target_midis = {
        int(round(freq_to_midi(p))) for p in chord_result["pitches"]
    }

    observed_by_midi = {}
    for vp in voice_params:
        m = int(round(freq_to_midi(vp["pitch"])))
        if m not in observed_by_midi:
            observed_by_midi[m] = []
        observed_by_midi[m].append(vp.get("inharmonicity", 0.0))

    frets_range = range(profile.max_fret + 1)
    valid_shapes = []

    for frets in itertools.product(frets_range, repeat=len(strummed_strings)):
        generated_midis = []
        generated_set = set()
        for string_idx, fret in zip(strummed_strings, frets):
            m = int(round(freq_to_midi(profile.tuning[string_idx]["freq"]))) + fret
            generated_midis.append(m)
            generated_set.add(m)

        if generated_set != target_midis:
            continue

        gen_by_midi = {}
        for s_idx, f, m in zip(strummed_strings, frets, generated_midis):
            if m not in gen_by_midi:
                gen_by_midi[m] = []
            gen_by_midi[m].append({"string": s_idx, "fret": f})

        shape_possible = True
        inharm_total = 0.0
        inharm_details = {}

        for m, gen_strings in gen_by_midi.items():
            obs_stretches = observed_by_midi.get(m, [])
            if not obs_stretches:
                shape_possible = False
                break

            best_cluster_score = -1.0
            best_cluster_details = {}

            while len(obs_stretches) < len(gen_strings):
                obs_stretches.append(obs_stretches[0])

            for p_stretches in itertools.permutations(obs_stretches, len(gen_strings)):
                cluster_score = 0.0
                cluster_details = {}
                for g_dict, obs_s in zip(gen_strings, p_stretches):
                    s_idx = g_dict["string"]
                    f = g_dict["fret"]
                    s = inharmonicity_score(obs_s, s_idx, f, profile)
                    cluster_score += s
                    cluster_details[s_idx] = {
                        "fret":             f,
                        "observed_stretch":  obs_s,
                        "expected_stretch":  expected_inharmonicity(profile, s_idx, f),
                        "score":             s,
                    }
                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_cluster_details = cluster_details

            inharm_total += best_cluster_score
            inharm_details.update(best_cluster_details)

        if not shape_possible:
            continue

        non_zero_frets = [f for f in frets if f > 0]
        span = (max(non_zero_frets) - min(non_zero_frets)) if non_zero_frets else 0
        total_fret = sum(frets)

        # Audio comparison for this shape
        shape_audio_loss = 0.0
        shape_audio_sim = 0.0
        if target_audio is not None:
            try:
                from synthesis_engine import SynthesisEngine
                engine = SynthesisEngine(instrument=profile.name, sample_rate=sample_rate)
                shape_pitches = [
                    profile.tuning[s]["freq"] * (2 ** (f / 12.0))
                    for s, f in zip(strummed_strings, frets)
                ]
                shape_stretches = [
                    expected_inharmonicity(profile, s, f)
                    for s, f in zip(strummed_strings, frets)
                ]
                cand_synth = engine.synthesize_candidate(
                    pitches=shape_pitches,
                    duration=target_audio.shape[-1] / sample_rate,
                    stretches=shape_stretches,
                )
                shape_comp = engine.compare_audio(cand_synth, target_audio)
                shape_audio_loss = shape_comp["stft_loss"]
                shape_audio_sim = shape_comp["similarity"]
            except Exception:
                pass

        valid_shapes.append({
            "frets":            frets,
            "span":             span,
            "total_fret":       total_fret,
            "inharm_score":     inharm_total,
            "inharm_details":   inharm_details,
            "audio_loss":       shape_audio_loss,
            "audio_similarity": shape_audio_sim,
        })

    if not valid_shapes:
        return chord_result

    # Sort: prefer audio match when available, else inharmonicity
    if target_audio is not None and any(s["audio_similarity"] > 0 for s in valid_shapes):
        valid_shapes.sort(
            key=lambda x: (-x["audio_similarity"], -x["inharm_score"], x["span"], x["total_fret"])
        )
    else:
        valid_shapes.sort(
            key=lambda x: (-x["inharm_score"], x["span"], x["total_fret"])
        )

    best_shape = valid_shapes[0]
    mapping = {string_idx: fret
               for string_idx, fret in zip(strummed_strings, best_shape["frets"])}

    return {
        "target_pitches":     chord_result["pitches"],
        "target_midis":       list(target_midis),
        "strummed_strings":   strummed_strings,
        "mapping":            mapping,
        "span":               best_shape["span"],
        "inharm_score":       best_shape["inharm_score"],
        "inharmonicity_scores": best_shape["inharm_details"],
        "audio_loss":         best_shape.get("audio_loss", 0.0),
        "audio_similarity":   best_shape.get("audio_similarity", 0.0),
        "all_valid_shapes":   valid_shapes[:5],
        "instrument":         profile.name,
    }


# ---------------------------------------------------------------------------
# Utility: map chord to tab (simple brute-force without inharmonicity scoring)
# ---------------------------------------------------------------------------

def map_chord_to_tab(target_pitches_hz: List[float],
                     strummed_strings: List[int],
                     profile: InstrumentProfile) -> dict:
    """
    Given a list of detected pitches and a list of strings that were physically
    strummed, brute-force searches for an assignment of frets on that instrument.
    """
    target_midis = {int(round(freq_to_midi(p))) for p in target_pitches_hz}
    valid_shapes = []
    frets_range = range(profile.max_fret + 1)

    for frets in itertools.product(frets_range, repeat=len(strummed_strings)):
        generated_midis = set()
        for string_idx, fret in zip(strummed_strings, frets):
            open_midi = int(round(freq_to_midi(profile.tuning[string_idx]["freq"])))
            generated_midis.add(open_midi + fret)

        if generated_midis == target_midis:
            non_zero_frets = [f for f in frets if f > 0]
            span = (max(non_zero_frets) - min(non_zero_frets)) if non_zero_frets else 0
            valid_shapes.append({
                "frets":      frets,
                "span":       span,
                "total_fret": sum(frets),
            })

    if not valid_shapes:
        return {"error": f"No valid fingering found for these pitches on {profile.display_name}."}

    valid_shapes.sort(key=lambda x: (x["span"], x["total_fret"]))
    best_shape = valid_shapes[0]
    mapping = {s: best_shape["frets"][i] for i, s in enumerate(strummed_strings)}

    return {
        "target_pitches":   target_pitches_hz,
        "target_midis":     list(target_midis),
        "strummed_strings": strummed_strings,
        "mapping":          mapping,
        "span":             best_shape["span"],
        "instrument":       profile.name,
    }
