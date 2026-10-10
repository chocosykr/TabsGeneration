import numpy as np
import math

# Standard Re-entrant Ukulele Tuning (High G)
# String 4 is G4, String 3 is C4, String 2 is E4, String 1 is A4
UKULELE_TUNING = {
    4: {"name": "G4", "freq": 392.00},
    3: {"name": "C4", "freq": 261.63},
    2: {"name": "E4", "freq": 329.63},
    1: {"name": "A4", "freq": 440.00}
}

MAX_FRET = 15

# Empirically calibrated from 3 recordings (open-string stretch, fret-3 stretch):
#   String 4 open: obs ~0.001-0.004  |  String 2 fret 3: obs ~0.095-0.100
#   String 3 open: obs ~0.000-0.002  |  String 3 fret 4: obs ~0.095-0.100
#
# Model: stretch(fret) = stretch_0 * r^fret
# where r is fit so that stretch(3) = 0.095 given stretch(0) = 0.002
#   r = (0.095 / 0.002) ^ (1/3) ≈ 3.60
#
# Per-string open-string stretch (fret 0) from empirical observations:
_STRETCH_OPEN = {
    4: 0.002,   # G4: observed ~0.001-0.004
    3: 0.001,   # C4: observed ~0.000-0.002
    2: 0.001,   # E4: observed ~0.000-0.002
    1: 0.002,   # A4: observed ~0.001-0.004
}

# Per-fret growth ratio: calibrated from open→fret-3 data
# We want even the lowest open stretch (0.001) to saturate near 0.095 by fret 3.
# r = (0.095 / 0.001)^(1/3) ≈ 4.56
_STRETCH_RATIO = 4.56

# Sigmoid ceiling used in karplus_strong.py
_STRETCH_MAX = 0.1


def freq_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)

def midi_to_freq(midi: float) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def expected_inharmonicity(string_idx: int, fret: int) -> float:
    """
    Computes the expected all-pass stretch coefficient for a given string/fret.

    Uses the two-point calibrated model:
        stretch(fret) = stretch_open * RATIO^fret
    clamped at the sigmoid ceiling (0.1).
    """
    s0 = _STRETCH_OPEN.get(string_idx, 0.002)
    return min(s0 * (_STRETCH_RATIO ** fret), _STRETCH_MAX)


def decay_score(observed_decay: float, string_idx: int) -> float:
    """
    Decay is primarily driven by strum dynamics, not string identity,
    so we deliberately leave this as a neutral (zero) contribution.
    It remains here as an extension point once per-string decay priors
    are calibrated from a larger dataset.
    """
    return 0.0


def inharmonicity_score(observed_inharm: float, string_idx: int, fret: int) -> float:
    """
    Scores how well the observed stretch matches the expected inharmonicity
    for (string_idx, fret) using a Gaussian similarity.

    Score is in [0, 1]: 1.0 = perfect match, decays as observation deviates
    from the physically expected value.

    Sigma is set to 0.04: wide enough to bridge the open (≈0.002) vs fretted
    (≈0.095) clusters, while still giving good separation between adjacent frets.
    """
    expected = expected_inharmonicity(string_idx, fret)
    sigma = 0.04
    diff = observed_inharm - expected
    return float(math.exp(-0.5 * (diff / sigma) ** 2))


def map_parameters_to_tab(optimized_pitch: float,
                           optimized_decay: float,
                           optimized_inharm: float) -> dict:
    """
    Maps a single continuous optimized pitch to discrete (string, fret) coordinates.
    Uses calibrated inharmonicity scoring to disambiguate among candidates
    that produce the same MIDI note on different strings.
    """
    target_midi = freq_to_midi(optimized_pitch)
    target_midi_rounded = round(target_midi)

    candidates = []

    for string_idx, data in UKULELE_TUNING.items():
        open_midi = freq_to_midi(data["freq"])
        fret = target_midi_rounded - open_midi
        fret = int(round(fret))

        if 0 <= fret <= MAX_FRET:
            candidates.append({
                "string": string_idx,
                "fret": fret,
                "open_name": data["name"]
            })

    if not candidates:
        return {"error": f"Pitch {optimized_pitch:.2f} Hz is out of bounds "
                         f"for a standard ukulele."}

    # Score the candidates: inharmonicity for fret disambiguation,
    # plus a tiny ergonomic tiebreaker (prefer lower frets).
    INHARMONICITY_WEIGHT = 1.0

    for candidate in candidates:
        s_inharm = inharmonicity_score(
            optimized_inharm, candidate["string"], candidate["fret"]
        )
        ergonomic_penalty = -candidate["fret"] * 0.001
        candidate["total_score"] = (INHARMONICITY_WEIGHT * s_inharm) + ergonomic_penalty

    candidates = sorted(candidates, key=lambda x: x["total_score"], reverse=True)
    best_candidate = candidates[0]

    return {
        "pitch_hz": optimized_pitch,
        "midi_note": target_midi_rounded,
        "string": best_candidate["string"],
        "fret": best_candidate["fret"],
        "all_candidates": candidates
    }


def refine_chord_with_inharmonicity(chord_result: dict,
                                    voice_params: list[dict],
                                    strummed_strings: list[int]) -> dict:
    """
    Post-optimization chord refinement.

    The optimizer provides a list of observed (pitch, stretch) for each active voice.
    We iterate through all valid chord shapes (frets on strings) that produce
    the exact same SET of target notes.

    For each shape, we compute the expected (pitch, stretch) for each string.
    We then match each string to an observed voice that has the SAME pitch
    (MIDI note). If multiple strings share the same pitch (unison), we try all
    permutations and pick the one with the best inharmonicity score.

    The shape with the highest total score wins.
    """
    import itertools

    target_midis = {
        int(round(freq_to_midi(p))) for p in chord_result["pitches"]
    }

    # Group observed voices by MIDI note
    observed_by_midi = {}
    for vp in voice_params:
        m = int(round(freq_to_midi(vp["pitch"])))
        if m not in observed_by_midi:
            observed_by_midi[m] = []
        observed_by_midi[m].append(vp["inharmonicity"])

    frets_range = range(MAX_FRET + 1)
    valid_shapes = []

    for frets in itertools.product(frets_range, repeat=len(strummed_strings)):
        generated_midis = []
        generated_set = set()
        for string_idx, fret in zip(strummed_strings, frets):
            m = int(round(freq_to_midi(UKULELE_TUNING[string_idx]["freq"]))) + fret
            generated_midis.append(m)
            generated_set.add(m)

        if generated_set != target_midis:
            continue

        # For this shape to be completely valid, it must have the exact same
        # number of voices for each MIDI note as the observation, or it must
        # be able to map every string to an observed voice.
        # Actually, because target sets match, we just map strings to observed voices.
        # Group generated strings by MIDI note
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
                
            # If we generated more strings for this note than observed voices,
            # we just duplicate the closest observed stretch to be forgiving (or vice versa).
            # But perfectly, len(gen_strings) == len(obs_stretches).
            # Let's compute the best bipartite matching (permutations) for this note cluster.
            best_cluster_score = -1.0
            best_cluster_details = {}
            
            # Pad obs_stretches if needed (e.g. 2 strings play G4 but optimizer only heard 1 clear peak)
            while len(obs_stretches) < len(gen_strings):
                obs_stretches.append(obs_stretches[0])
                
            # Permute the observed stretches to match to the generated strings
            for p_stretches in itertools.permutations(obs_stretches, len(gen_strings)):
                cluster_score = 0.0
                cluster_details = {}
                for g_dict, obs_s in zip(gen_strings, p_stretches):
                    s_idx = g_dict["string"]
                    f = g_dict["fret"]
                    s = inharmonicity_score(obs_s, s_idx, f)
                    cluster_score += s
                    cluster_details[s_idx] = {
                        "fret": f,
                        "observed_stretch": obs_s,
                        "expected_stretch": expected_inharmonicity(s_idx, f),
                        "score": s
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

        valid_shapes.append({
            "frets": frets,
            "span": span,
            "total_fret": total_fret,
            "inharm_score": inharm_total,
            "inharm_details": inharm_details
        })

    if not valid_shapes:
        return chord_result

    # Sort by inharmonicity score (primary), then ergonomics (secondary)
    valid_shapes.sort(key=lambda x: (-x["inharm_score"], x["span"], x["total_fret"]))
    best_shape = valid_shapes[0]

    mapping = {}
    for string_idx, fret in zip(strummed_strings, best_shape["frets"]):
        mapping[string_idx] = fret

    return {
        "target_pitches": chord_result["pitches"],
        "target_midis": list(target_midis),
        "strummed_strings": strummed_strings,
        "mapping": mapping,
        "span": best_shape["span"],
        "inharm_score": best_shape["inharm_score"],
        "inharmonicity_scores": best_shape["inharm_details"],
        "all_valid_shapes": valid_shapes[:5]
    }


def map_chord_to_tab(target_pitches_hz: list[float],
                     strummed_strings: list[int]) -> dict:
    """
    Given a list of detected pitches and a list of strings that were physically strummed,
    brute-force searches for an assignment of frets to exactly those strings such that
    the resulting set of notes perfectly matches the target pitches.
    """
    import itertools

    target_midis = {int(round(freq_to_midi(p))) for p in target_pitches_hz}

    valid_shapes = []
    frets_range = range(MAX_FRET + 1)

    for frets in itertools.product(frets_range, repeat=len(strummed_strings)):
        generated_midis = set()

        for string_idx, fret in zip(strummed_strings, frets):
            open_midi = int(round(freq_to_midi(UKULELE_TUNING[string_idx]["freq"])))
            generated_midis.add(open_midi + fret)

        if generated_midis == target_midis:
            non_zero_frets = [f for f in frets if f > 0]
            span = (max(non_zero_frets) - min(non_zero_frets)) if non_zero_frets else 0
            total_fret = sum(frets)

            valid_shapes.append({
                "frets": frets,
                "span": span,
                "total_fret": total_fret
            })

    if not valid_shapes:
        return {"error": "No valid fingering found for these pitches on the specified strings."}

    # Sort by minimum span (hand stretch), then by lowest frets overall
    valid_shapes.sort(key=lambda x: (x["span"], x["total_fret"]))
    best_shape = valid_shapes[0]

    mapping = {}
    for i, string_idx in enumerate(strummed_strings):
        mapping[string_idx] = best_shape["frets"][i]

    return {
        "target_pitches": target_pitches_hz,
        "target_midis": list(target_midis),
        "strummed_strings": strummed_strings,
        "mapping": mapping,
        "span": best_shape["span"]
    }
