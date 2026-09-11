import numpy as np

# Standard Re-entrant Ukulele Tuning (High G)
# String 4 is G4, String 3 is C4, String 2 is E4, String 1 is A4
UKULELE_TUNING = {
    4: {"name": "G4", "freq": 392.00},
    3: {"name": "C4", "freq": 261.63},
    2: {"name": "E4", "freq": 329.63},
    1: {"name": "A4", "freq": 440.00}
}

MAX_FRET = 15

def freq_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)

def midi_to_freq(midi: float) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12.0))

def map_parameters_to_tab(optimized_pitch: float, optimized_decay: float) -> dict:
    """
    Maps continuous optimized pitch to discrete (string, fret) coordinates.
    Since ukulele can have multiple ways to play the same pitch, 
    we find all candidates and choose the most ergonomic one (e.g. lowest fret),
    though decay can theoretically differentiate strings.
    """
    target_midi = freq_to_midi(optimized_pitch)
    target_midi_rounded = round(target_midi)
    
    candidates = []
    
    for string_idx, data in UKULELE_TUNING.items():
        open_midi = freq_to_midi(data["freq"])
        
        fret = target_midi_rounded - open_midi
        fret = int(round(fret))
        
        if 0 <= fret <= MAX_FRET:
            candidates.append({"string": string_idx, "fret": fret, "open_name": data["name"]})
            
    if not candidates:
        return {"error": f"Pitch {optimized_pitch:.2f} Hz is out of bounds for a standard ukulele."}
        
    # Heuristic: Pick the candidate with the lowest fret number (easiest to play generally)
    # A more advanced version would use optimized_decay to select the string since
    # thicker strings have different damping characteristics.
    candidates = sorted(candidates, key=lambda x: x["fret"])
    
    best_candidate = candidates[0]
    
    return {
        "pitch_hz": optimized_pitch,
        "midi_note": target_midi_rounded,
        "string": best_candidate["string"],
        "fret": best_candidate["fret"],
        "all_candidates": candidates
    }


def map_chord_to_tab(target_pitches_hz: list[float], strummed_strings: list[int]) -> dict:
    """
    Given a list of detected pitches and a list of strings that were physically strummed,
    brute-force searches for an assignment of frets to exactly those strings such that 
    the resulting set of notes perfectly matches the target pitches.
    """
    import itertools
    
    target_midis = {int(round(freq_to_midi(p))) for p in target_pitches_hz}
    
    valid_shapes = []
    
    # We evaluate all possible fret combinations for the strummed strings.
    # To keep it fast, max fret is bounded. 16 frets ^ 4 strings = 65536 combinations.
    frets_range = range(MAX_FRET + 1)
    
    for frets in itertools.product(frets_range, repeat=len(strummed_strings)):
        generated_midis = set()
        out_of_bounds = False
        
        for string_idx, fret in zip(strummed_strings, frets):
            open_midi = int(round(freq_to_midi(UKULELE_TUNING[string_idx]["freq"])))
            generated_midis.add(open_midi + fret)
            
        # The set of generated notes MUST EXACTLY EQUAL the set of target notes.
        # (If 4 strings are strummed but only 3 pitches were detected, it means 
        # one pitch must be doubled across two strings, which this handles beautifully).
        if generated_midis == target_midis:
            # Score the ergonomics of the shape
            non_zero_frets = [f for f in frets if f > 0]
            if not non_zero_frets:
                span = 0
            else:
                span = max(non_zero_frets) - min(non_zero_frets)
                
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
    
    # Format the result nicely
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
