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
