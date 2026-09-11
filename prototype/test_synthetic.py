import torch
import scipy.io.wavfile as wavfile
import numpy as np
import math
from karplus_strong import DifferentiableKarplusStrong, MultiVoiceKarplusStrong

def generate_test_audio(pitch_hz: float, output_path: str, sample_rate: int = 16000, decay: float = 0.98):
    """
    Generate a synthetic Karplus-Strong pluck at exactly pitch_hz.
    Works correctly with the log-pitch parameterisation in the model.
    """
    model = DifferentiableKarplusStrong(sample_rate=sample_rate, duration=1.0)

    with torch.no_grad():
        # Set raw parameter to log(pitch_hz) so exp() recovers exact frequency.
        model.log_pitch.copy_(torch.tensor(math.log(pitch_hz)))
        # Set decay
        model.logit_decay.copy_(torch.tensor(math.log(decay / (1 - decay))))
        model.log_energy.copy_(torch.tensor(0.0))

        y = model()

    # Normalise and save
    y = y / torch.max(torch.abs(y) + 1e-8)
    y_np = y.numpy()
    wavfile.write(output_path, sample_rate, (y_np * 32767).astype(np.int16))
    print(f"Generated {pitch_hz:.2f} Hz at '{output_path}'  (sr={sample_rate})")


def generate_multi_note_audio(pitches_hz: list[float],
                              output_path: str,
                              sample_rate: int = 16000,
                              decay: float = 0.98):
    """
    Generate a synthetic multi-note WAV by summing multiple Karplus-Strong voices.
    """
    num_voices = len(pitches_hz)
    model = MultiVoiceKarplusStrong(num_voices, sample_rate=sample_rate, duration=1.0)

    with torch.no_grad():
        for voice, pitch in zip(model.voices, pitches_hz):
            voice.log_pitch.copy_(torch.tensor(math.log(pitch)))
            voice.logit_decay.copy_(torch.tensor(math.log(decay / (1 - decay))))
            voice.log_energy.copy_(torch.tensor(0.0))

        y = model()

    # Normalise and save
    y = y / torch.max(torch.abs(y) + 1e-8)
    y_np = y.numpy()
    wavfile.write(output_path, sample_rate, (y_np * 32767).astype(np.int16))
    
    pitch_str = " + ".join(f"{p:.2f} Hz" for p in pitches_hz)
    print(f"Generated chord [{pitch_str}] at '{output_path}'  (sr={sample_rate})")


# Handy helper: convert string/fret to Hz using standard ukulele re-entrant tuning
UKULELE_OPEN_HZ = {4: 392.00, 3: 261.63, 2: 329.63, 1: 440.00}

def string_fret_to_hz(string: int, fret: int) -> float:
    return UKULELE_OPEN_HZ[string] * (2 ** (fret / 12))


def generate_two_note_audio(pitch1_hz: float, pitch2_hz: float,
                            output_path: str,
                            sample_rate: int = 16000,
                            decay1: float = 0.98, decay2: float = 0.98):
    generate_multi_note_audio([pitch1_hz, pitch2_hz], output_path, sample_rate, decay=(decay1+decay2)/2.0)


if __name__ == "__main__":
    # --- Single-note test files ---
    generate_test_audio(string_fret_to_hz(2, 0), "test_E4_s2f0.wav")
    generate_test_audio(string_fret_to_hz(1, 2), "test_B4_s1f2.wav")

    # --- Two-note (dyad) test files ---
    generate_two_note_audio(string_fret_to_hz(3, 0), string_fret_to_hz(2, 0), "test_two_C4_E4.wav")
    generate_two_note_audio(string_fret_to_hz(2, 0), string_fret_to_hz(1, 0), "test_two_E4_A4.wav")
    generate_two_note_audio(string_fret_to_hz(4, 0), string_fret_to_hz(4, 5), "test_two_G4_C5.wav")
    
    # --- Four-note (full chord) test files ---
    # Standard C-Major Chord: 0003
    # G4 (String 4 open), C4 (String 3 open), E4 (String 2 open), C5 (String 1 fret 3)
    generate_multi_note_audio([
        string_fret_to_hz(4, 0), 
        string_fret_to_hz(3, 0), 
        string_fret_to_hz(2, 0), 
        string_fret_to_hz(1, 3)
    ], "test_four_C_Major.wav")
