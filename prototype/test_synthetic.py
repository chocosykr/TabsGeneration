import torch
import scipy.io.wavfile as wavfile
import numpy as np
import math
from karplus_strong import DifferentiableKarplusStrong

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


# Handy helper: convert string/fret to Hz using standard ukulele re-entrant tuning
UKULELE_OPEN_HZ = {4: 392.00, 3: 261.63, 2: 329.63, 1: 440.00}

def string_fret_to_hz(string: int, fret: int) -> float:
    return UKULELE_OPEN_HZ[string] * (2 ** (fret / 12))


if __name__ == "__main__":
    # E4 open (String 2, Fret 0)
    generate_test_audio(string_fret_to_hz(2, 0), "test_E4_s2f0.wav")
    # B4 (String 1, Fret 2)
    generate_test_audio(string_fret_to_hz(1, 2), "test_B4_s1f2.wav")
