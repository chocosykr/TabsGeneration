import argparse
import scipy.io.wavfile as wavfile
import numpy as np
import torch

from pitch_estimation import estimate_initial_pitch, _extract_peak_energy_window
from optimization import optimize_parameters
from fretboard_mapper import map_parameters_to_tab

# Max audio length fed to the optimizer (seconds).
# Longer files slow down STFT loss massively; pitch is already estimated
# separately from the onset window, so we only need a short clean segment.
_MAX_OPT_SEC = 0.5


def process_audio(file_path: str, num_iterations: int = 50, lr: float = 0.5):
    print(f"Loading audio from {file_path}...")
    sample_rate, waveform_np = wavfile.read(file_path)

    # Convert to float32 and normalise
    waveform_np = waveform_np.astype(np.float32) / 32767.0

    # Mixdown stereo → mono
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=1)

    print(f"Loaded {len(waveform_np)} samples at {sample_rate} Hz "
          f"({len(waveform_np)/sample_rate:.2f}s).")

    # --- Step 1: Pitch estimation on onset window ---
    print("\n1. Estimating Initial Pitch (pYIN on onset window)...")
    waveform_full = torch.from_numpy(waveform_np).unsqueeze(0)
    initial_pitch = estimate_initial_pitch(waveform_full, sample_rate)
    print(f"   Estimated F0: {initial_pitch:.2f} Hz")

    # --- Truncate to onset window for optimization ---
    y_opt = _extract_peak_energy_window(waveform_np, sample_rate,
                                        window_sec=_MAX_OPT_SEC)
    max_samples = int(_MAX_OPT_SEC * sample_rate)
    y_opt = y_opt[:max_samples]
    waveform_opt = torch.from_numpy(y_opt).unsqueeze(0)
    print(f"   (Optimizer will use {len(y_opt)} samples = "
          f"{len(y_opt)/sample_rate:.2f}s onset window for speed)")

    # --- Step 2: Karplus-Strong optimisation ---
    print("\n2. Running Differentiable Karplus-Strong Optimization...")
    optimized_params = optimize_parameters(
        target_audio=waveform_opt,
        initial_pitch=initial_pitch,
        sample_rate=sample_rate,
        num_iterations=num_iterations,
        lr=lr
    )
    print(f"   Optimized Pitch: {optimized_params['pitch']:.2f} Hz")
    print(f"   Optimized Decay: {optimized_params['decay']:.4f}")

    # --- Step 3: Map to tablature ---
    print("\n3. Mapping Physical Parameters to Tablature...")
    tab_result = map_parameters_to_tab(optimized_params['pitch'],
                                       optimized_params['decay'])

    if "error" in tab_result:
        print(f"   ERROR: {tab_result['error']}")
    else:
        best = tab_result['all_candidates'][0]
        print(f"   -> Plucked String: {tab_result['string']} "
              f"(open {best['open_name']} string)")
        print(f"   -> Fret Number:    {tab_result['fret']}")
        print(f"   -> Note:           MIDI {tab_result['midi_note']}")

        print("\n   Alternative Fingering Candidates:")
        for cand in tab_result['all_candidates']:
            print(f"      String {cand['string']} "
                  f"(open {cand['open_name']}), Fret {cand['fret']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentiable Ukulele Tab Transcription MVP")
    parser.add_argument("audio_file", type=str,
                        help="Path to the monophonic ukulele WAV file")
    parser.add_argument("--iter", type=int, default=50,
                        help="Number of optimization iterations (default: 50)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate for Adam (default: 0.5)")
    args = parser.parse_args()

    process_audio(args.audio_file, args.iter, args.lr)
