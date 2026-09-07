import torch
import torch.nn.functional as F
from karplus_strong import DifferentiableKarplusStrong
from tqdm import tqdm
import math

# YIN is very accurate for clean single-note audio, so we trust it for pitch.
# The optimizer is used only to refine decay and excitation (timbre), not pitch.


def multi_scale_stft_loss(synth: torch.Tensor, target: torch.Tensor,
                          fft_sizes=[512, 1024, 2048]) -> torch.Tensor:
    """
    Multi-scale STFT loss combining Spectral Convergence and Log-Magnitude error.
    """
    if synth.ndim == 1:
        synth = synth.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    loss = torch.tensor(0.0)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft)

        mag_s = torch.abs(torch.stft(synth, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7
        mag_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7

        sc   = torch.norm(mag_t - mag_s, p="fro") / (torch.norm(mag_t, p="fro") + 1e-8)
        lmag = F.l1_loss(torch.log(mag_t), torch.log(mag_s))
        loss = loss + sc + lmag

    return loss / len(fft_sizes)


def optimize_parameters(target_audio: torch.Tensor, initial_pitch: float,
                        sample_rate: int = 16000,
                        num_iterations: int = 50, lr: float = 0.05):
    """
    Adam optimisation loop that fits the Karplus-Strong model to target_audio.
    """
    duration = target_audio.shape[-1] / sample_rate
    model = DifferentiableKarplusStrong(sample_rate=sample_rate, duration=duration)

    # Initialise log_pitch to the YIN estimate and FREEZE it.
    # Only decay and excitation_energy are optimized.
    log_pitch_anchor = math.log(max(initial_pitch, 20.0))
    with torch.no_grad():
        model.log_pitch.data.fill_(log_pitch_anchor)
    model.log_pitch.requires_grad_(False)  # Freeze pitch

    # Only optimise the timbre parameters
    optimizer = torch.optim.Adam(
        [model.logit_decay, model.log_energy], lr=lr
    )

    print(f"Starting optimisation  (pitch fixed at {initial_pitch:.2f} Hz, optimising decay/energy) ...")

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        synth = model()

        min_len = min(synth.shape[-1], target_audio.shape[-1])
        stft_loss = multi_scale_stft_loss(synth[:min_len], target_audio.squeeze()[:min_len])
        loss = stft_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % 10 == 0:
            p, d, _ = model.get_params()
            print(f"  iter {i+1:3d}  loss={loss.item():.4f}  "
                  f"pitch={p.item():.2f} Hz  decay={d.item():.4f}")

    # For fretboard mapping, use the original YIN pitch (most reliable)
    # Decay from optimizer gives timbre info for string disambiguation
    _, decay, energy = model.get_params()
    return {
        "pitch": initial_pitch,    # Use YIN estimate directly
        "decay": decay.item(),
        "excitation_energy": energy.item(),
    }
