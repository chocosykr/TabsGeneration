import torch
import torch.nn.functional as F
from karplus_strong import DifferentiableKarplusStrong
from tqdm import tqdm
import math
from typing import List

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


def optimize_multi_parameters(target_audio: torch.Tensor,
                              pitches: List[float],
                              sample_rate: int = 16000,
                              num_iterations: int = 50,
                              lr: float = 0.5,
                              active_mask: List[bool] = None) -> List[dict]:
    """
    Adam optimisation loop that fits a multi-voice Karplus-Strong model to
    `target_audio` (a mixed signal of multiple simultaneous notes).

    All pitches are frozen at the supplied `pitches` estimates. Only decay 
    and excitation_energy for each voice are optimised.

    `active_mask` (optional): a list of bools, one per voice. Voices marked
    False are simulated but kept SILENT (excitation energy frozen at ~0) —
    used when the model simulates more strings than were actually strummed.

    Returns a list of parameter dicts, one for each voice.
    """
    num_voices = len(pitches)
    if num_voices == 0:
        return []
    if active_mask is None:
        active_mask = [True] * num_voices
        
    duration = target_audio.shape[-1] / sample_rate
    from karplus_strong import MultiVoiceKarplusStrong
    model = MultiVoiceKarplusStrong(num_voices, sample_rate=sample_rate, duration=duration)

    # Initialise and freeze all pitches.
    optim_params = []
    for voice, pitch, active in zip(model.voices, pitches, active_mask):
        log_anchor = math.log(max(pitch, 20.0))
        with torch.no_grad():
            voice.log_pitch.data.fill_(log_anchor)
        voice.log_pitch.requires_grad_(False)

        if not active:
            # Silent simulated string: freeze excitation at ~0 so it never
            # contributes to the synthesised sound.
            with torch.no_grad():
                voice.log_energy.data.fill_(-20.0)
            voice.log_energy.requires_grad_(False)
            voice.logit_decay.requires_grad_(False)
            continue
        
        # Add trainable parameters to optimiser list
        optim_params.extend([voice.logit_decay, voice.log_energy])

    # Jointly optimise the timbre parameters across all voices.
    optimizer = torch.optim.Adam(optim_params, lr=lr)

    pitch_str = ", ".join(f"{p:.2f} Hz" for p in pitches)
    print(f"Starting {num_voices}-voice optimisation "
          f"(pitches: [{pitch_str}], optimising decay/energy for each) ...")

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        synth = model()

        min_len = min(synth.shape[-1], target_audio.shape[-1])
        stft_loss = multi_scale_stft_loss(synth[:min_len],
                                          target_audio.squeeze()[:min_len])
        stft_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % 10 == 0:
            decays = [v.get_params()[1].item() for v in model.voices]
            decay_str = ", ".join(f"d{v+1}={decays[v]:.4f}" for v in range(num_voices))
            print(f"  iter {i+1:3d}  loss={stft_loss.item():.4f}  {decay_str}")

    # Build return dictionaries
    results = []
    for i, voice in enumerate(model.voices):
        _, decay, energy = voice.get_params()
        results.append({
            "pitch": pitches[i],
            "decay": decay.item(),
            "excitation_energy": energy.item(),
        })
        print(f"   Voice {i+1} — Pitch: {pitches[i]:.2f} Hz, Decay: {decay.item():.4f}")

    return results
