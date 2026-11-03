import torch
import torch.nn.functional as F
from karplus_strong import DifferentiableKarplusStrong
from tqdm import tqdm
import math
from typing import List, Tuple, Union, Optional
from device_config import DEVICE

# YIN is very accurate for clean single-note audio, so we trust it for pitch.
# The optimizer is used to refine decay, excitation, and timbre.


def multi_scale_stft_loss(synth: torch.Tensor, target: torch.Tensor,
                          fft_sizes=[512, 1024, 2048]) -> torch.Tensor:
    """
    Multi-scale STFT loss combining Spectral Convergence and Log-Magnitude error.
    """
    if synth.ndim == 1:
        synth = synth.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    if synth.device != target.device:
        target = target.to(synth.device)

    loss = torch.tensor(0.0, device=synth.device)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=synth.device)

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


def compare_candidate_audio(synth: torch.Tensor,
                            target: torch.Tensor,
                            sample_rate: int = 16000,
                            fft_sizes=[512, 1024, 2048]) -> dict:
    """
    Directly compare the synthesized output audio of a candidate tab against
    the input tabs audio.

    Returns a dictionary of audio comparison metrics:
      - 'stft_loss': multi-scale STFT loss across fft_sizes
      - 'spectral_convergence': normalized spectral error (0 = identical)
      - 'log_mag_loss': L1 distance between log-magnitude spectra
      - 'waveform_correlation': normalized cosine similarity of waveforms (-1 to +1)
      - 'similarity': monotonic audio match score in (0, 1] derived from stft_loss
    """
    if synth.ndim == 1:
        synth = synth.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    if synth.device != target.device:
        target = target.to(synth.device)

    min_len = min(synth.shape[-1], target.shape[-1])
    s = synth[..., :min_len]
    t = target[..., :min_len]

    sc_list = []
    lmag_list = []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=s.device)

        mag_s = torch.abs(torch.stft(s, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7
        mag_t = torch.abs(torch.stft(t, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7

        sc = torch.norm(mag_t - mag_s, p="fro") / (torch.norm(mag_t, p="fro") + 1e-8)
        lmag = F.l1_loss(torch.log(mag_t), torch.log(mag_s))
        sc_list.append(sc)
        lmag_list.append(lmag)

    mean_sc = torch.stack(sc_list).mean()
    mean_lmag = torch.stack(lmag_list).mean()
    stft_loss = mean_sc + mean_lmag

    # Waveform correlation (cosine similarity of normalized audio)
    s_flat = s.squeeze()
    t_flat = t.squeeze()
    norm_s = torch.norm(s_flat) + 1e-8
    norm_t = torch.norm(t_flat) + 1e-8
    corr = torch.dot(s_flat, t_flat) / (norm_s * norm_t)

    # Monotonic audio similarity score in (0, 1]
    similarity = 1.0 / (1.0 + stft_loss)

    return {
        "stft_loss": float(stft_loss.item()),
        "spectral_convergence": float(mean_sc.item()),
        "log_mag_loss": float(mean_lmag.item()),
        "waveform_correlation": float(corr.item()),
        "similarity": float(similarity.item()),
    }


def synthesize_candidate_tab_audio(pitches: List[float],
                                   sample_rate: int = 16000,
                                   duration: float = 0.5,
                                   decays: List[float] = None,
                                   energies: List[float] = None,
                                   stretches: List[float] = None,
                                   active_mask: List[bool] = None) -> torch.Tensor:
    """
    Synthesize the acoustic output audio for a candidate tab without optimization,
    using specified or physically sensible defaults for decay, energy, and stretch.
    """
    num_voices = len(pitches)
    if num_voices == 0:
        return torch.zeros(int(sample_rate * duration), device=DEVICE)
    if active_mask is None:
        active_mask = [True] * num_voices

    from karplus_strong import MultiVoiceKarplusStrong
    model = MultiVoiceKarplusStrong(num_voices, sample_rate=sample_rate, duration=duration).to(DEVICE)

    with torch.no_grad():
        for i, (voice, pitch, active) in enumerate(zip(model.voices, pitches, active_mask)):
            voice.log_pitch.data.fill_(math.log(max(pitch, 20.0)))
            if not active:
                voice.log_energy.data.fill_(-20.0)
                continue

            if decays is not None and i < len(decays):
                d = max(0.5001, min(0.9998, decays[i]))
                sig = (d - 0.5) / 0.4999
                voice.logit_decay.data.fill_(math.log(sig / (1.0 - sig)))

            if energies is not None and i < len(energies):
                voice.log_energy.data.fill_(energies[i])

            if stretches is not None and i < len(stretches):
                s = max(1e-6, min(0.1 - 1e-6, stretches[i]))
                sig = s / 0.1
                voice.logit_stretch.data.fill_(math.log(sig / (1.0 - sig)))

        synth = model()
    return synth


def optimize_parameters(target_audio: torch.Tensor, initial_pitch: float,
                        sample_rate: int = 16000,
                        num_iterations: int = 50, lr: float = 0.05,
                        return_audio: bool = False):
    """
    Adam optimisation loop that fits the Karplus-Strong model to target_audio.
    """
    duration = target_audio.shape[-1] / sample_rate
    model = DifferentiableKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    target_audio = target_audio.to(DEVICE)

    # Initialise log_pitch to the YIN estimate and FREEZE it.
    # Only decay and excitation_energy are optimized.
    log_pitch_anchor = math.log(max(initial_pitch, 20.0))
    with torch.no_grad():
        model.log_pitch.data.fill_(log_pitch_anchor)
    model.log_pitch.requires_grad_(False)  # Freeze pitch

    # Only optimise the timbre parameters
    optimizer = torch.optim.Adam(
        [model.logit_decay, model.log_energy, model.logit_stretch], lr=lr
    )

    print(f"Starting optimisation  (pitch fixed at {initial_pitch:.2f} Hz, optimising decay/energy/stretch) ...")

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
            p, d, _, _ = model.get_params()
            print(f"  iter {i+1:3d}  loss={loss.item():.4f}  "
                  f"pitch={p.item():.2f} Hz  decay={d.item():.4f}")

    # Compare synthesized candidate output audio against target input audio
    with torch.no_grad():
        final_synth = model()
    min_len = min(final_synth.shape[-1], target_audio.shape[-1])
    comparison = compare_candidate_audio(final_synth[:min_len], target_audio.squeeze()[:min_len], sample_rate)

    # For fretboard mapping, use the original YIN pitch (most reliable)
    # Decay and stretch give timbre info for string disambiguation
    _, decay, _, stretch = model.get_params()
    res_dict = {
        "pitch": initial_pitch,    # Use YIN estimate directly
        "decay": decay.item(),
        "inharmonicity": stretch.item(),
        "audio_loss": comparison["stft_loss"],
        "audio_similarity": comparison["similarity"],
        "audio_metrics": comparison,
    }

    if return_audio:
        return res_dict, final_synth[:min_len], comparison
    return res_dict


def optimize_multi_parameters(target_audio: torch.Tensor,
                              pitches: List[float],
                              sample_rate: int = 16000,
                              num_iterations: int = 50,
                              lr: float = 0.5,
                              active_mask: List[bool] = None,
                              initial_stretches: List[float] = None,
                              return_audio: bool = False) -> Union[List[dict], Tuple[List[dict], torch.Tensor, dict]]:
    """
    Adam optimisation loop that fits a multi-voice Karplus-Strong model to
    `target_audio` (a mixed signal of multiple simultaneous notes).

    All pitches are frozen at the supplied `pitches` estimates. Only decay,
    excitation_energy, and stretch for each voice are optimised.

    `active_mask` (optional): a list of bools, one per voice. Voices marked
    False are simulated but kept SILENT (excitation energy frozen at ~0) -
    used when the model simulates more strings than were actually strummed.

    `initial_stretches` (optional): initial inharmonicity values for each voice,
    typically derived from the physical fret positions of the candidate tab.

    `return_audio`: if True, returns (results, synth_audio, comparison_dict)
    comparing the candidate tab's synthesized output audio directly with
    target_audio.
    """
    num_voices = len(pitches)
    if num_voices == 0:
        if return_audio:
            return [], torch.zeros(0, device=DEVICE), {}
        return []
    if active_mask is None:
        active_mask = [True] * num_voices

    # Move target to the configured device
    target_audio = target_audio.to(DEVICE)

    duration = target_audio.shape[-1] / sample_rate
    from karplus_strong import MultiVoiceKarplusStrong
    model = MultiVoiceKarplusStrong(num_voices, sample_rate=sample_rate, duration=duration).to(DEVICE)

    # Initialise and freeze all pitches.
    optim_params = []
    for i, (voice, pitch, active) in enumerate(zip(model.voices, pitches, active_mask)):
        log_anchor = math.log(max(pitch, 20.0))
        with torch.no_grad():
            voice.log_pitch.data.fill_(log_anchor)
        voice.log_pitch.requires_grad_(False)

        if initial_stretches is not None and i < len(initial_stretches) and active:
            s = max(1e-6, min(0.1 - 1e-6, initial_stretches[i]))
            sig = s / 0.1
            with torch.no_grad():
                voice.logit_stretch.data.fill_(math.log(sig / (1.0 - sig)))

        if not active:
            # Silent simulated string: freeze excitation at ~0 so it never
            # contributes to the synthesised sound.
            with torch.no_grad():
                voice.log_energy.data.fill_(-20.0)
            voice.log_energy.requires_grad_(False)
            voice.logit_decay.requires_grad_(False)
            voice.logit_stretch.requires_grad_(False)
            continue
        
        # Add trainable parameters to optimiser list
        optim_params.extend([voice.logit_decay, voice.log_energy, voice.logit_stretch])

    # Jointly optimise the timbre parameters across all voices.
    if num_iterations > 0 and len(optim_params) > 0:
        optimizer = torch.optim.Adam(optim_params, lr=lr)

        pitch_str = ", ".join(f"{p:.2f} Hz" for p in pitches)
        print(f"Starting {num_voices}-voice optimisation "
              f"(pitches: [{pitch_str}], optimising decay/energy/stretch for each) ...")

        for i in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            synth = model()

            min_len = min(synth.shape[-1], target_audio.shape[-1])
            stft_loss = multi_scale_stft_loss(synth[:min_len],
                                              target_audio.squeeze()[:min_len])

            # Add a penalty for muting strings (low decay) to prevent the optimizer
            # from "hiding" wrong notes by dropping their acoustic energy.
            decay_penalty = 0.0
            for voice, active in zip(model.voices, active_mask):
                if active:
                    _, decay, _, _ = voice.get_params()
                    # A decay of 0.95 or lower mutes the string extremely fast.
                    decay_penalty += torch.nn.functional.relu(0.95 - decay)
                    
            loss = stft_loss + (10.0 * decay_penalty)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (i + 1) % 10 == 0:
                decays = [v.get_params()[1].item() for v in model.voices]
                decay_str = ", ".join(f"d{v+1}={decays[v]:.4f}" for v in range(num_voices))
                print(f"  iter {i+1:3d}  loss={loss.item():.4f}  (stft={stft_loss.item():.4f})  {decay_str}")

    # Synthesize the final candidate output audio and compare with target input audio
    with torch.no_grad():
        final_synth = model()
    min_len = min(final_synth.shape[-1], target_audio.shape[-1])
    comparison = compare_candidate_audio(
        final_synth[:min_len], target_audio.squeeze()[:min_len], sample_rate
    )

    # Build return dictionaries
    results = []
    for i, voice in enumerate(model.voices):
        _, decay, _, stretch = voice.get_params()
        results.append({
            "pitch": pitches[i],
            "decay": decay.item(),
            "inharmonicity": stretch.item(),
            "audio_loss": comparison["stft_loss"],
            "audio_similarity": comparison["similarity"],
            "audio_metrics": comparison,
        })
        print(f"   Voice {i+1} - Pitch: {pitches[i]:.2f} Hz, Decay: {decay.item():.4f}, Stretch: {stretch.item():.4f}")

    print(f"   Candidate Output Audio vs Input Audio: STFT Loss = {comparison['stft_loss']:.4f}, "
          f"Spectral Convergence = {comparison['spectral_convergence']:.4f}, "
          f"Similarity Score = {comparison['similarity']:.4f}")

    if return_audio:
        return results, final_synth[:min_len], comparison
    return results
