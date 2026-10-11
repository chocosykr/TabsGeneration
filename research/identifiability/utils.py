import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal
from typing import Tuple, List, Dict, Optional
import math
from device_config import DEVICE

def multi_scale_stft_loss(synth: torch.Tensor, target: torch.Tensor,
                          fft_sizes=[512, 1024, 2048]) -> torch.Tensor:
    """
    Standard multi-scale STFT loss (Spectral Convergence + Log Magnitude)
    identical to prototype/optimization.py, guaranteeing exact consistency.
    """
    if synth.ndim == 1:
        synth = synth.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

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

def add_calibrated_noise(signal: torch.Tensor, snr_db: Optional[float], seed: Optional[int] = None) -> torch.Tensor:
    """
    Add Gaussian noise at a strictly calibrated Signal-to-Noise Ratio (SNR in dB).
    If snr_db is None or inf, returns clean signal.
    """
    if snr_db is None or math.isinf(snr_db):
        return signal.clone()
    
    if seed is not None:
        torch.manual_seed(seed)

    signal_power = torch.mean(signal ** 2)
    if signal_power == 0:
        return signal.clone()

    # SNR = 10 * log10(P_signal / P_noise) -> P_noise = P_signal / (10^(SNR/10))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)
    
    noise = torch.randn_like(signal) * noise_std
    return signal + noise

def estimate_partials_parabolic(waveform: np.ndarray, sample_rate: int, 
                                f0_nominal: float, num_partials: int = 6, 
                                n_fft: int = 65536) -> List[float]:
    """
    Estimate harmonic partial frequencies using Blackman-Harris windowing,
    high-resolution zero-padded FFT, and parabolic peak interpolation.
    
    Parabolic interpolation refinement:
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    f_refined = (k + p) * (sample_rate / n_fft)
    """
    # Use windowed signal
    window = scipy.signal.windows.blackmanharris(len(waveform))
    windowed = waveform * window
    
    # Zero pad for dense frequency grid
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    
    detected_partials = []
    
    for n in range(1, num_partials + 1):
        target_f = n * f0_nominal
        # Search window +/- 15% around nominal harmonic
        f_low = target_f * 0.85
        f_high = target_f * 1.15
        
        idx_low = np.searchsorted(freqs, f_low)
        idx_high = np.searchsorted(freqs, f_high)
        
        if idx_high <= idx_low or idx_high >= len(spectrum) - 1:
            detected_partials.append(np.nan)
            continue
            
        local_peak_idx = idx_low + np.argmax(spectrum[idx_low:idx_high])
        
        # Parabolic interpolation on log-magnitude
        k = local_peak_idx
        if 0 < k < len(spectrum) - 1:
            alpha = np.log(max(spectrum[k - 1], 1e-12))
            beta  = np.log(max(spectrum[k], 1e-12))
            gamma = np.log(max(spectrum[k + 1], 1e-12))
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-12:
                p = 0.5 * (alpha - gamma) / denom
                f_interp = (k + p) * (sample_rate / n_fft)
            else:
                f_interp = freqs[k]
        else:
            f_interp = freqs[k]
            
        detected_partials.append(float(f_interp))
        
    return detected_partials
