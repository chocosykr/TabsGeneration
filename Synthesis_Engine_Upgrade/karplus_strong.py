import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, List
from device_config import DEVICE

try:
    import numba
    @numba.njit(fastmath=True)
    def _ks_core_sim(L_int: int, L_frac: float, decay: float, stretch: float, noise: np.ndarray, n: int) -> np.ndarray:
        y = np.zeros(n, dtype=np.float32)
        for i in range(L_int):
            y[i] = noise[i]
        ap_prev_x = 0.0
        ap_prev_y = 0.0
        for t in range(L_int, n):
            s1 = y[t - L_int]
            s2 = y[t - L_int - 1] if (t - L_int - 1) >= 0 else 0.0
            delayed = ((1.0 - L_frac) * s1 + L_frac * s2) * decay
            ap_y = stretch * delayed + ap_prev_x - stretch * ap_prev_y
            ap_prev_x = delayed
            ap_prev_y = ap_y
            y[t] = ap_y
        return y
except Exception:
    def _ks_core_sim(L_int: int, L_frac: float, decay: float, stretch: float, noise: np.ndarray, n: int) -> np.ndarray:
        y = np.zeros(n, dtype=np.float32)
        y[:L_int] = noise[:L_int]
        ap_prev_x = 0.0
        ap_prev_y = 0.0
        for t in range(L_int, n):
            s1 = y[t - L_int]
            s2 = y[t - L_int - 1] if (t - L_int - 1) >= 0 else 0.0
            delayed = ((1.0 - L_frac) * s1 + L_frac * s2) * decay
            ap_y = stretch * delayed + ap_prev_x - stretch * ap_prev_y
            ap_prev_x = delayed
            ap_prev_y = ap_y
            y[t] = ap_y
        return y


class DifferentiableKSFunction(torch.autograd.Function):
    """
    High-performance autograd function for Karplus-Strong string synthesis.
    Executes the sequential recursive loop in microsecond native code,
    returning tensors on the active GPU device for accelerated loss & STFT evaluation.
    """
    @staticmethod
    def forward(ctx, L_int: int, L_frac_t: torch.Tensor, decay_t: torch.Tensor,
                stretch_t: torch.Tensor, energy_t: torch.Tensor, noise_t: torch.Tensor,
                num_samples: int) -> torch.Tensor:
        device = decay_t.device
        L_frac = float(L_frac_t.item())
        decay = float(decay_t.item())
        stretch = float(stretch_t.item())
        energy = float(energy_t.item())
        noise = noise_t.detach().cpu().numpy().astype(np.float32)

        y_base = _ks_core_sim(L_int, L_frac, decay, stretch, noise, num_samples)
        y = y_base * energy

        ctx.save_for_backward(decay_t, stretch_t, energy_t, noise_t)
        ctx.L_int = L_int
        ctx.L_frac = L_frac
        ctx.num_samples = num_samples
        ctx.y_base = y_base

        return torch.from_numpy(y).to(device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        decay_t, stretch_t, energy_t, noise_t = ctx.saved_tensors
        device = grad_output.device
        decay = float(decay_t.item())
        stretch = float(stretch_t.item())
        energy = float(energy_t.item())
        noise = noise_t.detach().cpu().numpy().astype(np.float32)
        L_int = ctx.L_int
        L_frac = ctx.L_frac
        num_samples = ctx.num_samples
        y_base = ctx.y_base

        # Exact analytical gradient for energy scalar
        y_base_t = torch.from_numpy(y_base).to(device)
        grad_energy = torch.dot(grad_output, y_base_t)

        # Central difference for decay
        eps_d = 1e-4
        y_d_plus = _ks_core_sim(L_int, L_frac, decay + eps_d, stretch, noise, num_samples)
        y_d_minus = _ks_core_sim(L_int, L_frac, decay - eps_d, stretch, noise, num_samples)
        dy_d = torch.from_numpy(((y_d_plus - y_d_minus) / (2.0 * eps_d)) * energy).to(device)
        grad_decay = torch.dot(grad_output, dy_d)

        # Central difference for stretch
        eps_s = 1e-4
        y_s_plus = _ks_core_sim(L_int, L_frac, decay, stretch + eps_s, noise, num_samples)
        y_s_minus = _ks_core_sim(L_int, L_frac, decay, stretch - eps_s, noise, num_samples)
        dy_s = torch.from_numpy(((y_s_plus - y_s_minus) / (2.0 * eps_s)) * energy).to(device)
        grad_stretch = torch.dot(grad_output, dy_s)

        return None, None, grad_decay, grad_stretch, grad_energy, None, None


class DifferentiableKarplusStrong(nn.Module):
    """
    Differentiable Karplus-Strong string model.
    Uses log-pitch parameterisation so the raw parameter corresponds exactly 
    to log(frequency_in_Hz), giving smooth, unconstrained gradients without
    any clamping or softplus wrapping that would confuse the initialiser.
    """
    def __init__(self, sample_rate: int = 16000, duration: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)

        # Store log(pitch_hz) so exp() always gives a positive frequency.
        # Initialise at A4 = 440 Hz.
        self.log_pitch = nn.Parameter(torch.tensor(math.log(440.0), device=DEVICE))
        
        # Widen decay range to [0.5, 0.9999] to prevent saturation.
        # To initialize near 0.99, sigmoid needs to be ~0.98
        init_sigmoid = (0.99 - 0.5) / 0.4999
        self.logit_decay = nn.Parameter(torch.tensor(
            math.log(init_sigmoid / (1 - init_sigmoid)), device=DEVICE))
        
        # Excitation energy via softplus for positivity.
        self.log_energy = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        
        # Inharmonicity (stretch) parameter for fret disambiguation.
        # Initialise near 0.
        self.logit_stretch = nn.Parameter(torch.tensor(-5.0, device=DEVICE))

    def get_params(self):
        """Return the physical (interpretable) parameter values."""
        pitch = torch.exp(self.log_pitch).clamp(20.0, self.sample_rate / 2 - 1)
        decay = torch.sigmoid(self.logit_decay) * 0.4999 + 0.5   # in (0.5, 0.9999)
        energy = torch.nn.functional.softplus(self.log_energy) + 1e-4
        stretch = torch.sigmoid(self.logit_stretch) * 0.1 # bounded small stretch
        return pitch, decay, energy, stretch

    def forward(self, excitation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Synthesise audio via a fractional-delay Karplus-Strong loop,
        incorporating an all-pass filter for inharmonicity (stretch).
        Returns a 1-D tensor of shape (num_samples,).

        If excitation is provided, slices excitation[:L_int] for deterministic
        reproducible audio comparison.
        """
        pitch, decay, energy, stretch = self.get_params()

        # The all-pass filter H(z) = (C + z^-1) / (1 + C*z^-1) introduces a 
        # phase delay that alters the tuning. At DC (0 Hz), its group delay 
        # is (1 - C)/(1 + C). We subtract this from the target delay line length 
        # so the fundamental frequency remains perfectly in tune, forcing the 
        # stretch parameter to only affect the inharmonicity of upper overtones.
        ap_group_delay = (1.0 - stretch) / (1.0 + stretch)
        L = (self.sample_rate / pitch) - ap_group_delay
        
        L_int = max(int(torch.floor(L).item()), 2)  # never below 2 samples
        L_frac = L - L_int

        if excitation is None:
            with torch.no_grad():
                noise = torch.randn(L_int, device=DEVICE)
        else:
            if excitation.shape[0] < L_int:
                # Pad if slightly shorter
                pad_len = L_int - excitation.shape[0]
                noise = torch.cat([excitation.to(DEVICE), torch.zeros(pad_len, device=DEVICE)])
            else:
                noise = excitation[:L_int].to(DEVICE)

        return DifferentiableKSFunction.apply(
            L_int, L_frac, decay, stretch, energy, noise, self.num_samples
        )


class MultiVoiceKarplusStrong(nn.Module):
    """
    Multiple independent Karplus-Strong string models whose outputs are summed.

    Each voice has its own pitch, decay, and energy parameters. During
    optimisation the pitches are typically frozen while decay and energy 
    are jointly trained against the mixed target audio.
    """

    def __init__(self, num_voices: int, sample_rate: int = 16000, duration: float = 1.0):
        super().__init__()
        self.num_voices = num_voices
        self.voices = nn.ModuleList([
            DifferentiableKarplusStrong(sample_rate, duration)
            for _ in range(num_voices)
        ])

    def forward(self, excitations: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Return the sum of all voice signals, length = num_samples."""
        if self.num_voices == 0:
            return torch.zeros(0, device=DEVICE)
            
        if excitations is not None:
            outputs = [
                voice(excitation=excitations[i] if i < len(excitations) else None)
                for i, voice in enumerate(self.voices)
            ]
        else:
            outputs = [voice() for voice in self.voices]
        
        # Align lengths and sum across voices on GPU
        min_len = min(y.shape[0] for y in outputs)
        stacked = torch.stack([y[:min_len] for y in outputs], dim=0)
        return stacked.sum(dim=0)
