import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import os
import sys

# Ensure prototype folder is in path for imports if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../prototype')))
from device_config import DEVICE

class ResearchKarplusStrong(nn.Module):
    """
    Research-grade Differentiable Karplus-Strong string model.
    Key properties:
    1. Supports injection of deterministic excitation vector: forward(excitation=...)
       Ensures exact reproducible comparisons between target and candidate synthesis.
    2. Master excitation slicing: accepts excitation longer than L_int and slices :L_int.
    3. Exposes explicit get_physical_params() and set_physical_params().
    4. Exact reproduction of the prototype's difference equations and delay logic.
    """
    def __init__(self, sample_rate: int = 16000, duration: float = 0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)

        # Learnable parameters (same transformations as prototype)
        self.log_pitch = nn.Parameter(torch.tensor(math.log(440.0), device=DEVICE))
        
        # Initialise decay near 0.99
        init_sigmoid = (0.99 - 0.5) / 0.4999
        self.logit_decay = nn.Parameter(torch.tensor(
            math.log(init_sigmoid / (1.0 - init_sigmoid)), device=DEVICE))
        
        # Excitation energy
        self.log_energy = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        
        # Stretch initialized near 0
        self.logit_stretch = nn.Parameter(torch.tensor(-5.0, device=DEVICE))

    def get_physical_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute interpretable physical values from raw logits."""
        pitch = torch.exp(self.log_pitch).clamp(20.0, self.sample_rate / 2 - 1)
        decay = torch.sigmoid(self.logit_decay) * 0.4999 + 0.5   # (0.5, 0.9999)
        energy = F.softplus(self.log_energy) + 1e-4
        stretch = torch.sigmoid(self.logit_stretch) * 0.1         # (0.0, 0.1)
        return pitch, decay, energy, stretch

    def set_physical_params(self, pitch: Optional[float] = None, 
                            decay: Optional[float] = None, 
                            energy: Optional[float] = None, 
                            stretch: Optional[float] = None):
        """Helper to accurately set physical parameters via inverse logit/exp."""
        with torch.no_grad():
            if pitch is not None:
                self.log_pitch.fill_(math.log(max(pitch, 20.0)))
            if decay is not None:
                decay_clamped = max(0.5001, min(0.9998, decay))
                sig = (decay_clamped - 0.5) / 0.4999
                self.logit_decay.fill_(math.log(sig / (1.0 - sig)))
            if energy is not None:
                # softplus inverse: x = log(exp(y - 1e-4) - 1)
                eff_energy = max(energy - 1e-4, 1e-6)
                if eff_energy > 20.0:
                    self.log_energy.fill_(eff_energy)
                else:
                    self.log_energy.fill_(math.log(math.exp(eff_energy) - 1.0))
            if stretch is not None:
                # stretch = sigmoid(z) * 0.1 -> sigmoid(z) = stretch / 0.1
                s_clamped = max(1e-6, min(0.1 - 1e-6, stretch))
                sig = s_clamped / 0.1
                self.logit_stretch.fill_(math.log(sig / (1.0 - sig)))

    def forward(self, excitation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Synthesise audio via Karplus-Strong loop.
        If excitation is None, samples random noise (matching prototype behavior).
        If excitation is provided, slices excitation[:L_int] for deterministic runs.
        """
        pitch, decay, energy, stretch = self.get_physical_params()

        # Prototype all-pass group delay compensation at DC
        ap_group_delay = (1.0 - stretch) / (1.0 + stretch)
        L = (self.sample_rate / pitch) - ap_group_delay
        
        L_int = max(int(torch.floor(L).item()), 2)
        L_frac = L - L_int

        if excitation is None:
            with torch.no_grad():
                noise = torch.randn(L_int, device=DEVICE)
        else:
            if excitation.shape[0] < L_int:
                raise ValueError(f"Excitation buffer length ({excitation.shape[0]}) < required L_int ({L_int})")
            noise = excitation[:L_int].to(DEVICE)

        # Initial delay line state scaled by energy
        y: list = [noise[i] * energy for i in range(L_int)]
        
        # State for all-pass filter
        ap_prev_x = torch.tensor(0.0, device=DEVICE)
        ap_prev_y = torch.tensor(0.0, device=DEVICE)

        for t in range(L_int, self.num_samples):
            s1 = y[t - L_int]
            s2 = y[t - L_int - 1] if (t - L_int - 1) >= 0 else torch.zeros_like(s1)
            
            # 1. Fractional delay (linear interpolation) + Decay (loop gain)
            delayed = ((1.0 - L_frac) * s1 + L_frac * s2) * decay
            
            # 2. All-pass filter for inharmonicity (stretch)
            # y[n] = stretch * x[n] + x[n-1] - stretch * y[n-1]
            ap_y = stretch * delayed + ap_prev_x - stretch * ap_prev_y
            
            ap_prev_x = delayed
            ap_prev_y = ap_y
            
            y.append(ap_y)

        return torch.stack(y)
