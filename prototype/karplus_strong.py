import torch
import torch.nn as nn
import math
from device_config import DEVICE

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

    def forward(self) -> torch.Tensor:
        """
        Synthesise audio via a fractional-delay Karplus-Strong loop,
        incorporating an all-pass filter for inharmonicity (stretch).
        Returns a 1-D tensor of shape (num_samples,).
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

        # Fixed noise excitation (scaled by energy)
        with torch.no_grad():
            noise = torch.randn(L_int, device=DEVICE)

        # Build signal as a Python list to avoid in-place tensor mutations
        y: list = [noise[i] * energy for i in range(L_int)]
        
        # State for the all-pass filter (scalar, same device as parameters)
        ap_prev_x = torch.tensor(0.0, device=DEVICE)
        ap_prev_y = torch.tensor(0.0, device=DEVICE)

        for t in range(L_int, self.num_samples):
            s1 = y[t - L_int]
            s2 = y[t - L_int - 1] if (t - L_int - 1) >= 0 else torch.zeros_like(s1)
            
            # 1. Fractional delay (linear interpolation) + Decay (low-pass)
            delayed = ((1.0 - L_frac) * s1 + L_frac * s2) * decay
            
            # 2. All-pass filter for inharmonicity (stretch)
            # Difference equation: y[n] = C * x[n] + x[n-1] - C * y[n-1]
            ap_y = stretch * delayed + ap_prev_x - stretch * ap_prev_y
            
            # Update state
            ap_prev_x = delayed
            ap_prev_y = ap_y
            
            y.append(ap_y)

        return torch.stack(y)


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

    def forward(self) -> torch.Tensor:
        """Return the sum of all voice signals, length = num_samples."""
        if self.num_voices == 0:
            return torch.zeros(0)
            
        outputs = [voice() for voice in self.voices]
        
        # Align lengths (they should match, but guard against off-by-one)
        min_len = min(y.shape[0] for y in outputs)
        
        # Sum all outputs
        mixed_signal = torch.zeros(min_len, device=DEVICE)
        for y in outputs:
            mixed_signal += y[:min_len]
            
        return mixed_signal
