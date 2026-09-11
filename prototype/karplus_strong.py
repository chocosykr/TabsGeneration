import torch
import torch.nn as nn
import math

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
        self.log_pitch = nn.Parameter(torch.tensor(math.log(440.0)))
        # Store logit(decay_norm) where decay_norm in (0,1) maps via sigmoid.
        # Initialise at decay=0.99.
        self.logit_decay = nn.Parameter(torch.tensor(math.log(0.99 / (1 - 0.99))))
        # Excitation energy via softplus for positivity.
        self.log_energy = nn.Parameter(torch.tensor(0.0))

    def get_params(self):
        """Return the physical (interpretable) parameter values."""
        pitch = torch.exp(self.log_pitch).clamp(20.0, self.sample_rate / 2 - 1)
        decay = torch.sigmoid(self.logit_decay) * 0.499 + 0.5   # in (0.5, 0.999)
        energy = torch.nn.functional.softplus(self.log_energy) + 1e-4
        return pitch, decay, energy

    def forward(self) -> torch.Tensor:
        """
        Synthesise audio via a fractional-delay Karplus-Strong loop.
        Returns a 1-D tensor of shape (num_samples,).
        """
        pitch, decay, energy = self.get_params()

        # Delay length L (continuous) and its integer/fractional parts
        L = self.sample_rate / pitch
        L_int = max(int(torch.floor(L).item()), 2)  # never below 2 samples
        L_frac = L - L_int

        # Fixed noise excitation (scaled by energy)
        with torch.no_grad():
            noise = torch.randn(L_int)

        # Build signal as a Python list to avoid in-place tensor mutations
        # which break autograd.
        y: list = [noise[i] * energy for i in range(L_int)]

        for t in range(L_int, self.num_samples):
            s1 = y[t - L_int]
            s2 = y[t - L_int - 1] if (t - L_int - 1) >= 0 else torch.zeros_like(s1)
            delayed = (1.0 - L_frac) * s1 + L_frac * s2
            y.append(delayed * decay)

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
        mixed_signal = torch.zeros(min_len, device=outputs[0].device)
        for y in outputs:
            mixed_signal += y[:min_len]
            
        return mixed_signal
