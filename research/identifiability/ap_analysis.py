import torch
import numpy as np

# Sample rate & pitch
sr = 16000
f0 = 392.0
partials = [392.0, 784.0, 1176.0, 1568.0, 1960.0, 2352.0]

for s in [0.001, 0.01, 0.03, 0.05, 0.08]:
    gd_dc = (1.0 - s) / (1.0 + s)
    # Loop over partials
    phase_delays = []
    group_delays = []
    for fn in partials:
        w = 2 * np.pi * fn / sr
        # H(e^jw) = (s + e^-jw) / (1 + s*e^-jw)
        num = s + np.exp(-1j * w)
        den = 1.0 + s * np.exp(-1j * w)
        H = num / den
        phase = np.angle(H)
        # Phase delay D_p(w) = -phase / w
        Dp = -phase / w
        # Group delay D_g(w) = (1 - s^2) / (1 + s^2 + 2*s*cos(w))
        Dg = (1 - s**2) / (1 + s**2 + 2 * s * np.cos(w))
        phase_delays.append(Dp)
        group_delays.append(Dg)
    print(f"Stretch {s:.3f}: DC_GD={gd_dc:.4f}")
    print(f"   Phase delays at partials: {[round(p, 4) for p in phase_delays]}")
    print(f"   Delta vs DC: {[round(p - gd_dc, 5) for p in phase_delays]}")
