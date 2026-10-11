import torch
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ResearchKarplusStrong
from utils import estimate_partials_parabolic
from device_config import DEVICE

def run_forward_model_audit():
    print("=" * 60)
    print("RUNNING EXPERIMENT FM-001: Forward Model Harmonic Audit")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.5
    f0_target = 392.0 # G4
    decay_target = 0.990
    energy_target = 1.0
    
    stretch_values = [0.0, 0.01, 0.03, 0.05, 0.08]
    
    # 1. Verify exact reproducibility with fixed excitation
    torch.manual_seed(42)
    master_excitation = torch.randn(1024, device=DEVICE)
    
    model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    model.set_physical_params(pitch=f0_target, decay=decay_target, energy=energy_target, stretch=0.03)
    
    out1 = model(excitation=master_excitation)
    out2 = model(excitation=master_excitation)
    diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"Reproducibility verification (max diff between identical forward calls): {diff:.2e}")
    assert diff == 0.0, "Forward model must be strictly deterministic with fixed excitation!"

    # Results dictionary
    audit_results = {
        "metadata": {
            "experiment_id": "FM-001",
            "f0_target": f0_target,
            "sample_rate": sample_rate,
            "duration": duration,
            "decay": decay_target,
            "energy": energy_target
        },
        "trials": []
    }

    plt.figure(figsize=(12, 8))
    
    for s_val in stretch_values:
        # Note: if s_val == 0.0, let's see how the model behaves
        # In ResearchKarplusStrong, set_physical_params handles s_val=0.0 cleanly.
        model.set_physical_params(pitch=f0_target, decay=decay_target, energy=energy_target, stretch=s_val)
        
        # Log effective delay properties
        pitch, decay, energy, stretch = model.get_physical_params()
        s_actual = stretch.item()
        
        ap_group_delay = (1.0 - s_actual) / (1.0 + s_actual)
        L = (sample_rate / pitch.item()) - ap_group_delay
        L_int = max(int(np.floor(L)), 2)
        L_frac = L - L_int
        
        audio = model(excitation=master_excitation).detach().cpu().numpy()
        
        # Estimate first 6 partials
        partials = estimate_partials_parabolic(audio, sample_rate, f0_target, num_partials=6)
        f1 = partials[0]
        
        # Calculate normalized ratios r_n = f_n / (n * f_1)
        ratios = [f_n / ((idx + 1) * f1) if not np.isnan(f_n) and not np.isnan(f1) else np.nan 
                  for idx, f_n in enumerate(partials)]
        
        # Compute RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))
        
        trial_data = {
            "stretch_requested": s_val,
            "stretch_actual": s_actual,
            "L": float(L),
            "L_int": int(L_int),
            "L_frac": float(L_frac),
            "ap_group_delay_dc": float(ap_group_delay),
            "f1": float(f1),
            "f1_drift_hz": float(f1 - f0_target),
            "partials": partials,
            "ratios": ratios,
            "rms": rms
        }
        audit_results["trials"].append(trial_data)
        
        print(f"\nStretch: {s_val:.4f} (Actual: {s_actual:.6f})")
        print(f"  Delay: L={L:.4f}, L_int={L_int}, L_frac={L_frac:.4f}, AP_GD={ap_group_delay:.4f}")
        print(f"  f1: {f1:.2f} Hz (Drift from {f0_target:.1f}Hz: {f1 - f0_target:+.2f} Hz)")
        partials_str = ", ".join(f"{p:.1f}" for p in partials)
        print(f"  Partials [f1..f6]: [{partials_str}]")
        ratios_str = ", ".join(f"{r:.4f}" for r in ratios)
        print(f"  Ratios r_n = f_n / (n*f1): [{ratios_str}]")

        # Plot spectrum
        fft_mag = np.abs(np.fft.rfft(audio, n=65536))
        freqs = np.fft.rfftfreq(65536, d=1.0/sample_rate)
        plt.plot(freqs, 20 * np.log10(np.maximum(fft_mag, 1e-6)), label=f"Stretch={s_val:.2f} (f1={f1:.1f}Hz)", alpha=0.7)

    plt.xlim(0, 3000)
    plt.ylim(-20, 80)
    plt.title(f"FM-001: Spectra Across Stretch Values (f0={f0_target} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/fm001_spectra.png", dpi=150)
    plt.close()

    # Plot partial inharmonicity ratios
    plt.figure(figsize=(10, 6))
    for trial in audit_results["trials"]:
        plt.plot(range(1, 7), trial["ratios"], marker='o', label=f"Stretch={trial['stretch_requested']}")
    plt.axhline(1.0, color='black', linestyle='--', alpha=0.5, label="Perfect Harmonicity (1.0)")
    plt.title("FM-001: Harmonic Partial Ratio r_n = f_n / (n * f_1)")
    plt.xlabel("Partial Index n")
    plt.ylabel("r_n")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/fm001_inharmonicity_ratios.png", dpi=150)
    plt.close()

    # Save JSON raw data
    with open("research/identifiability/data/fm001_results.json", "w") as f:
        json.dump(audit_results, f, indent=2)
    print("\nSaved FM-001 results to research/identifiability/data/fm001_results.json")
    print("Saved plots to research/identifiability/plots/fm001_*.png")

if __name__ == "__main__":
    run_forward_model_audit()
