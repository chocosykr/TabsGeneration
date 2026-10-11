import torch
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ResearchKarplusStrong
from utils import multi_scale_stft_loss, add_calibrated_noise
from device_config import DEVICE

def run_noise_sweeps():
    print("=" * 60)
    print("RUNNING EXPERIMENT NOI-001: Calibrated SNR Noise Robustness")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.5
    pitch = 392.0
    true_decay = 0.990
    true_energy = 1.0
    true_stretch = 0.030 # Fret 3 equivalent
    
    torch.manual_seed(101)
    excitation = torch.randn(1024, device=DEVICE)
    
    # Ground truth clean target
    target_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    target_model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=true_stretch)
    with torch.no_grad():
        clean_target = target_model(excitation=excitation)
        
    snr_levels = [None, 40.0, 30.0, 20.0, 10.0]
    seeds = [1, 2, 3] # 3 independent noise seeds per level
    
    results = {"snr_trials": []}
    
    for snr in snr_levels:
        snr_label = "Clean" if snr is None else f"{snr} dB"
        print(f"\n--- Testing SNR: {snr_label} ---")
        
        for seed in seeds:
            # Add calibrated noise
            noisy_target = add_calibrated_noise(clean_target, snr, seed=seed)
            
            # Isolated stretch recovery from init=0.005
            model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
            model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=0.005)
            model.log_pitch.requires_grad_(False)
            model.log_energy.requires_grad_(False)
            model.logit_decay.requires_grad_(False)
            model.logit_stretch.requires_grad_(True)
            
            opt = torch.optim.Adam([model.logit_stretch], lr=0.08)
            for _ in range(80):
                opt.zero_grad()
                synth = model(excitation=excitation)
                loss = multi_scale_stft_loss(synth, noisy_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.logit_stretch], 1.0)
                opt.step()
                
            _, _, _, rec_s = model.get_physical_params()
            rec_val = rec_s.item()
            err = abs(true_stretch - rec_val)
            trial_data = {
                "snr_db": snr,
                "seed": seed,
                "recovered_stretch": rec_val,
                "abs_error": err,
                "final_loss": loss.item()
            }
            results["snr_trials"].append(trial_data)
            print(f"Seed {seed} | Rec: {rec_val:.4f} | Err: {err:.4f} | Loss: {loss.item():.4f}")

    with open("research/identifiability/data/noi001_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved NOI-001 results to research/identifiability/data/noi001_results.json")

if __name__ == "__main__":
    run_noise_sweeps()
