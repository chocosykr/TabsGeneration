import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ResearchKarplusStrong
from utils import multi_scale_stft_loss, add_calibrated_noise
from device_config import DEVICE

def run_noisy_stretch_investigation():
    print("=" * 70)
    print("EXPERIMENT NOI-002: Detailed 40 dB Stretch Loss Landscape & Optimizer Audit")
    print("=" * 70)

    sample_rate = 16000
    duration = 0.5
    pitch = 392.0
    true_decay = 0.990
    true_energy = 1.0
    true_stretch = 0.030

    torch.manual_seed(101)
    excitation = torch.randn(1024, device=DEVICE)

    # Clean target
    clean_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    clean_model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=true_stretch)
    with torch.no_grad():
        clean_target = clean_model(excitation=excitation)

    # Stretch evaluation grid
    stretch_grid = np.linspace(0.0005, 0.095, 100)
    eval_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)

    # 1. Clean landscape baseline
    clean_losses = []
    for s in stretch_grid:
        eval_model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=s)
        with torch.no_grad():
            synth = eval_model(excitation=excitation)
            l = multi_scale_stft_loss(synth, clean_target).item()
            clean_losses.append(l)

    # 2. Noisy landscapes across 3 seeds at 40 dB SNR
    noise_seeds = [1, 2, 3]
    noisy_landscapes = {}
    
    plt.figure(figsize=(12, 7))
    plt.plot(stretch_grid, clean_losses, 'k--', label='Clean Target Landscape (Baseline)', linewidth=2)

    for seed in noise_seeds:
        noisy_target = add_calibrated_noise(clean_target, snr_db=40.0, seed=seed)
        losses_at_seed = []
        for s in stretch_grid:
            eval_model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=s)
            with torch.no_grad():
                synth = eval_model(excitation=excitation)
                l = multi_scale_stft_loss(synth, noisy_target).item()
                losses_at_seed.append(l)
        noisy_landscapes[f"seed_{seed}"] = losses_at_seed
        
        min_idx = np.argmin(losses_at_seed)
        min_s = stretch_grid[min_idx]
        loss_at_gt = losses_at_seed[np.argmin(np.abs(stretch_grid - true_stretch))]
        print(f"Seed {seed} Landscape: Min at S = {min_s:.4f} (Loss = {losses_at_seed[min_idx]:.4f}) | Loss at GT (0.030) = {loss_at_gt:.4f} | Loss at S=0.095 = {losses_at_seed[-1]:.4f}")
        
        plt.plot(stretch_grid, losses_at_seed, label=f'40 dB SNR (Seed {seed}) - Min at S={min_s:.4f}')

    plt.axvline(true_stretch, color='green', linestyle=':', label='Ground Truth Stretch (0.030)', linewidth=2)
    plt.title("NOI-002: Multi-Scale STFT Loss vs Stretch under 40 dB Measurement Noise")
    plt.xlabel("Stretch Parameter")
    plt.ylabel("Multi-Scale STFT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/noi002_40db_stretch_landscapes.png", dpi=150)
    plt.close()

    # 3. Optimizer Trajectory Inspection (Track gradient at step 0)
    print("\n--- Optimizer Convergence & Gradient Diagnostic ---")
    opt_diag = {}
    for seed in noise_seeds:
        noisy_target = add_calibrated_noise(clean_target, snr_db=40.0, seed=seed)
        model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
        # Initialize at true stretch 0.030!
        model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=0.030)
        model.log_pitch.requires_grad_(False)
        model.logit_decay.requires_grad_(False)
        model.log_energy.requires_grad_(False)
        model.logit_stretch.requires_grad_(True)

        # Check gradient at true stretch
        synth = model(excitation=excitation)
        loss = multi_scale_stft_loss(synth, noisy_target)
        loss.backward()
        grad_at_gt = model.logit_stretch.grad.item()
        print(f"Seed {seed} at S=0.030: Loss={loss.item():.4f}, dLoss/dlogit={grad_at_gt:.6f}")
        opt_diag[f"seed_{seed}"] = {
            "loss_at_gt": loss.item(),
            "grad_at_gt": grad_at_gt
        }

    results = {
        "metadata": {
            "experiment_id": "NOI-002",
            "f0": pitch,
            "true_stretch": true_stretch,
            "snr_db": 40.0,
            "stretch_grid": stretch_grid.tolist()
        },
        "clean_landscape": clean_losses,
        "noisy_landscapes": noisy_landscapes,
        "optimizer_diagnostic": opt_diag
    }

    with open("research/identifiability/data/noi002_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved NOI-002 results to research/identifiability/data/noi002_results.json")
    print("Saved plot to research/identifiability/plots/noi002_40db_stretch_landscapes.png")

if __name__ == "__main__":
    run_noisy_stretch_investigation()
