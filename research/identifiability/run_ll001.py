import torch
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ResearchKarplusStrong
from utils import multi_scale_stft_loss
from device_config import DEVICE

def run_loss_landscape_experiment():
    print("=" * 60)
    print("RUNNING EXPERIMENT LL-001: Loss Landscape Topography")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.5
    f0_target = 392.0 # G4
    true_decay = 0.990
    true_energy = 1.0
    true_stretch = 0.030
    
    # 1. Deterministic Excitation
    torch.manual_seed(123)
    excitation = torch.randn(1024, device=DEVICE)
    
    # 2. Synthesize Ground Truth Target
    target_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    target_model.set_physical_params(pitch=f0_target, decay=true_decay, energy=true_energy, stretch=true_stretch)
    with torch.no_grad():
        target_audio = target_model(excitation=excitation)
    
    # 3. Model for Evaluation
    eval_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    
    # -------------------------------------------------------------
    # SWEEP 1: Stretch 1D Sweep (all other parameters fixed at truth)
    # -------------------------------------------------------------
    stretch_grid = np.linspace(0.0005, 0.095, 100)
    stretch_losses_fixed_exc = []
    stretch_losses_rand_exc = []
    
    print("\nEvaluating Stretch 1D Landscape around true_stretch = 0.030...")
    for s in stretch_grid:
        eval_model.set_physical_params(pitch=f0_target, decay=true_decay, energy=true_energy, stretch=s)
        with torch.no_grad():
            # A: Fixed excitation
            synth_fixed = eval_model(excitation=excitation)
            loss_fixed = multi_scale_stft_loss(synth_fixed, target_audio).item()
            stretch_losses_fixed_exc.append(loss_fixed)
            
            # B: Random excitation (to quantify stochastic noise floor)
            synth_rand = eval_model(excitation=None)
            loss_rand = multi_scale_stft_loss(synth_rand, target_audio).item()
            stretch_losses_rand_exc.append(loss_rand)
            
    # -------------------------------------------------------------
    # SWEEP 2: Decay 1D Sweep (all other parameters fixed at truth)
    # -------------------------------------------------------------
    decay_grid = np.linspace(0.90, 0.9995, 100)
    decay_losses_fixed_exc = []
    decay_losses_rand_exc = []
    
    print("Evaluating Decay 1D Landscape around true_decay = 0.990...")
    for d in decay_grid:
        eval_model.set_physical_params(pitch=f0_target, decay=d, energy=true_energy, stretch=true_stretch)
        with torch.no_grad():
            synth_fixed = eval_model(excitation=excitation)
            loss_fixed = multi_scale_stft_loss(synth_fixed, target_audio).item()
            decay_losses_fixed_exc.append(loss_fixed)
            
            synth_rand = eval_model(excitation=None)
            loss_rand = multi_scale_stft_loss(synth_rand, target_audio).item()
            decay_losses_rand_exc.append(loss_rand)

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    # Stretch Plot
    plt.figure(figsize=(10, 6))
    plt.plot(stretch_grid, stretch_losses_fixed_exc, 'b-', label='Fixed Excitation (Deterministic)', linewidth=2)
    plt.plot(stretch_grid, stretch_losses_rand_exc, 'r--', label='Random Excitation (Prototype behavior)', alpha=0.6)
    plt.axvline(true_stretch, color='green', linestyle=':', label=f'True Stretch ({true_stretch})', linewidth=2)
    min_idx_s = np.argmin(stretch_losses_fixed_exc)
    plt.plot(stretch_grid[min_idx_s], stretch_losses_fixed_exc[min_idx_s], 'go', markersize=8, label=f'Loss Min ({stretch_grid[min_idx_s]:.4f})')
    plt.title(f"LL-001: Multi-Scale STFT Loss vs. Stretch (Target S={true_stretch})")
    plt.xlabel("Stretch Parameter")
    plt.ylabel("Multi-Scale STFT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/ll001_stretch_landscape.png", dpi=150)
    plt.close()
    
    # Decay Plot
    plt.figure(figsize=(10, 6))
    plt.plot(decay_grid, decay_losses_fixed_exc, 'b-', label='Fixed Excitation (Deterministic)', linewidth=2)
    plt.plot(decay_grid, decay_losses_rand_exc, 'r--', label='Random Excitation (Prototype behavior)', alpha=0.6)
    plt.axvline(true_decay, color='green', linestyle=':', label=f'True Decay ({true_decay})', linewidth=2)
    min_idx_d = np.argmin(decay_losses_fixed_exc)
    plt.plot(decay_grid[min_idx_d], decay_losses_fixed_exc[min_idx_d], 'go', markersize=8, label=f'Loss Min ({decay_grid[min_idx_d]:.4f})')
    plt.title(f"LL-001: Multi-Scale STFT Loss vs. Decay (Target g={true_decay})")
    plt.xlabel("Decay Parameter")
    plt.ylabel("Multi-Scale STFT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/ll001_decay_landscape.png", dpi=150)
    plt.close()

    # Save data
    results = {
        "metadata": {
            "experiment_id": "LL-001",
            "f0_target": f0_target,
            "true_decay": true_decay,
            "true_energy": true_energy,
            "true_stretch": true_stretch
        },
        "stretch_sweep": {
            "grid": stretch_grid.tolist(),
            "loss_fixed": stretch_losses_fixed_exc,
            "loss_rand": stretch_losses_rand_exc,
            "arg_min_fixed": float(stretch_grid[min_idx_s]),
            "min_loss_fixed": float(stretch_losses_fixed_exc[min_idx_s])
        },
        "decay_sweep": {
            "grid": decay_grid.tolist(),
            "loss_fixed": decay_losses_fixed_exc,
            "loss_rand": decay_losses_rand_exc,
            "arg_min_fixed": float(decay_grid[min_idx_d]),
            "min_loss_fixed": float(decay_losses_fixed_exc[min_idx_d])
        }
    }
    
    with open("research/identifiability/data/ll001_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary of Loss Landscape Results ---")
    print(f"Stretch Ground Truth: {true_stretch}")
    print(f"Stretch Loss Minimum (Fixed Excitation): {stretch_grid[min_idx_s]:.4f} (Loss = {stretch_losses_fixed_exc[min_idx_s]:.6f})")
    print(f"Stretch Loss Dynamic Range (Fixed): Max {max(stretch_losses_fixed_exc):.4f} vs Min {min(stretch_losses_fixed_exc):.4f}")
    print(f"Stretch Loss Noise Floor (Rand Excitation): ~{np.mean(stretch_losses_rand_exc):.4f} (Swamps fixed landscape by {np.mean(stretch_losses_rand_exc)/max(stretch_losses_fixed_exc):.1f}x!)")
    print()
    print(f"Decay Ground Truth: {true_decay}")
    print(f"Decay Loss Minimum (Fixed Excitation): {decay_grid[min_idx_d]:.4f} (Loss = {decay_losses_fixed_exc[min_idx_d]:.6f})")
    print(f"Decay Loss Dynamic Range (Fixed): Max {max(decay_losses_fixed_exc):.4f} vs Min {min(decay_losses_fixed_exc):.4f}")

if __name__ == "__main__":
    run_loss_landscape_experiment()
