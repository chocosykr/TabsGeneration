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

def optimize_single_parameter(target_audio: torch.Tensor,
                              pitch: float,
                              param_name: str,
                              initial_val: float,
                              fixed_decay: float = 0.990,
                              fixed_energy: float = 1.0,
                              fixed_stretch: float = 0.030,
                              excitation: torch.Tensor = None,
                              sample_rate: int = 16000,
                              duration: float = 0.5,
                              num_iterations: int = 80,
                              lr: float = 0.05):
    """
    Isolated optimizer fitting ONLY a single parameter (either 'stretch' or 'decay').
    Freezes all other parameters strictly.
    """
    model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    model.set_physical_params(pitch=pitch, decay=fixed_decay, energy=fixed_energy, stretch=fixed_stretch)
    
    # Freeze everything except target parameter
    model.log_pitch.requires_grad_(False)
    model.log_energy.requires_grad_(False)
    
    if param_name == "stretch":
        model.logit_decay.requires_grad_(False)
        model.set_physical_params(stretch=initial_val)
        model.logit_stretch.requires_grad_(True)
        trainable = [model.logit_stretch]
    elif param_name == "decay":
        model.logit_stretch.requires_grad_(False)
        model.set_physical_params(decay=initial_val)
        model.logit_decay.requires_grad_(True)
        trainable = [model.logit_decay]
    else:
        raise ValueError(f"Unknown param {param_name}")
        
    optimizer = torch.optim.Adam(trainable, lr=lr)
    
    loss_history = []
    val_history = []
    
    for it in range(num_iterations):
        optimizer.zero_grad()
        synth = model(excitation=excitation)
        loss = multi_scale_stft_loss(synth, target_audio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        _, d, _, s = model.get_physical_params()
        if param_name == "stretch":
            val_history.append(s.item())
        else:
            val_history.append(d.item())
            
    final_val = val_history[-1]
    return {
        "final_val": final_val,
        "final_loss": loss_history[-1],
        "initial_val": initial_val,
        "loss_history": loss_history,
        "val_history": val_history
    }

def run_isolated_recovery_experiments():
    print("=" * 60)
    print("RUNNING EXPERIMENT ST-001 (Stretch Recovery) & DC-001 (Decay Recovery)")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.5
    pitch = 392.0
    true_energy = 1.0
    
    # Deterministic excitation
    torch.manual_seed(777)
    excitation = torch.randn(1024, device=DEVICE)
    
    # ------------------------------------------------------------------
    # PART A: STRETCH-ONLY RECOVERY (ST-001)
    # Ground truth values: 0.001 (fret 0), 0.030 (fret 3), 0.080 (fret 7)
    # Initializations tested for each: [0.005, 0.020, 0.060]
    # ------------------------------------------------------------------
    true_stretches = [0.001, 0.030, 0.080]
    initial_stretches = [0.005, 0.020, 0.060]
    fixed_decay = 0.990
    
    st_results = {"trials": []}
    
    print("\n--- Running Stretch-Only Recovery Trials (ST-001) ---")
    for s_true in true_stretches:
        # Generate target
        t_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
        t_model.set_physical_params(pitch=pitch, decay=fixed_decay, energy=true_energy, stretch=s_true)
        with torch.no_grad():
            target_audio = t_model(excitation=excitation)
            
        for s_init in initial_stretches:
            res = optimize_single_parameter(
                target_audio, pitch=pitch, param_name="stretch",
                initial_val=s_init, fixed_decay=fixed_decay, fixed_energy=true_energy,
                excitation=excitation, lr=0.08, num_iterations=80
            )
            err = abs(s_true - res["final_val"])
            trial = {
                "s_true": s_true,
                "s_init": s_init,
                "s_rec": res["final_val"],
                "abs_error": err,
                "rel_error": err / s_true if s_true > 0 else 0.0,
                "final_loss": res["final_loss"],
                "loss_history": res["loss_history"],
                "val_history": res["val_history"]
            }
            st_results["trials"].append(trial)
            print(f"GT: {s_true:.3f} | Init: {s_init:.3f} -> Rec: {res['final_val']:.4f} | Err: {err:.4f} | Loss: {res['final_loss']:.4f}")

    # Plot ST-001 convergence curves
    plt.figure(figsize=(12, 6))
    for trial in st_results["trials"]:
        plt.plot(trial["val_history"], label=f"GT={trial['s_true']}, Init={trial['s_init']}")
    plt.title("ST-001: Stretch-Only Recovery Trajectories Across Initializations")
    plt.xlabel("Iteration")
    plt.ylabel("Recovered Stretch")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/st001_trajectories.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # PART B: DECAY-ONLY RECOVERY (DC-001)
    # Ground truth values: 0.950, 0.980, 0.995, 0.999
    # Initializations tested for each: [0.930, 0.970, 0.998]
    # ------------------------------------------------------------------
    true_decays = [0.950, 0.980, 0.995, 0.999]
    initial_decays = [0.930, 0.970, 0.998]
    fixed_stretch = 0.030
    
    dc_results = {"trials": []}
    
    print("\n--- Running Decay-Only Recovery Trials (DC-001) ---")
    for d_true in true_decays:
        t_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
        t_model.set_physical_params(pitch=pitch, decay=d_true, energy=true_energy, stretch=fixed_stretch)
        with torch.no_grad():
            target_audio = t_model(excitation=excitation)
            
        for d_init in initial_decays:
            res = optimize_single_parameter(
                target_audio, pitch=pitch, param_name="decay",
                initial_val=d_init, fixed_decay=d_init, fixed_energy=true_energy,
                fixed_stretch=fixed_stretch, excitation=excitation, lr=0.08, num_iterations=80
            )
            err = abs(d_true - res["final_val"])
            trial = {
                "d_true": d_true,
                "d_init": d_init,
                "d_rec": res["final_val"],
                "abs_error": err,
                "final_loss": res["final_loss"],
                "loss_history": res["loss_history"],
                "val_history": res["val_history"]
            }
            dc_results["trials"].append(trial)
            print(f"GT: {d_true:.4f} | Init: {d_init:.4f} -> Rec: {res['final_val']:.4f} | Err: {err:.5f} | Loss: {res['final_loss']:.4f}")

    # Plot DC-001 convergence curves
    plt.figure(figsize=(12, 6))
    for trial in dc_results["trials"]:
        plt.plot(trial["val_history"], label=f"GT={trial['d_true']}, Init={trial['d_init']}")
    plt.title("DC-001: Decay-Only Recovery Trajectories Across Initializations")
    plt.xlabel("Iteration")
    plt.ylabel("Recovered Decay")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("research/identifiability/plots/dc001_trajectories.png", dpi=150)
    plt.close()

    # Save data
    with open("research/identifiability/data/st001_results.json", "w") as f:
        json.dump(st_results, f, indent=2)
    with open("research/identifiability/data/dc001_results.json", "w") as f:
        json.dump(dc_results, f, indent=2)
        
    print("\nSuccessfully saved ST-001 and DC-001 results and plots.")

if __name__ == "__main__":
    run_isolated_recovery_experiments()
