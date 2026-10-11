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

def optimize_joint(target_audio: torch.Tensor,
                   pitch: float,
                   params_to_train: list,
                   inits: dict,
                   fixed_params: dict,
                   excitation: torch.Tensor,
                   sample_rate: int = 16000,
                   duration: float = 0.5,
                   num_iterations: int = 80,
                   lr: float = 0.05):
    """
    Joint parameter optimizer for coupling analysis.
    """
    model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    
    # Set all parameters initially to fixed/target values
    model.set_physical_params(
        pitch=pitch,
        decay=fixed_params.get("decay", 0.990),
        energy=fixed_params.get("energy", 1.0),
        stretch=fixed_params.get("stretch", 0.030)
    )
    
    # Freeze all
    model.log_pitch.requires_grad_(False)
    model.logit_decay.requires_grad_(False)
    model.log_energy.requires_grad_(False)
    model.logit_stretch.requires_grad_(False)
    
    trainable = []
    # Set requested initializations and unfreeze
    if "decay" in params_to_train:
        model.set_physical_params(decay=inits["decay"])
        model.logit_decay.requires_grad_(True)
        trainable.append(model.logit_decay)
    if "energy" in params_to_train:
        model.set_physical_params(energy=inits["energy"])
        model.log_energy.requires_grad_(True)
        trainable.append(model.log_energy)
    if "stretch" in params_to_train:
        model.set_physical_params(stretch=inits["stretch"])
        model.logit_stretch.requires_grad_(True)
        trainable.append(model.logit_stretch)
        
    optimizer = torch.optim.Adam(trainable, lr=lr)
    
    loss_history = []
    for _ in range(num_iterations):
        optimizer.zero_grad()
        synth = model(excitation=excitation)
        loss = multi_scale_stft_loss(synth, target_audio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        loss_history.append(loss.item())
        
    p, d, e, s = model.get_physical_params()
    return {
        "final_decay": d.item(),
        "final_energy": e.item(),
        "final_stretch": s.item(),
        "final_loss": loss_history[-1],
        "loss_history": loss_history
    }

def run_parameter_coupling():
    print("=" * 60)
    print("RUNNING EXPERIMENT PC-001: Parameter Coupling Analysis")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.5
    pitch = 392.0
    
    true_decay = 0.990
    true_energy = 1.0
    true_stretch = 0.030
    
    torch.manual_seed(999)
    excitation = torch.randn(1024, device=DEVICE)
    
    # Ground truth target
    target_model = ResearchKarplusStrong(sample_rate=sample_rate, duration=duration).to(DEVICE)
    target_model.set_physical_params(pitch=pitch, decay=true_decay, energy=true_energy, stretch=true_stretch)
    with torch.no_grad():
        target_audio = target_model(excitation=excitation)
        
    results = {"configurations": {}}
    
    # -------------------------------------------------------------
    # Config 1: Decay + Energy
    # -------------------------------------------------------------
    print("\n--- Config 1: Joint Recovery of [Decay, Energy] (Stretch fixed at 0.030) ---")
    inits_1 = {"decay": 0.950, "energy": 0.5}
    res_1 = optimize_joint(target_audio, pitch, ["decay", "energy"], inits_1, {"stretch": true_stretch}, excitation)
    print(f"GT: decay={true_decay}, energy={true_energy}")
    print(f"Rec: decay={res_1['final_decay']:.4f}, energy={res_1['final_energy']:.4f} | Loss: {res_1['final_loss']:.4f}")
    results["configurations"]["decay_energy"] = {
        "gt": {"decay": true_decay, "energy": true_energy},
        "init": inits_1,
        "recovered": {"decay": res_1["final_decay"], "energy": res_1["final_energy"]},
        "loss": res_1["final_loss"]
    }
    
    # -------------------------------------------------------------
    # Config 2: Stretch + Energy
    # -------------------------------------------------------------
    print("\n--- Config 2: Joint Recovery of [Stretch, Energy] (Decay fixed at 0.990) ---")
    inits_2 = {"stretch": 0.005, "energy": 0.5}
    res_2 = optimize_joint(target_audio, pitch, ["stretch", "energy"], inits_2, {"decay": true_decay}, excitation)
    print(f"GT: stretch={true_stretch}, energy={true_energy}")
    print(f"Rec: stretch={res_2['final_stretch']:.4f}, energy={res_2['final_energy']:.4f} | Loss: {res_2['final_loss']:.4f}")
    results["configurations"]["stretch_energy"] = {
        "gt": {"stretch": true_stretch, "energy": true_energy},
        "init": inits_2,
        "recovered": {"stretch": res_2["final_stretch"], "energy": res_2["final_energy"]},
        "loss": res_2["final_loss"]
    }

    # -------------------------------------------------------------
    # Config 3: Stretch + Decay
    # -------------------------------------------------------------
    print("\n--- Config 3: Joint Recovery of [Stretch, Decay] (Energy fixed at 1.0) ---")
    inits_3 = {"stretch": 0.005, "decay": 0.950}
    res_3 = optimize_joint(target_audio, pitch, ["stretch", "decay"], inits_3, {"energy": true_energy}, excitation)
    print(f"GT: stretch={true_stretch}, decay={true_decay}")
    print(f"Rec: stretch={res_3['final_stretch']:.4f}, decay={res_3['final_decay']:.4f} | Loss: {res_3['final_loss']:.4f}")
    results["configurations"]["stretch_decay"] = {
        "gt": {"stretch": true_stretch, "decay": true_decay},
        "init": inits_3,
        "recovered": {"stretch": res_3["final_stretch"], "decay": res_3["final_decay"]},
        "loss": res_3["final_loss"]
    }

    # -------------------------------------------------------------
    # Config 4: All Three Jointly [Stretch, Decay, Energy]
    # -------------------------------------------------------------
    print("\n--- Config 4: Joint Recovery of All Three [Stretch, Decay, Energy] ---")
    inits_4 = {"stretch": 0.005, "decay": 0.950, "energy": 0.5}
    res_4 = optimize_joint(target_audio, pitch, ["stretch", "decay", "energy"], inits_4, {}, excitation, num_iterations=120)
    print(f"GT: stretch={true_stretch}, decay={true_decay}, energy={true_energy}")
    print(f"Rec: stretch={res_4['final_stretch']:.4f}, decay={res_4['final_decay']:.4f}, energy={res_4['final_energy']:.4f} | Loss: {res_4['final_loss']:.4f}")
    results["configurations"]["all_three"] = {
        "gt": {"stretch": true_stretch, "decay": true_decay, "energy": true_energy},
        "init": inits_4,
        "recovered": {"stretch": res_4["final_stretch"], "decay": res_4["final_decay"], "energy": res_4["final_energy"]},
        "loss": res_4["final_loss"]
    }

    with open("research/identifiability/data/pc001_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved PC-001 results to research/identifiability/data/pc001_results.json")

if __name__ == "__main__":
    run_parameter_coupling()
