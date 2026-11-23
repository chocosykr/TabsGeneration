import torch
import random
from device_config import DEVICE
from karplus_strong import DifferentiableKarplusStrong
from optimization import optimize_multi_parameters

def run_experiment(pitch_hz: float, true_decay: float, noise_std: float = 0.0):
    print(f"\n--- Synthetic Decay Test: (Pitch: {pitch_hz:.1f}Hz) ---")
    print(f"Ground Truth -> decay: {true_decay:.4f}")
    
    # Generate true audio
    model = DifferentiableKarplusStrong(sample_rate=16000, duration=0.5).to(DEVICE)
    with torch.no_grad():
        model.log_pitch.fill_(torch.log(torch.tensor(pitch_hz)))
        
        # Reverse map decay
        init_sigmoid = (true_decay - 0.5) / 0.4999
        import math
        model.logit_decay.fill_(math.log(init_sigmoid / (1 - init_sigmoid)))
        
        # Keep stretch at 0 to avoid conflating the two parameters
        stretch_sigmoid = 1e-6 / 0.1
        model.logit_stretch.fill_(math.log(stretch_sigmoid / (1 - stretch_sigmoid)))
        
        # Energy
        model.log_energy.fill_(1.0)
        
        target_audio = model().cpu()
        
        # Add noise
        if noise_std > 0:
            target_audio += torch.randn_like(target_audio) * noise_std
            
    # Optimize
    target_audio = target_audio.unsqueeze(0)
    res = optimize_multi_parameters(
        target_audio, 
        [pitch_hz], 
        sample_rate=16000, 
        num_iterations=100, 
        lr=0.1
    )
    
    rec_decay = res[0]["decay"]
    
    print(f"Recovered    -> decay: {rec_decay:.4f}")
    print(f"Decay Error  : {abs(true_decay - rec_decay):.5f}")

if __name__ == "__main__":
    # Test cases: decay values ranging from heavily damped (0.95) to ringing (0.999)
    run_experiment(pitch_hz=392.00, true_decay=0.950)
    run_experiment(pitch_hz=466.16, true_decay=0.980)
    run_experiment(pitch_hz=587.33, true_decay=0.995)
    run_experiment(pitch_hz=659.25, true_decay=0.999)
    
    # Test with noise
    print("\n\n>>> REPEATING WITH MEASUREMENT NOISE (std=0.01) <<<")
    run_experiment(pitch_hz=392.00, true_decay=0.950, noise_std=0.01)
    run_experiment(pitch_hz=466.16, true_decay=0.980, noise_std=0.01)
    run_experiment(pitch_hz=587.33, true_decay=0.995, noise_std=0.01)
    run_experiment(pitch_hz=659.25, true_decay=0.999, noise_std=0.01)
