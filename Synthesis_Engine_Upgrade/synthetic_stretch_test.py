import torch
import random
from device_config import DEVICE
from karplus_strong import DifferentiableKarplusStrong
from optimization import optimize_multi_parameters

def run_experiment(fret: int, pitch_hz: float, true_decay: float, true_stretch: float, noise_std: float = 0.0):
    print(f"\n--- Synthetic Test: Fret {fret} (Pitch: {pitch_hz:.1f}Hz) ---")
    print(f"Ground Truth -> decay: {true_decay:.4f}, stretch: {true_stretch:.4f}")
    
    # Generate true audio
    model = DifferentiableKarplusStrong(sample_rate=16000, duration=0.5).to(DEVICE)
    with torch.no_grad():
        model.log_pitch.fill_(torch.log(torch.tensor(pitch_hz)))
        
        # Reverse map decay and stretch to logits
        init_sigmoid = (true_decay - 0.5) / 0.4999
        import math
        model.logit_decay.fill_(math.log(init_sigmoid / (1 - init_sigmoid)))
        
        stretch_sigmoid = true_stretch / 0.1
        # Avoid log(0)
        stretch_sigmoid = max(1e-6, min(1.0 - 1e-6, stretch_sigmoid))
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
    rec_stretch = res[0]["inharmonicity"]
    
    print(f"Recovered    -> decay: {rec_decay:.4f}, stretch: {rec_stretch:.4f}")
    print(f"Stretch Error: {abs(true_stretch - rec_stretch):.5f}")

if __name__ == "__main__":
    # Test cases: G string (fret 0, 3, 7)
    # Using arbitrary true stretch values to see if the optimizer can recover them
    run_experiment(fret=0, pitch_hz=392.00, true_decay=0.995, true_stretch=0.001)
    run_experiment(fret=3, pitch_hz=466.16, true_decay=0.990, true_stretch=0.030)
    run_experiment(fret=7, pitch_hz=587.33, true_decay=0.985, true_stretch=0.080)
    
    # Test with noise
    print("\n\n>>> REPEATING WITH MEASUREMENT NOISE (std=0.01) <<<")
    run_experiment(fret=0, pitch_hz=392.00, true_decay=0.995, true_stretch=0.001, noise_std=0.01)
    run_experiment(fret=3, pitch_hz=466.16, true_decay=0.990, true_stretch=0.030, noise_std=0.01)
    run_experiment(fret=7, pitch_hz=587.33, true_decay=0.985, true_stretch=0.080, noise_std=0.01)
