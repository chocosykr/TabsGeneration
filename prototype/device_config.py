"""
device_config.py - Dedicated GPU & PyTorch device configuration.

Configures execution to run on the laptop's dedicated GPU (NVIDIA CUDA).
"""

import os
import shutil
import subprocess
import torch

# ------------------------------------------------------------------------------
# CONFIGURE HERE
# ------------------------------------------------------------------------------
# Force execution on dedicated GPU (False = CPU mode, True = Dedicated GPU mode)
USE_GPU: bool = False

# Specific GPU index (0 = default dedicated GPU on laptops with hybrid graphics)
GPU_INDEX: int = 0
# ------------------------------------------------------------------------------


def _detect_physical_nvidia_gpu() -> str:
    """Check if a physical NVIDIA dedicated GPU is present on the laptop."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return ""
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return out
    except Exception:
        return ""


def _resolve_device() -> torch.device:
    env = os.environ.get("UKULELE_DEVICE", "").strip().lower()
    if env:
        d = torch.device(env)
        if d.type == "cuda" and not torch.cuda.is_available():
            print("[device_config] WARNING: UKULELE_DEVICE=cuda requested, but PyTorch cannot access CUDA.")
            return torch.device("cpu")
        return d

    if not USE_GPU:
        print("[device_config] Device mode: CPU active (USE_GPU = False).")
        return torch.device("cpu")

    if USE_GPU:
        if torch.cuda.is_available():
            # Select dedicated GPU
            device_str = f"cuda:{GPU_INDEX}" if torch.cuda.device_count() > GPU_INDEX else "cuda"
            device = torch.device(device_str)

            # Performance optimizations for modern NVIDIA RTX architectures
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

            gpu_name = torch.cuda.get_device_name(device)
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"[device_config] Dedicated GPU active: {gpu_name} ({vram_gb:.1f} GB VRAM) on {device}")
            return device
        else:
            # Detect whether physical dedicated GPU is present
            gpu_info = _detect_physical_nvidia_gpu()
            print("[device_config] NOTICE: Execution configured for dedicated GPU (USE_GPU = True).")
            if gpu_info:
                print(f"[device_config] Detected Hardware: Dedicated GPU '{gpu_info}' is installed.")
                print(f"[device_config] PyTorch build currently installed: {torch.__version__} (CPU-only build).")
                print("[device_config] To activate dedicated GPU acceleration, install the CUDA build of PyTorch:")
                print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n")
            else:
                print("[device_config] CUDA not available - falling back to CPU.\n")
            return torch.device("cpu")

    return torch.device("cpu")


DEVICE: torch.device = _resolve_device()
