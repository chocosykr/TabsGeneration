"""
device_config.py — Single source of truth for PyTorch device selection.

To switch between CPU and GPU, change USE_GPU below.
On Kaggle (GPU session), set USE_GPU = True.
On CPU-only machines, set USE_GPU = False.

You can also override without editing this file:
    UKULELE_DEVICE=cpu  python main.py ...
    UKULELE_DEVICE=cuda python main.py ...
"""

import os
import torch

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURE HERE
# ──────────────────────────────────────────────────────────────────────────────
USE_GPU: bool = False   # ← flip to True on Kaggle / any CUDA machine
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    # Environment variable takes highest priority
    env = os.environ.get("UKULELE_DEVICE", "").strip().lower()
    if env:
        d = torch.device(env)
        if d.type == "cuda" and not torch.cuda.is_available():
            print("[device_config] WARNING: UKULELE_DEVICE=cuda but CUDA not "
                  "available — falling back to CPU.")
            return torch.device("cpu")
        return d

    if USE_GPU:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[device_config] WARNING: USE_GPU=True but CUDA not "
                  "available — falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")


DEVICE: torch.device = _resolve_device()

print(f"[device_config] Using device: {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(DEVICE)})"
         if DEVICE.type == "cuda" else ""))
