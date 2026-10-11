# Parameter Coupling Analysis (Experiment PC-001)

**Experiment ID:** `PC-001`  
**Date:** 2026-10-11  
**Target File:** `research/identifiability/run_pc001.py`  
**Data File:** `research/identifiability/data/pc001_results.json`  

---

## 1. Executive Summary

Having established that `decay`, `energy`, and `stretch` are each identifiable in strict isolation, this experiment tested **joint recovery** to determine whether parameter coupling (compensation between amplitude/decay and inharmonicity) obscures the ground truth parameters.

### Key Finding:
**Under deterministic fixed excitation, Adam accurately recovers all parameter combinations jointly without getting trapped in coupled valleys or suffering parameter compensation.**

Even when all three parameters were initialized far from truth:
- **Stretch:** Ground truth $0.0300$, initialized at $0.0050$ $\longrightarrow$ **Recovered $0.0299$** (relative error $0.3\%$).
- **Decay:** Ground truth $0.9900$, initialized at $0.9500$ $\longrightarrow$ **Recovered $0.9899$** (relative error $0.01\%$).
- **Energy:** Ground truth $1.0000$, initialized at $0.5000$ $\longrightarrow$ **Recovered $0.9984$** (relative error $0.16\%$).

---

## 2. Quantitative Results

| Configuration | Parameters Optimized | Ground Truth | Initial Value | Recovered Value | Final STFT Loss |
|---|---|---|---|---|---|
| **Config 1** | Decay, Energy (Stretch fixed at $0.030$) | $g = 0.9900$<br>$E = 1.0000$ | $0.9500$<br>$0.5000$ | **$g = 0.9900$**<br>**$E = 0.9941$** | $0.0938$ |
| **Config 2** | Stretch, Energy (Decay fixed at $0.990$) | $S = 0.0300$<br>$E = 1.0000$ | $0.0050$<br>$0.5000$ | **$S = 0.0302$**<br>**$E = 0.9942$** | $0.1049$ |
| **Config 3** | Stretch, Decay (Energy fixed at $1.000$) | $S = 0.0300$<br>$g = 0.9900$ | $0.0050$<br>$0.9500$ | **$S = 0.0301$**<br>**$g = 0.9899$** | $0.1089$ |
| **Config 4** | **All Three Jointly** | $S = 0.0300$<br>$g = 0.9900$<br>$E = 1.0000$ | $0.0050$<br>$0.9500$<br>$0.5000$ | **$S = 0.0299$**<br>**$g = 0.9899$**<br>**$E = 0.9984$** | $0.0938$ |

---

## 3. Scientific Implication

Parameter coupling does **not** destroy identifiability in the differentiable Karplus–Strong model when excitation is fixed. 
The multi-scale STFT magnitude loss provides orthogonal sensitivity axes:
- Spectral convergence norm captures broad temporal energy/envelope (constraining decay and energy).
- Log-magnitude spectral peak alignments constrain phase-dispersion and inharmonicity (constraining stretch).
