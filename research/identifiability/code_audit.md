# Code Audit: Differentiable Karplus-Strong Model & Identifiability Pipeline

**Audit Date:** 2026-10-11  
**Target Files:**
- `prototype/karplus_strong.py`
- `prototype/optimization.py`
- `prototype/synthetic_decay_test.py`
- `prototype/synthetic_stretch_test.py`

---

## 1. Executive Summary

This audit rigorously inspects the current PyTorch implementation of the Karplus–Strong (KS) physical synthesis loop, the multi-scale STFT loss function, and the synthetic identifiability test scripts.

**Key Finding:** The previous negative conclusion regarding stretch identifiability was methodologically invalid. The forward synthesis model was regenerating stochastic white noise on every single forward pass, ensuring that the optimizer compared against a moving, random target even at zero measurement noise. Furthermore, DC-group-delay compensation, non-differentiable integer delay truncations, and unconstrained joint optimization of coupled parameters (`decay`, `energy`, `stretch`) created severe confounding factors.

Below is the complete item-by-item technical audit.

---

## 2. Item A: Random Excitation & Reproducibility

### Confirmed Implementation Facts
* **File & Function:** `prototype/karplus_strong.py`, `DifferentiableKarplusStrong.forward()` (lines 64–65):
  ```python
  with torch.no_grad():
      noise = torch.randn(L_int, device=DEVICE)
  y: list = [noise[i] * energy for i in range(L_int)]
  ```
* Every invocation of `forward()` generates a newly sampled independent Gaussian random vector $\mathbf{w} \sim \mathcal{N}(0, I)$ of length $L_{\text{int}}$.
* In `synthetic_stretch_test.py` (lines 29, 37–43) and `optimization.py` (`optimize_multi_parameters` line 154):
  * Target generation calls `model()` once with excitation $\mathbf{w}_{\text{target}}$.
  * Optimization runs 100 iterations. At every iteration $k$, `model()` samples a new excitation $\mathbf{w}_k$.
* Even with `noise_std = 0.0`, the synthesized audio $\hat{\mathbf{y}}_k$ at iteration $k$ never matches $\mathbf{y}_{\text{target}}$ sample-for-sample or phase-for-phase, because $\mathbf{w}_k \neq \mathbf{w}_{\text{target}}$.

### Mathematical & Algorithmic Consequences
* The optimizer is not solving an inverse parameter-estimation problem:
  $$\min_\theta \mathcal{L}(\text{KS}(\theta, \mathbf{w}_0), \text{KS}(\theta^*, \mathbf{w}_0))$$
  Instead, it is solving a stochastic approximation problem:
  $$\min_\theta \mathbb{E}_{\mathbf{w}} \left[ \mathcal{L}(\text{KS}(\theta, \mathbf{w}), \text{KS}(\theta^*, \mathbf{w}_0)) \right]$$
* High-frequency details and fine phase-dispersion effects (which stretch controls) fluctuate wildly between iterations, generating noisy gradients that swamp the subtle physical effect of stretch.

### Mechanism for Controlled Experiments
* The forward model must accept an optional pre-allocated excitation buffer:
  `forward(excitation: Optional[torch.Tensor] = None)`.
* **Handling varying delay lengths ($L_{\text{int}}$):**
  When parameters such as `pitch` or `stretch` change during optimization, $L_{\text{int}}$ can vary.
  If an underlying master excitation vector $\mathbf{W}_{\text{base}}$ of length $N_{\text{max}} \ge \max(L_{\text{int}})$ is fixed, any forward pass needing length $L_{\text{int}}$ slices the first $L_{\text{int}}$ elements:
  $$\mathbf{w} = \mathbf{W}_{\text{base}}[:L_{\text{int}}]$$
  This guarantees strict deterministic consistency across forward passes while preserving default behavior when `excitation is None`.

---

## 3. Item B: Parameter Initializations, Transformations & Constraints

### Parameter Parameterization Map
In `prototype/karplus_strong.py` (`DifferentiableKarplusStrong`):

| Physical Parameter | Learnable Raw Tensor | Initialization Code | Transformation in `get_params()` | Initial Physical Value | Supported Physical Range |
|---|---|---|---|---|---|
| **Pitch** ($f_0$ Hz) | `self.log_pitch` | `torch.tensor(math.log(440.0))` | `torch.exp(log_pitch).clamp(20.0, sr/2 - 1)` | $440.0\text{ Hz}$ | $[20.0, 7999.0]\text{ Hz}$ |
| **Decay** ($g$) | `self.logit_decay` | `logit( (0.99 - 0.5) / 0.4999 )` $\approx 4.5849$ | `torch.sigmoid(logit_decay) * 0.4999 + 0.5` | $0.9900$ | $(0.5, 0.9999)$ |
| **Energy** ($E$) | `self.log_energy` | `torch.tensor(0.0)` | `F.softplus(log_energy) + 1e-4` | $\ln(1 + e^0) + 10^{-4} \approx 0.6932$ | $(10^{-4}, \infty)$ |
| **Stretch** ($S$) | `self.logit_stretch` | `torch.tensor(-5.0)` | `torch.sigmoid(logit_stretch) * 0.1` | $\sigma(-5.0) \cdot 0.1 \approx 0.000669$ | $(0.0, 0.1)$ |

### Discrepancies & Issues Identified
1. **Target Energy Mismatch:**
   In `synthetic_stretch_test.py` (line 27), the script sets:
   ```python
   model.log_energy.fill_(1.0)
   ```
   Physical target energy is $\text{softplus}(1.0) + 1e-4 \approx 1.3134$.
   However, the optimizer initializes `log_energy = 0.0`, corresponding to physical energy $0.6932$. The optimizer must spend iterations adjusting energy by almost a factor of 2, while jointly trying to infer decay and stretch.
2. **Stretch Zero Representation:**
   Because stretch uses a sigmoid transformation scaled by $0.1$:
   $$S = 0.1 \cdot \sigma(z) \in (0, 0.1)$$
   A true stretch of strictly $0.0$ cannot be reached for finite logit $z$. In `synthetic_decay_test.py` (lines 22–24), the code attempts to set stretch to zero using:
   ```python
   stretch_sigmoid = 1e-6 / 0.1
   model.logit_stretch.fill_(math.log(stretch_sigmoid / (1 - stretch_sigmoid)))
   ```
   This yields $S \approx 10^{-6}$. While numerically negligible, this clipping boundary behavior must be explicitly handled.

---

## 4. Item C: All-Pass Filter Recurrence & Delay Compensation

### Mathematical Formulation
The all-pass filter is implemented in `prototype/karplus_strong.py` (lines 81–87):
$$H_{\text{ap}}(z) = \frac{S + z^{-1}}{1 + S z^{-1}}$$
Difference equation implemented:
$$y[n] = S \cdot x[n] + x[n-1] - S \cdot y[n-1]$$
where $x[n]$ is the output of the fractional delay line (`delayed`), and $S = \text{stretch}$.

### The Delay Calculation & DC Compensation Flaw
Lines 57–58 of `karplus_strong.py`:
```python
ap_group_delay = (1.0 - stretch) / (1.0 + stretch)
L = (self.sample_rate / pitch) - ap_group_delay
```
1. **Mathematical Reality:**
   For $H_{\text{ap}}(z) = \frac{C + z^{-1}}{1 + C z^{-1}}$, the phase response is:
   $$\theta(\omega) = -\omega - 2 \arctan\left(\frac{C \sin \omega}{1 + C \cos \omega}\right)$$
   The phase delay is $D_p(\omega) = -\theta(\omega)/\omega$, and the group delay is $D_g(\omega) = -d\theta/d\omega$.
   At DC ($\omega = 0$):
   $$D_g(0) = D_p(0) = \frac{1 - C}{1 + C}$$
   However, string pitch is determined by the loop phase delay at the fundamental frequency $\omega_0 = 2\pi f_0 / f_s$, **not at $\omega = 0$**.
2. **Consequence:**
   Because $D_p(\omega_0) \neq D_g(0)$ for $\omega_0 > 0$, subtracting the DC group delay introduces a pitch error at the fundamental frequency $f_0$. As stretch varies from $0.001$ to $0.080$, $f_0$ itself shifts, altering the harmonic pitch baseline!

### Differentiability & Truncation
Lines 60–61 of `karplus_strong.py`:
```python
L_int = max(int(torch.floor(L).item()), 2)
L_frac = L - L_int
```
1. Calling `.item()` converts $L_{\text{int}}$ into a Python integer, detaching it from PyTorch's computational graph.
2. While $L_{\text{frac}}$ retains local gradient through $L$, crossing an integer threshold causes a step discontinuity where $L_{\text{int}}$ jumps by $\pm 1$ and $L_{\text{frac}}$ jumps by $\mp 1$. Gradient descent cannot smoothly navigate across integer delay boundaries.

---

## 5. Item D: Multi-Scale STFT Loss Function

### Specification
In `prototype/optimization.py` (`multi_scale_stft_loss` lines 13–40):
- **FFT sizes:** $[512, 1024, 2048]$ samples.
- **Hop lengths:** $N_{\text{fft}} / 4 \in [128, 256, 512]$ samples.
- **Window:** Hann window of length $N_{\text{fft}}$.
- **Magnitude extraction:**
  $$\text{mag} = |\text{STFT}(y)| + 10^{-7}$$
- **Loss terms:**
  $$\mathcal{L}_{\text{SC}} = \frac{\|\text{mag}_{\text{target}} - \text{mag}_{\text{synth}}\|_F}{\|\text{mag}_{\text{target}}\|_F + 10^{-8}}$$
  $$\mathcal{L}_{\text{logmag}} = \frac{1}{M} \|\ln(\text{mag}_{\text{target}}) - \ln(\text{mag}_{\text{synth}})\|_1$$
  $$\mathcal{L} = \frac{1}{3} \sum_{i=1}^3 (\mathcal{L}_{\text{SC}}^{(i)} + \mathcal{L}_{\text{logmag}}^{(i)})$$

### Observability Assessment
1. **Phase Neglect:** The loss evaluates only magnitudes $|\text{STFT}|$. All-pass filters preserve magnitude by definition ($|H_{\text{ap}}(e^{j\omega})| = 1$). Inside a closed Karplus–Strong feedback loop, the all-pass filter alters the resonance peak frequencies (inharmonicity). Therefore, the STFT magnitude loss can only observe stretch through the **shifts in spectral peak positions**, not through direct filter attenuation.
2. **STFT Resolution Limits:**
   At $f_s = 16000\text{ Hz}$, $N_{\text{fft}} = 2048$ yields frequency bin spacing $\Delta f = 16000 / 2048 \approx 7.81\text{ Hz}$. For subtle stretch values ($S = 0.001 \to 0.01$), upper partial shifts might be subtle relative to FFT bin resolution, demanding thorough investigation of loss landscape curvature.

---

## 6. Item E: Optimizer Configuration & Regularization Confounders

### Single-Voice Optimizer (`optimize_parameters`)
- Optimizes: `[logit_decay, log_energy, logit_stretch]`
- Frozen: `log_pitch` (fixed to initial pitch estimate)
- Learning rate: $0.05$ (or $0.1$ in test scripts)
- Iterations: 50–100 with gradient clipping norm $1.0$
- No explicit penalty terms.

### Multi-Voice Optimizer (`optimize_multi_parameters`)
- Line 162–169 of `prototype/optimization.py`:
  ```python
  decay_penalty = 0.0
  for voice, active in zip(model.voices, active_mask):
      if active:
          _, decay, _, _ = voice.get_params()
          decay_penalty += torch.nn.functional.relu(0.95 - decay)
  loss = stft_loss + (10.0 * decay_penalty)
  ```
- **Confounder:** This hard penalty applies a massive slope ($10.0$) against any decay $< 0.95$. In identifiability experiments testing decay recovery (such as $g = 0.950$), this penalty distorts the physical objective, artificially pushing recovery away from values below $0.95$.
- **Requirement:** This penalty must be strictly excluded from all scientific identifiability experiments.

---

## 7. Audit Conclusions & Roadmap

| Feature | Audit Finding | Status |
|---|---|---|
| Excitation Noise | Regenerated on every forward call; breaks deterministic recovery | **Must fix for experiments** |
| DC Group Delay Comp. | Approximates loop delay at DC instead of $f_0$; shifts fundamental | **Must measure in Exp 3A/3B** |
| Delay Truncation | `.item()` creates non-differentiable integer jumps | **Confirmed fact** |
| Parameter Coupling | Decay, energy, stretch optimized simultaneously | **Must decouple in Exps 4 & 5** |
| Multi-Voice Decay Penalty | Biases optimization above 0.95 | **Must disable in identifiability tests** |
| Noise Testing | Previously `std=0.01` without reference to signal RMS | **Must replace with calibrated SNR** |
