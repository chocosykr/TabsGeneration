"""
synthesis_engine.py — Instrument-Aware Synthesis Engine (Upgrade Layer).

This module provides the upgraded synthesis engine that wraps the existing
DifferentiableKarplusStrong / MultiVoiceKarplusStrong models with
instrument-specific configuration from InstrumentProfile.

Key responsibilities:
  1. Build a MultiVoiceKarplusStrong model correctly initialised for
     the selected instrument (decay, stretch, energy priors).
  2. Run the Adam optimisation loop with instrument-specific loss weights
     and penalty thresholds.
  3. Synthesize candidate tab audio with instrument-specific defaults.
  4. Expose a clean SynthesisEngine class that the rest of the pipeline
     (main.py, neural_transcription.py, chord_sequence.py) can call without
     knowing which instrument is active.

The existing optimization.py is NOT modified. This module delegates the
recursive KS loop to DifferentiableKarplusStrong exactly as before.
"""

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict

from device_config import DEVICE
from instrument_config import InstrumentProfile, get_profile
from karplus_strong import DifferentiableKarplusStrong, MultiVoiceKarplusStrong


# ---------------------------------------------------------------------------
# STFT loss (identical to optimization.py — defined here so this module is
# self-contained and does not create a circular import with optimization.py)
# ---------------------------------------------------------------------------

def _multi_scale_stft_loss(synth: torch.Tensor,
                           target: torch.Tensor,
                           fft_sizes: List[int] = None) -> torch.Tensor:
    """Multi-scale STFT loss (Spectral Convergence + Log-Magnitude L1)."""
    if fft_sizes is None:
        fft_sizes = [512, 1024, 2048]

    if synth.ndim == 1:
        synth = synth.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if synth.device != target.device:
        target = target.to(synth.device)

    loss = torch.tensor(0.0, device=synth.device)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=synth.device)
        mag_s = torch.abs(torch.stft(synth, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7
        mag_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7
        sc   = torch.norm(mag_t - mag_s, p="fro") / (torch.norm(mag_t, p="fro") + 1e-8)
        lmag = F.l1_loss(torch.log(mag_t), torch.log(mag_s))
        loss = loss + sc + lmag
    return loss / len(fft_sizes)


def _compare_audio(synth: torch.Tensor,
                   target: torch.Tensor,
                   sample_rate: int = 16000,
                   fft_sizes: List[int] = None) -> Dict:
    """
    Compare synthesized vs target audio. Returns the same dict structure
    as optimization.compare_candidate_audio for drop-in compatibility.
    """
    if fft_sizes is None:
        fft_sizes = [512, 1024, 2048]

    if synth.ndim == 1:
        synth = synth.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if synth.device != target.device:
        target = target.to(synth.device)

    min_len = min(synth.shape[-1], target.shape[-1])
    s = synth[..., :min_len]
    t = target[..., :min_len]

    sc_list, lmag_list = [], []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=s.device)
        mag_s = torch.abs(torch.stft(s, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7
        mag_t = torch.abs(torch.stft(t, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)) + 1e-7
        sc_list.append(torch.norm(mag_t - mag_s, p="fro") / (torch.norm(mag_t, p="fro") + 1e-8))
        lmag_list.append(F.l1_loss(torch.log(mag_t), torch.log(mag_s)))

    mean_sc   = torch.stack(sc_list).mean()
    mean_lmag = torch.stack(lmag_list).mean()
    stft_loss = mean_sc + mean_lmag

    s_flat = s.squeeze()
    t_flat = t.squeeze()
    corr = torch.dot(s_flat, t_flat) / (torch.norm(s_flat) + 1e-8) / (torch.norm(t_flat) + 1e-8)
    similarity = 1.0 / (1.0 + stft_loss)

    return {
        "stft_loss":           float(stft_loss.item()),
        "spectral_convergence": float(mean_sc.item()),
        "log_mag_loss":         float(mean_lmag.item()),
        "waveform_correlation": float(corr.item()),
        "similarity":           float(similarity.item()),
    }


# ===========================================================================
# SynthesisEngine — the upgraded, instrument-aware synthesis interface
# ===========================================================================

class SynthesisEngine:
    """
    Instrument-aware Karplus-Strong synthesis engine.

    Usage
    -----
    # Ukulele mode
    engine = SynthesisEngine(instrument="ukulele")

    # Guitar mode
    engine = SynthesisEngine(instrument="guitar")

    # Switch modes (clean — no parameter bleed)
    engine.set_instrument("guitar")

    The engine exposes:
      - synthesize_candidate()   : fast forward-pass with given parameters
      - optimize()               : Adam loop fitting KS model to target audio
      - compare_audio()          : STFT + correlation comparison
    """

    def __init__(self, instrument: str = "ukulele", sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._profile: Optional[InstrumentProfile] = None
        self.set_instrument(instrument)

    # ── Mode selection ────────────────────────────────────────────────────────

    def set_instrument(self, instrument: str) -> None:
        """
        Load a new instrument profile. All subsequent synthesis calls use
        this profile's parameters. No state from the previous instrument
        is retained.
        """
        self._profile = get_profile(instrument)
        print(f"[SynthesisEngine] Loaded instrument profile: "
              f"{self._profile.display_name}")

    @property
    def profile(self) -> InstrumentProfile:
        if self._profile is None:
            raise RuntimeError("No instrument profile loaded. Call set_instrument() first.")
        return self._profile

    @property
    def instrument_name(self) -> str:
        return self.profile.name

    # ── Core synthesis ────────────────────────────────────────────────────────

    def _init_model(self, model: MultiVoiceKarplusStrong,
                    pitches: List[float],
                    active_mask: List[bool],
                    initial_stretches: Optional[List[float]] = None) -> None:
        """
        Initialise a MultiVoiceKarplusStrong model with instrument-specific
        decay, stretch, and energy priors from the loaded profile.

        This is the key upgrade point: previously all voices were initialised
        with generic defaults. Now each instrument provides its own calibrated
        priors for decay, inharmonicity, and energy.
        """
        p = self.profile
        for i, (voice, pitch, active) in enumerate(zip(model.voices, pitches, active_mask)):
            # Always initialise pitch (frozen)
            with torch.no_grad():
                voice.log_pitch.data.fill_(math.log(max(pitch, 20.0)))
            voice.log_pitch.requires_grad_(False)

            if not active:
                # Silent voice: freeze at near-zero energy
                with torch.no_grad():
                    voice.log_energy.data.fill_(-20.0)
                voice.log_energy.requires_grad_(False)
                voice.logit_decay.requires_grad_(False)
                voice.logit_stretch.requires_grad_(False)
                continue

            # ── Instrument-specific decay initialisation ──────────────────
            # decay_init is already in logit space from InstrumentProfile
            with torch.no_grad():
                voice.logit_decay.data.fill_(p.decay_init)

            # ── Instrument-specific stretch initialisation ─────────────────
            # If a physics-based prior stretch is provided (from fretboard mapper),
            # use it. Otherwise fall back to the instrument's default stretch_init.
            if initial_stretches is not None and i < len(initial_stretches):
                s = max(1e-6, min(p.stretch_max - 1e-6, initial_stretches[i]))
                sig = s / p.stretch_max
                sig = max(1e-7, min(1 - 1e-7, sig))
                logit_s = math.log(sig / (1.0 - sig))
                with torch.no_grad():
                    voice.logit_stretch.data.fill_(logit_s)
            else:
                with torch.no_grad():
                    voice.logit_stretch.data.fill_(p.stretch_init)

            # ── Energy initialisation ──────────────────────────────────────
            with torch.no_grad():
                voice.log_energy.data.fill_(p.default_energy)

    # ── Optimisation ─────────────────────────────────────────────────────────

    def optimize(self,
                 target_audio: torch.Tensor,
                 pitches: List[float],
                 num_iterations: int = 50,
                 lr: float = 0.5,
                 active_mask: Optional[List[bool]] = None,
                 initial_stretches: Optional[List[float]] = None,
                 return_audio: bool = False
                 ) -> Union[List[Dict], Tuple[List[Dict], torch.Tensor, Dict]]:
        """
        Instrument-aware Adam optimisation loop.

        Fits a MultiVoiceKarplusStrong model to target_audio using parameters
        initialised from the active instrument profile.

        Parameters
        ----------
        target_audio : torch.Tensor, shape (1, T) or (T,)
        pitches      : list of fundamental frequencies (Hz), one per voice
        num_iterations : Adam steps
        lr           : Adam learning rate
        active_mask  : list of bools; False = silent voice (pitch padded)
        initial_stretches : physics-based inharmonicity priors (from fretboard)
        return_audio : if True returns (results, final_synth, comparison_dict)

        Returns
        -------
        List[dict] with keys: pitch, decay, inharmonicity, audio_loss, audio_similarity
        (optionally with final_synth tensor and comparison dict)
        """
        p = self.profile
        num_voices = len(pitches)

        if num_voices == 0:
            empty_comp = {"stft_loss": 0.0, "spectral_convergence": 0.0,
                          "log_mag_loss": 0.0, "waveform_correlation": 0.0, "similarity": 1.0}
            if return_audio:
                return [], torch.zeros(0, device=DEVICE), empty_comp
            return []

        if active_mask is None:
            active_mask = [True] * num_voices

        target_audio = target_audio.to(DEVICE)
        duration = target_audio.shape[-1] / self.sample_rate

        model = MultiVoiceKarplusStrong(
            num_voices, sample_rate=self.sample_rate, duration=duration
        ).to(DEVICE)

        # ── Instrument-specific initialisation ────────────────────────────
        self._init_model(model, pitches, active_mask, initial_stretches)

        # Collect trainable parameters (active voices only)
        optim_params = []
        for voice, active in zip(model.voices, active_mask):
            if active:
                optim_params.extend([
                    voice.logit_decay, voice.log_energy, voice.logit_stretch
                ])

        # ── Optimisation loop ─────────────────────────────────────────────
        if num_iterations > 0 and len(optim_params) > 0:
            optimizer = torch.optim.Adam(optim_params, lr=lr)

            pitch_str = ", ".join(f"{pp:.2f} Hz" for pp in pitches)
            print(f"[{p.display_name}] Starting {num_voices}-voice optimisation "
                  f"(pitches: [{pitch_str}], "
                  f"decay_init={p.default_decay:.4f}, "
                  f"optimising decay/energy/stretch)...")

            for i in tqdm(range(num_iterations)):
                optimizer.zero_grad()
                synth = model()

                min_len = min(synth.shape[-1], target_audio.shape[-1])
                stft_loss = _multi_scale_stft_loss(
                    synth[:min_len], target_audio.squeeze()[:min_len]
                )

                # ── Instrument-specific decay penalty ─────────────────────
                # Prevents the optimizer from silencing wrong notes by driving
                # their decay below the instrument's physical minimum sustain.
                decay_penalty = torch.tensor(0.0, device=DEVICE)
                for voice, active in zip(model.voices, active_mask):
                    if active:
                        _, decay, _, _ = voice.get_params()
                        decay_penalty = (decay_penalty +
                            torch.nn.functional.relu(p.decay_penalty_threshold - decay))

                loss = stft_loss + 10.0 * decay_penalty
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if (i + 1) % 10 == 0:
                    decays = [v.get_params()[1].item() for v in model.voices]
                    decay_str = ", ".join(f"d{v+1}={decays[v]:.4f}"
                                         for v in range(num_voices))
                    print(f"  iter {i+1:3d}  loss={loss.item():.4f}  "
                          f"(stft={stft_loss.item():.4f})  {decay_str}")

        # ── Final audio comparison ────────────────────────────────────────
        with torch.no_grad():
            final_synth = model()
        min_len = min(final_synth.shape[-1], target_audio.shape[-1])
        comparison = _compare_audio(
            final_synth[:min_len], target_audio.squeeze()[:min_len], self.sample_rate
        )

        # ── Build result dicts ────────────────────────────────────────────
        results = []
        for i, voice in enumerate(model.voices):
            _, decay, _, stretch = voice.get_params()
            results.append({
                "pitch":           pitches[i],
                "decay":           decay.item(),
                "inharmonicity":   stretch.item(),
                "audio_loss":      comparison["stft_loss"],
                "audio_similarity": comparison["similarity"],
                "audio_metrics":   comparison,
                "instrument":      p.name,
            })
            print(f"   Voice {i+1} - Pitch: {pitches[i]:.2f} Hz, "
                  f"Decay: {decay.item():.4f}, "
                  f"Stretch: {stretch.item():.4f}")

        print(f"   [{p.display_name}] Audio comparison: "
              f"STFT Loss={comparison['stft_loss']:.4f}, "
              f"Spectral Convergence={comparison['spectral_convergence']:.4f}, "
              f"Similarity={comparison['similarity']:.4f}")

        if return_audio:
            return results, final_synth[:min_len], comparison
        return results

    # ── Candidate synthesis (no gradient) ─────────────────────────────────────

    def synthesize_candidate(self,
                             pitches: List[float],
                             duration: float = 0.5,
                             decays: Optional[List[float]] = None,
                             energies: Optional[List[float]] = None,
                             stretches: Optional[List[float]] = None,
                             active_mask: Optional[List[bool]] = None
                             ) -> torch.Tensor:
        """
        Fast candidate synthesis without optimisation, using instrument-specific
        default parameters where explicit values are not supplied.

        Returns a 1-D torch.Tensor of shape (num_samples,) on DEVICE.
        """
        p = self.profile
        num_voices = len(pitches)

        if num_voices == 0:
            return torch.zeros(int(self.sample_rate * duration), device=DEVICE)
        if active_mask is None:
            active_mask = [True] * num_voices

        model = MultiVoiceKarplusStrong(
            num_voices, sample_rate=self.sample_rate, duration=duration
        ).to(DEVICE)

        with torch.no_grad():
            for i, (voice, pitch, active) in enumerate(zip(model.voices, pitches, active_mask)):
                voice.log_pitch.data.fill_(math.log(max(pitch, 20.0)))

                if not active:
                    voice.log_energy.data.fill_(-20.0)
                    continue

                # ── Instrument-specific decay ──────────────────────────────
                if decays is not None and i < len(decays):
                    d = max(0.5001, min(0.9998, decays[i]))
                    sig = (d - 0.5) / 0.4999
                    voice.logit_decay.data.fill_(math.log(sig / (1.0 - sig)))
                else:
                    # Use profile default
                    voice.logit_decay.data.fill_(p.decay_init)

                # ── Energy ────────────────────────────────────────────────
                if energies is not None and i < len(energies):
                    voice.log_energy.data.fill_(energies[i])
                else:
                    voice.log_energy.data.fill_(p.default_energy)

                # ── Instrument-specific stretch ────────────────────────────
                if stretches is not None and i < len(stretches):
                    s = max(1e-6, min(p.stretch_max - 1e-6, stretches[i]))
                    sig = s / p.stretch_max
                    sig = max(1e-7, min(1 - 1e-7, sig))
                    voice.logit_stretch.data.fill_(math.log(sig / (1.0 - sig)))
                else:
                    voice.logit_stretch.data.fill_(p.stretch_init)

            synth = model()
        return synth

    # ── Audio comparison (delegated) ───────────────────────────────────────────

    def compare_audio(self,
                      synth: torch.Tensor,
                      target: torch.Tensor,
                      fft_sizes: Optional[List[int]] = None) -> Dict:
        """Compare synthesized vs target audio. Returns standard metrics dict."""
        return _compare_audio(synth, target, self.sample_rate, fft_sizes)


# ===========================================================================
# Module-level convenience functions (drop-in replacements for optimization.py
# when instrument-aware behaviour is needed)
# ===========================================================================

def synthesize_candidate_tab_audio_for_instrument(
        pitches: List[float],
        instrument: str,
        sample_rate: int = 16000,
        duration: float = 0.5,
        decays: Optional[List[float]] = None,
        energies: Optional[List[float]] = None,
        stretches: Optional[List[float]] = None,
        active_mask: Optional[List[bool]] = None) -> torch.Tensor:
    """
    Synthesize candidate tab audio for a specific instrument.

    This is the upgraded replacement for optimization.synthesize_candidate_tab_audio.
    It loads the correct instrument profile and uses instrument-specific defaults
    for parameters not explicitly supplied.
    """
    engine = SynthesisEngine(instrument=instrument, sample_rate=sample_rate)
    return engine.synthesize_candidate(
        pitches=pitches,
        duration=duration,
        decays=decays,
        energies=energies,
        stretches=stretches,
        active_mask=active_mask
    )


def optimize_multi_parameters_for_instrument(
        target_audio: torch.Tensor,
        pitches: List[float],
        instrument: str,
        sample_rate: int = 16000,
        num_iterations: int = 50,
        lr: float = 0.5,
        active_mask: Optional[List[bool]] = None,
        initial_stretches: Optional[List[float]] = None,
        return_audio: bool = False
) -> Union[List[Dict], Tuple[List[Dict], torch.Tensor, Dict]]:
    """
    Instrument-aware optimisation wrapper.

    Drop-in replacement for optimization.optimize_multi_parameters with
    instrument-specific parameter initialisation.
    """
    engine = SynthesisEngine(instrument=instrument, sample_rate=sample_rate)
    return engine.optimize(
        target_audio=target_audio,
        pitches=pitches,
        num_iterations=num_iterations,
        lr=lr,
        active_mask=active_mask,
        initial_stretches=initial_stretches,
        return_audio=return_audio
    )
