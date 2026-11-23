# Synthesis Engine Upgrade — README

## Overview

This folder (`Synthesis_Engine_Upgrade/`) is an **isolated working copy** of `prototype/`.

The original prototype is preserved untouched at `TabsGeneration/prototype/`.  
All development and experimentation for the dual instrument-mode upgrade happens here.

---

## What Changed

### NEW FILES (the upgrade)

| File | Purpose |
|------|---------|
| `instrument_config.py` | **Core registry.** `InstrumentProfile` dataclass + `UKULELE_PROFILE` + `GUITAR_PROFILE` + `get_profile()`. All instrument-specific physics parameters live here. |
| `synthesis_engine.py` | **Upgraded synthesis engine.** `SynthesisEngine` class — instrument-aware KS initialization, optimisation loop, candidate synthesis. Mode switching via `set_instrument()`. |
| `instrument_fretboard.py` | **Instrument-aware fretboard mapper.** Upgraded versions of `map_parameters_to_tab`, `refine_chord_with_inharmonicity`, `map_chord_to_tab` — all parameterised by `InstrumentProfile`. |
| `instrument_pitch.py` | **Instrument-aware pitch estimation.** Profile-driven pYIN bounds, HSS search range, polyphonic pitch estimation. |
| `instrument_spectral.py` | **Instrument-aware NNLS scorer.** Profile-driven harmonic template (num_harmonics, gaussian_sigma). |
| `main_upgraded.py` | **Upgraded CLI entry point.** Adds `--instrument` flag; orchestrates the full pipeline with the selected instrument mode. |
| `test_synthesis_engine.py` | **Full test suite.** 8 test groups, 40+ tests covering correctness, isolation, quality, integration, and before/after metrics. |

### UNCHANGED FILES (original implementation)

| File | Status |
|------|--------|
| `karplus_strong.py` | Unchanged — `DifferentiableKarplusStrong` and `MultiVoiceKarplusStrong` |
| `optimization.py` | Unchanged — `optimize_multi_parameters`, `synthesize_candidate_tab_audio`, `compare_candidate_audio` |
| `pitch_estimation.py` | Unchanged — original ukulele-specific pYIN and HSS search |
| `spectral_template.py` | Unchanged — original NNLS scorer |
| `fretboard_mapper.py` | Unchanged — original ukulele fretboard mapper |
| `main.py` | Unchanged — original CLI |
| `neural_transcription.py` | Unchanged — Basic Pitch integration |
| `chord_sequence.py` | Unchanged — sequence transcription |
| `ui_server.py` | Unchanged — web UI |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │      instrument_config.py        │
                    │  InstrumentProfile (dataclass)   │
                    │  UKULELE_PROFILE                 │
                    │  GUITAR_PROFILE                  │
                    │  get_profile(name)               │
                    └──────────────┬──────────────────┘
                                   │ profile
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
  │ synthesis_      │  │ instrument_     │  │ instrument_spectral  │
  │ engine.py       │  │ pitch.py        │  │ .py                  │
  │                 │  │                 │  │                      │
  │ SynthesisEngine │  │ pYIN (profiled) │  │ NNLS (profiled)      │
  │ .optimize()     │  │ HSS (profiled)  │  │ num_harmonics from   │
  │ .synthesize_    │  │ bounds from     │  │ profile              │
  │  candidate()    │  │ profile         │  │ gaussian_sigma from  │
  │                 │  │                 │  │ profile              │
  └────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘
           │                   │                       │
           ▼                   ▼                       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    instrument_fretboard.py                       │
  │  map_parameters_to_tab(profile=...)                              │
  │  refine_chord_with_inharmonicity(profile=...)                    │
  │  map_chord_to_tab(profile=...)                                   │
  └──────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    main_upgraded.py                              │
  │  --instrument ukulele → loads UKULELE_PROFILE                   │
  │  --instrument guitar  → loads GUITAR_PROFILE                    │
  │  Full pipeline: load audio → estimate pitches → optimize →      │
  │  map to tab → export                                             │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Run ukulele mode (identical to original behaviour)

```bash
cd Synthesis_Engine_Upgrade
python main_upgraded.py path/to/audio.wav --instrument ukulele --notes 4
```

### Run guitar mode

```bash
python main_upgraded.py path/to/audio.wav --instrument guitar --notes 6 --strummed 1,2,3,4,5,6
```

### Single note (ukulele)

```bash
python main_upgraded.py path/to/audio.wav --instrument ukulele --notes 1
```

### Single note (guitar)

```bash
python main_upgraded.py path/to/audio.wav --instrument guitar --notes 1
```

### Run the test suite

```bash
cd Synthesis_Engine_Upgrade
python test_synthesis_engine.py
```

---

## Key Instrument Differences

### Ukulele Mode
| Parameter | Value | Physical Basis |
|-----------|-------|---------------|
| Tuning | G4-C4-E4-A4 (re-entrant) | Standard soprano/concert/tenor |
| Strings | 4 | Standard ukulele |
| fmin / fmax | 98 Hz / 1050 Hz | G2 safety margin to above A5 |
| Polyphonic fmin | 240 Hz | Below C4 → ghost sub-harmonics |
| MIDI range | 60–84 (C4–C6) | Full playable range |
| Initial decay | ≈ 0.990 | Shorter sustain (nylon, short scale) |
| Harmonics (NNLS) | 8 | Nylon: rapid harmonic roll-off |
| Gaussian sigma | 1.5 bins | Narrower spread for higher pitches |
| Ergo weight | 0.5 | Fret-height penalty (original calibration) |

### Guitar Mode
| Parameter | Value | Physical Basis |
|-----------|-------|---------------|
| Tuning | E2-A2-D3-G3-B3-E4 (standard) | Standard 6-string acoustic/electric |
| Strings | 6 | Standard guitar |
| fmin / fmax | 70 Hz / 1200 Hz | Below E2 to above G5 |
| Polyphonic fmin | 75 Hz | Slightly below E2 open |
| MIDI range | 40–79 (E2–G5) | Full playable range (15 frets) |
| Initial decay | ≈ 0.996 | Longer sustain (steel, longer scale ~650mm) |
| Harmonics (NNLS) | 12 | Steel: rich sustained upper partials |
| Gaussian sigma | 2.0 bins | Wider for lower-pitched fundamentals |
| Ergo weight | 0.6 | Slightly stronger penalty (6 strings) |

---

## Relationship to Original `prototype/`

This folder is a **complete copy** of `prototype/` plus the new upgrade files.

- `prototype/main.py` → unchanged, still ukulele-only
- `prototype/optimization.py` → unchanged, still used by `synthesis_engine.py` indirectly
- `Synthesis_Engine_Upgrade/main_upgraded.py` → dual-mode entry point
- `Synthesis_Engine_Upgrade/synthesis_engine.py` → upgraded engine (wraps KS model)
- `Synthesis_Engine_Upgrade/instrument_config.py` → profile registry

To integrate the upgrade into the main prototype, the new files (`instrument_config.py`,
`synthesis_engine.py`, `instrument_fretboard.py`, `instrument_pitch.py`,
`instrument_spectral.py`) can be copied into `prototype/` and `main.py` updated
to import from them.  The original files are not broken by this — they coexist.

---

## Limitations and Known Issues

1. **Guitar chord search is slower** — 6 strings × 16 frets × itertools.product is 16^6 ≈ 16.7M combinations. For chords, constrain `--strummed` to fewer strings (e.g. `--strummed 1,2,3,4`) or reduce `max_fret_span`.

2. **Guitar inharmonicity priors** — The stretch_open values for wound strings (E2, A2, D3) are estimated from published literature rather than calibrated empirically from guitar recordings. They are physically grounded but may need adjustment from real data.

3. **Guitar chord chord_sequence.py** — `chord_sequence.py` currently imports from `fretboard_mapper.py` (ukulele-only). Full guitar sequence support requires updating `chord_sequence.py` to accept an instrument profile. This is scoped as a follow-up.

4. **Neural transcription (Basic Pitch)** — `neural_transcription.py` uses `UKULELE_TUNING` hardcoded. For guitar neural mode, `instrument_fretboard.py`'s `GUITAR_PROFILE.tuning` should be used instead.

5. **UI server** — `ui_server.py` does not yet expose the instrument selector. Adding a frontend toggle is a UI follow-up task.
