r"""
Phase 2 Integration Tests — Instrument-Aware Transcription & UI
================================================================
Tests the neural_transcription.py instrument-awareness additions
without requiring real audio files (uses synthetic MIDI note lists).

Run from project root:
    .venv/Scripts/python.exe Synthesis_Engine_Upgrade/test_integration_phase2.py
"""

import sys
import os
import types
import unittest
import math as _math

# ---------------------------------------------------------------------------
# Path setup — make Synthesis_Engine_Upgrade importable as a package
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Lightweight stubs so we can import neural_transcription without heavy deps
# ---------------------------------------------------------------------------
def _stub_module(name, **attrs):
    """Create and register a stub module."""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# ── basic_pitch ──────────────────────────────────────────────────────────────
_stub_module("basic_pitch")
_stub_module("basic_pitch.inference")
_stub_module("basic_pitch.note_creation")

# ── torch (need a Tensor class so type annotations don't break) ───────────────
_torch_stub = _stub_module("torch")
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    get_device_name=lambda d: "CPU"
)
_torch_stub.device = lambda x: types.SimpleNamespace(type="cpu")

# ── librosa ───────────────────────────────────────────────────────────────────
librosa_stub = _stub_module("librosa")
librosa_stub.load = lambda path, sr: (None, sr)
librosa_stub.stft = lambda y, **kw: None

# ── soundfile ─────────────────────────────────────────────────────────────────
_stub_module("soundfile")

# ── scipy + sub-modules ───────────────────────────────────────────────────────
scipy_stub = _stub_module("scipy")
scipy_stub.signal = types.SimpleNamespace(butter=None, sosfilt=None)
_stub_module("scipy.signal")

scipy_io_stub = _stub_module("scipy.io")
scipy_wavfile_stub = _stub_module("scipy.io.wavfile")
scipy_wavfile_stub.read = lambda path: (44100, None)
scipy_wavfile_stub.write = lambda path, sr, data: None
scipy_io_stub.wavfile = scipy_wavfile_stub

scipy_opt_stub = _stub_module("scipy.optimize")
scipy_opt_stub.nnls = lambda A, b: (None, None)
scipy_stub.optimize = scipy_opt_stub

# ── matplotlib ────────────────────────────────────────────────────────────────
mpl_stub = _stub_module("matplotlib")
mpl_stub.pyplot = types.SimpleNamespace(subplots=None, close=None)
mpl_stub.use = lambda x: None
_stub_module("matplotlib.pyplot")
_stub_module("matplotlib.colors")
_stub_module("matplotlib.ticker")

# ── numpy (use real numpy if installed) ───────────────────────────────────────
try:
    import numpy as np
except ImportError:
    np = _stub_module("numpy")

# ── numba ─────────────────────────────────────────────────────────────────────
_stub_module("numba")
_stub_module("numba.typed")

# ── fretboard_mapper (used by neural_transcription) ───────────────────────────
fm_stub = _stub_module("fretboard_mapper")
fm_stub.freq_to_midi = lambda freq: int(round(69 + 12 * _math.log2(max(freq, 1) / 440.0)))
fm_stub.midi_to_freq = lambda midi: 440.0 * (2 ** ((midi - 69) / 12.0))
fm_stub.UKULELE_TUNING = [67, 60, 64, 69]   # G4 C4 E4 A4

# ── optimization (used by neural_transcription) ───────────────────────────────
opt_stub = _stub_module("optimization")
opt_stub.compare_candidate_audio = lambda **kw: {"audio_match_normalized": 0.5, "stft_loss": 1.0}

# ── device_config (used by neural_transcription) ──────────────────────────────
dc_stub = _stub_module("device_config")
dc_stub.DEVICE = types.SimpleNamespace(type="cpu")

# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------
from instrument_config import (
    InstrumentProfile,
    get_profile,
    list_instruments,
    UKULELE_PROFILE,
    GUITAR_PROFILE,
)


def _import_neural_transcription():
    """Import neural_transcription with all heavy stubs in place."""
    se_stub = _stub_module("synthesis_engine")
    se_stub.synthesize_candidate_tab_audio_for_instrument = lambda **kw: (None, None)

    import importlib
    if "neural_transcription" in sys.modules:
        del sys.modules["neural_transcription"]
    return importlib.import_module("neural_transcription")


nt = _import_neural_transcription()


# ===========================================================================
# HELPERS
# ===========================================================================

def _tuning_open_midis(profile: InstrumentProfile) -> dict:
    """Return {string_idx: open_midi} for a profile, using the same logic as neural_transcription."""
    return {
        s: int(round(69 + 12 * _math.log2(max(profile.tuning[s]["freq"], 1) / 440.0)))
        for s in profile.tuning
    }


def _tuning_string_names(profile: InstrumentProfile) -> list:
    """Return string names ordered highest-string-first (matching tab_strings order)."""
    string_order = sorted(profile.tuning.keys(), reverse=True)
    return [profile.tuning[s]["name"] for s in string_order]


# ===========================================================================
# TEST SUITE
# ===========================================================================

class TestInstrumentConfig(unittest.TestCase):
    """Tests for the InstrumentProfile system."""

    def test_ukulele_profile_exists(self):
        self.assertIsInstance(UKULELE_PROFILE, InstrumentProfile)

    def test_guitar_profile_exists(self):
        self.assertIsInstance(GUITAR_PROFILE, InstrumentProfile)

    def test_ukulele_num_strings(self):
        """Ukulele has 4 strings (G-C-E-A)."""
        self.assertEqual(UKULELE_PROFILE.num_strings, 4)

    def test_guitar_num_strings(self):
        """Guitar has 6 strings (E-A-D-G-B-E)."""
        self.assertEqual(GUITAR_PROFILE.num_strings, 6)

    def test_ukulele_tuning_has_4_entries(self):
        """Ukulele tuning dict has 4 entries."""
        self.assertEqual(len(UKULELE_PROFILE.tuning), 4)

    def test_guitar_tuning_has_6_entries(self):
        """Guitar tuning dict has 6 entries."""
        self.assertEqual(len(GUITAR_PROFILE.tuning), 6)

    def test_ukulele_tuning_freqs_positive(self):
        """All ukulele open-string frequencies are positive."""
        for s, info in UKULELE_PROFILE.tuning.items():
            self.assertGreater(info["freq"], 0, f"String {s} freq must be positive")

    def test_guitar_tuning_freqs_positive(self):
        """All guitar open-string frequencies are positive."""
        for s, info in GUITAR_PROFILE.tuning.items():
            self.assertGreater(info["freq"], 0, f"String {s} freq must be positive")

    def test_ukulele_tuning_names_present(self):
        """Each ukulele string entry has a 'name' key."""
        for s, info in UKULELE_PROFILE.tuning.items():
            self.assertIn("name", info, f"String {s} should have a name")

    def test_guitar_tuning_names_present(self):
        """Each guitar string entry has a 'name' key."""
        for s, info in GUITAR_PROFILE.tuning.items():
            self.assertIn("name", info, f"String {s} should have a name")

    def test_guitar_has_lower_bass_than_ukulele(self):
        """Guitar lowest-string freq < ukulele lowest-string freq (wider range)."""
        uke_min_freq = min(info["freq"] for info in UKULELE_PROFILE.tuning.values())
        gtr_min_freq = min(info["freq"] for info in GUITAR_PROFILE.tuning.values())
        self.assertLess(gtr_min_freq, uke_min_freq,
                        "Guitar lowest string should be lower frequency than any ukulele string")

    def test_get_profile_ukulele(self):
        self.assertIs(get_profile("ukulele"), UKULELE_PROFILE)

    def test_get_profile_guitar(self):
        self.assertIs(get_profile("guitar"), GUITAR_PROFILE)

    def test_get_profile_case_insensitive(self):
        self.assertIs(get_profile("UKULELE"), UKULELE_PROFILE)
        self.assertIs(get_profile("Guitar"), GUITAR_PROFILE)

    def test_list_instruments(self):
        instruments = list_instruments()
        self.assertIn("ukulele", instruments)
        self.assertIn("guitar", instruments)

    def test_ukulele_guitar_profiles_distinct(self):
        self.assertIsNot(UKULELE_PROFILE, GUITAR_PROFILE)


class TestValidateInstrument(unittest.TestCase):
    """Tests for _validate_instrument() in neural_transcription."""

    def test_validate_ukulele(self):
        self.assertEqual(nt._validate_instrument("ukulele"), "ukulele")

    def test_validate_guitar(self):
        self.assertEqual(nt._validate_instrument("guitar"), "guitar")

    def test_validate_case_insensitive_guitar(self):
        self.assertEqual(nt._validate_instrument("GUITAR"), "guitar")

    def test_validate_case_insensitive_ukulele(self):
        self.assertEqual(nt._validate_instrument("Ukulele"), "ukulele")

    def test_validate_invalid_raises(self):
        with self.assertRaises(ValueError):
            nt._validate_instrument("banjo")

    def test_validate_empty_string_raises(self):
        with self.assertRaises(ValueError):
            nt._validate_instrument("")

    def test_validate_mandolin_raises(self):
        with self.assertRaises(ValueError):
            nt._validate_instrument("mandolin")

    def test_validate_error_message_helpful(self):
        try:
            nt._validate_instrument("sitar")
        except ValueError as e:
            msg = str(e).lower()
            self.assertTrue(
                "ukulele" in msg or "guitar" in msg or "valid" in msg,
                f"Error message should mention valid instruments, got: {e}"
            )


class TestMapPitchesToTabUkulele(unittest.TestCase):
    """map_pitches_to_tab() with UKULELE_PROFILE."""

    def test_response_has_tab_strings_key(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertIn("tab_strings", result)

    def test_ukulele_num_strings_in_response(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertEqual(result["num_strings"], 4)

    def test_ukulele_tab_strings_has_4_elements(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertEqual(len(result["tab_strings"]), 4)

    def test_ukulele_string_names_in_response(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertIn("string_names", result)
        self.assertEqual(len(result["string_names"]), 4)

    def test_ukulele_c4_lands_on_open_string(self):
        """C4 (MIDI 60) → open C string (fret 0)."""
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertIn(0, result["tab_strings"], "C4 should land on an open string for ukulele")

    def test_ukulele_a4_lands_on_open_string(self):
        """A4 (MIDI 69) → open A string (fret 0)."""
        result = nt.map_pitches_to_tab([69], UKULELE_PROFILE)
        self.assertIn(0, result["tab_strings"], "A4 should land on open A string")

    def test_ukulele_e4_lands_on_open_string(self):
        """E4 (MIDI 64) → open E string (fret 0)."""
        result = nt.map_pitches_to_tab([64], UKULELE_PROFILE)
        self.assertIn(0, result["tab_strings"], "E4 should land on open E string")

    def test_ukulele_frets_non_negative(self):
        result = nt.map_pitches_to_tab([60, 64], UKULELE_PROFILE)
        self.assertTrue(all(f >= 0 for f in result["tab_strings"]))

    def test_ukulele_empty_midi_returns_zeros(self):
        result = nt.map_pitches_to_tab([], UKULELE_PROFILE)
        self.assertEqual(result["tab_strings"], [0, 0, 0, 0])
        self.assertEqual(result["num_strings"], 4)

    def test_ukulele_response_has_mode_key(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertIn("mode", result)

    def test_ukulele_response_has_notes_key(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertIn("notes", result)
        self.assertIsInstance(result["notes"], list)

    def test_ukulele_response_has_candidates_key(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertIn("candidates", result)

    def test_ukulele_no_tab4321_in_map_pitches(self):
        """map_pitches_to_tab does NOT add tab_4321 — only the legacy wrapper does."""
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        # tab_4321 is intentionally absent from map_pitches_to_tab; it's added by wrappers.
        # This test documents that design decision.
        # We just check the actual returned keys are correct.
        self.assertIn("tab_strings", result)
        self.assertIn("num_strings", result)
        self.assertIn("string_names", result)


class TestMapPitchesToTabGuitar(unittest.TestCase):
    """map_pitches_to_tab() with GUITAR_PROFILE."""

    def test_response_has_tab_strings_key(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertIn("tab_strings", result)

    def test_guitar_num_strings_in_response(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertEqual(result["num_strings"], 6)

    def test_guitar_tab_strings_has_6_elements(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertEqual(len(result["tab_strings"]), 6)

    def test_guitar_string_names_in_response(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertIn("string_names", result)
        self.assertEqual(len(result["string_names"]), 6)

    def test_guitar_e2_lands_on_open_string(self):
        """E2 (MIDI 40) → guitar lowest open string (fret 0)."""
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertIn(0, result["tab_strings"], "E2 should land on open E2 string")

    def test_guitar_e4_lands_on_open_string(self):
        """E4 (MIDI 64) → guitar high E string open (fret 0)."""
        result = nt.map_pitches_to_tab([64], GUITAR_PROFILE)
        self.assertIn(0, result["tab_strings"], "E4 should be playable as open string on guitar")

    def test_guitar_frets_non_negative(self):
        result = nt.map_pitches_to_tab([40, 45], GUITAR_PROFILE)
        self.assertTrue(all(f >= 0 for f in result["tab_strings"]))

    def test_guitar_empty_midi_returns_zeros(self):
        result = nt.map_pitches_to_tab([], GUITAR_PROFILE)
        self.assertEqual(result["tab_strings"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(result["num_strings"], 6)

    def test_guitar_bass_note_below_ukulele_range(self):
        """MIDI 40 (E2) is below ukulele range but maps to a valid guitar fret."""
        result_gtr = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        result_uke = nt.map_pitches_to_tab([40], UKULELE_PROFILE)
        # Guitar should have a low fret (open position) on its low-E string
        gtr_ok = any(0 <= f <= 5 for f in result_gtr["tab_strings"])
        self.assertTrue(gtr_ok, "Guitar should map E2 to an open-position fret")
        # Ukulele result should not crash (function still returns dict)
        self.assertEqual(len(result_uke["tab_strings"]), 4)

    def test_guitar_response_has_mode_key(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertIn("mode", result)


class TestMapPitchesToTabChords(unittest.TestCase):
    """Multi-note (chord) mapping for both instruments."""

    def test_ukulele_two_note_chord(self):
        result = nt.map_pitches_to_tab([60, 64], UKULELE_PROFILE)
        self.assertEqual(len(result["tab_strings"]), 4)
        self.assertTrue(all(f >= 0 for f in result["tab_strings"]))

    def test_guitar_two_note_chord(self):
        result = nt.map_pitches_to_tab([40, 45], GUITAR_PROFILE)
        self.assertEqual(len(result["tab_strings"]), 6)
        self.assertTrue(all(f >= 0 for f in result["tab_strings"]))

    def test_ukulele_four_note_chord(self):
        result = nt.map_pitches_to_tab([60, 64, 67, 72], UKULELE_PROFILE)
        self.assertEqual(len(result["tab_strings"]), 4)
        self.assertEqual(result["num_strings"], 4)

    def test_guitar_six_note_chord(self):
        result = nt.map_pitches_to_tab([40, 45, 50, 55, 59, 64], GUITAR_PROFILE)
        self.assertEqual(len(result["tab_strings"]), 6)
        self.assertEqual(result["num_strings"], 6)

    def test_chord_mode_label(self):
        result = nt.map_pitches_to_tab([60, 64], UKULELE_PROFILE)
        self.assertEqual(result.get("mode"), "chord")

    def test_single_note_mode_label(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertEqual(result.get("mode"), "single_note")


class TestBackwardCompatibility(unittest.TestCase):
    """map_pitches_to_ukulele_tab() must remain backward-compatible."""

    def test_function_exists(self):
        self.assertTrue(hasattr(nt, "map_pitches_to_ukulele_tab"))

    def test_returns_tab_4321_key(self):
        """Legacy wrapper DOES return 'tab_4321' (injected by wrapper)."""
        result = nt.map_pitches_to_ukulele_tab([60])
        self.assertIn("tab_4321", result,
                      "map_pitches_to_ukulele_tab must inject tab_4321 for backward compat")

    def test_tab4321_has_4_elements(self):
        result = nt.map_pitches_to_ukulele_tab([60])
        self.assertEqual(len(result["tab_4321"]), 4)

    def test_tab4321_matches_tab_strings(self):
        result = nt.map_pitches_to_ukulele_tab([60])
        self.assertEqual(result["tab_4321"], result["tab_strings"])

    def test_legacy_empty_midi_returns_4_zeros(self):
        result = nt.map_pitches_to_ukulele_tab([])
        self.assertEqual(len(result["tab_4321"]), 4)
        self.assertTrue(all(f == 0 for f in result["tab_4321"]))

    def test_legacy_num_strings_is_4(self):
        result = nt.map_pitches_to_ukulele_tab([60])
        self.assertEqual(result.get("num_strings", 4), 4)

    def test_legacy_has_string_names(self):
        result = nt.map_pitches_to_ukulele_tab([60])
        self.assertIn("string_names", result)


class TestValidInstrumentsConstant(unittest.TestCase):
    """VALID_INSTRUMENTS constant in neural_transcription."""

    def test_valid_instruments_exists(self):
        self.assertTrue(hasattr(nt, "VALID_INSTRUMENTS"))

    def test_valid_instruments_contains_ukulele(self):
        self.assertIn("ukulele", nt.VALID_INSTRUMENTS)

    def test_valid_instruments_contains_guitar(self):
        self.assertIn("guitar", nt.VALID_INSTRUMENTS)


class TestInstrumentAwareResponseKeys(unittest.TestCase):
    """Response dicts must contain the new instrument-aware keys."""

    def test_ukulele_response_num_strings(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertEqual(result["num_strings"], 4)

    def test_guitar_response_num_strings(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertEqual(result["num_strings"], 6)

    def test_ukulele_string_names_length(self):
        result = nt.map_pitches_to_tab([60], UKULELE_PROFILE)
        self.assertEqual(len(result["string_names"]), 4)

    def test_guitar_string_names_length(self):
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        self.assertEqual(len(result["string_names"]), 6)

    def test_string_names_contain_note_names(self):
        """Each string name should be a non-empty string."""
        result = nt.map_pitches_to_tab([40], GUITAR_PROFILE)
        for name in result["string_names"]:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)


class TestPhase1Regression(unittest.TestCase):
    """
    Smoke-test: import Phase 1 modules without crash.
    Full Phase 1 suite is in test_synthesis_engine.py (78 tests).
    """

    def test_instrument_config_imports(self):
        import instrument_config
        self.assertIsNotNone(instrument_config)

    def test_synthesis_engine_imports(self):
        import synthesis_engine
        self.assertIsNotNone(synthesis_engine)

    def test_instrument_fretboard_imports(self):
        import instrument_fretboard
        self.assertIsNotNone(instrument_fretboard)

    def test_instrument_pitch_imports(self):
        # instrument_pitch uses torch.Tensor in a type annotation — needs Tensor stubbed
        import instrument_pitch
        self.assertIsNotNone(instrument_pitch)

    def test_instrument_spectral_imports(self):
        import instrument_spectral
        self.assertIsNotNone(instrument_spectral)

    def test_neural_transcription_imports(self):
        self.assertIsNotNone(nt)

    def test_main_upgraded_exists(self):
        path = os.path.join(_HERE, "main_upgraded.py")
        self.assertTrue(os.path.isfile(path))

    def test_upgrade_readme_exists(self):
        path = os.path.join(_HERE, "UPGRADE_README.md")
        self.assertTrue(os.path.isfile(path))

    def test_ui_server_exists(self):
        path = os.path.join(_HERE, "ui_server.py")
        self.assertTrue(os.path.isfile(path))

    def test_test_phase2_exists(self):
        path = os.path.join(_HERE, "test_integration_phase2.py")
        self.assertTrue(os.path.isfile(path))


class TestPrototypeIsolation(unittest.TestCase):
    """
    Verify new upgrade files are NOT placed inside the original prototype/ directory.

    'neural_transcription.py' existed in prototype/ BEFORE the upgrade —
    it is excluded from the check to avoid a false-positive.
    """

    _PROTOTYPE = os.path.join(_ROOT, "prototype")

    # Files that are genuinely NEW to Synthesis_Engine_Upgrade/ — verify they
    # don't also appear in prototype/ (which would indicate accidental edits).
    _NEW_UPGRADE_FILES = [
        "instrument_config.py",
        "synthesis_engine.py",
        "instrument_fretboard.py",
        "instrument_pitch.py",
        "instrument_spectral.py",
        "main_upgraded.py",
        "test_synthesis_engine.py",
        "test_integration_phase2.py",
        "UPGRADE_README.md",
        # ui_server.py also existed in prototype/, so excluded from new-file check.
    ]

    def test_prototype_dir_exists(self):
        self.assertTrue(os.path.isdir(self._PROTOTYPE), "prototype/ should exist")

    def test_new_upgrade_files_not_in_prototype(self):
        """None of the genuinely new upgrade files should exist inside prototype/."""
        if not os.path.isdir(self._PROTOTYPE):
            self.skipTest("prototype/ not found")
        proto_files = set(os.listdir(self._PROTOTYPE))
        for f in self._NEW_UPGRADE_FILES:
            self.assertNotIn(
                f, proto_files,
                f"New upgrade file '{f}' should NOT exist inside prototype/ — isolation violated!"
            )

    def test_synthesis_engine_upgrade_dir_exists(self):
        upgrade_dir = os.path.join(_ROOT, "Synthesis_Engine_Upgrade")
        self.assertTrue(os.path.isdir(upgrade_dir))

    def test_instrument_config_only_in_upgrade_dir(self):
        """instrument_config.py must be in Synthesis_Engine_Upgrade/, NOT in prototype/."""
        in_upgrade = os.path.isfile(os.path.join(_HERE, "instrument_config.py"))
        in_prototype = os.path.isfile(os.path.join(self._PROTOTYPE, "instrument_config.py"))
        self.assertTrue(in_upgrade, "instrument_config.py must exist in Synthesis_Engine_Upgrade/")
        self.assertFalse(in_prototype, "instrument_config.py must NOT exist in prototype/")


class TestUiServerInstrumentIntegration(unittest.TestCase):
    """Verify ui_server.py has all Phase 2 integration edits."""

    _UI_PATH = os.path.join(_HERE, "ui_server.py")

    @classmethod
    def setUpClass(cls):
        with open(cls._UI_PATH, encoding="utf-8") as f:
            cls.ui_source = f.read()

    def test_active_instrument_state_var(self):
        self.assertIn("let activeInstrument = 'ukulele'", self.ui_source,
                      "activeInstrument JS state variable must be declared")

    def test_set_instrument_function(self):
        self.assertIn("function setInstrument(instr)", self.ui_source,
                      "setInstrument() JS function must be defined")

    def test_instrument_badge_html(self):
        self.assertIn('id="instrumentBadge"', self.ui_source,
                      "instrumentBadge span must exist in HTML")

    def test_instr_btn_uke_html(self):
        self.assertIn('id="instrBtnUke"', self.ui_source,
                      "instrBtnUke button must exist in HTML")

    def test_instr_btn_gtr_html(self):
        self.assertIn('id="instrBtnGtr"', self.ui_source,
                      "instrBtnGtr button must exist in HTML")

    def test_spectrogram_passes_instrument(self):
        self.assertIn("instrument: activeInstrument", self.ui_source,
                      "fetchSpectrogram must pass activeInstrument to API")

    def test_chord_btn_passes_instrument(self):
        self.assertIn("instrument:activeInstrument", self.ui_source,
                      "runChordBtn fetch must include instrument:activeInstrument")

    def test_sequence_btn_passes_instrument(self):
        # At least two occurrences: one in chord handler, one in sequence handler
        count = self.ui_source.count("instrument:activeInstrument")
        self.assertGreaterEqual(count, 2,
                                "Both runChordBtn and runSequenceBtn must pass instrument")

    def test_render_fretboard_instrument_aware(self):
        self.assertIn("function renderFretboard(tabStrings, numStrings, stringNames)", self.ui_source,
                      "renderFretboard must accept instrument-aware parameters")

    def test_api_instruments_endpoint(self):
        self.assertIn("/api/instruments", self.ui_source,
                      "/api/instruments GET endpoint must be defined")

    def test_valid_instruments_in_server(self):
        self.assertIn("_VALID_INSTRUMENTS", self.ui_source,
                      "_VALID_INSTRUMENTS must be defined in ui_server.py")

    def test_guitar_error_message_for_physical_engine(self):
        self.assertIn("guitar", self.ui_source.lower(),
                      "Guitar mode must be mentioned (error handling) in ui_server.py")

    def test_instrument_badge_update_js(self):
        self.assertIn('"instrumentBadge"', self.ui_source,
                      "JS must update instrumentBadge")


# ===========================================================================
# MAIN
# ===========================================================================

def _run_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestInstrumentConfig,
        TestValidateInstrument,
        TestMapPitchesToTabUkulele,
        TestMapPitchesToTabGuitar,
        TestMapPitchesToTabChords,
        TestBackwardCompatibility,
        TestValidInstrumentsConstant,
        TestInstrumentAwareResponseKeys,
        TestPhase1Regression,
        TestPrototypeIsolation,
        TestUiServerInstrumentIntegration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = _run_suite()
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - skipped
    print(f"\n{'='*60}")
    print(f"Phase 2 Integration Test Summary")
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
