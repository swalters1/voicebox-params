"""Qwen (Qwen3-TTS Base) now exposes its sampling knobs through the §7 contract.

Until this, PyTorchTTSBackend.generate() was called with only
(text, prompt, language, instruct) — none of Qwen's sampling params were
reachable. The motivating use is per-unit expressiveness: the audiobook
pipeline renders one line at a time, so a shouted line can go hotter and a
calm aside cooler via per-request temperature.

The model isn't exercised (Qwen is a heavy optional dep); this pins the option
contract, which is where the gap was.
"""

import inspect

import pytest

from backend.backends.pytorch_backend import PyTorchTTSBackend
from backend.utils.param_spec import OptionError, resolve_options, validate_options

SPEC = PyTorchTTSBackend.PARAM_SPEC

# qwen-tts library defaults (its generate resolves user value over these).
LIBRARY_DEFAULTS = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
}


def test_generate_declares_options():
    """The switch: the escalation loop forwards tuning only to backends whose
    generate() declares an ``options`` arg (_backend_accepts_options)."""
    assert "options" in inspect.signature(PyTorchTTSBackend.generate).parameters


def test_spec_defaults_match_the_library():
    assert resolve_options(SPEC, {}, reject_unknown=False) == LIBRARY_DEFAULTS


def test_temperature_override_resolves():
    resolved = resolve_options(SPEC, {"temperature": 1.4}, reject_unknown=False)
    assert resolved["temperature"] == 1.4
    # untouched knobs keep library defaults
    assert resolved["repetition_penalty"] == 1.05


@pytest.mark.parametrize(
    "bad",
    [
        {"nope": 1},
        {"temperature": 9.0},
        {"repetition_penalty": 0.5},
        {"top_p": 2.0},
    ],
)
def test_bad_input_is_rejected_at_the_boundary(bad):
    with pytest.raises(OptionError):
        validate_options(SPEC, bad)


def test_param_names_are_qwen_generate_kwargs():
    """These names are passed straight into generate_voice_clone(**kwargs), so
    they must be the exact kwargs qwen-tts honors — a typo would silently no-op.
    Pinned as a literal because the library isn't importable in CI; if qwen-tts
    renames one, this is the reminder to update the spec."""
    assert {p.name for p in SPEC} == set(LIBRARY_DEFAULTS)
