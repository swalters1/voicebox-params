"""TADA (HumeAI) now exposes its inference options through the §7 contract.

Until this, ``HumeTadaBackend.generate()`` was called with only (prompt, text)
and none of tada's InferenceOptions were reachable — no pacing, no
expressiveness control. These tests pin the wiring: the spec is discoverable,
the pipeline will forward tuning (generate declares ``options``), an empty
request reproduces the library defaults exactly, and the request boundary
still rejects bad input.

The model itself isn't exercised (tada is a heavy optional dependency); this is
about the option contract, which is where the previous gaps lived.
"""

import inspect

import pytest

from backend.backends.hume_backend import HumeTadaBackend
from backend.utils.param_spec import OptionError, resolve_options, validate_options

SPEC = HumeTadaBackend.PARAM_SPEC

# hume-tada 0.1.x InferenceOptions defaults. If tada ever changes these, the
# spec must follow — an empty request must stay byte-identical to old behaviour.
LIBRARY_DEFAULTS = {
    "text_temperature": 0.6,
    "text_top_p": 0.9,
    "text_repetition_penalty": 1.1,
    "acoustic_cfg_scale": 1.6,
    "duration_cfg_scale": 1.0,
    "num_flow_matching_steps": 10,
    "speed_up_factor": None,
}


def test_generate_declares_options():
    """This is the switch: the escalation loop forwards tuning only to backends
    whose generate() declares an ``options`` arg (_backend_accepts_options)."""
    assert "options" in inspect.signature(HumeTadaBackend.generate).parameters


def test_spec_defaults_match_the_library():
    resolved = resolve_options(SPEC, {}, reject_unknown=False)
    assert resolved == LIBRARY_DEFAULTS, "empty request must equal tada's own defaults"


def test_overrides_resolve():
    resolved = resolve_options(
        SPEC, {"text_temperature": 0.9, "speed_up_factor": 1.5}, reject_unknown=False
    )
    assert resolved["text_temperature"] == 0.9
    assert resolved["speed_up_factor"] == 1.5


def test_none_default_is_not_range_checked():
    """speed_up_factor defaults to None (library 'off'); the None must survive
    resolution without tripping its 0.5–4.0 bounds."""
    assert resolve_options(SPEC, {}, reject_unknown=False)["speed_up_factor"] is None


@pytest.mark.parametrize(
    "bad",
    [
        {"nonsense": 1},
        {"text_temperature": 9.0},
        {"speed_up_factor": 99},
        {"num_flow_matching_steps": 0},
    ],
)
def test_bad_input_is_rejected_at_the_boundary(bad):
    with pytest.raises(OptionError):
        validate_options(SPEC, bad)


def test_spec_maps_cleanly_onto_inference_options_fields():
    """Every declared param name must be a real InferenceOptions field, so the
    backend's field-filter passes them through instead of silently dropping.

    Skips when tada isn't installed (the heavy dep isn't in CI); when it is,
    this catches a typo'd param name that would otherwise no-op forever.
    """
    tada = pytest.importorskip("tada.modules.tada")
    from dataclasses import fields

    known = {f.name for f in fields(tada.InferenceOptions)}
    declared = {p.name for p in SPEC}
    missing = declared - known
    assert not missing, f"PARAM_SPEC names not in InferenceOptions: {missing}"
