"""Declarative per-engine parameter specs and option resolution.

Each backend declares a ``PARAM_SPEC`` (a list of :class:`Param`) instead of
hard-coding sampling literals inside ``generate()``. One mechanism then serves
both audiences (FORK_NOTES §7): casual callers send ``{}`` and get the declared
defaults; advanced callers send overrides, and the same spec drives request
validation *and* the UI panel advertised by ``GET /engines``.

Resolution is layered (last wins), e.g.::

    engine defaults -> language/context -> profile overrides -> request options

Unknown keys are rejected loudly rather than silently dropped — silently
ignoring a misapplied option is exactly how a typo looks like it "works"
(FORK_NOTES §7e).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional


class OptionError(ValueError):
    """A supplied option is unknown for the engine or out of range.

    Raised by :func:`resolve_options`. HTTP boundaries translate this to a 422;
    keeping it a plain domain error means this module never imports FastAPI.
    """


@dataclass(frozen=True)
class Param:
    """One tunable parameter on an engine.

    Attributes:
        name: Keyword passed to the underlying model call.
        default: Value used when no layer overrides it.
        min/max: Inclusive numeric bounds (None to skip range checking).
        stage: ``"call"`` params vary per generate; ``"load"`` params
            (device/dtype/variant) change at model-load time and must not ride
            in the per-generate options blob.
        desc: Human-readable hint for the advanced-mode UI.
    """

    name: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    stage: str = "call"
    desc: str = ""


def resolve_options(
    spec: Iterable[Param],
    *layers: Optional[dict],
    reject_unknown: bool = True,
) -> dict:
    """Merge override *layers* over a spec's defaults into a full option set.

    Returns every parameter in *spec* with its resolved value (defaults filled),
    across all stages. Later layers win. With ``reject_unknown=True`` an
    unrecognized key raises :class:`OptionError`; out-of-range values always do.

    Backends call this with ``reject_unknown=False`` to defensively fill their
    own defaults; the request boundary calls it with ``reject_unknown=True`` to
    validate and echo.
    """
    allowed = {p.name: p for p in spec}
    out = {p.name: p.default for p in allowed.values()}

    for layer in layers:
        for key, value in (layer or {}).items():
            param = allowed.get(key)
            if param is None:
                if reject_unknown:
                    raise OptionError(
                        f"unknown option {key!r}; valid: {sorted(allowed)}"
                    )
                continue
            if (
                value is not None
                and param.min is not None
                and param.max is not None
                and not (param.min <= value <= param.max)
            ):
                raise OptionError(
                    f"{key}={value} out of range [{param.min}, {param.max}]"
                )
            out[key] = value

    return out


def filter_applicable(spec: Iterable[Param], overrides: Optional[dict]) -> dict:
    """Keep only *overrides* that are valid for this spec (leniently).

    Drops keys not in the spec AND keys whose value is out of range, without
    raising. Used for the profile layer of option resolution: a per-voice tuning
    may target a different default engine, so keys that don't apply (or don't
    fit) to the engine actually in use are silently ignored rather than failing
    a request the caller never mis-parameterized.
    """
    allowed = {p.name: p for p in spec}
    out = {}
    for key, value in (overrides or {}).items():
        param = allowed.get(key)
        if param is None:
            continue
        if (
            value is not None
            and param.min is not None
            and param.max is not None
            and not (param.min <= value <= param.max)
        ):
            continue
        out[key] = value
    return out


def split_call_options(spec: Iterable[Param], resolved: dict) -> dict:
    """Return only the ``stage=="call"`` entries of a resolved option set.

    Keeps load-time params (device/dtype/variant) out of the per-generate blob
    that gets spread into the model's ``generate()`` call.
    """
    call_names = {p.name for p in spec if p.stage == "call"}
    return {k: v for k, v in resolved.items() if k in call_names}


def spec_as_dicts(spec: Iterable[Param]) -> list[dict]:
    """Serialize a PARAM_SPEC for the ``GET /engines`` capability response."""
    return [asdict(p) for p in spec]
