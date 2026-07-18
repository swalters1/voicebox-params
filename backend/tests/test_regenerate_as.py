"""Tests for "Regenerate as ..." — recasting a generation to a different voice.

The plumbing is trivial; the inheritance rules are the subtle part (issue #4):

* ``tts_params`` ARE inherited — a recast should carry the same tuning.
* ``chars_per_second`` is NOT inherited on a recast. Pace is per-voice, and the
  verify gate predicts expected duration from it, so reusing the previous
  voice's pace makes a complete render look short — the same class of
  false-failure as the Whisper 30s window.
* The seed is ALWAYS re-rolled unless supplied. Determinism holds over
  ``(seed, text, ref_audio, params)``, so reusing the parent's seed reproduces
  byte-identical audio: useless as "another take", meaningless across voices.

These call the route function directly rather than through TestClient — the app
factory does real startup work (migrations, model registry) that this logic
doesn't need.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from backend.routes import generations as gen_routes


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    for pid, name in (("kate-child", "Kate (child)"), ("kate-adult", "Kate (adult)")):
        session.add(DBVoiceProfile(id=pid, name=name, default_engine="chatterbox_turbo"))
    session.commit()
    yield session
    session.close()


@pytest.fixture
def runs():
    """Capture the kwargs run_generation would be called with; render nothing."""
    calls: list[dict] = []

    def fake_run_generation(**kwargs):
        calls.append(kwargs)
        return None  # enqueue is stubbed too, so this is never awaited

    with (
        patch.object(gen_routes, "run_generation", fake_run_generation),
        patch.object(gen_routes, "enqueue_generation", lambda gid, coro: None),
        patch.object(gen_routes, "get_task_manager", lambda: MagicMock()),
    ):
        yield calls


def _seed_generation(db, *, profile_id="kate-child", gen_params=None, seed=1234):
    gen = DBGeneration(
        id="gen-1",
        profile_id=profile_id,
        text="The map is not the territory.",
        language="en",
        status="completed",
        engine="chatterbox_turbo",
        seed=seed,
        gen_params=gen_params,
    )
    db.add(gen)
    db.commit()
    return gen


async def _regen(db, body=None):
    return await gen_routes.regenerate_generation(
        "gen-1",
        models.RegenerateRequest(**body) if body else None,
        db,
    )


@pytest.mark.asyncio
async def test_plain_regenerate_keeps_the_voice(db, runs):
    _seed_generation(db)
    await _regen(db)

    (call,) = runs
    assert call["profile_id"] == "kate-child"
    assert call["recast_profile_id"] is None, "same-voice take must not be marked a recast"


@pytest.mark.asyncio
async def test_plain_regenerate_rerolls_the_seed(db, runs):
    """A take that reproduces the parent byte-for-byte is not a take.

    The route used to pass the parent's stored seed straight through. Once the
    fork started persisting a concrete seed on every render, that quietly made
    every regenerate deterministic — same seed + text + voice + params is
    byte-identical by construction.
    """
    _seed_generation(db, seed=999)
    await _regen(db)

    assert runs[0]["seed"] is None, "regenerate must roll a fresh seed, not reuse the parent's"


@pytest.mark.asyncio
async def test_explicit_seed_is_honoured(db, runs):
    _seed_generation(db)
    await _regen(db, {"seed": 4242})

    assert runs[0]["seed"] == 4242


@pytest.mark.asyncio
async def test_recast_switches_voice_and_marks_the_take(db, runs):
    _seed_generation(db)
    await _regen(db, {"profile_id": "kate-adult"})

    (call,) = runs
    assert call["profile_id"] == "kate-adult"
    assert call["recast_profile_id"] == "kate-adult", "recast take must record its own voice"


@pytest.mark.asyncio
async def test_recast_to_the_same_voice_is_not_a_recast(db, runs):
    """Passing the row's own profile is a plain take, not a misattributed recast."""
    _seed_generation(db)
    await _regen(db, {"profile_id": "kate-child"})

    assert runs[0]["recast_profile_id"] is None


@pytest.mark.asyncio
async def test_recast_inherits_tts_params(db, runs):
    _seed_generation(db, gen_params={"tts_params": {"temperature": 0.55}})
    await _regen(db, {"profile_id": "kate-adult"})

    assert runs[0]["tts_params"]["temperature"] == 0.55


@pytest.mark.asyncio
async def test_recast_drops_inherited_chars_per_second(db, runs):
    """The rule that keeps a recast's verify honest.

    Carrying the child voice's 17.6 cps to the adult voice would have verify
    predict the wrong duration and fail a complete render — while every other
    threshold the author tuned should still carry over.
    """
    _seed_generation(
        db, gen_params={"verify_config": {"chars_per_second": 17.6, "coverage_min": 0.9}}
    )
    await _regen(db, {"profile_id": "kate-adult", "verify": True})

    vc = runs[0]["verify_config"] or {}
    assert vc.get("chars_per_second") != 17.6, "stale per-voice pace must not carry over"
    assert vc.get("coverage_min") == 0.9, "unrelated tuning must still be inherited"


@pytest.mark.asyncio
async def test_same_voice_regenerate_keeps_chars_per_second(db, runs):
    """Only a RECAST invalidates the pace — a same-voice take keeps it."""
    _seed_generation(db, gen_params={"verify_config": {"chars_per_second": 17.6}})
    await _regen(db, {"verify": True})

    assert (runs[0]["verify_config"] or {}).get("chars_per_second") == 17.6


@pytest.mark.asyncio
async def test_recast_to_unknown_profile_404s(db, runs):
    gen = _seed_generation(db)

    with pytest.raises(HTTPException) as exc:
        await _regen(db, {"profile_id": "nope"})
    assert exc.value.status_code == 404

    db.refresh(gen)
    assert gen.status == "completed", "a rejected recast must not strand the row in 'generating'"
    assert not runs
