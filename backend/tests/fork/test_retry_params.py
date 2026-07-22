"""Retry must re-render with the params the row was rendered with.

Two defects, one root cause — the reproduce path re-derived instead of reading
what was stored:

1. ``/generate/{id}/retry`` called ``run_generation`` with no ``tts_params`` or
   ``verify_config``, so a unit deliberately rendered at temperature 1.15 came
   back at the engine default (0.9) — silently, in the audio only.
2. ``gen_params`` was only written AFTER a successful render, so a *failed* row
   (the only kind retry accepts) carried nothing to inherit. Verified against
   the real DB: 0 of the failed rows had gen_params. That made fixing (1) alone
   a no-op, which is why creation-time persistence is part of this.

Matters most for per-unit emotion arcs: a failed unit retried would drop out of
its arc exactly when it's most troublesome.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from backend.routes import generations as gen_routes
from backend.services import history

GID = "gen-retry-1"
ARC_PARAMS = {"temperature": 1.15, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05}


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    s.add(DBVoiceProfile(id="narrator", name="Narrator", default_engine="qwen"))
    s.commit()
    yield s
    s.close()


def _add_failed(session, gen_params):
    """A failed generation — the only state retry accepts."""
    session.add(
        DBGeneration(
            id=GID,
            profile_id="narrator",
            text="The festival grounds stretched wide across the town square.",
            language="en",
            audio_path="",
            duration=0,
            seed=1173744344,
            engine="qwen",
            model_size="1.7B",
            status="failed",
            gen_params=gen_params,
        )
    )
    session.commit()


@pytest.fixture
def runs():
    calls: list[dict] = []
    with (
        patch.object(gen_routes, "run_generation", lambda **kw: calls.append(kw)),
        patch.object(gen_routes, "enqueue_generation", lambda gid, coro: None),
        patch.object(gen_routes, "get_task_manager", lambda: MagicMock()),
    ):
        yield calls


@pytest.fixture
def client(session):
    app = FastAPI()
    app.include_router(gen_routes.router)
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def test_retry_inherits_stored_tts_params(session, client, runs):
    """The arc's temperature survives a retry instead of reverting to defaults."""
    _add_failed(session, {"engine": "qwen", "tts_params": ARC_PARAMS})

    assert client.post(f"/generate/{GID}/retry").status_code == 200

    assert len(runs) == 1
    assert runs[0].get("tts_params") == ARC_PARAMS, (
        "retry dropped the stored tts_params — the unit would re-render at "
        "engine defaults, silently leaving its emotion arc"
    )


def test_retry_inherits_verify_config(session, client, runs):
    """Verify settings are reproduced too, not silently reset."""
    vcfg = {"chars_per_second": 17.6, "max_attempts": 10}
    _add_failed(session, {"engine": "qwen", "tts_params": ARC_PARAMS, "verify_config": vcfg})

    assert client.post(f"/generate/{GID}/retry").status_code == 200

    assert runs[0].get("verify_config") == vcfg
    assert runs[0].get("verify") is True, "a row rendered with verify must retry with verify"


def test_retry_without_stored_params_is_graceful(session, client, runs):
    """Rows predating this fix (gen_params NULL) still retry, just with defaults."""
    _add_failed(session, None)

    assert client.post(f"/generate/{GID}/retry").status_code == 200

    assert runs[0].get("tts_params") is None
    assert runs[0].get("verify") is False


@pytest.mark.asyncio
async def test_create_generation_persists_gen_params(session):
    """A row is self-describing from birth — so a row that DIES still knows
    what it was meant to run with. Without this, retry has nothing to inherit."""
    record = {"engine": "qwen", "tts_params": ARC_PARAMS}

    await history.create_generation(
        profile_id="narrator",
        text="x",
        language="en",
        audio_path="",
        duration=0,
        seed=1,
        db=session,
        generation_id="gen-created",
        status="generating",
        engine="qwen",
        gen_params=record,
    )

    row = session.query(DBGeneration).filter_by(id="gen-created").first()
    assert row.gen_params == record, "resolved params must persist at creation, not only on success"
