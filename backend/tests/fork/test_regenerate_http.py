"""Regenerate over real HTTP — the layer the direct-call tests can't see.

``test_regenerate_as.py`` calls the route function directly with hand-built
objects. That verifies the inheritance logic but skips everything FastAPI does
in front of it: body parsing, an absent body, and Pydantic validation of what
the client actually sends.

That gap is not hypothetical. v0.6.3 shipped a 422 on plain Regenerate that the
direct-call tests could not have caught, because they passed a clean dict where
the real request carried a *persisted* config containing a key the spec didn't
advertise. These tests drive the endpoint the way the app does.

A minimal app mounting only the generations router is used here rather than
``create_app()``, which does real startup work (migrations, model registry)
irrelevant to request handling.
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

GID = "gen-http-1"


@pytest.fixture
def session():
    # TestClient runs the app in a worker thread and SQLite ":memory:" is
    # per-connection, so the default pool would hand that thread a fresh, empty
    # database. StaticPool keeps one shared connection.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    for pid, name in (("kate-child", "Kate (child)"), ("kate-adult", "Kate (adult)")):
        s.add(DBVoiceProfile(id=pid, name=name, default_engine="chatterbox_turbo"))
    s.commit()
    yield s
    s.close()


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


def _seed(session, *, gen_params=None, profile_id="kate-child"):
    session.add(
        DBGeneration(
            id=GID,
            profile_id=profile_id,
            text="The map is not the territory.",
            language="en",
            status="completed",
            engine="chatterbox_turbo",
            seed=1234,
            gen_params=gen_params,
        )
    )
    session.commit()


# The config the server persists for a verified render, matching a real v0.6.3
# generation row. chars_per_second is deliberately Kate's MEASURED 17.6 rather
# than the spec default of 16.0: if it were the default, an assertion on the
# resolved value could not distinguish "inherited" from "dropped and
# re-defaulted", and would pass either way.
STORED_VERIFY_CONFIG = {
    "coverage_min": 0.8,
    "duration_ratio_min": 0.55,
    "chars_per_second": 17.6,
    "model_size": "base",
    "min_words_for_check": 3,
    "ignore_leading_words": 1,
    "max_attempts": 10,
    "retry_temperature": 0.0,
    "split_enabled": True,
    "split_min_chars": 120,
    "join_silence_ms": 250,
    "max_split_depth": 2,
}


def test_post_with_no_body_at_all(client, session, runs):
    """Bare POST — how the endpoint behaved before it took a body."""
    _seed(session)
    r = client.post(f"/generate/{GID}/regenerate")
    assert r.status_code == 200, r.text
    assert runs[0]["recast_profile_id"] is None


def test_post_with_json_content_type_and_empty_body(client, session, runs):
    """What the app actually sends.

    apiClient.request() always sets Content-Type: application/json, and plain
    Regenerate passes no body — so the server receives a JSON content type with
    zero bytes. Adding an optional body model must not break that.
    """
    _seed(session)
    r = client.post(
        f"/generate/{GID}/regenerate",
        headers={"Content-Type": "application/json"},
        content=b"",
    )
    assert r.status_code == 200, r.text


def test_regenerate_of_a_verified_row_over_http(client, session, runs):
    """The v0.6.3 regression, driven the way the user hit it.

    A row rendered with verify on carries a stored config including
    ``max_split_depth`` — persisted by the server, but absent from
    VERIFY_PARAM_SPEC at the time. Feeding it back through strict validation
    returned 422 on a request the user typed nothing into.
    """
    _seed(session, gen_params={"verify_config": STORED_VERIFY_CONFIG})

    r = client.post(f"/generate/{GID}/regenerate")

    assert r.status_code == 200, f"regenerate of a verified row must not 422: {r.text}"
    vc = runs[0]["verify_config"] or {}
    assert vc.get("coverage_min") == 0.8
    # Same voice, so the measured pace is still valid and must carry over.
    assert vc.get("chars_per_second") == 17.6


def test_recast_of_a_verified_row_over_http(client, session, runs):
    _seed(session, gen_params={"verify_config": STORED_VERIFY_CONFIG})

    r = client.post(f"/generate/{GID}/regenerate", json={"profile_id": "kate-adult"})

    assert r.status_code == 200, r.text
    assert runs[0]["profile_id"] == "kate-adult"
    assert runs[0]["recast_profile_id"] == "kate-adult"
    # Pace is per-voice: Kate (child)'s measured 17.6 must not be used to judge
    # Kate (adult). It falls back to the spec default until re-measured.
    assert (runs[0]["verify_config"] or {}).get("chars_per_second") == 16.0


def test_caller_supplied_bad_key_still_422s(client, session, runs):
    """Leniency is for inherited values only — a typed typo must stay loud."""
    _seed(session)
    r = client.post(
        f"/generate/{GID}/regenerate",
        json={"verify": True, "verify_config": {"nonsense_key": 1}},
    )
    assert r.status_code == 422
    assert not runs


def test_unknown_generation_404s(client, session, runs):
    r = client.post("/generate/does-not-exist/regenerate")
    assert r.status_code == 404
    assert not runs


def test_malformed_json_body_is_rejected_cleanly(client, session, runs):
    """Garbage in the body should 422, not 500."""
    _seed(session)
    r = client.post(
        f"/generate/{GID}/regenerate",
        headers={"Content-Type": "application/json"},
        content=b"{not json",
    )
    assert r.status_code == 422
    assert not runs
