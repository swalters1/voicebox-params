"""Version rows must reach the API with every field they carry.

Both serializers build ``GenerationVersionResponse`` field-by-field rather than
from the ORM object, so a column added to the model is silently dropped unless
someone remembers to add a line in two places. That is exactly what happened to
``profile_id``: the migration ran, the recast wrote the value, the response
model declared the field, and the UI read it — but the serializers never
populated it, so every take rendered unlabelled.

Typechecking cannot catch this (the types are all correct) and the regenerate
tests cannot either (they assert on run_generation's kwargs, not on what the
API returns). Hence this file: assert on the SERIALIZED output.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import (
    Base,
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    VoiceProfile as DBVoiceProfile,
)
from backend.models import GenerationVersionResponse
from backend.services import history as history_svc, versions as versions_svc

GID = "gen-ser-1"


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    s.add(DBVoiceProfile(id="kate-child", name="Kate (child)"))
    s.add(DBVoiceProfile(id="kate-adult", name="Kate (adult)"))
    s.add(
        DBGeneration(
            id=GID,
            profile_id="kate-child",
            text="The map is not the territory.",
            language="en",
            status="completed",
        )
    )
    s.add(
        DBGenerationVersion(
            id="v1", generation_id=GID, label="take-1", audio_path="a.wav", is_default=False
        )
    )
    s.add(
        DBGenerationVersion(
            id="v2",
            generation_id=GID,
            label="take-2",
            audio_path="b.wav",
            is_default=True,
            profile_id="kate-adult",   # a recast take
            source_version_id="v1",
        )
    )
    s.commit()
    yield s
    s.close()


def test_response_model_declares_profile_id():
    assert "profile_id" in GenerationVersionResponse.model_fields


def test_list_versions_returns_profile_id(db):
    """The /generate/{id}/versions path."""
    by_id = {v.id: v for v in versions_svc.list_versions(GID, db)}

    assert by_id["v2"].profile_id == "kate-adult", "recast take lost its voice in serialization"
    assert by_id["v1"].profile_id is None, "an ordinary take carries no voice"


def _history_versions(db) -> dict:
    """Versions as the /history list serializes them.

    Targets ``_get_versions_for_generation`` directly: it is the second
    hand-written constructor and the one feeding the take list in the UI.
    ``get_generation`` does not attach versions at all, so it cannot stand in.
    """
    versions, _active = history_svc._get_versions_for_generation(GID, db)
    return {v.id: v for v in (versions or [])}


def test_history_returns_profile_id(db):
    """The path the UI take list actually renders from.

    This is the one that was broken: recasting worked, the value was stored,
    and the take still showed as a bare "take-2".
    """
    by_id = _history_versions(db)

    assert by_id["v2"].profile_id == "kate-adult", "recast take lost its voice in serialization"
    assert by_id["v1"].profile_id is None


def test_history_returns_source_version_id(db):
    """Dropped by the same constructor — it drives the 'from take-N' line."""
    assert _history_versions(db)["v2"].source_version_id == "v1"


def test_every_model_field_is_populated_by_both_serializers(db):
    """Guard against the NEXT field being dropped.

    Rather than enumerating fields by hand, compare the two hand-written
    constructors against the model: any declared field that comes back None
    from both, while the DB row has a value for it, is a dropped field.
    """
    row_v2 = db.query(DBGenerationVersion).filter_by(id="v2").one()
    from_versions = {v.id: v for v in versions_svc.list_versions(GID, db)}["v2"]
    from_history = _history_versions(db)["v2"]

    dropped = []
    for name in GenerationVersionResponse.model_fields:
        if name == "effects_chain":  # stored as JSON text, shape differs by design
            continue
        db_value = getattr(row_v2, name, None)
        if db_value is None:
            continue
        for label, resp in (("versions", from_versions), ("history", from_history)):
            if getattr(resp, name, None) is None:
                dropped.append(f"{label}.{name}")

    assert not dropped, f"fields present on the row but dropped in serialization: {dropped}"
