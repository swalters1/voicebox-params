"""Restoring the database from a backup.

Stage 3 of backup (#6). A backup that has never been restored is a hypothesis,
so the centrepiece here is the round trip: back up, change the database,
restore, and assert the change is gone.

Restore is also the one operation that loses data by SUCCEEDING, so the
guards get as much attention as the happy path.
"""

import sqlite3

import pytest
from sqlalchemy import create_engine

from backend.database.backup import backup_now
from backend.database.models import Base
from backend.database.restore import (
    PENDING_NAME,
    RestoreError,
    apply_pending_restore,
    cancel_pending_restore,
    pending_restore,
    stage_restore,
    validate_backup,
)


def _real_schema_db(path):
    """A database with the app's actual schema, so guards see real columns."""
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(bind=engine)
    engine.dispose()
    return path


def _profiles(db) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {r[0] for r in conn.execute("SELECT id FROM profiles")}
    finally:
        conn.close()


def _add_profile(db, pid: str):
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO profiles (id, name) VALUES (?, ?)", (pid, pid))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_round_trip_restores_the_earlier_state(tmp_path):
    """Back up, change the database, restore, and the change is gone.

    This is the test the whole feature exists to pass.
    """
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    _add_profile(db, "kate")

    backup = backup_now(db, "0.6.7", backups)

    # Time passes; the database moves on.
    _add_profile(db, "added-after-backup")
    assert _profiles(db) == {"kate", "added-after-backup"}

    stage_restore(backup, backups, Base.metadata)
    applied = apply_pending_restore(db, backups)

    assert applied is not None
    assert _profiles(db) == {"kate"}, "restore did not roll the database back"
    assert pending_restore(backups) is None, "staged file should be consumed"


def test_restore_saves_the_replaced_database_first(tmp_path):
    """Restoring the wrong backup must not be terminal."""
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    _add_profile(db, "kate")
    backup = backup_now(db, "0.6.7", backups)

    _add_profile(db, "work-i-would-hate-to-lose")

    stage_restore(backup, backups, Base.metadata)
    apply_pending_restore(db, backups)

    saved = list(backups.glob("voicebox-prerestore-*.db"))
    assert len(saved) == 1, "no pre-restore copy was kept"
    assert "work-i-would-hate-to-lose" in _profiles(saved[0])


def test_restore_survives_a_restart_cycle(tmp_path):
    """Staging happens while the app runs; applying happens on next start."""
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    _add_profile(db, "original")
    backup = backup_now(db, "0.6.7", backups)
    _add_profile(db, "later")

    stage_restore(backup, backups, Base.metadata)

    # Still staged — nothing has changed yet, which is the point.
    assert pending_restore(backups) is not None
    assert _profiles(db) == {"original", "later"}

    apply_pending_restore(db, backups)  # "restart"
    assert _profiles(db) == {"original"}


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_rejects_a_file_that_is_not_a_database(tmp_path):
    junk = tmp_path / "notes.db"
    junk.write_bytes(b"this is not a database")

    with pytest.raises(RestoreError):
        validate_backup(junk)


def test_rejects_a_database_without_core_tables(tmp_path):
    other = tmp_path / "other.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE unrelated (id TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(RestoreError, match="core tables"):
        validate_backup(other)


def test_rejects_a_backup_from_a_newer_version(tmp_path):
    """Migrations only run forward.

    A database from a future version carries columns this code cannot see, and
    restoring it would fail silently rather than loudly. The check inspects the
    SCHEMA, not the version in the filename — which is just a renameable string.
    """
    future = _real_schema_db(tmp_path / "future.db")
    conn = sqlite3.connect(future)
    conn.execute("ALTER TABLE generations ADD COLUMN some_future_column TEXT")
    conn.commit()
    conn.close()

    with pytest.raises(RestoreError, match="NEWER"):
        validate_backup(future, Base.metadata)


def test_accepts_a_backup_from_an_older_version(tmp_path):
    """Older is fine — migrations will bring it forward."""
    old = _real_schema_db(tmp_path / "old.db")
    conn = sqlite3.connect(old)
    # Simulate a column this app added later by dropping it from the backup.
    conn.execute("ALTER TABLE generations DROP COLUMN gen_params")
    conn.commit()
    conn.close()

    validate_backup(old, Base.metadata)  # must not raise


def test_staging_rejects_before_anything_is_touched(tmp_path):
    """Validation happens while a user is watching, not at startup."""
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    backups.mkdir()
    junk = backups / "voicebox-manual-9.9.9-20260101-000000.db"
    junk.write_bytes(b"not a database")

    with pytest.raises(RestoreError):
        stage_restore(junk, backups, Base.metadata)

    assert pending_restore(backups) is None
    assert db.exists()


def test_apply_never_raises_on_a_corrupt_staged_file(tmp_path):
    """A bad staged file must not stop the app from starting."""
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    backups.mkdir()
    _add_profile(db, "kate")
    (backups / PENDING_NAME).write_bytes(b"corrupt")

    assert apply_pending_restore(db, backups) is None
    assert _profiles(db) == {"kate"}, "live database was damaged by a bad restore"


def test_failed_restore_leaves_the_staged_file_for_a_retry(tmp_path):
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / PENDING_NAME).write_bytes(b"corrupt")

    apply_pending_restore(db, backups)

    assert pending_restore(backups) is not None


def test_nothing_staged_is_a_no_op(tmp_path):
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    _add_profile(db, "kate")

    assert apply_pending_restore(db, backups) is None
    assert _profiles(db) == {"kate"}


def test_cancel_discards_the_staged_restore(tmp_path):
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    _add_profile(db, "kate")
    backup = backup_now(db, "0.6.7", backups)
    _add_profile(db, "later")

    stage_restore(backup, backups, Base.metadata)
    assert cancel_pending_restore(backups) is True
    assert pending_restore(backups) is None

    apply_pending_restore(db, backups)
    assert _profiles(db) == {"kate", "later"}, "cancelled restore was applied anyway"


def test_staging_keeps_the_original_backup(tmp_path):
    """A mis-click should be cancellable without losing the backup."""
    db = _real_schema_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    _add_profile(db, "kate")
    backup = backup_now(db, "0.6.7", backups)

    stage_restore(backup, backups, Base.metadata)
    cancel_pending_restore(backups)

    assert backup.exists()
