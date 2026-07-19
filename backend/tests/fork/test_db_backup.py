"""Pre-migration database snapshots.

Stage 1 of backup (issue #6). The feature is small; the properties that matter
are: it captures the database BEFORE migrations, it keeps one snapshot per app
version so repeated launches can't evict the pre-upgrade copy, and it NEVER
prevents the app from starting.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from backend.database.backup import (
    DEFAULT_RETAIN,
    existing_snapshots,
    snapshot_before_migration,
)


def _make_db(path: Path, rows: int = 3) -> Path:
    """A real SQLite file, so the copy is exercised against actual content."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE generations (id TEXT PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO generations VALUES (?, ?)", [(f"g{i}", f"line {i}") for i in range(rows)]
    )
    conn.commit()
    conn.close()
    return path


def test_takes_a_snapshot(tmp_path):
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    snap = snapshot_before_migration(db, "0.6.7", backups)

    assert snap is not None and snap.exists()
    assert snap.name.startswith("voicebox-pre-0.6.7-")
    assert snap.suffix == ".db"


def test_snapshot_is_a_faithful_copy(tmp_path):
    """A backup that doesn't restore is a hypothesis — read the copy back."""
    db = _make_db(tmp_path / "voicebox.db", rows=5)
    backups = tmp_path / "backups"

    snap = snapshot_before_migration(db, "0.6.7", backups)

    assert snap.read_bytes() == db.read_bytes()
    conn = sqlite3.connect(snap)
    try:
        assert conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == 5
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_captures_pre_migration_state(tmp_path):
    """The point of the feature: the snapshot must predate the schema change."""
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    snap = snapshot_before_migration(db, "0.6.7", backups)

    # Now do what a migration does.
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE generations ADD COLUMN gen_params TEXT")
    conn.commit()
    conn.close()

    live_cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(generations)")}
    snap_cols = {r[1] for r in sqlite3.connect(snap).execute("PRAGMA table_info(generations)")}

    assert "gen_params" in live_cols
    assert "gen_params" not in snap_cols, "snapshot was taken after the migration"


def test_fresh_install_takes_nothing(tmp_path):
    """No database yet — nothing to protect, and no empty file left behind."""
    backups = tmp_path / "backups"

    assert snapshot_before_migration(tmp_path / "voicebox.db", "0.6.7", backups) is None
    assert existing_snapshots(backups) == []


def test_one_snapshot_per_version(tmp_path):
    """Repeated launches of the same version must not churn the retention window.

    Otherwise starting the app a handful of times would quietly evict the
    pre-upgrade copy that is the entire reason the feature exists.
    """
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    first = snapshot_before_migration(db, "0.6.7", backups)
    again = snapshot_before_migration(db, "0.6.7", backups)

    assert first is not None
    assert again is None, "second launch of the same version re-snapshotted"
    assert len(existing_snapshots(backups)) == 1


def test_new_version_takes_a_new_snapshot(tmp_path):
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    snapshot_before_migration(db, "0.6.7", backups)
    snapshot_before_migration(db, "0.6.8", backups)

    names = [p.name for p in existing_snapshots(backups)]
    assert len(names) == 2
    assert any("0.6.7" in n for n in names) and any("0.6.8" in n for n in names)


def test_retention_prunes_oldest_first(tmp_path):
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    for minor in range(8):
        snapshot_before_migration(db, f"0.6.{minor}", backups, retain=3)

    remaining = [p.name for p in existing_snapshots(backups)]
    assert len(remaining) == 3
    # Oldest go first; the three newest versions survive.
    assert all(any(f"0.6.{m}" in n for n in remaining) for m in (5, 6, 7))
    assert not any("0.6.0" in n for n in remaining)


def test_default_retention_is_applied(tmp_path):
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    for minor in range(DEFAULT_RETAIN + 3):
        snapshot_before_migration(db, f"0.7.{minor}", backups)

    assert len(existing_snapshots(backups)) == DEFAULT_RETAIN


def test_failure_never_raises(tmp_path, monkeypatch):
    """Runs on the startup path: a failed backup must not stop the app booting.

    A missed backup is bad; a machine that won't start is worse.
    """
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    def boom(*a, **kw):
        raise OSError("No space left on device")

    monkeypatch.setattr("backend.database.backup.shutil.copy2", boom)

    assert snapshot_before_migration(db, "0.6.7", backups) is None


def test_unrelated_files_are_left_alone(tmp_path):
    """Pruning must only ever touch files it created."""
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"
    backups.mkdir()
    bystander = backups / "my-own-manual-backup.db"
    bystander.write_bytes(b"precious")

    for minor in range(6):
        snapshot_before_migration(db, f"0.6.{minor}", backups, retain=2)

    assert bystander.exists(), "pruning deleted a file it did not create"
    assert bystander.read_bytes() == b"precious"


def test_init_db_snapshots_before_migrating(tmp_path, monkeypatch):
    """Wiring check: the call must sit between engine creation and migrations.

    Placement is the whole reason a plain file copy is safe here, so pin the
    ordering rather than trusting the source to stay put.
    """
    from backend import config
    from backend.database import session as session_mod

    from sqlalchemy import create_engine

    from backend.database.models import Base

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config.set_data_dir(data_dir)

    # Build the pre-existing database from the real models: run_migrations is
    # stubbed below to record ordering, so the schema has to be valid already
    # for init_db's later steps (backfill, seeding) to run.
    seed_engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(bind=seed_engine)
    seed_engine.dispose()

    order: list[str] = []

    import backend.database.backup as backup_mod

    orig = backup_mod.snapshot_before_migration

    def traced_snapshot(*a, **kw):
        order.append("snapshot")
        return orig(*a, **kw)

    monkeypatch.setattr(backup_mod, "snapshot_before_migration", traced_snapshot)
    monkeypatch.setattr(
        session_mod, "run_migrations", lambda engine: order.append("migrate")
    )

    session_mod.init_db()

    assert order[:2] == ["snapshot", "migrate"], f"wrong order: {order}"
    assert len(existing_snapshots(config.get_backups_dir())) == 1
