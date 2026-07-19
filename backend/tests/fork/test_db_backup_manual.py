"""User-initiated backups, taken while the database is live.

Stage 2 of backup (#6). The distinction from the pre-migration snapshot is the
whole point: that one runs before anything has connected, so a file copy is
safe. This one runs while the server is serving, where a copy can catch a torn
mid-transaction state. Hence ``VACUUM INTO``.
"""

import sqlite3

import pytest

from backend.database.backup import backup_now, list_backups, snapshot_before_migration


def _make_db(path, rows: int = 3):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE generations (id TEXT PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO generations VALUES (?, ?)", [(f"g{i}", f"line {i}") for i in range(rows)]
    )
    conn.commit()
    conn.close()
    return path


def test_backup_now_produces_a_valid_database(tmp_path):
    db = _make_db(tmp_path / "voicebox.db", rows=7)
    backups = tmp_path / "backups"

    out = backup_now(db, "0.6.7", backups)

    assert out.exists() and out.name.startswith("voicebox-manual-0.6.7-")
    conn = sqlite3.connect(out)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == 7
    finally:
        conn.close()


def test_backup_is_consistent_with_an_open_writer(tmp_path):
    """The reason this path exists.

    Hold an open connection with an UNCOMMITTED write, then back up. The backup
    must contain the committed state only — never the half-written row — and
    must be a valid database rather than a torn copy.
    """
    db = _make_db(tmp_path / "voicebox.db", rows=3)
    backups = tmp_path / "backups"

    writer = sqlite3.connect(db, isolation_level="DEFERRED")
    try:
        writer.execute("INSERT INTO generations VALUES ('uncommitted', 'not yet')")
        # deliberately NOT committed while the backup runs
        out = backup_now(db, "0.6.7", backups)
    finally:
        writer.rollback()
        writer.close()

    conn = sqlite3.connect(out)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        ids = {r[0] for r in conn.execute("SELECT id FROM generations")}
    finally:
        conn.close()

    assert "uncommitted" not in ids, "backup captured an uncommitted write"
    assert len(ids) == 3


def test_backup_now_raises_when_there_is_no_database(tmp_path):
    """User-initiated: a silent no-op would be worse than an error."""
    with pytest.raises(FileNotFoundError):
        backup_now(tmp_path / "nope.db", "0.6.7", tmp_path / "backups")


def test_manual_backups_are_not_pruned_by_snapshots(tmp_path):
    """Deliberate user backups must not be evicted by automatic retention."""
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    mine = backup_now(db, "0.6.7", backups)
    for minor in range(8):
        snapshot_before_migration(db, f"0.7.{minor}", backups, retain=2)

    assert mine.exists(), "automatic retention deleted a manual backup"


def test_list_backups_reports_both_kinds(tmp_path):
    db = _make_db(tmp_path / "voicebox.db")
    backups = tmp_path / "backups"

    snapshot_before_migration(db, "0.6.6", backups)
    backup_now(db, "0.6.7", backups)

    entries = list_backups(backups)
    by_kind = {e["kind"]: e for e in entries}

    assert set(by_kind) == {"automatic", "manual"}
    assert by_kind["automatic"]["version"] == "0.6.6"
    assert by_kind["manual"]["version"] == "0.6.7"
    assert all(e["size_bytes"] > 0 for e in entries)


def test_list_backups_ignores_foreign_files(tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / "notes.txt").write_text("hello")
    (backups / "some-other.db").write_bytes(b"x")

    assert list_backups(backups) == []


def test_list_backups_on_missing_directory(tmp_path):
    assert list_backups(tmp_path / "does-not-exist") == []
