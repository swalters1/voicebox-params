"""Automatic database snapshots taken before a schema migration.

The risk this covers is narrow and real: an app upgrade runs additive
migrations against a database holding voices and thousands of generations, and
until now the only protection was a human remembering to copy the data
directory first.

Why a plain file copy is correct HERE and nowhere else
------------------------------------------------------
This runs from :func:`backend.database.session.init_db` immediately before
``run_migrations()``. At that moment SQLAlchemy has created an engine but not
yet connected (it connects lazily), so the process holds **zero open
connections and has performed zero writes** — the file on disk is quiescent.
No WAL journal mode is configured either, so there is no ``-wal``/``-shm``
sidecar to keep consistent.

A user-triggered backup, taken while the server is live, has none of those
guarantees: a copy can catch a torn mid-transaction state that restores
corrupt. That path must use ``VACUUM INTO`` or the sqlite3 backup API instead.
Do not copy this function's approach there.

Failure policy
--------------
This runs on the startup path, so it must never prevent the app from booting.
Every failure is logged and swallowed: a missed backup is bad, a machine that
won't start is worse.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# How many snapshots to keep. Each is a full copy of the database, so this
# trades disk against history depth.
DEFAULT_RETAIN = 5

_PREFIX = "voicebox-pre-"
_MANUAL_PREFIX = "voicebox-manual-"
_SUFFIX = ".db"
# voicebox-pre-<version>-<YYYYmmdd-HHMMSS>.db
_NAME_RE = re.compile(rf"^{re.escape(_PREFIX)}(?P<version>.+)-\d{{8}}-\d{{6}}{re.escape(_SUFFIX)}$")
_MANUAL_RE = re.compile(
    rf"^{re.escape(_MANUAL_PREFIX)}(?P<version>.+)-\d{{8}}-\d{{6}}{re.escape(_SUFFIX)}$"
)


def _safe_version(version: str) -> str:
    # Version can contain dots but not path separators; keep it readable.
    return version.replace("/", "-").replace("\\", "-")


def _snapshot_name(version: str, when: datetime) -> str:
    return f"{_PREFIX}{_safe_version(version)}-{when.strftime('%Y%m%d-%H%M%S')}{_SUFFIX}"


def _manual_name(version: str, when: datetime) -> str:
    return f"{_MANUAL_PREFIX}{_safe_version(version)}-{when.strftime('%Y%m%d-%H%M%S')}{_SUFFIX}"


def existing_snapshots(backups_dir: Path) -> list[Path]:
    """Snapshots in *backups_dir*, oldest first (name sorts chronologically)."""
    if not backups_dir.is_dir():
        return []
    return sorted(p for p in backups_dir.glob(f"{_PREFIX}*{_SUFFIX}") if _NAME_RE.match(p.name))


def _has_snapshot_for_version(backups_dir: Path, version: str) -> bool:
    for p in existing_snapshots(backups_dir):
        m = _NAME_RE.match(p.name)
        if m and m.group("version") == version:
            return True
    return False


def _prune(backups_dir: Path, retain: int) -> None:
    snapshots = existing_snapshots(backups_dir)
    for old in snapshots[: max(0, len(snapshots) - retain)]:
        try:
            old.unlink()
            logger.info("Pruned old database snapshot: %s", old.name)
        except OSError as e:
            logger.warning("Could not prune snapshot %s: %s", old.name, e)


def snapshot_before_migration(
    db_path: Path,
    version: str,
    backups_dir: Path,
    *,
    retain: int = DEFAULT_RETAIN,
) -> Optional[Path]:
    """Copy *db_path* into *backups_dir* before migrations run.

    Takes at most ONE snapshot per app version: the point is to capture the
    database as it stood before a given upgrade touched it, so repeated
    launches of the same version must not churn through the retention window
    and evict the pre-upgrade copy that actually matters.

    Returns the snapshot path, or None if nothing was taken (fresh install,
    already snapshotted for this version, or a failure — all non-fatal).
    """
    try:
        if not db_path.exists():
            logger.debug("No database at %s yet; nothing to snapshot", db_path)
            return None

        if _has_snapshot_for_version(backups_dir, version):
            logger.debug("Database snapshot for v%s already exists; skipping", version)
            return None

        backups_dir.mkdir(parents=True, exist_ok=True)
        target = backups_dir / _snapshot_name(version, datetime.now())

        # copy2 preserves mtime, which makes the snapshot's age obvious in a
        # file listing even though the name already carries a timestamp.
        shutil.copy2(db_path, target)

        size_mb = target.stat().st_size / 1_000_000
        logger.info(
            "Database snapshot before v%s migration: %s (%.1f MB)", version, target.name, size_mb
        )

        _prune(backups_dir, retain)
        return target

    except Exception as e:  # never block startup on a backup failure
        logger.warning("Could not snapshot database before migration: %s", e)
        return None


def backup_now(db_path: Path, version: str, backups_dir: Path) -> Path:
    """Back up a LIVE database on demand (the Settings button).

    Unlike :func:`snapshot_before_migration`, this runs while the server is
    serving: connections are open and a write may be in flight. A file copy
    here can capture a torn mid-transaction state that restores corrupt, so
    this uses SQLite's ``VACUUM INTO`` instead — it takes a read transaction
    and writes a consistent, already-compacted database.

    Raises on failure. This one is user-initiated, so a silent no-op would be
    worse than an error: the caller must be able to say it didn't work.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"No database at {db_path}")

    backups_dir.mkdir(parents=True, exist_ok=True)
    target = backups_dir / _manual_name(version, datetime.now())
    if target.exists():  # same-second collision
        raise FileExistsError(f"Backup {target.name} already exists; try again")

    conn = sqlite3.connect(str(db_path))
    try:
        # Parameterised: the path is interpolated by SQLite, not by us.
        conn.execute("VACUUM INTO ?", (str(target),))
    finally:
        conn.close()

    logger.info("Manual database backup: %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
    return target


def list_backups(backups_dir: Path) -> list[dict]:
    """Every backup in *backups_dir*, newest first.

    Covers both kinds so the UI can show one list: ``automatic`` snapshots taken
    before a migration, and ``manual`` ones the user asked for.
    """
    if not backups_dir.is_dir():
        return []

    entries: list[dict] = []
    for path in backups_dir.glob(f"voicebox-*{_SUFFIX}"):
        auto = _NAME_RE.match(path.name)
        manual = _MANUAL_RE.match(path.name)
        match = auto or manual
        if not match:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": path.name,
                "kind": "automatic" if auto else "manual",
                "version": match.group("version"),
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime),
            }
        )

    entries.sort(key=lambda e: e["name"], reverse=True)
    return entries
