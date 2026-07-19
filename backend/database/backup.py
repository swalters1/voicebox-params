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
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# How many snapshots to keep. Each is a full copy of the database, so this
# trades disk against history depth.
DEFAULT_RETAIN = 5

_PREFIX = "voicebox-pre-"
_SUFFIX = ".db"
# voicebox-pre-<version>-<YYYYmmdd-HHMMSS>.db
_NAME_RE = re.compile(rf"^{re.escape(_PREFIX)}(?P<version>.+)-\d{{8}}-\d{{6}}{re.escape(_SUFFIX)}$")


def _snapshot_name(version: str, when: datetime) -> str:
    # Version can contain dots but not path separators; keep it readable.
    safe = version.replace("/", "-").replace("\\", "-")
    return f"{_PREFIX}{safe}-{when.strftime('%Y%m%d-%H%M%S')}{_SUFFIX}"


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
