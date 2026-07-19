"""Restoring the database from a backup.

Stage 3 of backup (#6). This is the one operation that can lose data by
SUCCEEDING — a restore of the wrong file replaces everything — so it is
deliberately the most cautious code in the backup feature.

Two-phase by design
-------------------
The server holds ``voicebox.db`` open, so it cannot overwrite it in place.
Rather than closing the engine and performing surgery on a live database, a
restore is **staged** while the app runs and **applied on the next startup**,
at the very top of ``init_db()`` before any engine exists. That is the same
quiescent moment that makes the pre-migration snapshot safe: no connections,
no writes, nothing to tear.

The failure mode this buys is "the restart didn't restore", which is
recoverable and obvious. The alternative — swapping a database out from under
a running server — fails as "half-restored", which is neither.

Safety copy
-----------
Applying a restore writes the CURRENT database aside first. If you restore the
wrong backup, the state you just replaced is still on disk.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Staged restore lives beside the backups. The leading dot keeps it out of the
# backup listing's glob.
PENDING_NAME = ".pending-restore.db"

_PRERESTORE_PREFIX = "voicebox-prerestore-"

# Tables the app cannot function without — a file missing these is not a
# Voicebox database, whatever its name says.
_REQUIRED_TABLES = {"profiles", "generations"}


class RestoreError(ValueError):
    """A backup was rejected. The message is shown to the user."""


def _table_columns(conn: sqlite3.Connection) -> dict[str, set[str]]:
    tables = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not r[0].startswith("sqlite_")
    }
    return {t: {c[1] for c in conn.execute(f"PRAGMA table_info('{t}')")} for t in tables}


def validate_backup(path: Path, metadata=None) -> None:
    """Reject anything that isn't a restorable Voicebox database.

    Checks, in order: the file exists, SQLite can read it, it passes
    ``integrity_check``, it has the core tables, and its schema is not NEWER
    than this app's. That last one matters because migrations only run forward
    — restoring a future database would leave columns the code cannot see, and
    silently so.

    The schema is inspected directly rather than trusting the version in the
    filename, which is just a string a user can rename.
    """
    if not path.exists():
        raise RestoreError(f"Backup not found: {path.name}")

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise RestoreError(f"Not a readable database: {e}")

    try:
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError as e:
            raise RestoreError(f"Not a valid SQLite database: {e}")
        if integrity != "ok":
            raise RestoreError(f"Database failed integrity check: {integrity}")

        columns = _table_columns(conn)
        missing = _REQUIRED_TABLES - set(columns)
        if missing:
            raise RestoreError(
                f"Backup is missing core tables ({', '.join(sorted(missing))}); "
                "it does not look like a Voicebox database"
            )

        if metadata is not None:
            known = {t.name: set(t.columns.keys()) for t in metadata.sorted_tables}
            unknown: list[str] = []
            for table, cols in columns.items():
                if table not in known:
                    unknown.append(table)
                    continue
                unknown += [f"{table}.{c}" for c in sorted(cols - known[table])]
            if unknown:
                raise RestoreError(
                    "Backup was made by a NEWER version of Voicebox "
                    f"(unrecognised: {', '.join(unknown[:5])}"
                    f"{'…' if len(unknown) > 5 else ''}). "
                    "Update the app before restoring it."
                )
    finally:
        conn.close()


def stage_restore(backup_path: Path, backups_dir: Path, metadata=None) -> Path:
    """Validate *backup_path* and stage it to be applied on next startup.

    Returns the staged path. Raises :class:`RestoreError` if the backup is not
    fit to restore — validation happens HERE, while a user is watching, rather
    than at startup where a rejection would be a log line nobody reads.
    """
    validate_backup(backup_path, metadata)

    backups_dir.mkdir(parents=True, exist_ok=True)
    pending = backups_dir / PENDING_NAME

    # Copy rather than move: the backup stays in the list, so a mis-click can be
    # cancelled and re-done without losing the file.
    shutil.copy2(backup_path, pending)
    logger.info("Restore staged from %s; will be applied on next start", backup_path.name)
    return pending


def pending_restore(backups_dir: Path) -> Optional[Path]:
    """The staged restore, if one is waiting."""
    pending = backups_dir / PENDING_NAME
    return pending if pending.exists() else None


def cancel_pending_restore(backups_dir: Path) -> bool:
    """Discard a staged restore. Returns True if one was discarded."""
    pending = pending_restore(backups_dir)
    if not pending:
        return False
    try:
        pending.unlink()
        logger.info("Staged restore cancelled")
        return True
    except OSError as e:
        logger.warning("Could not cancel staged restore: %s", e)
        return False


def _safety_copy(db_path: Path, backups_dir: Path) -> Optional[Path]:
    """Preserve the database that is about to be replaced."""
    if not db_path.exists():
        return None
    target = backups_dir / f"{_PRERESTORE_PREFIX}{datetime.now():%Y%m%d-%H%M%S}.db"
    try:
        shutil.copy2(db_path, target)
        logger.info("Saved pre-restore copy: %s", target.name)
        return target
    except OSError as e:
        # Refuse to proceed: a restore without a way back is exactly what this
        # feature exists to prevent.
        raise RestoreError(f"Could not save a pre-restore copy, aborting restore: {e}")


def apply_pending_restore(db_path: Path, backups_dir: Path, *, attempts: int = 5) -> Optional[Path]:
    """Apply a staged restore. Call at the TOP of init_db, before any engine.

    Returns the restored path, or None if nothing was staged. Never raises:
    a restore that cannot be applied must not stop the app from starting — the
    staged file is left in place so the next launch tries again.

    On Windows a lingering handle from the previous process can block the
    replace for a moment, hence the short retry.
    """
    pending = pending_restore(backups_dir)
    if not pending:
        return None

    try:
        # Re-validate: the file has been sitting on disk since staging.
        validate_backup(pending)

        _safety_copy(db_path, backups_dir)

        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                os.replace(pending, db_path)  # atomic within a volume
                logger.info("Database restored from staged backup")
                return db_path
            except OSError as e:  # Windows: file still held by the old process
                last_error = e
                time.sleep(0.2 * (attempt + 1))

        raise RestoreError(f"Could not replace the database after {attempts} attempts: {last_error}")

    except Exception as e:
        # Leave the staged file alone so the next start retries.
        logger.error("Pending restore was NOT applied: %s", e)
        return None
