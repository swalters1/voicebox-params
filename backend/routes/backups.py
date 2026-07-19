"""Database backup endpoints.

Stage 2 of backup (#6). The automatic pre-migration snapshot needs no API — it
fires on startup — but a user-initiated backup does, and it takes a different
code path: the server is live, so it uses ``VACUUM INTO`` rather than a file
copy (see ``database/backup.py`` for why that distinction matters).
"""

import logging

from fastapi import APIRouter, HTTPException

from .. import config, models
from ..database.backup import backup_now, list_backups

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/backups", response_model=models.BackupListResponse)
async def get_backups():
    """List database backups, newest first (automatic and manual)."""
    entries = list_backups(config.get_backups_dir())
    return models.BackupListResponse(
        backups=[models.BackupResponse(**e) for e in entries],
        directory=str(config.get_backups_dir()),
    )


@router.post("/backups", response_model=models.BackupResponse)
async def create_backup():
    """Back up the database now.

    Safe to call while the server is serving requests — ``VACUUM INTO`` takes a
    consistent snapshot under a read transaction rather than copying bytes that
    might be mid-write.
    """
    from .. import __version__

    try:
        path = backup_now(config.get_db_path(), __version__, config.get_backups_dir())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # User-initiated: surface the failure rather than pretending it worked.
        logger.exception("Manual backup failed")
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")

    stat = path.stat()
    from datetime import datetime

    return models.BackupResponse(
        name=path.name,
        kind="manual",
        version=__version__,
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime),
    )
