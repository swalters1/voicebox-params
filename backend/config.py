"""
Configuration module for voicebox backend.

Handles data directory configuration for production bundling.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Allow users to override the HuggingFace model download directory.
# Set VOICEBOX_MODELS_DIR to an absolute path before starting the server.
# This sets HF_HUB_CACHE so all huggingface_hub downloads go to that path.
_custom_models_dir = os.environ.get("VOICEBOX_MODELS_DIR")
if _custom_models_dir:
    os.environ["HF_HUB_CACHE"] = _custom_models_dir
    logger.info("Model download path set to: %s", _custom_models_dir)

# Default data directory (used in development)
_data_dir = Path("data").resolve()


def _path_relative_to_any_data_dir(path: Path) -> Path | None:
    """Extract the path within a data dir from an absolute or relative path."""
    parts = path.parts
    for idx, part in enumerate(parts):
        if part != "data":
            continue

        tail = parts[idx + 1 :]
        if tail:
            return Path(*tail)
        return Path()

    return None


def set_data_dir(path: str | Path):
    """
    Set the data directory path.

    Args:
        path: Path to the data directory
    """
    global _data_dir
    _data_dir = Path(path).resolve()
    _data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory set to: %s", _data_dir)


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path to the data directory
    """
    return _data_dir


def to_storage_path(path: str | Path) -> str:
    """Convert a filesystem path to a DB-safe path relative to the data dir.

    Always emits POSIX (forward-slash) separators so the stored value is
    portable across operating systems — a DB written on Windows must resolve
    when the same data dir is opened on Linux (e.g. a Docker container).
    """
    resolved_path = Path(path).resolve()

    relative_to_any_data_dir = _path_relative_to_any_data_dir(resolved_path)
    if relative_to_any_data_dir is not None:
        return relative_to_any_data_dir.as_posix()

    try:
        return resolved_path.relative_to(_data_dir).as_posix()
    except ValueError:
        return str(resolved_path)


def resolve_storage_path(path: str | Path | None) -> Path | None:
    """Resolve a DB-stored path against the configured data dir."""
    if path is None:
        return None

    # Older records (and any DB created on Windows) may use backslash
    # separators. Normalize to POSIX before parsing so they resolve on any OS;
    # backslash is never a character in the relative paths we store, so this is
    # safe. Windows still accepts the forward slashes in the result.
    stored_path = Path(str(path).replace("\\", "/"))
    # Empty paths (e.g. failed generations) must not resolve to the data
    # dir itself, which exists and would defeat the callers' 404 guards.
    # Path("") is truthy, so check parts rather than the raw value.
    if not stored_path.parts:
        return None
    if stored_path.is_absolute():
        rebased_path = _path_relative_to_any_data_dir(stored_path)
        if rebased_path is not None:
            candidate = (_data_dir / rebased_path).resolve()
            if candidate.exists() or not stored_path.exists():
                return candidate

        return stored_path

    # 0.3.0 records sometimes stored relative paths with the data-dir name
    # baked in (e.g. "data/profiles/..."). Joining those directly with
    # _data_dir produces a spurious "<data_dir>/data/profiles/..." nest.
    if stored_path.parts and stored_path.parts[0] == "data":
        stored_path = (
            Path(*stored_path.parts[1:]) if len(stored_path.parts) > 1 else Path()
        )

    return (_data_dir / stored_path).resolve()


def get_db_path() -> Path:
    """Get database file path."""
    return _data_dir / "voicebox.db"


def get_profiles_dir() -> Path:
    """Get profiles directory path."""
    path = _data_dir / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generations_dir() -> Path:
    """Get generations directory path."""
    path = _data_dir / "generations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_captures_dir() -> Path:
    """Get captures directory path."""
    path = _data_dir / "captures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    """Get cache directory path."""
    path = _data_dir / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """Get models directory path."""
    path = _data_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_backups_dir() -> Path:
    """Get backups directory path (database snapshots)."""
    path = _data_dir / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Voicebox Cloud (backup & sync). Two hosts: the web app owns auth + device
# pairing (voicebox.sh), the API owns sync + account endpoints
# (api.voicebox.sh). Override both for local development, e.g.
# VOICEBOX_CLOUD_URL=http://localhost:17592 VOICEBOX_CLOUD_API_URL=http://localhost:17593
def get_cloud_web_url() -> str:
    """Base URL of the Voicebox Cloud web app (auth + /connect + exchange)."""
    return os.environ.get("VOICEBOX_CLOUD_URL", "https://voicebox.sh").rstrip("/")


def get_cloud_api_url() -> str:
    """Base URL of the Voicebox Cloud API (bearer-authenticated sync/account)."""
    return os.environ.get("VOICEBOX_CLOUD_API_URL", "https://api.voicebox.sh").rstrip("/")


# Where the prebuilt GPU backend binaries (CUDA/ROCm sidecars + lib bundles) are
# downloaded from: "<base>/<tag>/<asset>". Single source of truth, overridable
# via VOICEBOX_RELEASES_URL so a fork (or a mirror) points at its own releases
# WITHOUT patching the download services — a hardcoded per-repo URL otherwise
# breaks every fork the moment it ships its own backend build.
_DEFAULT_RELEASES_URL = "https://github.com/swalters1/voicebox-params/releases/download"


def get_releases_base_url() -> str:
    """Base URL for downloading prebuilt backend binaries (GPU sidecars)."""
    return os.environ.get("VOICEBOX_RELEASES_URL", _DEFAULT_RELEASES_URL).rstrip("/")
