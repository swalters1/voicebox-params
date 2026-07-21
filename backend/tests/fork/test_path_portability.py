"""Storage paths must be OS-portable.

`to_storage_path` used to return `str(Path)`, which emits OS-native separators
— so a DB written on Windows stored `generations\\uuid.wav`. On Linux
`Path("generations\\uuid.wav")` is a single filename containing a literal
backslash, so `resolve_storage_path` could not find the file and every imported
render / profile sample resolved to a nonexistent path (surfaced moving a
real Windows DB into the Docker container).

Fix is two-sided: write with `.as_posix()`, and normalize `\\`->`/` on read so
existing Windows DBs keep working. These tests pin both directions; the read
test is red-first on any POSIX host (CI runs on Linux).
"""

from pathlib import Path

import pytest

from backend import config


@pytest.fixture
def data_dir(tmp_path):
    """Point config at a temp data dir with a real generations/ file."""
    original = config.get_data_dir()
    config.set_data_dir(tmp_path)
    (tmp_path / "generations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "generations" / "clip.wav").write_bytes(b"RIFF")
    try:
        yield tmp_path
    finally:
        config.set_data_dir(original)


def test_to_storage_path_is_posix(data_dir):
    """A path under the data dir is stored with forward slashes, no backslashes."""
    stored = config.to_storage_path(data_dir / "generations" / "clip.wav")
    assert stored == "generations/clip.wav"
    assert "\\" not in stored


def test_resolve_accepts_backslash_paths(data_dir):
    """A legacy Windows-style stored path resolves to the real file on any OS."""
    resolved = config.resolve_storage_path("generations\\clip.wav")
    assert resolved is not None
    assert resolved.as_posix().endswith("generations/clip.wav")
    assert resolved.exists()


def test_resolve_accepts_posix_paths(data_dir):
    """The forward-slash form (what we now write) also resolves."""
    resolved = config.resolve_storage_path("generations/clip.wav")
    assert resolved is not None
    assert resolved.exists()


def test_roundtrip_windows_db_on_posix(data_dir):
    """End-to-end: a value a Windows build would have stored resolves back."""
    windows_value = "generations\\clip.wav"  # what str(Path) produced on Windows
    resolved = config.resolve_storage_path(windows_value)
    assert resolved is not None and resolved.exists()
    # and re-storing the resolved path yields the portable POSIX form
    assert config.to_storage_path(resolved) == "generations/clip.wav"
