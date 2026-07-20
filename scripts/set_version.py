#!/usr/bin/env python3
"""Set (or check) the app version across every file that carries one.

A release used to mean hand-editing five files and then pushing a matching tag.
That is exactly the kind of chore that gets a digit wrong, and the failure is
nasty: the build goes green, the installer works, and then a user's app 404s
trying to download its GPU backend from a release tag that does not exist.

Three of these files genuinely matter:

  * tauri/src-tauri/tauri.conf.json  — CI reads it for the release tag and the
    sidecar upload target (``tagName: v__VERSION__``)
  * backend/__init__.py              — builds the sidecar download URL as
    {releases_url}/v{__version__}/..., and the Rust shell version-checks the
    downloaded backend against the app version
  * the git tag                      — triggers the release workflow

The rest are metadata. They are stamped anyway, because "unmanaged" turned out
to mean "silently wrong": tauri/src-tauri/Cargo.toml sat at 0.5.0 for four
releases and nobody noticed, since Tauri reads the version from
tauri.conf.json. A file that carries a version and is never updated is worse
than one that has none — it reads as fact.

Usage
-----
    python scripts/set_version.py 0.7.0     # stamp every file
    python scripts/set_version.py --check   # verify they all agree

``--check`` is the same comparison the release workflow makes against the git
tag, so it can be run locally before tagging.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (path, regex with a single capturing group around the version)
TARGETS: list[tuple[Path, re.Pattern]] = [
    (ROOT / "backend" / "__init__.py", re.compile(r'(?m)^__version__ = "([^"]+)"')),
    (ROOT / "tauri" / "src-tauri" / "tauri.conf.json", re.compile(r'"version":\s*"([^"]+)"')),
    (ROOT / "tauri" / "src-tauri" / "Cargo.toml", re.compile(r'(?m)^version = "([^"]+)"')),
    (ROOT / "package.json", re.compile(r'"version":\s*"([^"]+)"')),
    (ROOT / "app" / "package.json", re.compile(r'"version":\s*"([^"]+)"')),
    (ROOT / "tauri" / "package.json", re.compile(r'"version":\s*"([^"]+)"')),
]

# The three that break a release if they disagree; the rest are metadata.
CRITICAL = {"backend/__init__.py", "tauri/src-tauri/tauri.conf.json"}

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for path, pattern in TARGETS:
        if not path.exists():
            found[rel(path)] = None
            continue
        match = pattern.search(path.read_text(encoding="utf-8"))
        found[rel(path)] = match.group(1) if match else None
    return found


def stamp(version: str) -> int:
    changed = 0
    for path, pattern in TARGETS:
        if not path.exists():
            print(f"  !! missing   {rel(path)}")
            continue
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if not match:
            print(f"  !! no match  {rel(path)}")
            continue
        if match.group(1) == version:
            print(f"     unchanged {rel(path)}")
            continue
        # Replace only the captured group, so surrounding syntax is untouched.
        start, end = match.span(1)
        path.write_text(text[:start] + version + text[end:], encoding="utf-8")
        print(f"  -> {match.group(1)} => {version}  {rel(path)}")
        changed += 1

    # A JSON file we mangled would fail the build much later and confusingly.
    for path, _ in TARGETS:
        if path.suffix == ".json" and path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"  !! {rel(path)} is no longer valid JSON: {e}")
                return -1
    return changed


def check() -> int:
    found = read_versions()
    width = max(len(k) for k in found)
    for name, value in found.items():
        flag = "*" if name in CRITICAL else " "
        print(f"  {flag} {name:<{width}}  {value or '(not found)'}")

    values = {v for v in found.values() if v}
    missing = [k for k, v in found.items() if v is None]

    print()
    if missing:
        print(f"MISSING in: {', '.join(missing)}")
    if len(values) == 1 and not missing:
        print(f"OK — all files agree on {values.pop()}")
        print("(* = a mismatch here breaks the GPU backend download)")
        return 0
    print(f"MISMATCH — found {sorted(values)}")
    print("Run: python scripts/set_version.py <version>")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", nargs="?", help="version to set, e.g. 0.7.0")
    parser.add_argument("--check", action="store_true", help="verify all files agree")
    args = parser.parse_args()

    if args.check or not args.version:
        return check()

    if not SEMVER.match(args.version):
        # Tauri validates semver and MSI's ProductVersion is 3-part numeric, so
        # a 4-part or suffixed version fails much later, in the bundler.
        print(f"error: {args.version!r} is not major.minor.patch")
        return 2

    print(f"Setting version to {args.version}\n")
    if stamp(args.version) < 0:
        return 1
    print()
    return check()


if __name__ == "__main__":
    sys.exit(main())
