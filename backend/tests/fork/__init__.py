"""Tests added by the voicebox-params fork.

CI runs this DIRECTORY wholesale rather than a list of filenames, so a new test
file here is picked up automatically. That is deliberate: the previous
approach pinned filenames in .github/workflows/ci.yml, and twice a new file was
added without updating the list — producing a green run that had silently
skipped the very tests written to catch a bug.

Upstream's tests remain in the parent directory. They predate this fork and
have not been audited for whether they pass in CI's lean dependency
environment, which is why they are not run here yet.
"""
