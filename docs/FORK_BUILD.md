# Building the fork to installable `.exe`s

This closes the loop: the fork compiles to the same Windows installers + sidecar
binaries as upstream, so releases follow normal branch/merge/tag patterns.

## Architecture (what a build produces)

Voicebox is a **Tauri** app (Rust shell + web frontend) that talks over HTTP to a
**Python FastAPI backend** shipped as a bundled **sidecar** binary. A build makes:

- `voicebox-server[-cuda|-rocm].exe` — the PyInstaller-frozen backend (this repo's
  `backend/`), one variant per accelerator. This is the `.exe` you already run in
  production at `%APPDATA%\sh.voicebox.app\backends\...`.
- `voicebox-mcp.exe` — the MCP shim sidecar.
- The Tauri installer (NSIS `.exe` / MSI) that bundles the shell + frontend + sidecars.

## Prerequisites

- **bun** `1.3.8` (the repo's package manager; also provides a node ≥20 runtime — the
  frontend requires node ≥20.19).
- **Rust** toolchain (for `tauri build`).
- **Python 3.10** + a **full** backend venv — *all* engines, not just turbo, so the
  sidecar bundles them. Use `just setup-python` (creates `backend/venv`, installs
  `requirements.txt`, then `chatterbox-tts --no-deps` / `hume-tada --no-deps`).
- **CUDA 12.8** toolkit for the `--cuda` sidecar (matches `cu128` in `release.yml`).
- **Tauri updater signing keys** — *only if you re-enable signed auto-updates*. The
  fork ships with `bundle.createUpdaterArtifacts: false` (see "Signing" below), so a
  plain unsigned installer builds with **no key needed**. Upstream defaulted to
  `"v1Compatible"`, which made `tauri build` fail on any machine/CI without
  `TAURI_SIGNING_PRIVATE_KEY`.

> The dev server we run for the pipeline (`D:\code\voicebox-dev`) is a **turbo-only**
> venv — fine for running, **not** for building a shippable sidecar. A release build
> needs the full dependency set so every engine is present in the frozen `.exe`.

## Local build

```bash
bun install                      # workspace frontend deps (app/web/tauri/landing)
bun run typecheck                # verify the frontend (see UI changes below)

# One shot: build sidecars + the installer
bun run build                    # = scripts/build-server.sh && cd tauri && bun run tauri build
```

Or step-by-step:

```bash
bun run build:server             # build_binary.py -> dist/ -> copy to tauri/src-tauri/binaries/
cd tauri && bun run tauri build  # bundle shell + frontend + sidecars into installers
```

For the **CUDA** sidecar specifically (what production ships), the Windows path in the
`justfile` runs `python backend/build_binary.py --cuda` (onedir) then
`scripts/package_cuda.py` to stage the CUDA libs.

## CI / release (the normal pattern)

Existing GitHub Actions already do all of this — the fork inherits them:

- **`.github/workflows/release.yml`** — triggers on **git tag push** (and manual
  `workflow_dispatch`). Builds the CPU, **CUDA (`--cuda`, cu128)**, and ROCm sidecars,
  then the Tauri installers, and publishes the release.
- **`.github/workflows/build-windows.yml`** — manual Windows build.
- **`.github/workflows/ci.yml`** — PR checks.

So the release flow is standard:

```
feature branch  ->  PR into main  ->  merge  ->  git tag vX.Y.Z  ->  push tag
                                                              -> release.yml builds all installers
```

Enable Actions on the fork (`swalters1/voicebox-params`) and it runs on push/tag with
no extra setup, aside from secrets: the Tauri **signing key** + password (release
signing/updater) must be added as repo secrets, mirroring upstream's.

## Signing / auto-updates

The fork sets `bundle.createUpdaterArtifacts: false` in
`tauri/src-tauri/tauri.conf.json` so CI and local builds produce a normal
(unsigned) installer without a signing key — fine for installing and running.
To ship **signed auto-updates** instead, generate your own key
(`cd tauri && bun tauri signer generate -w ~/.tauri/voicebox.key`), put its public
key in `plugins.updater.pubkey`, set `createUpdaterArtifacts` back to
`"v1Compatible"`, and add `TAURI_SIGNING_PRIVATE_KEY` (+ `_PASSWORD`) as repo
secrets. Do **not** reuse upstream's pubkey — you don't hold its private key.

## Fork-specific build notes

1. **New backend modules are pinned for PyInstaller.** `backend/build_binary.py` now
   lists `backend.utils.param_spec`, `backend.utils.verify`, and
   `backend.utils.chunked_tts` as `--hidden-import`s (some are imported lazily, which
   PyInstaller's static analysis misses). Without these the frozen `.exe` would
   `ImportError` at runtime on the verify/advanced-options paths.
2. **Frontend changes are pure additions** (advanced-mode panel + verify controls +
   `verified` badge) — no new npm deps, so `bun install` is unchanged.
3. **App identity.** `tauri/src-tauri/tauri.conf.json` keeps `identifier`
   `sh.voicebox.app` and `productName` `Voicebox`. If you want the fork to install
   **alongside** production instead of replacing it, change the `identifier` (and
   ideally `productName`) and use your own signing keys. To ship it as a drop-in
   update to the existing install, keep both and reuse the upstream signing key.

## Merge plan to get here

The work lives on `feat/verify-loop`. To move to normal release patterns:

1. Open a PR: `feat/verify-loop` → `main` on the fork; review the 13 commits.
2. Merge. Tag `v0.5.0-fork.1` (or your scheme) and push the tag.
3. `release.yml` builds the installers; download from the release, or run
   `bun run build` locally for an unsigned local `.exe`.
