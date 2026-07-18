# Voicebox — Fork Changelog & Acknowledgments

## First, a thank-you

This project is a fork of **[Voicebox](https://github.com/jamiepine/voicebox)** by
**Jamie Pine** and the Voicebox contributors — an open-source AI voice studio,
generously released under the MIT license.

None of the work below would exist without that foundation. The engine abstraction,
the Tauri app, the profile/cloning system, the chunked TTS pipeline, the capture and
dictation flows, the whole thoughtful architecture — all of it came first, and all of
it made this specialization *possible* rather than a from-scratch slog. Building on top
of it was a pleasure precisely because the original is clean, well-factored, and clearly
made with care. Thank you. This fork is meant as a tribute to that work, not a
replacement for it.

If you're here for the general voice studio, use upstream — it's excellent. This fork
exists for one specific, demanding job described below.

## Why this fork exists

The motivating use case is **long-form audiobook narration**: rendering entire books
across ~two dozen distinct character voices plus a narrator, chapter after chapter.

At that scale, a failure mode that is a shrug for a one-off clip becomes a wall. TTS
models occasionally sample an end-of-sequence token early and **truncate** a line —
dropping the last words. At the length of a single utterance that's rare; across tens of
thousands of utterances it's constant, and every truncation is a manual re-render. Stock
Voicebox renders beautifully; it just had no *self-checking* layer, and its inference
parameters were fixed constants with no way to tune or record them.

So this fork adds a **reliability and tunability layer** on top of the existing engines —
additive, backward-compatible, and drop-in. It does not change how upstream works; it
gives ambitious users a way to make renders reproducible, tunable, and *verified*.

## What's different in this fork

Every change is additive — existing endpoints, requests, and behavior are untouched
unless you opt in.

### 1. Reproducible renders
Every render now resolves a **concrete seed** (a random one when you don't supply it) and
persists it. Previously an auto-seeded render couldn't be reproduced; now the same
`(text, seed, params)` yields byte-identical audio — confirmed on GPU. A blessed take can
be reproduced forever, and the generation row *is* the manifest.

### 2. A declarative advanced-options contract
Each engine's tunable surface is now **data, not hardcoded literals**. Backends declare a
`PARAM_SPEC`; `GET /engines`, `/verify/params`, and `/transcribe/params` advertise it;
requests carry per-engine `tts_params` / `verify_config` / transcribe `options`, validated
against the spec. Unknown or out-of-range keys are **rejected loudly** (422) instead of
silently ignored — the quiet failure that used to make a misapplied option look like it
worked. Options resolve in layers: engine defaults → language → per-voice profile → request.

### 3. A loop-back verify system
Opt in with `verify: true` and each rendered chunk is **transcribed back and checked**
against the intended text. The gate is built around the real failure modes: it keys on
**duration shortfall** (the true truncation signal) plus word-coverage that deliberately
ignores the leading token and never demands an exact transcript match — because ASR mangles
names and drops the first word of short clips. No false-passes, and no false-rejects on
good audio.

### 4. Staged escalation (the reliability payoff)
On a verify failure, a unit escalates cheapest-first: **seed-retry** the whole unit → an
optional **lower-temperature** retry → **split at a clause boundary into the safe zone and
re-render each piece**, joined with a sized silence (not a crossfade, which would eat the
sentence pause). Seed-retry rescues an unlucky roll on a right-sized unit; the split rescues
the chronically-hard ones that no seed can land. A top-level `verified` flag lets a caller
refuse to ship an all-attempts-failed render.

### 5. Everything is recorded
Each render persists a `gen_params` manifest — resolved inference params, per-chunk seeds,
the winning stage (`seed` / `temp` / `split-N`), and the full per-attempt verify report —
readable back over the API. You can always see *why* a unit needed help and reproduce it.

### 6. Advanced-mode UI
The desktop app gains an advanced panel that **builds itself from the capability
endpoints** — a control per declared parameter, a verify toggle, and a verified/unverified
badge in history. New engine or new knob → the panel updates for free.

## Relationship to upstream

- **License:** MIT, same as upstream.
- **Compatibility:** Additive and backward-compatible — the fork stays a drop-in.
- **Staying current:** upstream is tracked as the `upstream` remote; improvements flow in
  via `git fetch upstream && git merge upstream/main`.
- **Credit:** the original design and the great majority of the codebase remain Jamie
  Pine's and the Voicebox contributors' work. Please support and star the
  [upstream project](https://github.com/jamiepine/voicebox).

Thanks again for building something worth forking. 🙏
