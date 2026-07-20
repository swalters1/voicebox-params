# Voicebox (params fork) — built for long-form narration

A fork of [Voicebox](https://github.com/jamiepine/voicebox) that adds a
**reliability and tunability layer** for rendering audiobooks: work that runs for
hours, across dozens of character voices, where a defect rate that is invisible
on a single clip becomes a wall.

Everything here is additive. Existing endpoints, requests, and behaviour are
unchanged unless you opt in.

---

## Why the fork exists

Stock Voicebox renders beautifully. The problem is not quality — it is **scale
and self-knowledge**.

TTS models occasionally sample an end-of-sequence token early and **truncate** a
line, dropping the last words. On one clip that is a curiosity. Across tens of
thousands of utterances in a book, it is constant, and every truncation is a
manual re-render that someone has to notice first.

Stock Voicebox had no way to notice. It also had no way to reproduce a good
take, and its inference parameters were fixed constants with no way to tune or
record them. Three gaps, all of which only matter at length:

| Gap | Consequence over a book |
|---|---|
| No self-checking | Truncations ship silently; you find them by listening |
| No reproducibility | A great take can't be recreated; auto-seeded renders were lost |
| Fixed parameters | No way to tune per voice, and no record of what produced a render |

This fork closes all three, so a long render can be trusted rather than
supervised.

---

## The goal

**A render should be a pure function of `(text, profile, engine, resolved_options, seed)`
— recorded, reproducible, and verified before it ships.**

Everything below follows from that sentence.

---

## What's different

### 1. Reproducible renders

Every render resolves a **concrete seed** — a random one when you don't supply
it — and persists it. Previously an auto-seeded render could never be
reproduced. Now the same inputs yield **byte-identical** audio, confirmed on
GPU.

The generation row *is* the manifest. A blessed take can be recreated forever.

### 2. A declarative options contract

Each engine's tunable surface is **data, not hardcoded literals**. Backends
declare a `PARAM_SPEC`; the API advertises it; requests carry overrides that are
validated against it.

```
GET /engines           → per-engine params (temperature, top_k, top_p, …)
GET /verify/params     → the verify gate's 12 knobs
GET /transcribe/params → Whisper decode options
```

Options resolve in layers, last wins:

```
engine defaults → language defaults → per-voice profile → per-request
```

Unknown or out-of-range keys are **rejected loudly (422)** rather than silently
ignored — the quiet failure that used to make a misapplied option look like it
worked. Profile and language layers are lenient (a per-voice tuning may target a
different engine); the request layer is strict.

### 3. A loop-back verify system

Opt in with `verify: true` and each rendered chunk is **transcribed back and
checked** against the intended text.

The gate is built around the real failure modes, not the obvious ones:

- **Duration shortfall** is the primary signal — a truncated clip stops short of
  the duration its text implies at the speaker's pace.
- **Word coverage** is secondary, and deliberately **ignores the leading token**,
  because Whisper drops the first word of short clips as an artifact of padding.
- It never requires an exact transcript match. ASR normalises numbers, swaps
  homophones, and mangles invented names — a strict match false-rejects
  constantly.

### 4. Staged escalation

On a verify failure, a unit escalates **cheapest-first**:

1. **Seed-retry** the whole unit (budget: `max_attempts`, default 10)
2. **Lower-temperature** retry (optional, off by default — unproven)
3. **Split** at a clause boundary into the safe zone and re-render each piece,
   joined with **sized silence** (not a crossfade, which eats the sentence pause)

The maths behind the staging: retries sample repeatedly from a *fixed* success
probability, while splitting *raises* that probability by shortening the
sequence. For a typical unit, 3–4 retries is 94–99%. For a chronically hard one
(observed success ≈ 1 in 6), even 16 retries is only 95% — which is why the
split exists, and why it comes last: it changes the pacing, so it is a treatment
of last resort.

A top-level `verified` flag lets a caller refuse to ship an all-attempts-failed
render.

### 5. Everything is recorded

Each render persists a `gen_params` manifest: resolved inference params,
per-chunk seeds, the winning stage (`seed` / `temp` / `split-N`), and the full
per-attempt verify report. You can always see *why* a unit needed help.

### 6. Recasting and auditioning — "Regenerate as …"

Characters age between books, and lines get cast to the wrong speaker. A
generation can be re-rendered in a **different voice**, stored as a comparable
take under the same row rather than scattered across history.

That makes it an **audition tool**: select a candidate voice, recast a known
line, listen, repeat. Takes are labelled with the voice that rendered them.

The inheritance rules matter here:

- `tts_params` **are** inherited — same tuning, different speaker
- `chars_per_second` is **not** — pace is per-voice, and reusing the old voice's
  pace makes a complete render look short to the verify gate
- The seed is **re-rolled** — determinism holds over
  `(seed, text, ref_audio, params)`, so a seed carries no meaning across voices

### 7. Backup and restore

Long projects accumulate irreplaceable state: cloned voices, thousands of
generations, per-voice tuning. This fork protects it.

- **Automatic pre-upgrade snapshot** — the database is copied before any schema
  migration runs, one per app version, keeping the last 5. It fires at the one
  moment the data is genuinely at risk and nobody is thinking about backups.
- **Back up now** — on demand, safe while the server is busy (`VACUUM INTO`,
  which takes a consistent snapshot rather than copying bytes mid-write).
- **Restore** — staged while the app runs, applied at startup before anything
  connects. The database being replaced is **saved aside first**, and the
  restore aborts if that copy can't be made.
- Backups from a **newer** version are refused — migrations only run forward.

Rendered audio is deliberately **not** in the backup. Because every generation
stores its seed and resolved params, audio regenerates byte-identically:
`voicebox.db` + `profiles/` is a *complete logical backup* at ~1.6 MB instead of
many gigabytes.

### 8. Advanced-mode UI

The desktop app gains an advanced panel that **builds itself from the capability
endpoints** — a control per declared parameter, a verify toggle, and a
verified/unverified badge in history. A new engine or a new knob updates the
panel for free, with no UI change.

---

## Field notes

Lessons that cost real time. Worth knowing before tuning anything.

**Whisper's 30-second window cuts both ways.** The same single-window call that
pads short clips also *stops* at 30 s on long ones. Transcribing a 60 s render
whole returns only its opening and looks exactly like catastrophic truncation —
clean prefix, hard stop. Never transcribe more than ~25 s in one call. The
verify loop slices automatically; any QA tooling you write must too.

**Prevention beats every runtime fix.** Right-sizing units at plan time (~260
chars) keeps the success probability high, so the loop rarely escalates past a
seed retry. The verify loop is a safety net for an unlucky roll on a
*right-sized* unit — it cannot rescue a chronically oversized one.

**Measure `chars_per_second` per voice.** The default of 16.0 is a structural
placeholder, not a tuned value. Real measurements have ranged 17.6–20.5. Too low
a value makes complete renders look short.

**Garble and truncation are different failures.** Chunks over ~800 chars come
back rushed with repeated or dropped words at deterministic spots. That is the
chunker, not sampling — no seed or temperature fixes it.

**Don't trust n=1.** Seed × text interaction is enormous; the same text has run
1.4 s–18.7 s across seeds. Any claim that a setting is "better" needs a rate over
many seeds and several texts per length bucket.

---

## Relationship to upstream

- **License:** MIT, same as upstream.
- **Compatibility:** additive and backward-compatible — this stays a drop-in.
- **Staying current:** upstream is tracked as the `upstream` remote; improvements
  flow in with `git fetch upstream && git merge upstream/main`.
- **Credit:** the original design and the great majority of the codebase are
  Jamie Pine's and the Voicebox contributors' work. The engine abstraction, the
  Tauri app, the profile and cloning system, the chunked TTS pipeline — all of it
  came first, and made this specialisation possible rather than a from-scratch
  slog.

If you want a general-purpose voice studio, **use upstream** — it's excellent.
This fork exists for one specific, demanding job.

---

## Where things live

| Concern | Path |
|---|---|
| Verify gate + STT windowing | `backend/utils/verify.py` |
| Chunking, escalation, seeds | `backend/utils/chunked_tts.py` |
| Options contract | `backend/utils/param_spec.py` |
| Backup / restore | `backend/database/backup.py`, `restore.py` |
| Advanced panel | `app/src/components/Generation/AdvancedGeneratePanel.tsx` |
| Backups UI | `app/src/components/ServerTab/BackupSection.tsx` |
| Fork tests | `backend/tests/fork/` (CI runs the directory) |

**Pipeline contract.** Send per character:

```json
{
  "verify": true,
  "verify_config": {
    "chars_per_second": 17.6,
    "coverage_min": 0.80,
    "max_attempts": 10,
    "split_enabled": true,
    "split_min_chars": 120
  }
}
```

Then read `gen_params.verified` and **refuse to ship anything `false`**.
`gen_params.verify[*].stage` tells you whether a unit needed a seed retry or a
split.
