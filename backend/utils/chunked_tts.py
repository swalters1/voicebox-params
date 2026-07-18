"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.
"""

import inspect
import logging
import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("voicebox.chunked-tts")

# Upper bound for auto-generated seeds. Kept within int32 so the value round-
# trips cleanly through torch.manual_seed and the ``Generation.seed`` integer
# column regardless of platform.
_MAX_SEED = 2**31 - 1

# Stride between a chunk's verify-retry seeds. A large prime keeps a chunk's
# retry seeds clear of neighbouring chunks' base seeds (which are base+i), while
# staying a deterministic function of the starting seed — so the entire retry
# sequence replays from one integer (FORK_NOTES §2).
_RETRY_STRIDE = 1_000_003

# Stride between the seeds handed to split-and-recurse pieces. A different prime
# from _RETRY_STRIDE keeps split-piece seeds clear of a unit's retry walk while
# staying deterministic.
_SPLIT_STRIDE = 10_000_019


def resolve_seed(seed: int | None) -> int:
    """Return *seed* if given, otherwise a fresh random seed.

    Renders must always run with a concrete seed so the exact value can be
    recorded and replayed later. Passing ``None`` down to the backends left
    the RNG in its ambient state and produced un-reproducible audio.
    """
    return seed if seed is not None else random.randint(0, _MAX_SEED)


@dataclass
class ChunkedTTSResult:
    """Result of a (possibly chunked) TTS generation.

    Carries the assembled audio plus the seeds actually used, so callers can
    persist a reproducible record of the render. ``verify`` is populated only
    when a verification hook runs (see ``generate_chunked``'s ``verify_fn``).
    """

    audio: np.ndarray
    sample_rate: int
    seed: int
    chunk_seeds: List[int] = field(default_factory=list)
    verify: List[dict] | None = None
    # True/False when a verifier ran (all chunks passed), else None. Lets a
    # caller refuse to ship an all-attempts-failed render (FORK_NOTES §8).
    verified: bool | None = None


@dataclass
class EscalationConfig:
    """Staged verify-loop escalation strategy (FORK_NOTES §9).

    Cheapest first: seed-retry the whole unit, then (optionally) retry at a lower
    temperature, then split at a boundary into the safe zone and recurse — each
    stage attacks the same early-EOS probability at a different cost.
    """

    max_attempts: int = 10          # seed-retry budget per stage (§8: 3 is too few)
    retry_temperature: float = 0.0  # stage 2 temp; 0 = off (unproven, §9 HYPOTHESIS)
    split_enabled: bool = True      # stage 3: split-and-recurse (the proven fix)
    split_min_chars: int = 120      # split target — cross into the safe zone, not a 5% shave
    join_silence_ms: int = 250      # sized pause at split joins (NOT a crossfade — §9)
    max_split_depth: int = 2        # recursion guard

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
    }
)

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence-end (``.!?`` not preceded by an abbreviation and not
    inside brackets) → clause boundary (``;:,—``) → whitespace → hard cut.

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to split at the last real sentence ending
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            # Absolute fallback: hard cut but avoid splitting inside a tag
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last sentence-ending punctuation in *text*.

    Skips periods that follow common abbreviations (``Dr.``, ``Mr.``, etc.)
    and periods inside bracket tags (``[laugh]``).  Also handles CJK
    sentence-ending punctuation (``。！？``).
    """
    best = -1
    # ASCII sentence ends
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        char = text[pos]
        # Skip periods after abbreviations
        if char == ".":
            # Walk backwards to find the preceding word
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            # Skip decimal numbers (digit immediately before the period)
            if word_start >= 0 and text[word_start].isdigit():
                continue
        # Skip if we're inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    # CJK sentence-ending punctuation
    for m in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if m.start() > best:
            best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation."""
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = m.start()
        # Skip if inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    for m in _PARA_TAG_RE.finditer(text):
        if m.start() < pos < m.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result


def _backend_accepts_options(backend) -> bool:
    """True if the backend's ``generate`` declares an ``options`` argument.

    Lets us forward per-request tuning only to engines that support it, so
    untuned backends keep working with the same call as before.
    """
    try:
        return "options" in inspect.signature(backend.generate).parameters
    except (ValueError, TypeError):
        return False


def join_with_silence(
    chunks: List[np.ndarray],
    sample_rate: int,
    silence_ms: int = 250,
) -> np.ndarray:
    """Concatenate audio pieces with a fixed silence gap — no crossfade.

    Used to join split-and-recurse pieces (FORK_NOTES §9): the crossfade join
    used elsewhere *eats* the inter-sentence pause, which is fine for garble
    avoidance but wrong for prose pacing. A sized silence preserves the beat.
    """
    pieces = [np.asarray(c, dtype=np.float32) for c in chunks if c is not None and len(c) > 0]
    if not pieces:
        return np.array([], dtype=np.float32)
    if len(pieces) == 1:
        return pieces[0]
    gap = np.zeros(max(0, int(sample_rate * silence_ms / 1000)), dtype=np.float32)
    out: List[np.ndarray] = [pieces[0]]
    for p in pieces[1:]:
        if gap.size:
            out.append(gap)
        out.append(p)
    return np.concatenate(out)


async def _render_unit(
    backend,
    text: str,
    voice_prompt: dict,
    language: str,
    seed: int,
    instruct: str | None,
    trim_fn,
    verify_fn,
    options: dict | None,
    esc: EscalationConfig,
    depth: int = 0,
) -> dict:
    """Render one text unit with staged escalation (FORK_NOTES §9).

    Stages, cheapest first: (1) seed-retry the whole unit; (2) optionally retry
    at a lower temperature; (3) split at a boundary into the safe zone and run
    each piece back through this same function, joining with a sized silence.

    Returns a dict: ``{audio, sample_rate, verified, stage, seed, attempts,
    splits}``. ``verified`` is None when no verifier ran, else whether the unit
    ultimately passed. ``audio`` is always populated (best effort on failure).
    """

    async def _seed_retry(opts: dict | None, stage: str):
        """Seed-retry at fixed *opts*. Returns (audio, sr, seed, attempts, ok)."""
        extra = {}
        if opts and _backend_accepts_options(backend):
            extra["options"] = opts
        atts: list = []
        cur = seed
        audio = None
        sr = 0
        for i in range(esc.max_attempts):
            audio, sr = await backend.generate(text, voice_prompt, language, cur, instruct, **extra)
            if trim_fn is not None:
                audio = trim_fn(audio, sr)
            if verify_fn is None:
                return audio, sr, cur, None, True
            ok, detail = await verify_fn(text, audio, sr)
            atts.append({"seed": cur, "stage": stage, "ok": ok, **detail})
            if ok:
                return audio, sr, cur, atts, True
            cur = (seed + (i + 1) * _RETRY_STRIDE) % _MAX_SEED
        return audio, sr, (atts[-1]["seed"] if atts else cur), atts, False

    all_attempts: list = []

    # --- Stage 1: seed-retry, whole unit --------------------------------------
    audio, sr, used_seed, atts, ok = await _seed_retry(options, "seed")
    if atts:
        all_attempts += atts
    if verify_fn is None:
        return {"audio": audio, "sample_rate": sr, "verified": None,
                "stage": "seed", "seed": used_seed, "attempts": None, "splits": None}
    if ok:
        return {"audio": audio, "sample_rate": sr, "verified": True,
                "stage": "seed", "seed": used_seed, "attempts": all_attempts, "splits": None}

    # --- Stage 2: lower temperature, retry (opt-in) ---------------------------
    if esc.retry_temperature and esc.retry_temperature > 0:
        opts2 = dict(options or {})
        opts2["temperature"] = esc.retry_temperature
        a2, s2, seed2, atts2, ok2 = await _seed_retry(opts2, "temp")
        if atts2:
            all_attempts += atts2
        if ok2:
            return {"audio": a2, "sample_rate": s2, "verified": True, "stage": "temp",
                    "seed": seed2, "attempts": all_attempts, "splits": None}
        audio, sr, used_seed = a2, s2, seed2  # keep latest as the fallback render

    # --- Stage 3: split at a boundary into the safe zone, recurse -------------
    if esc.split_enabled and depth < esc.max_split_depth:
        pieces = split_text_into_chunks(text, esc.split_min_chars)
        if len(pieces) > 1:
            logger.info("Escalating: splitting %d-char unit into %d pieces", len(text), len(pieces))
            audios: List[np.ndarray] = []
            reports: list = []
            sr_final = sr or 24000
            all_ok = True
            for pi, piece in enumerate(pieces):
                sub_seed = (seed + (pi + 1) * _SPLIT_STRIDE) % _MAX_SEED
                sub = await _render_unit(backend, piece, voice_prompt, language, sub_seed,
                                         instruct, trim_fn, verify_fn, options, esc, depth + 1)
                audios.append(sub["audio"])
                sr_final = sub["sample_rate"] or sr_final
                all_ok = all_ok and bool(sub["verified"])
                reports.append({"piece": pi, "chars": len(piece), "verified": sub["verified"],
                                "stage": sub["stage"], "seed": sub["seed"],
                                "attempts": sub["attempts"], "splits": sub["splits"]})
            joined = join_with_silence(audios, sr_final, esc.join_silence_ms)
            return {"audio": joined, "sample_rate": sr_final, "verified": all_ok,
                    "stage": f"split-{len(pieces)}", "seed": seed,
                    "attempts": all_attempts, "splits": reports}

    # --- Exhausted: return the best (still-failing) render, flagged unverified -
    return {"audio": audio, "sample_rate": sr, "verified": False,
            "stage": "exhausted", "seed": used_seed, "attempts": all_attempts, "splits": None}


def _chunk_report(index: int, unit: dict) -> dict:
    """Shape a unit's escalation result into a per-chunk verify record."""
    return {
        "chunk": index,
        "verified": unit["verified"],
        "stage": unit["stage"],
        "seed": unit["seed"],
        "attempts": unit["attempts"],
        "splits": unit["splits"],
    }


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
    verify_fn=None,
    escalation: EscalationConfig | None = None,
    options: dict | None = None,
) -> ChunkedTTSResult:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()``.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    A concrete seed is always resolved before generation (a random one when
    *seed* is ``None``) so the exact value can be recorded and replayed.

    When *verify_fn* is supplied, each chunk runs through the staged escalation
    (:func:`_render_unit`, FORK_NOTES §9): seed-retry, optional lower-temperature
    retry, then split-and-recurse with a sized-pause join. *escalation* tunes
    the budgets/strategy (defaults if omitted). Note the two join types: the
    caller-level chunks are joined with a crossfade (garble avoidance), while an
    escalation split joins its pieces with a sized silence (prose pacing).

    Returns
    -------
    ChunkedTTSResult
        Assembled audio, resolved base seed, per-chunk seeds, the per-chunk
        verification report, and ``verified`` (all chunks passed / None).
    """
    resolved_seed = resolve_seed(seed)
    esc = escalation or EscalationConfig()
    chunks = split_text_into_chunks(text, max_chunk_chars)

    if len(chunks) <= 1:
        # Short text — single chunk. Fall back to the raw text when the
        # splitter returned nothing (e.g. whitespace-only input).
        chunk_text = chunks[0] if chunks else text
        unit = await _render_unit(backend, chunk_text, voice_prompt, language,
                                  resolved_seed, instruct, trim_fn, verify_fn, options, esc)
        verified = None if verify_fn is None else bool(unit["verified"])
        return ChunkedTTSResult(
            audio=unit["audio"],
            sample_rate=unit["sample_rate"],
            seed=unit["seed"],
            chunk_seeds=[unit["seed"]],
            verify=[_chunk_report(0, unit)] if verify_fn is not None else None,
            verified=verified,
        )

    # Long text — chunked generation
    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    audio_chunks: List[np.ndarray] = []
    chunk_seeds: List[int] = []
    verify_report: List[dict] = []
    sample_rate: int | None = None
    all_verified = True

    for i, chunk_text in enumerate(chunks):
        logger.info("Generating chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk_text))
        # Vary the seed per chunk to avoid correlated RNG artefacts, but keep it
        # deterministic so the same (text, seed) pair always reproduces.
        chunk_seed = resolved_seed + i
        unit = await _render_unit(backend, chunk_text, voice_prompt, language,
                                  chunk_seed, instruct, trim_fn, verify_fn, options, esc)

        audio_chunks.append(np.asarray(unit["audio"], dtype=np.float32))
        chunk_seeds.append(unit["seed"])
        if verify_fn is not None:
            verify_report.append(_chunk_report(i, unit))
            all_verified = all_verified and bool(unit["verified"])
        if sample_rate is None:
            sample_rate = unit["sample_rate"]

    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return ChunkedTTSResult(
        audio=audio,
        sample_rate=sample_rate,
        seed=resolved_seed,
        chunk_seeds=chunk_seeds,
        verify=verify_report if verify_report else None,
        verified=all_verified if verify_fn is not None else None,
    )
