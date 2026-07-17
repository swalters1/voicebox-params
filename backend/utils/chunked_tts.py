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


def _backend_accepts_params(backend) -> bool:
    """True if the backend's ``generate`` declares a ``params`` argument.

    Lets us forward per-request tuning only to engines that support it, so
    untuned backends keep working with the same call as before.
    """
    try:
        return "params" in inspect.signature(backend.generate).parameters
    except (ValueError, TypeError):
        return False


async def _generate_one_chunk(
    backend,
    chunk_text: str,
    voice_prompt: dict,
    language: str,
    seed: int,
    instruct: str | None,
    trim_fn,
    verify_fn,
    max_verify_attempts: int,
    gen_params: dict | None = None,
) -> Tuple[np.ndarray, int, int, list | None]:
    """Generate a single chunk, retrying with fresh seeds if verification fails.

    Returns ``(audio, sample_rate, seed_used, attempts)`` where *attempts* is
    ``None`` when no ``verify_fn`` was supplied, or a list of per-attempt
    ``{"seed", "ok", ...}`` records otherwise. The audio returned is always the
    last attempt; on total failure that is the final (still-rejected) render so
    the pipeline degrades gracefully rather than dropping a chunk.
    """
    attempts: list = []
    current_seed = seed
    audio: np.ndarray | None = None
    sample_rate = 0

    # Forward tuning params only to backends that accept them.
    extra = {}
    if gen_params and _backend_accepts_params(backend):
        extra["params"] = gen_params

    total_tries = max_verify_attempts if verify_fn is not None else 1
    for attempt in range(total_tries):
        audio, sample_rate = await backend.generate(
            chunk_text,
            voice_prompt,
            language,
            current_seed,
            instruct,
            **extra,
        )
        if trim_fn is not None:
            audio = trim_fn(audio, sample_rate)

        if verify_fn is None:
            return audio, sample_rate, current_seed, None

        ok, detail = await verify_fn(chunk_text, audio, sample_rate)
        attempts.append({"seed": current_seed, "ok": ok, **detail})
        if ok:
            return audio, sample_rate, current_seed, attempts

        logger.info(
            "Chunk verification failed (attempt %d/%d), re-seeding: %s",
            attempt + 1,
            total_tries,
            detail,
        )
        # Re-seed with a fresh random value for the next attempt.
        current_seed = random.randint(0, _MAX_SEED)

    return audio, sample_rate, attempts[-1]["seed"], attempts


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
    max_verify_attempts: int = 3,
    gen_params: dict | None = None,
) -> ChunkedTTSResult:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()``.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    A concrete seed is always resolved before generation (a random one when
    *seed* is ``None``) so the exact value can be recorded and replayed. The
    resolved seeds are returned on the result.

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()``; *seed* is resolved to a concrete
        value first.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).
    verify_fn : callable | None
        Optional async ``(chunk_text, audio, sample_rate) -> (ok, detail)``
        hook. When supplied, a chunk whose verification returns ``ok=False``
        is re-generated with a fresh seed up to *max_verify_attempts* times.
        ``detail`` is a dict merged into the per-attempt record.
    max_verify_attempts : int
        Maximum attempts per chunk when *verify_fn* is supplied (default 3).

    Returns
    -------
    ChunkedTTSResult
        Assembled audio, sample rate, resolved base seed, per-chunk seeds, and
        (when verified) the per-chunk verification report.
    """
    resolved_seed = resolve_seed(seed)
    chunks = split_text_into_chunks(text, max_chunk_chars)

    if len(chunks) <= 1:
        # Short text — single chunk. Fall back to the raw text when the
        # splitter returned nothing (e.g. whitespace-only input).
        chunk_text = chunks[0] if chunks else text
        audio, sample_rate, seed_used, attempts = await _generate_one_chunk(
            backend,
            chunk_text,
            voice_prompt,
            language,
            resolved_seed,
            instruct,
            trim_fn,
            verify_fn,
            max_verify_attempts,
            gen_params,
        )
        return ChunkedTTSResult(
            audio=audio,
            sample_rate=sample_rate,
            seed=seed_used,
            chunk_seeds=[seed_used],
            verify=[{"chunk": 0, "attempts": attempts}] if attempts is not None else None,
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

    for i, chunk_text in enumerate(chunks):
        logger.info(
            "Generating chunk %d/%d (%d chars)",
            i + 1,
            len(chunks),
            len(chunk_text),
        )
        # Vary the seed per chunk to avoid correlated RNG artefacts,
        # but keep it deterministic so the same (text, seed) pair
        # always produces the same output.
        chunk_seed = resolved_seed + i

        chunk_audio, chunk_sr, seed_used, attempts = await _generate_one_chunk(
            backend,
            chunk_text,
            voice_prompt,
            language,
            chunk_seed,
            instruct,
            trim_fn,
            verify_fn,
            max_verify_attempts,
            gen_params,
        )

        audio_chunks.append(np.asarray(chunk_audio, dtype=np.float32))
        chunk_seeds.append(seed_used)
        if attempts is not None:
            verify_report.append({"chunk": i, "attempts": attempts})
        if sample_rate is None:
            sample_rate = chunk_sr

    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return ChunkedTTSResult(
        audio=audio,
        sample_rate=sample_rate,
        seed=resolved_seed,
        chunk_seeds=chunk_seeds,
        verify=verify_report if verify_report else None,
    )
