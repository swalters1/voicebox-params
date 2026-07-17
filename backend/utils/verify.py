"""Loop-back verification of rendered TTS audio.

Transcribes each rendered chunk with Whisper and compares the transcript to
the intended text, so gross render failures (dropped clauses, truncation, or
hallucinated noise) can be detected and re-seeded. Plugs into
``generate_chunked`` as its ``verify_fn`` hook.

The comparison is deliberately tolerant: ASR is not ground truth (it drops
filler words, mangles proper nouns, and expands numbers), so an exact word
count would false-fail on good audio. Instead it combines a normalized fuzzy
similarity with a word-count ratio to catch the *gross* failures that actually
matter, not benign drift.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("voicebox.verify")

_WORD_RE = re.compile(r"[a-z0-9']+")

# Whisper decode options tuned to avoid the short-text hallucination problem
# when transcribing our own (often short) chunk output for verification.
_VERIFY_WHISPER_OPTIONS = {
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.3,
}


def _normalize(text: str) -> list[str]:
    """Lowercase and tokenize to comparable word units (drops punctuation)."""
    return _WORD_RE.findall(text.lower())


@dataclass
class VerifyConfig:
    """Thresholds and STT settings for chunk verification.

    Attributes:
        similarity_threshold: Minimum normalized fuzzy similarity (0..1).
        word_ratio_min: Minimum transcript/expected word-count ratio. Catches
            gross dropouts and truncation where a whole clause vanishes.
        model_size: Whisper model used for verification (small = fast).
        language: Optional language hint for transcription.
        min_words_for_check: Expected chunks shorter than this (in words) are
            accepted without a strict check, since ASR is unreliable on very
            short clips and would cause spurious re-seeds.
    """

    similarity_threshold: float = 0.70
    word_ratio_min: float = 0.60
    model_size: str = "base"
    language: Optional[str] = None
    min_words_for_check: int = 2


def compare_texts(expected: str, got: str, cfg: VerifyConfig) -> Tuple[bool, dict]:
    """Compare intended text against a transcript. Returns ``(ok, detail)``."""
    exp = _normalize(expected)
    got_words = _normalize(got)

    similarity = SequenceMatcher(None, " ".join(exp), " ".join(got_words)).ratio()
    word_ratio = (len(got_words) / len(exp)) if exp else 1.0

    if len(exp) < cfg.min_words_for_check:
        # Too short to judge reliably — accept rather than churn seeds.
        ok = True
    else:
        ok = (
            similarity >= cfg.similarity_threshold
            and word_ratio >= cfg.word_ratio_min
        )

    detail = {
        "transcript": got[:200],
        "similarity": round(similarity, 3),
        "word_ratio": round(word_ratio, 3),
        "expected_words": len(exp),
        "got_words": len(got_words),
    }
    return ok, detail


def make_chunk_verifier(stt_backend, cfg: Optional[VerifyConfig] = None):
    """Build an async ``verify_fn(chunk_text, audio, sample_rate)`` closure.

    The returned function writes the rendered chunk to a temp WAV, transcribes
    it via *stt_backend*, and compares. It fails **open**: if transcription
    itself errors, the chunk is accepted (``ok=True``) so a flaky verifier
    never blocks a render — the render is still returned either way.
    """
    cfg = cfg or VerifyConfig()

    async def verify_fn(chunk_text: str, audio: np.ndarray, sample_rate: int):
        import soundfile as sf

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, audio, sample_rate, format="WAV")
            transcript = await stt_backend.transcribe(
                tmp_path,
                cfg.language,
                cfg.model_size,
                _VERIFY_WHISPER_OPTIONS,
            )
        except Exception as e:  # fail open — never block a render on STT issues
            logger.warning("verification transcribe failed, accepting chunk: %s", e)
            return True, {"error": str(e)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        ok, detail = compare_texts(chunk_text, transcript, cfg)
        if not ok:
            logger.info(
                "chunk verify mismatch sim=%.2f word_ratio=%.2f exp=%d got=%d",
                detail["similarity"],
                detail["word_ratio"],
                detail["expected_words"],
                detail["got_words"],
            )
        return ok, detail

    return verify_fn
