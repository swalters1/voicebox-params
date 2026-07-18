"""Loop-back verification of rendered TTS audio.

Transcribes each rendered chunk with Whisper and compares the transcript to
the intended text, so a truncated render (EOS sampled early → words missing at
the END) can be detected and re-seeded. Plugs into ``generate_chunked`` as its
``verify_fn`` hook.

Gate design is driven by hard-won findings (see docs/FORK_NOTES.md §5):

* Stock Voicebox transcription is bare HF ``WhisperForConditionalGeneration``
  with greedy decode over a clip padded to 30s of mel. On SHORT clips it drops
  the **leading** word from the transcript even though the audio contains it.
  So the gate MUST NOT depend on the first token, and MUST NOT require an exact
  transcript match (Whisper also normalizes numbers, homophones, and mangles
  invented names).
* The real failure mode is **truncation** — words missing at the end — best
  seen as a **duration shortfall** against the speaker's pace. That is the
  primary signal here; word-coverage (excluding the leading token) is the
  secondary one.

Thresholds are deliberately conservative STRUCTURAL defaults, not tuned
constants — per FORK_NOTES §4 they need a rate measured over many seeds × texts
per length bucket before any value is trusted. Tune via VerifyConfig.
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

from .param_spec import Param, resolve_options

logger = logging.getLogger("voicebox.verify")

_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> list[str]:
    """Lowercase and tokenize to comparable word units (drops punctuation)."""
    return _WORD_RE.findall(text.lower())


@dataclass
class VerifyConfig:
    """Thresholds and STT settings for chunk verification.

    Attributes:
        coverage_min: Minimum fraction of expected words (excluding the leading
            token) that must appear, in order, in the transcript. Catches
            content dropped mid/late.
        duration_ratio_min: Minimum ratio of actual audio duration to the
            duration expected from the text at ``chars_per_second``. The
            primary truncation signal — a clip that stops short trips this.
        chars_per_second: Speaker pace estimate used to predict duration.
            FORK_NOTES §4 measured fragment medians ~15.3-17.0 cps; this is a
            single global stand-in for a per-speaker measurement (see TODO).
        model_size: Whisper model used for verification.
        language: Language forced during transcription. Forcing it skips
            autodetection, which misbehaves on padded silence (FORK_NOTES §5).
        min_words_for_check: Expected chunks shorter than this (in words) are
            accepted without a strict check — ASR is unreliable on very short
            clips and would cause spurious re-seeds.
        ignore_leading_words: Number of leading expected words to drop before
            computing coverage, to absorb Whisper's leading-word drop artifact.
    """

    coverage_min: float = 0.80
    duration_ratio_min: float = 0.55
    chars_per_second: float = 16.0
    model_size: str = "base"
    language: Optional[str] = None
    min_words_for_check: int = 3
    ignore_leading_words: int = 1
    # TODO(fork): replace chars_per_second with a per-profile pace measured from
    # the reference audio or prior blessed renders — global cps is a rough stand-in.


# Declarative tuning surface for the verify gate, mirroring the engine PARAM_SPEC
# contract (FORK_NOTES §7). These are STRUCTURAL defaults, not tuned constants —
# §4 is emphatic they need a rate measured over many seeds × texts per length
# bucket. Exposed per-request and advertised via GET /verify/params so the
# audiobook pipeline can sweep them. ``language`` is derived from the generation,
# so it is not part of this spec.
VERIFY_PARAM_SPEC = [
    # --- gate thresholds (how a render is judged) ---
    Param("coverage_min", 0.80, 0.0, 1.0, desc="Min fraction of expected words (excl. leading token) found"),
    Param("duration_ratio_min", 0.55, 0.0, 2.0, desc="Min actual/expected audio duration (truncation floor)"),
    Param("chars_per_second", 16.0, 1.0, 60.0, desc="Speaker pace used to predict duration (measure per-voice)"),
    Param("min_words_for_check", 3, 0, 100, desc="Expected chunks shorter than this accepted unchecked"),
    Param("ignore_leading_words", 1, 0, 10, desc="Leading expected words dropped before coverage (ASR artifact)"),
    Param("model_size", "base", desc="Whisper model used for verification"),
    # --- escalation strategy (what to do on failure — FORK_NOTES §9) ---
    Param("max_attempts", 10, 1, 24, desc="Seed-retry budget per escalation stage"),
    Param("retry_temperature", 0.0, 0.0, 2.0, desc="Stage-2 lowered temperature (0 = off; unproven)"),
    Param("split_enabled", True, desc="Stage 3: split a failing unit at a boundary and recurse"),
    Param("split_min_chars", 120, 40, 260, desc="Split target size — cross into the safe zone"),
    Param("join_silence_ms", 250, 0, 2000, desc="Sized pause at split joins (not a crossfade)"),
]

# Which resolved keys feed the gate (VerifyConfig) vs the escalation strategy.
_GATE_KEYS = {
    "coverage_min", "duration_ratio_min", "chars_per_second",
    "min_words_for_check", "ignore_leading_words", "model_size",
}
_ESCALATION_KEYS = {
    "max_attempts", "retry_temperature", "split_enabled", "split_min_chars", "join_silence_ms",
}


def build_verify_config(options: Optional[dict], language: Optional[str]) -> VerifyConfig:
    """Build the gate VerifyConfig from resolved options + the generation language.

    Fills defaults from VERIFY_PARAM_SPEC defensively (unknown keys ignored —
    validation already happened at the request boundary). Only gate keys are
    used; escalation keys go to :func:`build_escalation_config`.
    """
    resolved = resolve_options(VERIFY_PARAM_SPEC, options or {}, reject_unknown=False)
    gate = {k: v for k, v in resolved.items() if k in _GATE_KEYS}
    return VerifyConfig(language=language, **gate)


def build_escalation_config(options: Optional[dict]):
    """Build the EscalationConfig (loop strategy) from resolved verify options."""
    from .chunked_tts import EscalationConfig

    resolved = resolve_options(VERIFY_PARAM_SPEC, options or {}, reject_unknown=False)
    esc = {k: v for k, v in resolved.items() if k in _ESCALATION_KEYS}
    return EscalationConfig(**esc)


def evaluate(
    expected_text: str,
    transcript: str,
    audio_duration_sec: float,
    cfg: VerifyConfig,
) -> Tuple[bool, dict]:
    """Judge one rendered chunk. Returns ``(ok, detail)``.

    Never gates on the first token and never requires an exact match — see the
    module docstring for why.
    """
    exp = _normalize(expected_text)
    got = _normalize(transcript)

    # Too short to judge reliably — accept rather than churn seeds.
    if len(exp) < cfg.min_words_for_check:
        return True, {
            "transcript": transcript[:200],
            "skipped": "too_short",
            "expected_words": len(exp),
            "got_words": len(got),
        }

    # Word coverage, excluding the leading token(s): Whisper's leading-word drop
    # on short padded clips is an artifact, not a render failure.
    exp_core = exp[cfg.ignore_leading_words :] or exp
    sm = SequenceMatcher(None, exp_core, got, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    coverage = matched / len(exp_core) if exp_core else 1.0

    # Duration shortfall — the primary truncation signal.
    expected_sec = len(expected_text) / cfg.chars_per_second if cfg.chars_per_second else 0.0
    duration_ratio = (audio_duration_sec / expected_sec) if expected_sec > 0 else 1.0

    ok = coverage >= cfg.coverage_min and duration_ratio >= cfg.duration_ratio_min

    detail = {
        "transcript": transcript[:200],
        "coverage": round(coverage, 3),
        "duration_ratio": round(duration_ratio, 3),
        "duration_sec": round(audio_duration_sec, 3),
        "expected_sec": round(expected_sec, 3),
        "expected_words": len(exp),
        "got_words": len(got),
    }
    return ok, detail


def make_chunk_verifier(stt_backend, cfg: Optional[VerifyConfig] = None):
    """Build an async ``verify_fn(chunk_text, audio, sample_rate)`` closure.

    The returned function writes the rendered chunk to a temp WAV, transcribes
    it via *stt_backend*, and evaluates it. It fails **open**: if transcription
    itself errors, the chunk is accepted (``ok=True``) so a flaky verifier never
    blocks a render — the render is still returned either way.
    """
    cfg = cfg or VerifyConfig()

    async def verify_fn(chunk_text: str, audio: np.ndarray, sample_rate: int):
        import soundfile as sf

        duration_sec = (len(audio) / sample_rate) if sample_rate else 0.0

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, audio, sample_rate, format="WAV")
            transcript = await stt_backend.transcribe(
                tmp_path,
                cfg.language,
                cfg.model_size,
                # Force the language (skips autodetect on padded silence). No
                # thresholds here — the leading-word drop is padding+greedy, not
                # silence, so no decode option fixes it (FORK_NOTES §5).
                None,
            )
        except Exception as e:  # fail open — never block a render on STT issues
            logger.warning("verification transcribe failed, accepting chunk: %s", e)
            return True, {"error": str(e)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        ok, detail = evaluate(chunk_text, transcript, duration_sec, cfg)
        if not ok:
            logger.info(
                "chunk verify FAIL coverage=%.2f dur_ratio=%.2f exp_words=%d got_words=%d",
                detail.get("coverage", -1),
                detail.get("duration_ratio", -1),
                detail["expected_words"],
                detail["got_words"],
            )
        return ok, detail

    return verify_fn
