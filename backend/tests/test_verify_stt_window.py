"""Regression tests for the Whisper 30s-window trap in the verify loop.

Whisper decodes ONE 30s mel window per call. Transcribing a longer render in a
single call silently returns only its opening — which is shaped exactly like
mass truncation (clean prefix, hard stop) and fails audio that is perfectly
fine. At the default ``max_chunk_chars=800`` and a real measured ~17 cps a
chunk is ~45s, so this was the DEFAULT verify path, not an edge case.

These tests pin the fix: audio is sliced under the window before transcription
and the transcripts joined.
"""

import numpy as np
import pytest

from backend.utils.verify import (
    STT_WINDOW_SEC,
    VerifyConfig,
    evaluate,
    make_chunk_verifier,
    split_for_stt,
)

SR = 16_000


def _tone(seconds: float, sample_rate: int = SR) -> np.ndarray:
    """Non-silent audio of a given length (content is irrelevant here)."""
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


class _WindowLimitedSTT:
    """Stand-in for Whisper that reproduces the 30s truncation behaviour.

    Transcribes only the first 30 seconds of whatever it is handed, mapping
    audio duration back to words at a fixed pace.
    """

    HARD_WINDOW_SEC = 30.0

    def __init__(self, words: list[str], total_duration: float):
        self.words = words
        self.pace = len(words) / total_duration  # words per second
        self.calls: list[float] = []

    async def transcribe(self, path, language, model_size, options):
        import soundfile as sf

        audio, sample_rate = sf.read(path)
        duration = len(audio) / sample_rate
        self.calls.append(duration)

        heard = min(duration, self.HARD_WINDOW_SEC)
        # Which words of the whole utterance this slice covers.
        start = int(sum(self.calls[:-1]) * self.pace)
        count = int(heard * self.pace)
        return " ".join(self.words[start : start + count])


def test_short_audio_is_not_sliced():
    """The common case stays a single call — no added cost, no new seams."""
    pieces = split_for_stt(_tone(10.0), SR)
    assert len(pieces) == 1
    assert pieces[0].shape[0] == int(10.0 * SR)


def test_long_audio_is_sliced_under_the_window():
    audio = _tone(95.0)
    pieces = split_for_stt(audio, SR)

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.shape[0] / SR <= STT_WINDOW_SEC + 0.01

    # Lossless: every sample survives, in order.
    assert sum(p.shape[0] for p in pieces) == audio.shape[0]
    assert np.array_equal(np.concatenate(pieces), audio)


@pytest.mark.parametrize("duration", [0.5, 24.9, 25.1, 45.0, 60.6, 284.0])
def test_slices_never_exceed_window(duration):
    for piece in split_for_stt(_tone(duration), SR):
        assert piece.shape[0] / SR <= STT_WINDOW_SEC + 0.01


def test_degenerate_inputs_do_not_crash():
    assert len(split_for_stt(np.zeros(0, dtype=np.float32), SR)) == 1
    assert len(split_for_stt(_tone(60.0), 0)) == 1  # unknown sample rate


@pytest.mark.asyncio
async def test_good_long_render_passes_verification():
    """The bug, end to end: a complete 60s render must not be judged truncated.

    Mirrors the real case — 203 words, ~60s, every word present. Transcribed in
    one call the verifier sees only the first 30s (~66% coverage) and fails it,
    triggering the whole escalation ladder on correct audio.
    """
    words = [f"word{i}" for i in range(203)]
    text = " ".join(words)
    duration = 60.6

    stt = _WindowLimitedSTT(words, duration)
    verify_fn = make_chunk_verifier(stt, VerifyConfig(chars_per_second=17.6))

    ok, detail = await verify_fn(text, _tone(duration), SR)

    assert ok, f"complete render judged truncated: {detail}"
    assert detail["coverage"] > 0.95
    assert detail["stt_windows"] > 1, "long audio should have been sliced"
    assert max(stt.calls) <= STT_WINDOW_SEC + 0.01, "a call exceeded the window"


@pytest.mark.asyncio
async def test_genuine_truncation_still_fails():
    """The guard must not blind the gate: real truncation is still caught.

    Same 203-word text, but the render stops a third of the way in. Without
    this, "slice everything" would be indistinguishable from "pass everything".
    """
    words = [f"word{i}" for i in range(203)]
    text = " ".join(words)

    rendered = 20.0  # only the opening actually got rendered
    stt = _WindowLimitedSTT(words[:67], rendered)
    verify_fn = make_chunk_verifier(stt, VerifyConfig(chars_per_second=17.6))

    ok, detail = await verify_fn(text, _tone(rendered), SR)

    assert not ok, f"truncated render passed: {detail}"


def test_window_truncation_would_fail_the_gate():
    """Pins WHY this matters: the 30s-only transcript fails a good render.

    If this ever stops failing, the gate has been loosened to the point where
    it can no longer detect a two-thirds content loss.
    """
    words = [f"word{i}" for i in range(203)]
    text = " ".join(words)
    first_30s_only = " ".join(words[:100])

    ok, detail = evaluate(text, first_30s_only, 60.6, VerifyConfig(chars_per_second=17.6))

    assert not ok
    assert detail["coverage"] < 0.6
