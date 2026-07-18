"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

from typing import Optional
from ..backends import get_stt_backend, STTBackend
from ..utils.param_spec import Param

# Declarative Whisper decode-option surface (FORK_NOTES §7f). The engine-
# agnostic openai-whisper option names both STT backends understand; the
# PyTorch backend maps them to HF generate kwargs. Unlike the TTS specs, these
# are validate-only (see param_spec.validate_options) — Whisper's own defaults
# stand unless a caller explicitly overrides one, so the `default` values here
# are informational (what the engine does by default) and are NOT force-applied.
# Model size and language stay first-class Form fields with their own pickers,
# so this spec is the "advanced decode" panel only.
WHISPER_PARAM_SPEC = [
    Param("temperature", 0.0, 0.0, 1.0, desc="Decode temperature (0 = greedy)"),
    Param("no_speech_threshold", 0.6, 0.0, 1.0, desc="Silence probability to treat a segment as no-speech"),
    Param("logprob_threshold", -1.0, -20.0, 0.0, desc="Avg log-prob below which a segment is treated as failed"),
    Param("compression_ratio_threshold", 2.4, 0.0, 100.0, desc="Gzip ratio above which output is treated as repetitive"),
    Param("condition_on_previous_text", True, desc="Feed prior text as context (off avoids short-text loops)"),
]


def get_whisper_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).
    
    Returns:
        STT backend instance
    """
    return get_stt_backend()


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    backend = get_stt_backend()
    backend.unload_model()
