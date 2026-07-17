"""Transcription endpoints."""

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import models
from ..services import transcribe
from ..services.task_queue import create_background_task
from ..utils.tasks import get_task_manager

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


@router.get("/transcribe/params")
async def transcribe_params():
    """Advertise the Whisper decode-option surface for the advanced-mode UI.

    Mirrors GET /engines but for transcription (FORK_NOTES §7f). Send chosen
    values as the ``options`` JSON field on /transcribe.
    """
    from ..utils.param_spec import spec_as_dicts

    return {"param_spec": spec_as_dicts(transcribe.WHISPER_PARAM_SPEC)}


@router.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
    options: str | None = Form(None),
):
    """Transcribe audio file to text.

    ``options`` is an optional JSON string of Whisper decode overrides validated
    against WHISPER_PARAM_SPEC (see GET /transcribe/params), e.g.
    ``{"no_speech_threshold": 0.2}``. Unknown/out-of-range keys -> 400. Only the
    keys you set are applied; the resolved set is echoed on the response.
    """
    whisper_options: dict | None = None
    if options:
        import json

        from ..utils.param_spec import validate_options, OptionError

        try:
            parsed = json.loads(options)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")
        try:
            whisper_options = validate_options(transcribe.WHISPER_PARAM_SPEC, parsed)
        except OptionError as e:
            raise HTTPException(status_code=400, detail=str(e))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        from ..utils.audio import load_audio
        from ..backends import WHISPER_HF_REPOS

        audio, sr = await asyncio.to_thread(load_audio, tmp_path)
        duration = len(audio) / sr

        whisper_model = transcribe.get_whisper_model()
        model_size = model if model else whisper_model.model_size

        valid_sizes = list(WHISPER_HF_REPOS.keys())
        if model_size not in valid_sizes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model size '{model_size}'. Must be one of: {', '.join(valid_sizes)}",
            )

        already_loaded = whisper_model.is_loaded() and whisper_model.model_size == model_size
        if not already_loaded and not whisper_model._is_model_cached(model_size):
            progress_model_name = f"whisper-{model_size}"
            task_manager = get_task_manager()

            async def download_whisper_background():
                try:
                    await whisper_model.load_model_async(model_size)
                    task_manager.complete_download(progress_model_name)
                except Exception as e:
                    task_manager.error_download(progress_model_name, str(e))

            task_manager.start_download(progress_model_name)
            create_background_task(download_whisper_background())

            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"Whisper model {model_size} is being downloaded. Please wait and try again.",
                    "model_name": progress_model_name,
                    "downloading": True,
                },
            )

        text = await whisper_model.transcribe(
            tmp_path, language, model_size, whisper_options
        )

        return models.TranscriptionResponse(
            text=text,
            duration=duration,
            options=whisper_options or None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
