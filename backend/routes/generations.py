"""TTS generation endpoints."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import history, personality, profiles, tts
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services.generation import run_generation
from ..services.task_queue import cancel_generation as cancel_generation_job, enqueue_generation
from ..utils.audio import load_audio
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_tts_options(
    engine: str,
    language: str,
    profile_overrides: dict | None,
    request_overrides: dict | None,
) -> dict | None:
    """Resolve options for an engine, layering language, profile, then request.

    Resolution order (last wins, FORK_NOTES §7b):
        engine PARAM_SPEC defaults -> language defaults -> profile.option_overrides
            -> request.tts_params

    Language and profile layers are applied leniently (keys not in this engine's
    spec, or out of range for it, are dropped — a per-voice tuning may target a
    different default engine). Request overrides are strict: an unknown key or
    out-of-range value is a 422.
    """
    from ..backends import get_param_spec, get_engine_language_defaults
    from ..utils.param_spec import resolve_options, filter_applicable, OptionError

    spec = get_param_spec(engine)
    if not spec:
        # Engine has no tunable params; a non-empty request override is a client
        # error. Profile overrides for a mismatched engine are ignored.
        if request_overrides:
            raise HTTPException(
                status_code=422,
                detail=f"engine {engine!r} accepts no tts_params",
            )
        return None
    try:
        # engine defaults -> language -> profile (all lenient: keys not in THIS
        # engine's spec, or out of its range, are dropped so they never 422 an
        # unrelated request).
        lang_defaults = get_engine_language_defaults(engine, language)
        base = resolve_options(
            spec,
            filter_applicable(spec, lang_defaults),
            filter_applicable(spec, profile_overrides),
        )
        # + request (strict: unknown key / out-of-range -> 422)
        return resolve_options(spec, base, request_overrides or {})
    except OptionError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _resolve_verify_options(overrides: dict | None, *, strict: bool = True) -> dict | None:
    """Resolve verify-gate overrides against VERIFY_PARAM_SPEC.

    *strict* (the request layer) rejects unknown/out-of-range keys with a 422 —
    the §7e discipline that makes a typo loud. Set it False for INHERITED
    values: a stored record is not user input, and it may legitimately carry
    keys this spec doesn't advertise, so a stale key there must not fail a
    request the user never typed.
    """
    if not overrides:
        return None
    from ..utils.param_spec import resolve_options, filter_applicable, OptionError
    from ..utils.verify import VERIFY_PARAM_SPEC

    if not strict:
        overrides = filter_applicable(VERIFY_PARAM_SPEC, overrides)
        if not overrides:
            return None
    try:
        return resolve_options(VERIFY_PARAM_SPEC, overrides)
    except OptionError as e:
        raise HTTPException(status_code=422, detail=str(e))

IMPORTED_AUDIO_PROFILE_NAME = "Imported Audio"
IMPORT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
IMPORT_AUDIO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


def _get_or_create_import_profile(db: Session) -> DBVoiceProfile:
    """Singleton profile every imported audio clip points at — keeps the
    Generation FK happy without making profile_id nullable across the schema."""
    row = (
        db.query(DBVoiceProfile)
        .filter(DBVoiceProfile.name == IMPORTED_AUDIO_PROFILE_NAME)
        .first()
    )
    if row is not None:
        return row
    row = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=IMPORTED_AUDIO_PROFILE_NAME,
        description="External audio imported into a story timeline.",
        language="en",
        voice_type="import",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _resolve_generation_engine(data: models.GenerationRequest, profile) -> str:
    return data.engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


@router.get("/verify/params")
async def verify_params():
    """Advertise the verify-gate tuning surface for the advanced-mode UI.

    Mirrors GET /engines but for the loop-back verifier (FORK_NOTES §7f). Send
    the chosen values as ``verify_config`` on /generate with ``verify: true``.
    """
    from ..utils.param_spec import spec_as_dicts
    from ..utils.verify import VERIFY_PARAM_SPEC

    return {"param_spec": spec_as_dicts(VERIFY_PARAM_SPEC)}


@router.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile."""
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from ..backends import engine_has_model_sizes

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Resolve/validate tuning options BEFORE creating the generation row, so a
    # bad option 422s cleanly instead of orphaning a "generating" row.
    resolved_options = _resolve_tts_options(
        engine, data.language, getattr(profile, "option_overrides", None), data.tts_params
    )
    resolved_verify = _resolve_verify_options(data.verify_config)

    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None

    text = data.text
    source = "manual"
    if data.personality and getattr(profile, "personality", None):
        try:
            llm_result = await personality.rewrite_as_profile(profile.personality, data.text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        text = llm_result.text.strip()
        if not text:
            raise HTTPException(status_code=500, detail="LLM produced empty output; nothing to speak.")
        source = "personality_speak"

    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=data.instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size if engine_has_model_sizes(engine) else None,
        source=source,
        # Record the resolved params up front so the row is self-describing even
        # if the render dies. A completed generate overwrites this with the full
        # record; a FAILED row keeps it, which is the only reason retry has
        # anything to reproduce from.
        gen_params={
            "engine": engine,
            "tts_params": resolved_options or {},
            **({"verify_config": resolved_verify} if resolved_verify else {}),
        },
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=text,
    )

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    else:
        import json as _json

        profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
        if profile_obj and profile_obj.effects_chain:
            try:
                effects_chain_config = _json.loads(profile_obj.effects_chain)
            except Exception:
                pass

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=data.profile_id,
            text=text,
            language=data.language,
            engine=engine,
            model_size=model_size,
            seed=data.seed,
            normalize=data.normalize,
            effects_chain=effects_chain_config,
            instruct=data.instruct,
            mode="generate",
            max_chunk_chars=data.max_chunk_chars,
            crossfade_ms=data.crossfade_ms,
            tts_params=resolved_options,
            verify=data.verify,
            verify_config=resolved_verify,
        )
    )

    return generation


@router.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    # Reproduce from what the row was RENDERED with, never from current config.
    # Re-deriving would silently swap a deliberately-set temperature for the
    # engine default — invisible in the row, audible only in the output. Stored
    # values are already resolved, so they pass through as-is.
    stored = (gen.gen_params or {}) if isinstance(gen.gen_params, dict) else {}
    stored_tts = stored.get("tts_params") or None
    stored_verify_cfg = stored.get("verify_config") or None

    gen.status = "generating"
    gen.error = None
    gen.audio_path = ""
    gen.duration = 0
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or "1.7B",
            seed=gen.seed,
            instruct=gen.instruct,
            mode="retry",
            tts_params=stored_tts,
            verify=bool(stored_verify_cfg),
            verify_config=stored_verify_cfg,
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(
    generation_id: str,
    data: Optional[models.RegenerateRequest] = None,
    db: Session = Depends(get_db),
):
    """Re-run TTS and save the result as a new take.

    With no body (or no ``profile_id``) this is a plain regenerate: same voice,
    another take. With ``profile_id`` it is a **recast** — the same text in a
    different voice, stored as a comparable take under the same generation.

    Inheritance rules (FORK_NOTES / issue #4):

    * ``tts_params`` are inherited from the parent's resolved record — a recast
      should sound like the same tuning, just a different speaker.
    * ``chars_per_second`` is deliberately NOT inherited on a recast. Pace is
      per-voice, and verify predicts expected duration from it; reusing the
      previous voice's pace makes a complete render look short.
    * The seed is ALWAYS re-rolled unless explicitly supplied. Determinism holds
      over ``(seed, text, ref_audio, params)``, so reusing the parent's seed
      reproduces byte-identical audio — useless for "another take", and
      meaningless across voices since the reference audio changed.
    """
    data = data or models.RegenerateRequest()

    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    recast_profile_id = data.profile_id if data.profile_id != gen.profile_id else None
    render_profile_id = recast_profile_id or gen.profile_id

    engine = gen.engine or "qwen"

    profile = await profiles.get_profile(render_profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if recast_profile_id:
        # The new voice may not support the parent's engine — fail before the
        # row flips to "generating".
        try:
            profiles.validate_profile_engine(profile, engine)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    parent_params = (gen.gen_params or {}) if isinstance(gen.gen_params, dict) else {}

    # Whether the option sets came from the CALLER (validate strictly — a typo
    # should be loud) or were INHERITED from the parent's stored record
    # (validate leniently — the user didn't type these, and a record may carry
    # keys the spec doesn't advertise, e.g. escalation's max_split_depth).
    tts_overrides = data.tts_params
    tts_from_caller = tts_overrides is not None
    if not tts_from_caller:
        tts_overrides = parent_params.get("tts_params") or None

    verify_overrides = data.verify_config
    verify_from_caller = verify_overrides is not None
    if not verify_from_caller:
        inherited = parent_params.get("verify_config") or {}
        if isinstance(inherited, dict):
            # Drop the gate's derived/per-voice values: chars_per_second is the
            # speaker's pace and must be re-derived for a different voice.
            verify_overrides = {
                k: v
                for k, v in inherited.items()
                if k not in ("language",) and not (recast_profile_id and k == "chars_per_second")
            } or None

    if tts_from_caller:
        resolved_tts = _resolve_tts_options(
            engine, gen.language, getattr(profile, "option_overrides", None), tts_overrides
        )
    else:
        # Inherited: fold the stored params in as a lenient layer so a stale or
        # cross-engine key is dropped rather than 422-ing a plain regenerate.
        from ..utils.param_spec import filter_applicable
        from ..backends import get_param_spec

        spec = get_param_spec(engine)
        resolved_tts = _resolve_tts_options(
            engine,
            gen.language,
            {
                **(getattr(profile, "option_overrides", None) or {}),
                **(filter_applicable(spec, tts_overrides) if spec else {}),
            },
            None,
        )
    resolved_verify = _resolve_verify_options(verify_overrides, strict=verify_from_caller)

    gen.status = "generating"
    gen.error = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=render_profile_id,
        text=gen.text,
    )

    version_id = str(uuid.uuid4())

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=render_profile_id,
            text=gen.text,
            language=gen.language,
            engine=engine,
            model_size=gen.model_size or "1.7B",
            seed=data.seed,
            instruct=gen.instruct,
            mode="regenerate",
            version_id=version_id,
            tts_params=resolved_tts,
            verify=bool(data.verify),
            verify_config=resolved_verify,
            recast_profile_id=recast_profile_id,
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post("/generate/{generation_id}/cancel")
async def cancel_generation(generation_id: str, db: Session = Depends(get_db)):
    """Cancel a queued or running generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") not in ("loading_model", "generating"):
        raise HTTPException(status_code=400, detail="Only active generations can be cancelled")

    cancellation_state = cancel_generation_job(generation_id)
    if cancellation_state is None:
        # Row says active but the worker is no longer tracking it — the gen
        # coroutine exited without writing a terminal status (most often a
        # SQLite lock racing with the failed-status write inside the worker's
        # exception handler). Fail the row here so the user can move on.
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation orphaned by worker",
        )
        return {"message": "Orphaned generation cleared"}

    if cancellation_state == "queued":
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation cancelled",
        )
        return {"message": "Queued generation cancelled"}

    return {"message": "Generation cancellation requested"}


@router.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates."""
    import json

    async def event_stream():
        try:
            while True:
                db.expire_all()
                gen = db.query(DBGeneration).filter_by(id=generation_id).first()
                if not gen:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                    return

                payload = {
                    "id": gen.id,
                    "status": gen.status or "completed",
                    "duration": gen.duration,
                    "error": gen.error,
                    # Agent-originated sources ("mcp", "rest") skip main-window
                    # autoplay — the floating pill plays those directly.
                    "source": gen.source,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if (gen.status or "completed") in ("completed", "failed"):
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for generation %s", generation_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate/stream")
async def stream_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech and stream the WAV audio directly without saving to disk."""
    from ..backends import get_tts_backend_for_engine, ensure_model_cached_or_raise, load_engine_model, engine_needs_trim

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    resolved_options = _resolve_tts_options(
        engine, data.language, getattr(profile, "option_overrides", None), data.tts_params
    )
    tts_model = get_tts_backend_for_engine(engine)
    model_size = data.model_size or "1.7B"

    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    voice_prompt = await profiles.create_voice_prompt_for_profile(
        data.profile_id,
        db,
        engine=engine,
    )

    from ..utils.chunked_tts import generate_chunked

    trim_fn = None
    if engine_needs_trim(engine):
        from ..utils.audio import trim_tts_output

        trim_fn = trim_tts_output

    result = await generate_chunked(
        tts_model,
        data.text,
        voice_prompt,
        language=data.language,
        seed=data.seed,
        instruct=data.instruct,
        max_chunk_chars=data.max_chunk_chars,
        crossfade_ms=data.crossfade_ms,
        trim_fn=trim_fn,
        options=resolved_options,
    )
    audio, sample_rate = result.audio, result.sample_rate

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    elif profile.effects_chain:
        import json as _json

        try:
            effects_chain_config = _json.loads(profile.effects_chain)
        except Exception:
            effects_chain_config = None

    if effects_chain_config:
        from ..utils.effects import apply_effects

        audio = apply_effects(audio, sample_rate, effects_chain_config)

    if data.normalize:
        from ..utils.audio import normalize_audio

        audio = normalize_audio(audio)

    wav_bytes = tts.audio_to_wav_bytes(audio, sample_rate)

    async def _wav_stream():
        try:
            chunk_size = 64 * 1024
            for i in range(0, len(wav_bytes), chunk_size):
                yield wav_bytes[i : i + chunk_size]
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("Client disconnected during audio stream")

    return StreamingResponse(
        _wav_stream(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@router.post("/generate/import", response_model=models.GenerationResponse)
async def import_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an external audio file as a generation row.

    Designed for the story timeline so users can drop in music or other
    non-TTS audio. The row points at a singleton "Imported Audio" profile
    so the existing generation/story plumbing keeps working unchanged."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMPORT_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(IMPORT_AUDIO_EXTENSIONS)}",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > IMPORT_AUDIO_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {IMPORT_AUDIO_MAX_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    generation_id = str(uuid.uuid4())
    target = config.get_generations_dir() / f"{generation_id}{suffix}"
    target.write_bytes(audio_bytes)

    try:
        audio, sr = load_audio(str(target))
        duration = float(len(audio) / sr) if sr else 0.0
    except Exception as decode_err:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {decode_err}",
        ) from decode_err

    profile = _get_or_create_import_profile(db)
    display_name = Path(file.filename or "Imported audio").stem or "Imported audio"

    return await history.create_generation(
        profile_id=profile.id,
        text=display_name,
        language="en",
        audio_path=config.to_storage_path(target),
        duration=duration,
        seed=None,
        db=db,
        generation_id=generation_id,
        status="completed",
        engine="import",
        model_size=None,
        source="import",
    )
