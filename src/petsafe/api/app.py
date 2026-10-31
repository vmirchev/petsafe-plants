"""FastAPI endpoints for the trained and Meta vision comparison flows."""

import asyncio
import json
import os
import base64
import hashlib
import hmac
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from PIL import Image

from petsafe.api.schemas import (
    CompareResponse,
    MuseResponse,
    PlantAssessment,
    SegmentationResult,
    SparkToxicityOpinion,
    TrainedPrediction,
)
from petsafe.api.services import (
    MAX_UPLOAD_BYTES,
    MAX_SEPARATE_PLANTS,
    MODEL_NAMES,
    RUNS_DIR,
    ProgressCallback,
    assess_with_spark,
    ask_spark_toxicity_opinion,
    available_models,
    load_uploaded_image,
    predict_trained_flow,
    segment_and_assess,
)
from petsafe.api.admin_store import MetaUnavailable, record_investigation, save_settings, stats

APP_ENV = os.getenv("PETSAFE_APP_ENV", "development").strip().lower()
if APP_ENV not in {"development", "production"}:
    raise ValueError("PETSAFE_APP_ENV must be 'development' or 'production'.")


def _client_payload(value):
    """Omit diagnostic model metadata from production API responses."""
    if APP_ENV != "production":
        return value
    if isinstance(value, dict):
        return {
            key: _client_payload(item)
            for key, item in value.items()
            if key not in {"raw_meta_response", "sam_response", "api_usage", "cache_hit"}
        }
    if isinstance(value, list):
        return [_client_payload(item) for item in value]
    return value


def _client_model(value):
    """Apply the same production filtering to non-streaming endpoints."""
    if APP_ENV != "production":
        return value
    return type(value).model_validate(_client_payload(value.model_dump(mode="json")))


app = FastAPI(
    title="PetSafe Plants Inference API",
    version="0.1.0",
    description=(
        "Compare a local DINOv3 classifier with the separate Muse Spark "
        "plus optional SAM 3.1 flow, then attach source-backed toxicity lookups. "
        "Results are experimental and are not veterinary advice."
    ),
    docs_url=None if APP_ENV == "production" else "/docs",
    redoc_url=None if APP_ENV == "production" else "/redoc",
    openapi_url=None if APP_ENV == "production" else "/openapi.json",
)
origins = [origin.strip() for origin in os.getenv(
    "PETSAFE_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(MetaUnavailable)
async def meta_unavailable_handler(_request: Request, exc: MetaUnavailable) -> JSONResponse:
    return JSONResponse(status_code=429 if exc.code == "quota_exceeded" else 503,
                        content={"code": exc.code, "detail": exc.code})


@app.get("/healthz")
async def health() -> dict[str, str | int]:
    """Report service readiness without exposing credentials."""
    if APP_ENV == "production":
        return {"status": "ok"}
    return {
        "status": "ok",
        "trained_artifacts": "available" if available_models() else "missing",
        "max_separate_plants": MAX_SEPARATE_PLANTS,
    }


@app.get("/api/config")
async def frontend_config() -> dict[str, str]:
    """Tell the static client which presentation mode the server selected."""
    return {"environment": APP_ENV}


@app.get("/api/availability")
async def availability() -> dict:
    """Report whether new paid Meta calls can be made."""
    current = stats()
    code = "api_disabled" if not current["enabled"] else "quota_exceeded" if not current["remaining"] else "available"
    return {"code": code, "github_url": "https://github.com/vmirchev/petsafe-plants",
            "demo_url": os.getenv("PETSAFE_DEMO_URL", "https://www.youtube.com/")}


class AdminLogin(BaseModel):
    password: str


class AdminSettings(BaseModel):
    enabled: bool
    daily_limit: int = Field(ge=0, le=1_000_000)


def _admin_secret() -> bytes:
    value = os.getenv("PETSAFE_SESSION_SECRET", "")
    if len(value) < 32:
        raise HTTPException(503, "Admin session secret is not configured.")
    return value.encode()


def _require_admin(request: Request) -> None:
    token = request.cookies.get("petsafe_admin", "")
    try:
        payload, signature = token.rsplit(".", 1)
        expected = hmac.new(_admin_secret(), payload.encode(), hashlib.sha256).hexdigest()
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode()
        expires = int(decoded.split(":", 1)[0])
    except (ValueError, UnicodeError, IndexError):
        raise HTTPException(401, "Admin login required.") from None
    if not hmac.compare_digest(signature, expected) or expires < time.time():
        raise HTTPException(401, "Admin login required.")


@app.get("/admin")
async def admin_page() -> FileResponse:
    """Serve the login shell; all dashboard data requires a signed session."""
    return FileResponse(Path(__file__).resolve().parents[3] / "web" / "admin.html")


@app.post("/admin/api/login")
async def admin_login(credentials: AdminLogin, request: Request, response: Response) -> dict:
    """Create a signed, HttpOnly admin session after password verification."""
    configured = os.getenv("PETSAFE_ADMIN_PASSWORD", "")
    _admin_secret()
    if not configured or not hmac.compare_digest(credentials.password, configured):
        raise HTTPException(401, "Invalid admin password.")
    payload = base64.urlsafe_b64encode(f"{int(time.time()) + 28800}:{secrets.token_hex(16)}".encode()).decode().rstrip("=")
    signature = hmac.new(_admin_secret(), payload.encode(), hashlib.sha256).hexdigest()
    response.set_cookie("petsafe_admin", f"{payload}.{signature}", max_age=28800,
                        httponly=True, secure=APP_ENV == "production" or request.url.scheme == "https",
                        samesite="strict", path="/admin")
    return {"ok": True}


@app.post("/admin/api/logout", dependencies=[Depends(_require_admin)])
async def admin_logout(response: Response) -> dict:
    response.delete_cookie("petsafe_admin", path="/admin")
    return {"ok": True}


@app.get("/admin/api/stats", dependencies=[Depends(_require_admin)])
async def admin_stats() -> dict:
    return stats()


@app.post("/admin/api/settings", dependencies=[Depends(_require_admin)])
async def admin_settings(settings: AdminSettings) -> dict:
    save_settings(settings.enabled, settings.daily_limit)
    return stats()


@app.get("/api/models")
async def list_models() -> dict[str, object]:
    """List notebook 03 model architectures and locally available artifacts."""
    return {
        "available": available_models(),
        "supported": list(MODEL_NAMES),
    }


async def _read_image(upload: UploadFile):
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    try:
        image, normalized = load_uploaded_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return image, normalized


def _model_name(value: str) -> str:
    if value not in MODEL_NAMES:
        raise HTTPException(status_code=422, detail=f"model_name must be one of {MODEL_NAMES}.")
    if value not in available_models():
        raise HTTPException(
            status_code=503,
            detail=f"No saved artifacts for {value}. Configure PETSAFE_RUNS_DIR.",
        )
    return value


@app.post("/api/flows/trained")
async def trained_flow(
    image_file: UploadFile = File(...),
    model_name: str = Form("dinov3"),
) -> TrainedPrediction:
    """Run one selected notebook 03 classifier and return its top classes."""
    name = _model_name(model_name)
    image, _normalized = await _read_image(image_file)
    try:
        record_investigation()
        return _client_model(await predict_trained_flow(name, image))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/flows/muse-spark")
async def muse_spark_flow(image_file: UploadFile = File(...)) -> PlantAssessment:
    """Run Muse Spark on the original image and return its structured JSON."""
    _image, normalized = await _read_image(image_file)
    try:
        record_investigation()
        return _client_model(await assess_with_spark(normalized))
    except MetaUnavailable:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Muse Spark request failed: {exc}") from exc


@app.post("/api/flows/muse-toxicity-opinion")
async def muse_toxicity_opinion(
    image_file: UploadFile = File(...),
    plant_name: str = Form(...),
    animal: str = Form(...),
) -> SparkToxicityOpinion:
    """Return an explicitly unverified toxicity concern for one animal."""
    if animal not in {"cats", "dogs"}:
        raise HTTPException(status_code=422, detail="animal must be cats or dogs.")
    if not plant_name.strip() or len(plant_name) > 160:
        raise HTTPException(status_code=422, detail="plant_name must be 1–160 characters.")
    _image, normalized = await _read_image(image_file)
    try:
        return await ask_spark_toxicity_opinion(normalized, plant_name.strip(), animal)
    except MetaUnavailable:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Muse Spark request failed: {exc}") from exc


@app.post("/api/flows/compare", response_model=CompareResponse)
async def compare_flows(
    image_file: UploadFile = File(...),
    model_name: str = Form("dinov3"),
    force_sam: bool = Form(False),
    skip_sam: bool = Form(False),
    sam_prompt: str = Form("auto"),
) -> CompareResponse:
    """Run the DINOv3 classifier and separate Muse Spark/SAM flows."""
    if force_sam and skip_sam:
        raise HTTPException(status_code=422, detail="force_sam and skip_sam cannot both be true.")
    if sam_prompt not in {"auto", "flower", "plant"}:
        raise HTTPException(status_code=422, detail="sam_prompt must be 'auto', 'flower', or 'plant'.")
    name = _model_name(model_name)
    image, normalized = await _read_image(image_file)
    return _client_model(await _run_comparison(image, normalized, name, force_sam, skip_sam, sam_prompt))


async def _run_comparison(
    image: Image.Image,
    normalized: bytes,
    name: str,
    force_sam: bool,
    skip_sam: bool,
    sam_prompt: str,
    emit: ProgressCallback | None = None,
) -> CompareResponse:
    async def publish(event: str, data: dict) -> None:
        if emit:
            await emit(event, data)

    await publish("analyzing_scene", {"image_width": image.width, "image_height": image.height})

    async def classifier():
        result = await predict_trained_flow(name, image)
        await publish("trained_ready", result.model_dump(mode="json"))
        return result

    async def muse_path() -> tuple[PlantAssessment, SegmentationResult]:
        return await _run_muse_path(
            image, normalized, force_sam, skip_sam, sam_prompt,
            trained_model_classes(name), publish,
        )

    # SAM can reveal specimens as soon as Muse Spark has assessed the scene.
    # The independent DINOv3 classifier path may still be running.
    tasks = [asyncio.create_task(classifier()), asyncio.create_task(muse_path())]
    try:
        trained, (spark, segmentation) = await asyncio.gather(*tasks)
    except MetaUnavailable:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Initial comparison failed: {exc}") from exc
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return CompareResponse(
        trained_flow=trained,
        muse_spark_flow=spark,
        segmentation=segmentation,
        image_width=image.width,
        image_height=image.height,
    )


async def _run_muse_path(
    image: Image.Image,
    normalized: bytes,
    force_sam: bool,
    skip_sam: bool,
    sam_prompt: str,
    class_names: list[str],
    publish: ProgressCallback,
) -> tuple[PlantAssessment, SegmentationResult]:
    """Assess the scene and selected SAM regions without requiring a classifier."""
    record_investigation()
    spark = await assess_with_spark(normalized)
    if spark.separate_plant_count > MAX_SEPARATE_PLANTS:
        reason = (
            f"Muse Spark reported {spark.separate_plant_count} separate plants "
            "in this photo. Inspect one plant at a time in closer photos."
        )
        spark = spark.model_copy(update={
            "suggest_segmentation": False,
            "segmentation_reason": reason,
            "another_photo_needed": True,
            "photo_instructions": (
                "Take a closer, sharp photo of one individual plant, including its flowers, "
                "leaves, and stem. Repeat for the other plants."
            ),
            "photo_reason": reason,
        })
        await publish("scene_analyzed", {
            **spark.model_dump(mode="json"),
            "max_separate_plants": MAX_SEPARATE_PLANTS,
        })
        return spark, SegmentationResult(
            ran=False,
            status="skipped",
            reason=[reason],
        )
    await publish("scene_analyzed", {
        **spark.model_dump(mode="json"),
        "max_separate_plants": MAX_SEPARATE_PLANTS,
    })
    selected_target = spark.segmentation_target if sam_prompt == "auto" else sam_prompt
    selected_prompt = "flower" if selected_target == "flower" else "plant"
    reasons: list[str] = []
    if spark.suggest_segmentation:
        reasons.append("Muse Spark recommended a closer look at plant regions.")
    if spark.distinct_plant_count > 1:
        reasons.append("Multiple distinct plant types were reported.")
    if spark.bouquet_type in {"mixed_species", "uncertain"}:
        reasons.append("The bouquet may contain different plant types.")
    if any(group.needs_inspection and group.identification_confidence == "low"
           for group in spark.plant_groups):
        reasons.append("A low-confidence plant group needs inspection.")

    should_segment = force_sam or (bool(reasons) and not skip_sam)
    if force_sam and not reasons:
        reasons.append("SAM was requested explicitly.")
    if sam_prompt != "auto" and should_segment:
        reasons.append("The SAM target was selected manually.")
    if skip_sam:
        segmentation = SegmentationResult(
            ran=False,
            status="skipped",
            reason=reasons or ["SAM was disabled for this request."],
        )
    elif not should_segment:
        segmentation = SegmentationResult(
            ran=False,
            status="skipped",
            reason=["No trigger condition was met; the original image assessment is shown."],
        )
    else:
        try:
            await publish("segmenting", {"reason": reasons, "prompt": selected_prompt})
            segments, counts = await segment_and_assess(
                image, normalized, class_names, prompt=selected_prompt, emit=publish
            )
            segmentation = SegmentationResult(
                ran=True,
                status="completed" if segments else "no_masks",
                reason=reasons,
                prompt=selected_prompt,
                segments=segments,
                **counts,
            )
            if not segments and any(
                group.needs_inspection or group.identification_confidence == "low"
                for group in spark.plant_groups
            ) and not spark.another_photo_needed:
                spark = spark.model_copy(update={
                    "another_photo_needed": True,
                    "photo_instructions": "Take a closer, sharper photo showing the uncertain plant and its leaves or flowers.",
                    "photo_reason": "No useful isolated region was available for another visual assessment.",
                })
        except MetaUnavailable:
            raise
        except Exception as exc:
            # Keep the scene result usable if SAM or a crop reassessment fails.
            segmentation = SegmentationResult(
                ran=True,
                status="failed",
                reason=reasons,
                prompt=selected_prompt,
                error=str(exc),
            )
    return spark, segmentation


@app.post("/api/flows/muse-stream")
async def stream_muse(
    image_file: UploadFile = File(...),
    force_sam: bool = Form(False),
    skip_sam: bool = Form(False),
    sam_prompt: str = Form("auto"),
) -> StreamingResponse:
    """Stream Muse Spark, optional SAM, and regional Spark results without model artifacts."""
    if force_sam and skip_sam:
        raise HTTPException(422, "force_sam and skip_sam cannot both be true.")
    if sam_prompt not in {"auto", "flower", "plant"}:
        raise HTTPException(422, "sam_prompt must be auto, flower, or plant.")
    image, normalized = await _read_image(image_file)

    async def events():
        queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

        async def emit(event: str, data: dict) -> None:
            await queue.put((event, _client_payload(data)))

        async def run():
            try:
                await emit("analyzing_scene", {
                    "image_width": image.width, "image_height": image.height,
                })
                spark, segmentation = await _run_muse_path(
                    image, normalized, force_sam, skip_sam, sam_prompt, [], emit,
                )
                result = MuseResponse(
                    muse_spark_flow=spark,
                    segmentation=segmentation,
                    image_width=image.width,
                    image_height=image.height,
                )
                await emit("completed", result.model_dump(mode="json"))
            except MetaUnavailable as exc:
                await emit("error", {"code": exc.code, "message": exc.code})
            except Exception as exc:
                await emit("error", {"message": str(exc)})

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                if event in {"completed", "error"}:
                    break
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/flows/compare-stream")
async def stream_comparison(
    image_file: UploadFile = File(...),
    model_name: str = Form("dinov3"),
    force_sam: bool = Form(False),
    skip_sam: bool = Form(False),
    sam_prompt: str = Form("auto"),
) -> StreamingResponse:
    """Stream real comparison events over an SSE response to a multipart POST."""
    if force_sam and skip_sam:
        raise HTTPException(422, "force_sam and skip_sam cannot both be true.")
    if sam_prompt not in {"auto", "flower", "plant"}:
        raise HTTPException(422, "sam_prompt must be auto, flower, or plant.")
    name = _model_name(model_name)
    image, normalized = await _read_image(image_file)

    async def events():
        queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

        async def emit(event: str, data: dict) -> None:
            await queue.put((event, _client_payload(data)))

        async def run():
            try:
                result = await _run_comparison(
                    image, normalized, name, force_sam, skip_sam, sam_prompt, emit
                )
                await emit("completed", result.model_dump(mode="json"))
            except MetaUnavailable as exc:
                await emit("error", {"code": exc.code, "message": exc.code})
            except Exception as exc:
                await emit("error", {"message": str(exc)})

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                if event in {"completed", "error"}:
                    break
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def trained_model_classes(name: str) -> list[str]:
    """Read a model's class order for Spark taxonomy comparison."""
    import json

    mapping_path = RUNS_DIR / name / "class_mapping.json"
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return list(mapping["index_to_class"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Invalid class mapping: {mapping_path}") from exc


WEB_DIR = Path(__file__).resolve().parents[3] / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
