"""FastAPI endpoints for the trained and Meta vision comparison flows."""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from petsafe.api.schemas import (
    CompareResponse,
    PlantAssessment,
    SegmentationResult,
    TrainedPrediction,
)
from petsafe.api.services import (
    MAX_UPLOAD_BYTES,
    MODEL_NAMES,
    RUNS_DIR,
    assess_with_spark,
    available_models,
    load_uploaded_image,
    predict_trained,
    segment_and_assess,
)
from petsafe.api.toxicity import lookup_toxicity

app = FastAPI(
    title="PetSafe Plants Inference API",
    version="0.1.0",
    description=(
        "Compare notebook 03 plant classifiers with Muse Spark and optional SAM 3.1 "
        "segmentation, then attach a local source-backed toxicity reference lookup. "
        "Results are experimental and are not veterinary advice."
    ),
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


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Report service readiness without exposing credentials."""
    return {"status": "ok", "trained_artifacts": "available" if available_models() else "missing"}


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
    model_name: str = Form("efficientnet_b2"),
) -> TrainedPrediction:
    """Run one selected notebook 03 classifier and return its top classes."""
    name = _model_name(model_name)
    image, _normalized = await _read_image(image_file)
    try:
        prediction = await predict_trained(name, image)
        return prediction.model_copy(
            update={"toxicity": lookup_toxicity(prediction.predicted_class)}
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/flows/muse-spark")
async def muse_spark_flow(image_file: UploadFile = File(...)) -> PlantAssessment:
    """Run Muse Spark on the original image and return its structured JSON."""
    _image, normalized = await _read_image(image_file)
    try:
        return await assess_with_spark(normalized)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Muse Spark request failed: {exc}") from exc


@app.post("/api/flows/compare", response_model=CompareResponse)
async def compare_flows(
    image_file: UploadFile = File(...),
    model_name: str = Form("efficientnet_b2"),
    confidence_threshold: float = Form(0.55, ge=0.0, le=1.0),
    force_sam: bool = Form(False),
    skip_sam: bool = Form(False),
    sam_prompt: str = Form("flower"),
) -> CompareResponse:
    """Run both flows and conditionally segment/re-assess the uploaded image."""
    if force_sam and skip_sam:
        raise HTTPException(status_code=422, detail="force_sam and skip_sam cannot both be true.")
    if sam_prompt not in {"flower", "plant"}:
        raise HTTPException(status_code=422, detail="sam_prompt must be 'flower' or 'plant'.")
    name = _model_name(model_name)
    image, normalized = await _read_image(image_file)
    try:
        trained, spark = await asyncio.gather(
            predict_trained(name, image),
            assess_with_spark(normalized),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Initial comparison failed: {exc}") from exc
    trained = trained.model_copy(
        update={"toxicity": lookup_toxicity(trained.predicted_class)}
    )

    reasons: list[str] = []
    if spark.flower_count_estimate > 1:
        reasons.append("Muse Spark estimated multiple visible flowers.")
    if spark.identification_confidence == "low":
        reasons.append("Muse Spark reported low plant-identification confidence.")
    if spark.flower_count_confidence == "low":
        reasons.append("Muse Spark reported low flower-count confidence.")
    if trained.confidence < confidence_threshold:
        reasons.append(
            f"The trained model probability ({trained.confidence:.3f}) is below "
            f"the configured threshold ({confidence_threshold:.3f})."
        )
    if spark.suggest_segmentation and not reasons:
        reasons.append("Muse Spark recommended segmentation based on the image structure.")

    should_segment = force_sam or (bool(reasons) and not skip_sam)
    if force_sam and not reasons:
        reasons.append("SAM was requested explicitly.")
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
            segments = await segment_and_assess(
                image, normalized, trained_model_classes(name), prompt=sam_prompt
            )
            segmentation = SegmentationResult(
                ran=True,
                status="completed" if segments else "no_masks",
                reason=reasons,
                prompt=sam_prompt,
                segments=segments,
            )
        except Exception as exc:
            # Keep the two initial results usable if SAM or a crop reassessment fails.
            segmentation = SegmentationResult(
                ran=True,
                status="failed",
                reason=reasons,
                prompt=sam_prompt,
                error=str(exc),
            )

    return CompareResponse(
        trained_flow=trained,
        muse_spark_flow=spark,
        segmentation=segmentation,
        image_width=image.width,
        image_height=image.height,
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
