"""Model loading and inference services used by the HTTP routes."""

import asyncio
import base64
import io
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from petsafe.api.result_cache import cache_result, get_cached_result, image_digest
from petsafe.api.prompts import PLANT_ASSESSMENT_PROMPT
from petsafe.api.schemas import (
    ClassScore,
    FlowerDetection,
    PlantAssessment,
    SegmentAssessment,
    SparkToxicityOpinion,
    SparkUsage,
    TrainedPrediction,
)
from petsafe.api.toxicity import load_toxicity_database, lookup_toxicity

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

MODEL_NAMES = ("siglip2",)
DATASET_CLASSES = 18
MAX_UPLOAD_BYTES = int(os.getenv("PETSAFE_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_IMAGE_SIDE = int(os.getenv("PETSAFE_MAX_IMAGE_SIDE", "2048"))
MAX_SEGMENTS = int(os.getenv("PETSAFE_MAX_SEGMENTS", "6"))
MIN_MASK_PIXELS = int(os.getenv("PETSAFE_MIN_MASK_PIXELS", "300"))
MIN_MASK_AREA_RATIO = 0.001
MIN_CROP_SIDE = 32
MAX_FLOWER_DETECTIONS = int(os.getenv("PETSAFE_MAX_FLOWER_DETECTIONS", "6"))
FLOWER_DETECTION_THRESHOLD = float(os.getenv("PETSAFE_FLOWER_DETECTION_THRESHOLD", "0.05"))
DEFAULT_RUNS_DIR = Path("runs/plant_export_v20261026_2355546_holdout10_seed42")
RUNS_DIR = Path(os.getenv("PETSAFE_RUNS_DIR", str(DEFAULT_RUNS_DIR))).expanduser()
MODEL_API_BASE = "https://api.meta.ai/v1"
SPARK_MODEL = os.getenv("PETSAFE_SPARK_MODEL", "muse-spark-1.3")
SPARK_MAX_COMPLETION_TOKENS = int(
    os.getenv("PETSAFE_SPARK_MAX_COMPLETION_TOKENS", "3000")
)
SAM_MODEL = "sam-3.1"
_MODEL_LOCK = Lock()
_DETECTOR_LOCK = Lock()

_CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}
_PLANT_GROUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "integer"},
        "plant_identification": {"type": "string"},
        "scientific_name": {"type": "string"},
        "toxicity_lookup_key": {"type": "string"},
        "identification_confidence": _CONFIDENCE,
        "visual_evidence": {"type": "array", "items": {"type": "string"}},
        "needs_inspection": {"type": "boolean"},
    },
    "required": ["id", "plant_identification", "scientific_name", "toxicity_lookup_key",
                 "identification_confidence", "visual_evidence", "needs_inspection"],
}
ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "plant_image_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_bouquet": {"type": "boolean"},
                "bouquet_type": {"type": "string", "enum": ["single_species", "mixed_species", "uncertain", "not_bouquet"]},
                "distinct_plant_count": {"type": "integer"},
                "plant_groups": {"type": "array", "items": _PLANT_GROUP_SCHEMA},
                "suggest_segmentation": {"type": "boolean"},
                "segmentation_target": {"type": "string", "enum": ["flower", "plant"]},
                "segmentation_target_reason": {"type": "string"},
                "segmentation_reason": {"type": "string"},
                "another_photo_needed": {"type": "boolean"},
                "photo_instructions": {"type": "string"},
                "photo_reason": {"type": "string"},
            },
            "required": [
                "is_bouquet", "bouquet_type", "distinct_plant_count", "plant_groups",
                "suggest_segmentation", "segmentation_target", "segmentation_target_reason",
                "segmentation_reason",
                "another_photo_needed", "photo_instructions", "photo_reason",
            ],
        },
    },
}

def available_models() -> list[str]:
    """Return trained model folders with the files required for inference."""
    return [
        name
        for name in MODEL_NAMES
        if all((RUNS_DIR / name / filename).is_file() for filename in (
            "best_head.pt",
            "class_mapping.json",
            "config.json",
        ))
    ]


def load_uploaded_image(data: bytes) -> tuple[Image.Image, bytes]:
    """Validate, orient, normalize, and bound an uploaded image."""
    if not data:
        raise ValueError("The uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Upload a readable JPEG, PNG, or WebP image.") from exc
    if image.width < 2 or image.height < 2:
        raise ValueError("Image dimensions must be at least 2 by 2 pixels.")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    return image, buffer.getvalue()


class TrainedModel:
    """One loaded frozen encoder and its trained classification head."""

    def __init__(self, name: str) -> None:
        if name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model_name {name!r}; choose from {MODEL_NAMES}.")
        run_dir = RUNS_DIR / name
        required = [run_dir / filename for filename in (
            "best_head.pt",
            "class_mapping.json",
            "config.json",
        )]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                f"Missing trained artifacts for {name} under {run_dir}. "
                "Set PETSAFE_RUNS_DIR to notebook 03's experiment folder."
            )

        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["training"]
        mapping = json.loads((run_dir / "class_mapping.json").read_text(encoding="utf-8"))
        self.class_names = mapping["index_to_class"]
        if len(self.class_names) != DATASET_CLASSES:
            raise ValueError(f"Expected {DATASET_CLASSES} classes in {run_dir / 'class_mapping.json'}.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1
            self.encoder = efficientnet_b2(weights=weights)
            self.encoder.classifier = nn.Identity()
            self.processor = weights.transforms()
            self.encoder_kind = "torchvision"
        else:
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor

            model_id = config["hf_dinov3_id"] if name == "dinov3" else config["hf_siglip2_id"]
            token = os.getenv("HF_TOKEN") if name == "dinov3" else None
            processor_class = AutoImageProcessor if name == "dinov3" else AutoProcessor
            try:
                self.processor = processor_class.from_pretrained(model_id, token=token)
                self.encoder = AutoModel.from_pretrained(model_id, token=token)
            except OSError as exc:
                if name == "dinov3":
                    raise RuntimeError(
                        "DINOv3 access failed. Accept the Hugging Face model terms and set HF_TOKEN."
                    ) from exc
                raise
            self.encoder_kind = name
        self.encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        state = torch.load(required[0], map_location="cpu", weights_only=True)
        embedding_size = int(state["0.weight"].shape[0])
        output_classes = int(state["2.weight"].shape[0])
        if output_classes != len(self.class_names):
            raise ValueError("Saved classifier output dimension does not match the class mapping.")
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Dropout(0.2),
            nn.Linear(embedding_size, output_classes),
        )
        self.head.load_state_dict(state)
        self.head.to(self.device).eval()
        self.name = name

    def _embedding(self, image: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            if self.encoder_kind == "torchvision":
                pixels = self.processor(image).unsqueeze(0).to(self.device)
                return self.encoder(pixels)
            inputs = self.processor(images=[image], return_tensors="pt")
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            if self.encoder_kind == "siglip2":
                output = self.encoder.get_image_features(**inputs)
            else:
                output = self.encoder(**inputs, return_dict=True)
            if torch.is_tensor(output):
                return output
            for attribute in ("image_embeds", "pooler_output"):
                value = getattr(output, attribute, None)
                if torch.is_tensor(value):
                    return value
            hidden = getattr(output, "last_hidden_state", None)
            if torch.is_tensor(hidden):
                return hidden[:, 0]
        raise TypeError(f"No image embedding tensor returned for {self.name}.")

    def predict(self, image: Image.Image, top_k: int = 5) -> TrainedPrediction:
        """Run the notebook 03 encoder and saved head on one image."""
        features = self._embedding(image).float()
        with torch.inference_mode():
            probabilities = self.head(features).softmax(dim=1)[0]
        count = min(top_k, len(self.class_names))
        scores, indices = probabilities.topk(count)
        predictions = [
            ClassScore(class_name=self.class_names[int(index)], probability=float(score))
            for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
        ]
        return TrainedPrediction(
            model_name=self.name,
            predicted_class=predictions[0].class_name,
            confidence=predictions[0].probability,
            top_predictions=predictions,
        )


@lru_cache(maxsize=1)
def get_trained_model(name: str) -> TrainedModel:
    """Load at most one model to limit GPU and host memory use."""
    with _MODEL_LOCK:
        return TrainedModel(name)


async def predict_trained(name: str, image: Image.Image) -> TrainedPrediction:
    """Run blocking torch and model-loading work off the async event loop."""
    rgb = image.convert("RGB")
    image_key = image_digest(
        rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes()
    )
    cache_key = f"trained:{name}:{image_key}"
    cached = get_cached_result(cache_key)
    if cached is not None:
        return TrainedPrediction.model_validate(cached).model_copy(update={"cache_hit": True})

    prediction = await asyncio.to_thread(lambda: get_trained_model(name).predict(rgb))
    cache_result(cache_key, prediction.model_dump(mode="json"))
    return prediction


@lru_cache(maxsize=1)
def get_flower_detector():
    """Load Grounding DINO tiny once, on the first trained-flow request."""
    from transformers import pipeline

    return pipeline(
        "zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-tiny",
        device=0 if torch.cuda.is_available() else -1,
    )


def _flower_boxes(image: Image.Image) -> list[tuple[tuple[int, int, int, int], float]]:
    """Detect and deduplicate flower boxes before crop classification."""
    with _DETECTOR_LOCK:
        detections = get_flower_detector()(
            image, candidate_labels=["flower"], threshold=FLOWER_DETECTION_THRESHOLD
        )
    candidates = []
    for detection in detections:
        box = detection["box"]
        bounds = (
            max(0, int(box["xmin"])), max(0, int(box["ymin"])),
            min(image.width, int(box["xmax"])), min(image.height, int(box["ymax"])),
        )
        if bounds[2] - bounds[0] < 24 or bounds[3] - bounds[1] < 24:
            continue
        candidates.append((bounds, float(detection["score"])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept = []
    for bounds, score in candidates:
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        overlaps = False
        for other, _ in kept:
            intersection = (
                max(0, min(bounds[2], other[2]) - max(bounds[0], other[0]))
                * max(0, min(bounds[3], other[3]) - max(bounds[1], other[1]))
            )
            other_area = (other[2] - other[0]) * (other[3] - other[1])
            if intersection / (area + other_area - intersection) > 0.5:
                overlaps = True
                break
        if overlaps:
            continue
        kept.append((bounds, score))
        if len(kept) >= MAX_FLOWER_DETECTIONS:
            break
    return kept


async def predict_trained_flow(name: str, image: Image.Image) -> TrainedPrediction:
    """Classify the whole image and each zero-shot flower crop independently."""
    full = await predict_trained(name, image)
    full = full.model_copy(update={"toxicity": lookup_toxicity(full.predicted_class)})
    box_key = (
        f"flower-boxes:v1:{FLOWER_DETECTION_THRESHOLD}:{MAX_FLOWER_DETECTIONS}:"
        f"{image.width}:{image.height}:{image_digest(image.tobytes())}"
    )
    cached_boxes = get_cached_result(box_key)
    try:
        if cached_boxes is None:
            boxes = await asyncio.to_thread(_flower_boxes, image)
            cache_result(box_key, {"boxes": boxes})
        else:
            boxes = cached_boxes["boxes"]
    except Exception as exc:
        return full.model_copy(update={"detection_status": "failed", "detection_error": str(exc)})
    detections = []
    for bounds, score in boxes:
        crop = image.crop(bounds)
        prediction = await predict_trained(name, crop)
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=85)
        detections.append(FlowerDetection(
            bounds=bounds,
            detector_score=score,
            crop_data_url=_data_url(buffer.getvalue()),
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            top_predictions=prediction.top_predictions,
            toxicity=lookup_toxicity(prediction.predicted_class),
        ))
    return full.model_copy(update={
        "detections": detections,
        "detection_status": "completed" if detections else "no_flowers",
    })


def _data_url(data: bytes, media_type: str = "image/jpeg") -> str:
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


TOXICITY_OPINION_PROMPT = (
    "You are answering an optional, unverified question about possible pet toxicity. "
    "You have no reference database or live sources in this task. The plant name "
    "is only a visual hypothesis and may be wrong. Use general model knowledge "
    "to say whether there may be a toxicity concern for the named animal. "
    "Return possible_concern only when you recall a plausible hazard; otherwise "
    "return unknown. Never claim a plant is safe, non-toxic, or verified. "
    "Do not invent citations, symptoms, treatments, or a confidence percentage. "
    "Keep the explanation brief and state the key uncertainty."
)
TOXICITY_OPINION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "unverified_pet_toxicity_opinion",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "risk_signal": {"type": "string", "enum": ["possible_concern", "unknown"]},
                "explanation": {"type": "string"},
                "uncertainty": {"type": "string"},
            },
            "required": ["risk_signal", "explanation", "uncertainty"],
        },
    },
}


async def ask_spark_toxicity_opinion(
    image_bytes: bytes, plant_name: str, animal: str,
) -> SparkToxicityOpinion:
    """Ask Muse Spark for a non-authoritative concern signal without local lookup."""
    prompt_hash = image_digest(TOXICITY_OPINION_PROMPT.encode("utf-8"))
    cache_key = (
        f"spark-toxicity-opinion:v1:{SPARK_MODEL}:{prompt_hash}:"
        f"{image_digest(image_bytes)}:"
        f"{image_digest(f'{plant_name}:{animal}'.encode('utf-8'))}"
    )
    cached = get_cached_result(cache_key)
    if cached is not None:
        return SparkToxicityOpinion.model_validate(cached)
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY in the API server environment.")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=MODEL_API_BASE, api_key=api_key, timeout=120, max_retries=2)
    try:
        response = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[
                {"role": "developer", "content": TOXICITY_OPINION_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        f"Visual hypothesis: {plant_name}. Animal: {animal}. "
                        "Could this plant pose a toxicity concern for this animal?"
                    )},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                ]},
            ],
            response_format=TOXICITY_OPINION_SCHEMA,
            extra_body={"prompt_cache_key": f"petsafe-toxicity-opinion-{prompt_hash[:20]}"},
            max_completion_tokens=450,
            reasoning_effort="minimal",
        )
    finally:
        await client.close()
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("Muse Spark returned a truncated toxicity opinion.")
    if not choice.message.content:
        raise RuntimeError("Muse Spark returned an empty toxicity opinion.")
    opinion = SparkToxicityOpinion.model_validate_json(choice.message.content)
    cache_result(cache_key, opinion.model_dump(mode="json"))
    return opinion


async def assess_with_spark(image_bytes: bytes, extra_instruction: str = "") -> PlantAssessment:
    """Return Muse Spark's schema-validated multi-plant assessment."""
    supported_classes = ", ".join(
        item["class_name"] for item in load_toxicity_database()["plants"]
    )
    developer_prompt = (
        PLANT_ASSESSMENT_PROMPT + "\nAllowed toxicity_lookup_key values: "
        + supported_classes + ", unknown."
    )
    prompt_hash = image_digest(developer_prompt.encode("utf-8"))
    cache_key = (
        f"spark:v4:{SPARK_MODEL}:{prompt_hash}:"
        f"{image_digest(image_bytes)}:"
        f"{image_digest(extra_instruction.encode('utf-8'))}"
    )
    cached = get_cached_result(cache_key)
    if cached is not None:
        assessment = PlantAssessment.model_validate(cached)
        return assessment.model_copy(
            update={
                "plant_groups": _with_toxicity(assessment).plant_groups,
                "cache_hit": True,
            }
        )

    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY in the API server environment.")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=MODEL_API_BASE, api_key=api_key, timeout=120, max_retries=2)
    try:
        response = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": extra_instruction or "Assess this photograph."},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                ]},
            ],
            response_format=ASSESSMENT_SCHEMA,
            extra_body={"prompt_cache_key": f"petsafe-plants-{prompt_hash[:20]}"},
            max_completion_tokens=SPARK_MAX_COMPLETION_TOKENS,
            reasoning_effort="minimal",
        )
    finally:
        await client.close()
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(
            "Muse Spark reached the completion-token limit and returned truncated JSON. "
            "Increase PETSAFE_SPARK_MAX_COMPLETION_TOKENS and restart the API."
        )
    content = choice.message.content
    if not content:
        raise RuntimeError("Muse Spark returned an empty assessment.")
    assessment = PlantAssessment.model_validate_json(content)
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    cached_tokens = (
        details.get("cached_tokens") if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    assessment = assessment.model_copy(update={
        "api_usage": SparkUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_hit_ratio=(cached_tokens / input_tokens
                             if cached_tokens is not None and input_tokens else None),
        ),
        "raw_meta_response": response.model_dump(mode="json"),
    })
    # Keep toxicity deterministic and source-backed; it is never requested from Spark.
    cache_result(cache_key, assessment.model_dump(mode="json"))
    return _with_toxicity(assessment)


def _with_toxicity(assessment: PlantAssessment) -> PlantAssessment:
    """Attach verified local lookups to each visual hypothesis."""
    groups = []
    for group in assessment.plant_groups:
        key = group.scientific_name if group.scientific_name.casefold() != "unknown" else group.toxicity_lookup_key
        lookup = lookup_toxicity(key)
        if lookup.matched_class is None and group.toxicity_lookup_key.casefold() != "unknown":
            lookup = lookup_toxicity(group.toxicity_lookup_key)
        groups.append(group.model_copy(update={"toxicity": lookup}))
    return assessment.model_copy(update={"plant_groups": groups})


@dataclass(eq=False)
class MaskCandidate:
    """A decoded, image-aligned mask before regional reassessment."""

    object_id: str
    bounds: tuple[int, int, int, int]
    mask: np.ndarray
    pixels: int
    confidence: float | None


def _mask_candidate(image: Image.Image, record: Any) -> MaskCandidate | None:
    """Map a SAM mask into a cropped image-aligned boolean array."""
    from meta_sam_parser import decode_mask_to_raster

    local = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    local = local.reshape(record.mask.height, record.mask.width)
    bounds = record.bounds
    left = max(0, int(np.floor(bounds.left)))
    top = max(0, int(np.floor(bounds.top)))
    right = min(image.width, int(np.ceil(bounds.right)))
    bottom = min(image.height, int(np.ceil(bounds.bottom)))
    if right <= left or bottom <= top:
        return None
    local_mask = Image.fromarray(local * 255).resize(
        (right - left, bottom - top), resample=Image.Resampling.NEAREST
    )
    mask = np.asarray(local_mask) > 0
    ys, xs = np.where(mask)
    pixels = len(xs)
    if pixels < max(MIN_MASK_PIXELS, int(image.width * image.height * MIN_MASK_AREA_RATIO)):
        return None
    if pixels > image.width * image.height * 0.9:
        return None
    if xs.max() - xs.min() + 1 < MIN_CROP_SIDE or ys.max() - ys.min() + 1 < MIN_CROP_SIDE:
        return None
    crop_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    tight_bounds = (left + crop_box[0], top + crop_box[1], left + crop_box[2], top + crop_box[3])
    raw_confidence = getattr(record, "confidence", None)
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and not np.isfinite(confidence):
        confidence = None
    return MaskCandidate(str(record.object_id), tight_bounds,
                         mask[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]],
                         pixels, confidence)


def _mask_intersection(a: MaskCandidate, b: MaskCandidate) -> int:
    left = max(a.bounds[0], b.bounds[0])
    top = max(a.bounds[1], b.bounds[1])
    right = min(a.bounds[2], b.bounds[2])
    bottom = min(a.bounds[3], b.bounds[3])
    if right <= left or bottom <= top:
        return 0
    a_slice = a.mask[top-a.bounds[1]:bottom-a.bounds[1], left-a.bounds[0]:right-a.bounds[0]]
    b_slice = b.mask[top-b.bounds[1]:bottom-b.bounds[1], left-b.bounds[0]:right-b.bounds[0]]
    return int(np.count_nonzero(a_slice & b_slice))


def _mask_quality(candidate: MaskCandidate, image_area: int) -> float:
    box = candidate.bounds
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    size = min(1.0, (candidate.pixels / max(1, image_area * 0.02)) ** 0.5)
    fill = candidate.pixels / max(1, box_area)
    confidence = max(0.0, min(1.0, candidate.confidence)) if candidate.confidence is not None else 0.5
    return 0.5 * size + 0.2 * fill + 0.3 * confidence


def _select_mask_candidates(candidates: list[MaskCandidate], image_area: int) -> list[MaskCandidate]:
    """Remove near-identical masks and select diverse, readable regions."""
    ranked = sorted(candidates, key=lambda item: _mask_quality(item, image_area), reverse=True)
    unique: list[MaskCandidate] = []
    for candidate in ranked:
        duplicate = False
        for kept in unique:
            intersection = _mask_intersection(candidate, kept)
            union = candidate.pixels + kept.pixels - intersection
            area_ratio = min(candidate.pixels, kept.pixels) / max(candidate.pixels, kept.pixels)
            if intersection / max(1, union) >= 0.8 or (
                area_ratio >= 0.65
                and intersection / max(1, min(candidate.pixels, kept.pixels)) >= 0.92
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    selected: list[MaskCandidate] = []
    while unique and len(selected) < MAX_SEGMENTS:
        best = max(unique, key=lambda item: _mask_quality(item, image_area) +
                   0.5 * (1 - max((_mask_intersection(item, prior) / item.pixels
                                   for prior in selected), default=0)))
        selected.append(best)
        unique.remove(best)
    return selected


def _encode_mask_crop(image: Image.Image, candidate: MaskCandidate) -> bytes:
    """Create a transparent PNG only for a selected region."""
    rgba = np.asarray(image.crop(candidate.bounds).convert("RGBA")).copy()
    rgba[:, :, 3] = candidate.mask.astype(np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


async def segment_and_assess(
    image: Image.Image,
    image_bytes: bytes,
    class_names: list[str],
    prompt: str = "flower head",
    emit: ProgressCallback | None = None,
) -> tuple[list[SegmentAssessment], dict[str, int]]:
    """Filter SAM masks before asking Muse Spark about informative crops."""
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY in the API server environment.")
    from meta_sam_parser import image_segmentation_format, parse_responses_stream
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=MODEL_API_BASE, api_key=api_key, timeout=180, max_retries=2)
    try:
        events = await client.responses.create(
            model=SAM_MODEL,
            input=[{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _data_url(image_bytes)},
                ],
            }],
            stream=True,
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        )
        parsed = parse_responses_stream(events, image_segmentation_format())
        result = await parsed.final_result()
    finally:
        await client.close()

    names = {name.casefold() for name in class_names}
    candidates: list[MaskCandidate] = []
    mask_count = 0
    for record in result.records:
        if getattr(record, "kind", None) != "mask":
            continue
        mask_count += 1
        candidate = _mask_candidate(image, record)
        if candidate is not None:
            candidates.append(candidate)
    selected = _select_mask_candidates(candidates, image.width * image.height)
    segments: list[SegmentAssessment] = []
    crops = [(candidate, _encode_mask_crop(image, candidate)) for candidate in selected]
    if emit:
        await emit("segments_ready", {
            "regions": [
                {
                    "object_id": candidate.object_id,
                    "bounds": candidate.bounds,
                    "crop_data_url": _data_url(crop_bytes, "image/png"),
                }
                for candidate, crop_bytes in crops
            ],
            "mask_count": mask_count,
            "filtered_count": mask_count - len(selected),
        })
        if crops:
            await emit("identifying_regions", {"count": len(crops)})
    for candidate, crop_bytes in crops:
        assessment = await assess_with_spark(
            crop_bytes,
            "This is a transparent-background crop of a SAM-selected plant region. "
            "Assess only visible plant types in this crop and preserve uncertainty.",
        )
        segments.append(SegmentAssessment(
            object_id=candidate.object_id,
            bounds=candidate.bounds,
            crop_data_url=_data_url(crop_bytes, "image/png"),
            assessment=assessment,
            in_trained_taxonomy=any(
                group.plant_identification.casefold() in names for group in assessment.plant_groups
            ),
        ))
        if emit:
            await emit("region_assessed", segments[-1].model_dump(mode="json"))
    return segments, {"mask_count": mask_count, "filtered_count": mask_count - len(selected),
                      "assessed_count": len(segments)}
