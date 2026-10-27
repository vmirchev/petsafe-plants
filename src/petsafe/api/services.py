"""Model loading and inference services used by the HTTP routes."""

import asyncio
import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from petsafe.api.schemas import (
    ClassScore,
    PlantAssessment,
    SegmentAssessment,
    TrainedPrediction,
)
from petsafe.api.toxicity import load_toxicity_database, lookup_toxicity

MODEL_NAMES = ("efficientnet_b2", "dinov3", "siglip2")
DATASET_CLASSES = 18
MAX_UPLOAD_BYTES = int(os.getenv("PETSAFE_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_IMAGE_SIDE = int(os.getenv("PETSAFE_MAX_IMAGE_SIDE", "2048"))
MAX_SEGMENTS = int(os.getenv("PETSAFE_MAX_SEGMENTS", "6"))
MIN_MASK_PIXELS = int(os.getenv("PETSAFE_MIN_MASK_PIXELS", "300"))
DEFAULT_RUNS_DIR = Path("runs/plant_export_v20261026_2355546_holdout10_seed42")
RUNS_DIR = Path(os.getenv("PETSAFE_RUNS_DIR", str(DEFAULT_RUNS_DIR))).expanduser()
MODEL_API_BASE = "https://api.meta.ai/v1"
SPARK_MODEL = os.getenv("PETSAFE_SPARK_MODEL", "muse-spark-1.3")
SPARK_MAX_COMPLETION_TOKENS = int(
    os.getenv("PETSAFE_SPARK_MAX_COMPLETION_TOKENS", "1200")
)
SAM_MODEL = "sam-3.1"
_MODEL_LOCK = Lock()

ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "plant_image_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "plant_identification": {"type": "string"},
                "identification_confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "flower_count_estimate": {"type": "integer", "minimum": 0},
                "flower_count_confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "flower_count_notes": {"type": "string"},
                "suggest_segmentation": {"type": "boolean"},
                "segmentation_reason": {"type": "string"},
                "visual_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "plant_identification",
                "identification_confidence",
                "flower_count_estimate",
                "flower_count_confidence",
                "flower_count_notes",
                "suggest_segmentation",
                "segmentation_reason",
                "visual_evidence",
            ],
        },
    },
}

ASSESSMENT_PROMPT = (
    "Inspect the plant photograph and return the requested structured assessment. "
    "Estimate the count of distinct visible blossoms, not leaves or petals; use 0 "
    "when none are visible. If uncertain, provide a best integer estimate and mark "
    "count confidence low. Identify the plant as a visual hypothesis. Recommend "
    "segmentation when multiple flowers/plants or overlapping parts would benefit "
    "from separate inspection. Never infer pet safety or toxicity."
)


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
    return await asyncio.to_thread(get_trained_model(name).predict, image)


def _data_url(data: bytes, media_type: str = "image/jpeg") -> str:
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


async def assess_with_spark(image_bytes: bytes, extra_instruction: str = "") -> PlantAssessment:
    """Return Muse Spark's schema-validated plant and flower-count assessment."""
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY in the API server environment.")
    from openai import AsyncOpenAI

    supported_classes = ", ".join(
        item["class_name"] for item in load_toxicity_database()["plants"]
    )
    identity_instruction = (
        f" Known lookup plant labels: {supported_classes}. When one is visually supported, "
        "return its exact label as the plant identification. If none fit or the image is "
        "ambiguous, return 'Unknown'."
    )
    client = AsyncOpenAI(base_url=MODEL_API_BASE, api_key=api_key, timeout=120, max_retries=2)
    try:
        response = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": ASSESSMENT_PROMPT + identity_instruction + " " + extra_instruction},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                ],
            }],
            response_format=ASSESSMENT_SCHEMA,
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
    # Keep toxicity deterministic and source-backed; it is never requested from Spark.
    return assessment.model_copy(
        update={
            "toxicity": lookup_toxicity(assessment.plant_identification),
            "raw_meta_response": response.model_dump(mode="json"),
        }
    )


def _crop_from_mask(
    image: Image.Image, mask_record: Any
) -> tuple[tuple[int, int, int, int], np.ndarray, bytes] | None:
    """Map SAM's box-local raster to image pixels and encode a transparent crop."""
    from meta_sam_parser import decode_mask_to_raster

    local = np.frombuffer(decode_mask_to_raster(mask_record.mask), dtype=np.uint8)
    local = local.reshape(mask_record.mask.height, mask_record.mask.width)
    bounds = mask_record.bounds
    left = max(0, int(np.floor(bounds.left)))
    top = max(0, int(np.floor(bounds.top)))
    right = min(image.width, int(np.ceil(bounds.right)))
    bottom = min(image.height, int(np.ceil(bounds.bottom)))
    if right <= left or bottom <= top:
        return None
    local_mask = Image.fromarray(local * 255).resize(
        (right - left, bottom - top), resample=Image.Resampling.NEAREST
    )
    full_mask = np.zeros((image.height, image.width), dtype=bool)
    full_mask[top:bottom, left:right] = np.asarray(local_mask) > 0
    ys, xs = np.where(full_mask)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    crop_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.asarray(image.convert("RGB"))
    rgba[:, :, 3] = full_mask.astype(np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(rgba).crop(crop_box).save(buffer, format="PNG", optimize=True)
    return crop_box, full_mask, buffer.getvalue()


async def segment_and_assess(
    image: Image.Image,
    image_bytes: bytes,
    class_names: list[str],
    prompt: str = "flower",
) -> list[SegmentAssessment]:
    """Use Meta SAM 3.1, then ask Muse Spark to assess each usable crop."""
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
            metadata={"mask_encoding": "one_bit"},
        )
        parsed = parse_responses_stream(events, image_segmentation_format())
        result = await parsed.final_result()
    finally:
        await client.close()

    names = {name.casefold() for name in class_names}
    segments: list[SegmentAssessment] = []
    for record in result.records:
        if getattr(record, "kind", None) != "mask":
            continue
        prepared = _crop_from_mask(image, record)
        if prepared is None:
            continue
        crop_box, _mask, crop_bytes = prepared
        assessment = await assess_with_spark(
            crop_bytes,
            "This is a transparent-background crop of one SAM-selected flower region. "
            "Assess only the visible selected region and preserve uncertainty.",
        )
        segments.append(SegmentAssessment(
            object_id=record.object_id,
            bounds=crop_box,
            crop_data_url=_data_url(crop_bytes, "image/png"),
            assessment=assessment,
            in_trained_taxonomy=assessment.plant_identification.casefold() in names,
        ))
        if len(segments) >= MAX_SEGMENTS:
            break
    return segments
