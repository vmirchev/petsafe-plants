"""Deterministic lookups against the project's reviewed toxicity reference."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from petsafe.api.schemas import ToxicityEvidence, ToxicityResult

DEFAULT_TOXICITY_DB = (
    Path(__file__).resolve().parents[3] / "data" / "toxicity" / "plant_toxicity.json"
)


def _normalize(value: str) -> str:
    """Normalize names for exact, case-insensitive alias matching."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


@lru_cache(maxsize=1)
def load_toxicity_database() -> dict[str, Any]:
    """Load and minimally validate the local reference JSON once per process."""
    path = Path(os.getenv("PETSAFE_TOXICITY_DB", str(DEFAULT_TOXICITY_DB))).expanduser()
    try:
        database = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Toxicity reference not found at {path}. Set PETSAFE_TOXICITY_DB to its location."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Toxicity reference at {path} is not valid JSON: {exc}") from exc

    plants = database.get("plants")
    if not isinstance(plants, list) or not plants:
        raise RuntimeError(f"Toxicity reference at {path} has no plants list.")
    return database


def lookup_toxicity(plant_name: str | None) -> ToxicityResult:
    """Match a visual/model identity to a reviewed entry without inventing a status.

    A unique exact alias is preferred. If the output contains a longer phrase,
    a unique whole-token alias may match; ambiguous or unknown text abstains.
    """
    database = load_toxicity_database()
    if not plant_name or _normalize(plant_name) in {"unknown", "unidentified", "none"}:
        return ToxicityResult(
            toxicity_status="unknown",
            potential_hazard=None,
            lookup_note="No specific plant identity was returned; toxicity was not looked up.",
        )

    candidate = _normalize(plant_name)
    exact: list[dict[str, Any]] = []
    contained: list[dict[str, Any]] = []
    for plant in database["plants"]:
        names = {
            _normalize(str(name))
            for name in [plant.get("class_name"), plant.get("scientific_name"), *plant.get("aliases", [])]
            if name
        }
        if candidate in names:
            exact.append(plant)
        elif any(f" {name} " in f" {candidate} " for name in names if name):
            contained.append(plant)

    matches = exact if exact else contained
    # Multiple entries may share a common alias. Only select an unambiguous class.
    unique = {plant["class_name"]: plant for plant in matches}
    if len(unique) != 1:
        return ToxicityResult(
            toxicity_status="unknown",
            potential_hazard=None,
            lookup_note=(
                "The identity did not uniquely match a reviewed class; no toxicity status was inferred."
            ),
        )

    plant = next(iter(unique.values()))
    sources = database.get("sources", {})
    evidence = [
        ToxicityEvidence(
            source_id=item["source_id"],
            source_match=item["source_match"],
            evidence_taxon=item["evidence_taxon"],
            evidence_status=item["evidence_status"],
            match_scope=item["match_scope"],
            source_url=item.get("source_url") or sources.get(item["source_id"], {}).get("url", ""),
        )
        for item in plant.get("evidence", [])
    ]
    return ToxicityResult(
        toxicity_status=plant["toxicity_status"],
        potential_hazard=plant.get("potential_hazard"),
        matched_class=plant["class_name"],
        scientific_name=plant.get("scientific_name"),
        taxonomic_scope=plant.get("taxonomic_scope"),
        evidence=evidence,
        lookup_note=plant.get("review_note") or plant.get("source_qualification"),
    )
