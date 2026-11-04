# PetSafe Plants

**Beyond plant identification: AI-powered investigation with source-backed pet toxicity information.**

Built for the **Meta Global AI Developer Hackathon 2026** · Indie Developer · *Build for Your World*.

PetSafe Plants was inspired by my cat Misho, who became ill after drinking water from a vase containing lilies. A photograph can contain several overlapping plants; identifying the most obvious flower does not mean every plant has been identified.

PetSafe Plants analyzes the full photograph, examines difficult regions when useful, and matches candidate plant identities to a curated veterinary toxicity reference database. **The AI does not generate toxicity classifications.**

## How it works

1. **Meta Muse Spark** assesses the full image, proposes plant identities, and reports uncertainty.
2. **Meta SAM 3.1** optionally segments overlapping or uncertain plant regions.
3. **Muse Spark** reassesses selected crops; the application combines findings while preserving unresolved identities.
4. **Local taxonomic and toxicity lookup** links candidate taxa to source-backed information for cats and dogs.

Simple scenes can skip segmentation to reduce unnecessary API calls. The web interface visualizes real investigation stages and extracted SAM specimens.

> **Important:** Visual identification can be incorrect or incomplete. A `non_toxic` reference classification does not guarantee safety. PetSafe Plants is not veterinary advice. If exposure is suspected, contact a veterinarian promptly.

## Run locally

**Requirements:** Git, Docker Engine with Docker Compose, and a Meta Model API key with access to the required models. Model calls may incur charges.

```bash
git clone https://github.com/vmirchev/petsafe-plants.git
cd petsafe-plants
cp .env.example .env
mkdir -p local-models
```

Edit `.env` and set:

```dotenv
MODEL_API_KEY=your-meta-model-api-key
PETSAFE_RUNS_HOST_DIR=./local-models
PETSAFE_APP_ENV=development
```

Start the application:

```bash
docker compose up --build -d api
```

Open **http://127.0.0.1:8000/** and upload a photo or select a built-in sample. The first Docker build may be large because it installs ML dependencies. The `local-models/` folder can remain empty: **the main Muse Spark + SAM investigation does not require trained model weights**.

```bash
docker compose logs -f api   # View logs
docker compose down          # Stop the app (preserves named volumes)
```

### Optional: Admin controls

To enable `/admin`, set `PETSAFE_ADMIN_PASSWORD` and `PETSAFE_SESSION_SECRET` in `.env`. Generate a session secret with `openssl rand -hex 32`, then recreate the container:

```bash
docker compose up -d --force-recreate api
```

The admin page shows API usage and can disable new Meta calls or change the combined rolling 24-hour call limit. Limits and counters persist in SQLite. Call quotas are **not** exact dollar-spending caps.

### Optional: DINOv3 comparison

The **`/compare.html`** page also uses a trained Meta DINOv3 classifier. Its model artifacts are **not included** in this repository. To use the comparison, set `PETSAFE_RUNS_HOST_DIR` to an experiment directory containing `dinov3/best_head.pt`, `dinov3/config.json`, and `dinov3/class_mapping.json`. You may also need `HF_TOKEN` and accepted model-access terms. None of this is required for the main `/` page.

**Without Docker:** With Python 3.11+ and [`uv`](https://docs.astral.sh/uv/), run `uv sync --extra api`, set `MODEL_API_KEY`, and start:

```bash
uv run uvicorn petsafe.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

## Models and engineering

| Component | Role |
|---|---|
| **Meta Muse Spark** | Multimodal scene assessment and regional identification |
| **Meta SAM 3.1** | Conditional segmentation of plant regions |
| **Meta DINOv3 ViT-S/16** | Optional 18-class image-classification baseline |
| **Local reference database** | Deterministic taxonomic matching and cat/dog toxicity lookups |

The application implements streamed investigation events, SAM-mask selection and duplicate filtering, result caching, server-side API usage quotas, and an admin kill switch. Meta API credentials stay on the backend.

## Experiments and limitations

The `notebooks/` directory documents dataset exploration, training, evaluation, model comparisons, and Muse Spark/SAM experiments. The research dataset covers **18 plant classes**: 1,295 source image records, 1,292 after exact deduplication, divided into 890 training, 111 validation, 111 test, and 180 **internal holdout** images. The source archive and trained weights are not distributed here.

Earlier EfficientNet-B2 and SigLIP2 experiments are preserved for comparison but are not used in live inference. Internal classification results do **not** establish the reliability of the full pipeline in real-world pet-safety situations.

## Source code and toxicity references

- `src/petsafe/` — Python application and AI pipeline
- `web/` — Investigation interface and sample images
- `tests/` — Automated tests
- `notebooks/` — Experiments and evaluations
- `data/toxicity/` — Curated reference lookup

The toxicity database contains **101 plant entries** with separate cat and dog information, source links, and taxonomic qualifications, drawing primarily on [ASPCA](https://www.aspca.org/pet-care/animal-poison-control/cats-plant-list) and other veterinary references. These sources do not endorse PetSafe Plants. Unmatched or insufficiently supported identifications remain unresolved rather than being presented as safe.

## License

Original application code is released under the [MIT License](LICENSE). Third-party models, photographs, and source-derived reference materials remain subject to their respective rights and terms; source citation does not itself grant redistribution rights.

---

*PetSafe Plants is an independent research and demonstration project, not affiliated with or endorsed by Meta or ASPCA.*
