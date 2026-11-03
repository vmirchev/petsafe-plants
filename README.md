# 🌱 PetSafe Plants

**Beyond plant identification: AI-powered investigation with source-backed pet toxicity information.**

Built for the **Meta Global AI Developer Hackathon 2026** · Indie Developer · *Build for Your World*.

PetSafe Plants was inspired by my cat Misho, who became ill after drinking water from a vase containing lilies. The project explores a practical problem: a photograph can contain several overlapping plants, and recognizing the most obvious flower does not mean every plant has been identified.

**PetSafe Plants investigates the entire scene, examines difficult regions when useful, and checks candidate identities against a curated veterinary toxicity reference database.** It does not use an AI model to invent toxicity classifications.

## How it works

1. **Muse Spark** assesses the uploaded photograph, proposes plant identities, and flags uncertainty.
2. **SAM 3.1**, when needed, isolates relevant regions in mixed or difficult scenes.
3. **Muse Spark** reassesses selected crops; the application combines findings and keeps unresolved identities explicit.
4. **Local taxonomic and toxicity lookup** connects candidate plants to source-backed information for cats and dogs.

The animated investigation interface shows real processing stages and extracted SAM specimens. Simple scenes can skip segmentation to avoid unnecessary API calls.

> **Important:** Plant identification may be wrong or incomplete. A `non_toxic` reference classification is not a guarantee of safety. PetSafe Plants does not replace veterinary advice. If exposure is suspected, contact a veterinarian promptly.

## Run locally (Muse Spark + SAM demo)

**Requirements:** Git, Docker Engine with Docker Compose, and a **Meta Model API key** with access to the required models. Model requests can incur charges.

```bash
git clone https://github.com/vmirchev/petsafe-plants.git
cd petsafe-plants
cp .env.example .env
mkdir -p local-models
```

Edit `.env` and set at least:

```dotenv
MODEL_API_KEY=your-meta-model-api-key
PETSAFE_RUNS_HOST_DIR=./local-models
PETSAFE_APP_ENV=development
```

`local-models/` can be empty for the **Muse Spark investigation** at `/`; Docker Compose requires this directory to exist because it also supports the optional trained-model comparison. No trained weights are needed for the main investigation.

Start the app:

```bash
docker compose up --build -d api
```

Open **http://127.0.0.1:8000/**, upload a photograph, or try a sample image. The first build can be large because the image includes ML dependencies.

Useful commands:

```bash
docker compose logs -f api      # Follow logs
docker compose down             # Stop (preserves named volumes)
```

**Optional: admin and usage limits.** Add `PETSAFE_ADMIN_PASSWORD` and `PETSAFE_SESSION_SECRET` to `.env` (generate the secret with `openssl rand -hex 32`), restart with `docker compose up -d --force-recreate api`, then open `/admin`. The admin panel can disable new Meta calls and set a combined rolling 24-hour API-call limit. Settings and counters persist in SQLite. The default call limit is 100, and there is a separate lifetime limit of 1,000 attempted Meta calls. Neither is an exact dollar-spending cap.

**Optional: trained-model comparison.** `/compare.html` additionally uses a DINOv3 classifier. Its training artifacts are **not** distributed in this repository. To use it, set `PETSAFE_RUNS_HOST_DIR` to a directory containing the saved DINOv3 run under `dinov3/` (`best_head.pt`, `config.json`, `class_mapping.json`). You may also need `HF_TOKEN` and to accept the model's access terms. The main `/` demo does **not** need these artifacts.

**Without Docker:** With Python 3.11+ and [uv](https://docs.astral.sh/uv/), run `uv sync --extra api`, export `MODEL_API_KEY`, then start `uv run uvicorn petsafe.api.app:app --app-dir src --host 127.0.0.1 --port 8000`. Optional trained-model paths are configured with `PETSAFE_RUNS_DIR`.

## Models and engineering

| Component | Role |
|---|---|
| **Meta Muse Spark** | Multimodal scene assessment and regional identification |
| **Meta SAM 3.1** | Conditional plant/flower segmentation |
| **Meta DINOv3 ViT-S/16** | Optional 18-class single-image comparison baseline |
| **Local reference database** | Deterministic taxonomic matching and cat/dog toxicity information |

The API includes streaming investigation events, mask selection and duplicate filtering, local result caching, server-side Meta call quotas, and an admin kill switch. Meta API credentials remain server-side. The UI includes an optional developer inspector to examine model outputs and usage.

## Experiments and limitations

The repository also contains notebooks for dataset exploration, training, evaluation, DINOv3 comparison, and Muse Spark/SAM experiments. The research dataset covers **18 plant classes**: 1,295 source records, 1,292 after exact deduplication, split into 890 train, 111 validation, 111 test, and a 180-image **internal** holdout. The image dataset and trained weights are excluded from the repository because of size and third-party rights.

Earlier EfficientNet-B2 and SigLIP2 experiments are retained for comparison; they are **not** used in the live application. Classification accuracy on this dataset does **not** establish real-world pet-safety reliability, and the complete multi-stage pipeline has not been validated as a safety-critical system.

## Source code and data

The app's source is in `src/petsafe/`, the web interface in `web/`, tests in `tests/`, and experiments in `notebooks/`. The curated toxicity lookup includes **101 plant entries**, with cat and dog classifications, taxonomic qualifications, and source links, primarily from [ASPCA](https://www.aspca.org/pet-care/animal-poison-control/cats-plant-list) and other veterinary references. These references are evidence for the reported classifications, not endorsements of the application.

**Third-party rights:** The original photographs, pretrained models, and external reference material retain their respective rights and licenses. Citation of a source does not itself grant permission to reproduce its content.

## License

The original application code is released under the **[MIT License](LICENSE)**. Third-party models, images, and source-derived reference content are not relicensed by MIT.

---

*PetSafe Plants is an independent research and demonstration project. It is not affiliated with or endorsed by ASPCA or Meta.*
