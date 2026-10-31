# PetSafe Plants

**AI-powered plant identification and pet toxicity detection using Meta's multimodal models.**

Built for the **Meta Global AI Developer Hackathon 2026** (October 26 – November 4), Indie Developer track.

## About the Project

PetSafe Plants aims to help pet owners identify potentially toxic plants from photographs.

The idea was inspired by a personal experience: my cat became ill after drinking water from a vase containing lilies. I wanted to explore how computer vision and multimodal AI could help pet owners recognize similar risks.

Plant identification is not always straightforward. Visually similar species, unfamiliar camera angles, poor lighting, and mixed bouquets can produce incorrect predictions.

The project explores whether combining specialized computer vision models with Meta's multimodal models can improve identification and help recognize uncertain predictions.

**Current stage:** Dataset preparation, model experimentation, and evaluation.

PetSafe Plants does not provide veterinary advice or a validated toxicity assessment. The app's 101-entry cat and dog toxicity reference is a curated source lookup; it does not confirm the visual identification or guarantee safety.

## Research Objectives

The project investigates several questions:

- How well do pretrained computer vision models generalize to unfamiliar plant photographs?
- How does Meta Muse Spark compare with conventional image classifiers?
- Can SAM 3.1 segmentation improve identification when multiple plants appear in an image?
- Can model disagreement and additional visual evidence help detect uncertain identifications?
- How can plant identification be connected to verified pet-toxicity information without presenting uncertain predictions as facts?

## Models

The initial experiments use frozen pretrained image encoders with trained classification heads.

| Model | Role |
|---|---|
| Meta DINOv3 ViT-S/16 | Active full-photo classifier with a trained 18-class head |
| Meta Muse Spark | Multimodal identification comparison |
| Meta SAM 3.1 | Conditional segmentation and multi-plant analysis |

The objective is not simply to maximize classification accuracy but to understand how the models behave on difficult photographs.

Notebook 04 preserves an earlier comparison with EfficientNet-B2 and Google SigLIP2. SigLIP2 tied DINOv3 at 96.4% validation accuracy and had lower validation loss (0.144 vs. 0.213). On the 111-image internal test split, SigLIP2 scored 98.2% accuracy and 98.4% macro F1; DINOv3 scored 97.3% and 97.5%; EfficientNet-B2 scored 81.1% and 78.7%. These are single-seed results, and possible cross-split near-duplicates could inflate the scores. **The active app now uses only Meta models:** DINOv3 ViT-S/16, Muse Spark, and SAM 3.1. Historical comparison results are retained as experiment records, not used for live inference.

## Dataset

The current dataset contains **18 plant classes**, with images collected from Wikimedia Commons and iNaturalist. The images were selected from sources permitting reuse under their respective licenses. Links to the original images and their sources can be provided upon request.

The source archive, `v20260925_155046.zip`, contains 1,295 image records. After removing three exact duplicates, 1,292 unique images remain.

### Dataset Splits

A reproducible split is generated using random seed `42`.

| Split | Images | Purpose |
|---|---:|---|
| Training | 890 | Train classification heads |
| Validation | 111 | Model selection |
| Test | 111 | Internal evaluation |
| Holdout | 180 | Additional evaluation |
| **Total** | **1,292** | |

The holdout contains 10 randomly selected images from each of the 18 classes.

**Important:** All images originate from the same source archive. The holdout is therefore an internal holdout, not an independently collected external dataset.

The original train/validation/test assignments are retained as provenance but are not used in the new experimental split.

Exact duplicates are removed using SHA-256 hashes. Potential near-duplicates and uncertain labels still require inspection.

The holdout must remain untouched during model selection and hyperparameter tuning.

## Notebooks

The repository contains Jupyter notebooks for reproducible ML experiments and a small FastAPI service for interactive inference comparisons.

| Notebook | Purpose |
|---|---|
| `01_Plant_export_exploration.ipynb` | Dataset exploration, image properties, duplicates, and split analysis |
| `02_Another_Dataset.ipynb` | Was removed before initial commit as some photos from dataset, could not be used. |
| `03_Train_Plant_Fetcher_Export.ipynb` | Feature extraction and training the Meta DINOv3 classifier head |
| `04_Compare_Training_Runs.ipynb` | Historical comparison of three saved training runs and evaluation metrics |
| `05_Test_Inference.ipynb` | DINOv3 evaluation on the 180-image internal holdout (notebook section number 06) |
| `06_Compare_Muse_Spark.ipynb` | DINOv3 versus Meta Muse Spark (notebook section number 05) |
| `07_Upload_Spark_SAM_Flow.ipynb` | Interactive image upload, Muse Spark identification, optional SAM 3.1 segmentation, and Spark on segmented crops |
| `08_DINOv3_Error_Review_Meta.ipynb` | Reviews every saved DINOv3 internal-test mistake plus five seeded correct examples with Muse Spark; caches responses in Drive |

Notebook 06 evaluates an **internal holdout**, not an independent external dataset.

The notebooks are designed primarily for Google Colab.

Some experiments require GPU access, approved model access, or API credentials.

### Running the Notebooks

The original dataset archive is stored locally and is not included in this repository.

The preparation workflow produces:

`plant_export_v20260925_155046_holdout10_seed42.zip`

The archive contains the four dataset splits, class mappings, metadata, and provenance manifests.

For Colab, the expected Google Drive locations are:

**Original dataset:**

`MyDrive/ML/Flower Classifier/data/v20260925_155046.zip`

**Prepared dataset and experiment outputs:**

`MyDrive/PetSafe Plants/`

Run notebook 01 for dataset exploration, followed by notebook 03 for model training.

Notebook 04 compares saved training results. The Muse Spark comparison is in `06_Compare_Muse_Spark.ipynb`; internal holdout inference is in `05_Test_Inference.ipynb`.

Muse Spark experiments require a Meta Model API key.

API calls may incur charges.

## Inference API

The API compares two independent Meta model paths. **Left:** DINOv3 ViT-S/16 extracts full-image features and the saved 18-class head predicts one plant identity. **Right:** Muse Spark assesses the scene → SAM 3.1 optionally isolates useful regions → Muse Spark reassesses them. SAM is used only on the right. Both paths attach local source-backed toxicity lookups to their candidate identities. The DINOv3 probability is not calibrated safety confidence, and its full-image prediction cannot identify every plant in a mixed scene. Set `PETSAFE_RUNS_DIR` to the experiment directory containing `dinov3/`; the model needs `best_head.pt`, `config.json`, and `class_mapping.json` from notebook 03. Saved classifier weights and the dataset remain outside this repository. DINOv3 access through Hugging Face requires accepted model terms and `HF_TOKEN` when the encoder is not already cached.

After each visual identification, the API performs a deterministic local lookup in `data/toxicity/plant_toxicity.json` and returns separate cat and dog statuses, taxonomic scope, possible clinical signs, and source links alongside the prediction. The top-level `toxicity_status` and `clinical_effects` fields remain the cat result for existing clients; `dog_toxicity` contains the separately reviewed dog result. The 101 plant labels are a curated selection of commonly encountered house, garden, and bouquet plants, not a statistically ranked global list. Spark does not generate toxicity claims. The table is a limited reference, not a diagnosis or a guarantee of safety; unknown or unmatched identities remain unknown. `Hippeastrum` and `Amaryllis` have separate cited entries because the common name “amaryllis” refers to both; that common name alone gives an ambiguous lookup. Set `PETSAFE_TOXICITY_DB` to use a different copy. Docker includes this JSON file in the API image.

All 101 plant entries have separate cat and dog reviews with source links, the source taxon, possible clinical signs where reported, and exposure or taxonomic qualifiers. The records use the separate [ASPCA cat](https://www.aspca.org/pet-care/animal-poison-control/cats-plant-list) and [dog](https://www.aspca.org/pet-care/animal-poison-control/dogs-plant-list) lists, [ASPCA plant profiles](https://www.aspca.org/pet-care/aspca-poison-control/toxic-and-non-toxic-plants/), the [Merck Veterinary Manual](https://www.merckvetmanual.com/toxicology/poisonous-plants/houseplants-and-ornamentals-toxic-to-animals), and [Pet Poison Helpline profiles](https://www.petpoisonhelpline.com/poison/rubber-tree-plant/) for croton and rubber tree. `Lilium` and `Hemerocallis` are listed as toxic to cats and non-toxic to dogs; the cat kidney-failure warning is never copied to the dog result. A `not_listed` clinical-effects record means the linked source gives no plant-specific signs, not that ingestion is symptom-free. The classification and clinical-effects reporting status are separate: ASPCA lists money tree as non-toxic yet mentions possible gastrointestinal signs. No uniform severity score is assigned because the sources do not provide a consistent plant-by-plant scale, and effects depend on the animal, plant part, and exposure. Where a source names only one species within a broader class, the class status remains `unknown` and the narrower scope is shown.

The suggested [Kaggle ASPCA Plant Toxicity dataset](https://www.kaggle.com/datasets/varoooooon/aspca-plant-toxicity) was considered as a discovery lead. Its file details and reuse license could not be independently verified here, so no Kaggle rows were copied into this repository. Each included record cites the underlying veterinary source directly.

The API keeps up to 256 inference result payloads in a process-local LRU cache, keyed by SHA-256 image hashes and model/request type. It caches trained-model predictions and Muse Spark assessments, including segmented-crop Spark responses. It does not store uploaded image bytes and the cache clears when the API process restarts. Set `PETSAFE_RESULT_CACHE_SIZE=0` to disable caching or change the limit; SAM segmentation itself is rerun when requested. Meta's separate prompt cache is automatic; the API sends a stable key for its shared developer prompt. A Meta cached-token count of zero means no provider cache reuse was reported for that request, while “not reported” means the provider omitted the cached-token detail.

Install the optional server dependencies and configure credentials and paths in the server environment:

```bash
uv sync --extra api
export PETSAFE_RUNS_DIR="/path/to/plant_export_v20261026_2355546_holdout10_seed42"
export MODEL_API_KEY="your-meta-model-api-key"
# Optional; defaults to 3000 completion tokens for multi-plant structured responses.
export PETSAFE_SPARK_MAX_COMPLETION_TOKENS="3000"
export HF_TOKEN="your-hugging-face-token"
uv run uvicorn petsafe.api.app:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/` for the Muse Spark investigation. It runs Muse Spark on the uploaded image, conditionally calls SAM 3.1, then reassesses selected regions with Muse Spark. This page does not load or require trained model artifacts. Open `http://127.0.0.1:8000/compare.html` for the two-flow comparison with DINOv3. Both pages share the static client code and are served by FastAPI without a Node.js build step.

### Botanical investigation workspace

The main page offers three synthetic sample photos below the upload canvas. Selecting one starts the normal investigation. The served copies live in `web/demo/`; they are AI-generated examples and do not establish plant identity or pet safety.

The browser uses [Motion](https://motion.dev/) (MIT licensed) for spring-style thought bubbles, native Web Animations for three moving search circles, and a CSS sweep over the photograph while a scan is active. Motion loads from a pinned jsDelivr URL, with a native animation fallback; no Node.js build step, hosted fonts, or paid visual assets are required. The photo moves into the center of the investigation canvas when a scan starts. Actual SAM specimens float over the photo when masks arrive; the comparison report appears below the canvas as model results arrive. Scanning questions are investigation prompts, never inferred findings or estimated progress percentages. The sweep and search circles are decorative and do not claim to mark identified plants.

The main page reads real events from `/api/flows/muse-stream`; the comparison page uses `/api/flows/compare-stream`. Both can emit `analyzing_scene`, `scene_analyzed`, optional `segmenting`, one `specimen_ready` event per selected SAM crop, `segments_ready`, `identifying_regions`, individual `region_assessed` events, then `completed` or `error`. Only the comparison emits `trained_ready`. SAM's image response must finish before masks can be selected; after that, each encoded crop is sent to the browser immediately instead of waiting for all selected crops. Its Muse Spark → SAM branch runs independently of the DINOv3 classifier, so real SAM crops can appear while DINOv3 is still working; `completed` waits for both branches. After the complete payload arrives, the browser spends 1.8 seconds presenting the scan and specimen transition before rendering the final report; source-backed hazard warnings appear immediately. This presentation pause is skipped for reduced motion and never delays API processing. The ordinary JSON comparison endpoint remains available. Scene and regional events already include their synchronous local toxicity lookups, so the UI does not simulate a separate lookup delay.

Selected transparent SAM crops animate from their real image bounds into movable specimen cards over the photo. Drag a card or use arrow keys to rearrange it within the canvas. Hover, focus, or drag a specimen to highlight its original mask and connecting line; select it for identification details, uncertainty, and source links. Only selected masks are sent to the browser; the existing filter counts describe omitted masks. Raw Meta responses and token/cache statistics remain available in the reports. Warnings appear when evidence arrives, independently of animations.

Use **Expand canvas** to give the photo most of the viewport height; click **Restore canvas** or press Escape to return. After a report completes, **Investigate another plant** opens the file picker without first clearing the current image. Choosing a file archives the previous complete or partial report in this tab and starts the new investigation automatically; cancelling the picker leaves the current report visible. **Rerun this photo** is inside the Developer Inspector and replaces the current report with a fresh run using the selected options. Each image is assessed independently; the app does not send earlier reports as conversational context. History disappears on reload or tab closure. Replacing/removing a photo cancels its stream and ignores stale responses; already running local inference threads or provider requests may still finish. Reduced-motion preferences disable decorative movement by default. The **Animations** control shows when the system preference has disabled motion and lets you explicitly enable it for this page; the choice is saved in the browser. Keyboard users can open and close the specimen inspection dialog.

Run the dependency-free state/stream tests with Node.js 20 or later:

```bash
node --test tests/web-investigation.test.mjs
```

For a browser smoke check with the API running: upload a single plant and select **Skip segmentation**, then a mixed scene with **Always segment**. Confirm that the photo centers during scanning, results appear below it, real crops float over the photo, cards can be dragged or moved with arrow keys, and source lines follow the right locations. Open a crop in the inspection dialog. Also check **Automatic**, a no-plant image, unavailable/empty SAM results, a follow-up upload while inference is pending, a narrow mobile viewport, and the operating system's reduced-motion setting. No API results are simulated in the production client.

`PETSAFE_RUNS_DIR` should point to the run root, with model subdirectories directly beneath it. For example, the notebook writes `.../runs/plant_export_v20261026_2355546_holdout10_seed42/dinov3/best_head.pt`. Make the notebook's Google Drive run folder available to the API process by mounting that drive or copying the run artifacts to a local folder. DINOv3 ViT-S/16 is the only active trained classifier.

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /healthz` | API status and whether trained artifacts are discoverable |
| `GET /api/models` | Supported model names and those with artifacts installed |
| `POST /api/flows/trained` | DINOv3 full-image prediction and local toxicity-reference lookup; multipart fields `image_file` and optional `model_name` |
| `POST /api/flows/muse-spark` | Muse Spark JSON assessment and local toxicity-reference lookup on the uploaded image |
| `POST /api/flows/muse-stream` | Muse Spark scene assessment, conditional SAM, regional reassessment, and SSE progress; no classifier artifacts required |
| `POST /api/flows/compare` | Run both sides together, conditionally call SAM, reassess each segment with Muse Spark, and look up each returned identity |
| `POST /api/flows/compare-stream` | The same comparison with real SSE progress events; accepts the same multipart fields and ends with the complete comparison payload |

Open `http://127.0.0.1:8000/docs` for the interactive API schema. Example comparison request:

```bash
curl -X POST http://127.0.0.1:8000/api/flows/compare \
  -F "image_file=@plant.jpg" \
  -F "model_name=dinov3"
```

The comparison endpoint requests SAM when Muse Spark suggests segmentation, reports multiple distinct plant types, marks a bouquet mixed or uncertain, or marks a low-confidence group as needing inspection. Multiple flowers of one plant type alone do not trigger SAM. Spark also returns a structured `segmentation_target` and `segmentation_target_reason`: `flower` for flower heads or `plant` for whole plants. With the default `sam_prompt=auto`, that target is sent to SAM when segmentation runs; `sam_prompt=flower` or `sam_prompt=plant` overrides it manually. The corresponding SAM concept prompts are `flower head` and `plant`. Use `force_sam=true` or `skip_sam=true` to override the segmentation decision. After SAM, the API drops tiny or unreadable masks, removes near-duplicates using mask overlap, and selects up to six diverse regions before re-assessing them with Spark. The response reports the actual SAM prompt, mask counts before and after filtering, and the chosen target. If no useful region remains, the full-scene assessment is retained and an uncertain scene can request another photo. Spark returns plant groups, bouquet status, additional-photo guidance, and Meta prompt-cache token usage (`cached_input_tokens` and `cache_hit_ratio`). A toxicity lookup accompanies each plant group and includes reference links. Classifier probabilities are not calibrated confidence scores. The local result cache is separate from Meta's prompt cache; on a local cache hit, displayed token usage describes the original Meta request.

When Muse Spark reports more than 8 clearly separate whole plants, both Muse flows skip SAM and request closer photos of individual plants, even if Muse Spark did not request another photo. Set `PETSAFE_MAX_SEPARATE_PLANTS` in `.env` to change the limit; for example, `8` requests a new photo for 9 or more separate plants. This count includes separate pots of the same species, but excludes individual flowers and cut-flower bouquets. It is a model estimate and can be wrong. The limit also applies when SAM is manually forced. Recreate the API container after changing the setting: `docker compose up -d --force-recreate --no-deps api`.

The API accepts images up to 12 MB, normalizes the longest side to 2,048 pixels, and sends the normalized image to the selected Meta services. Keep the server private while it can make paid Model API calls; this initial API has no user authentication or rate limiting.

### Run with Docker Compose

Docker builds the API image, mounts saved model artifacts read-only, and bind-mounts `web/` read-only for fast UI editing. The image excludes datasets, notebooks, local environments, and model weights. Create a local Compose environment file, then set the artifact directory and API key there:

```bash
cp .env.example .env
```

Set `MODEL_API_KEY` and `PETSAFE_RUNS_HOST_DIR` in `.env`. The latter must point to the run root containing the model folders, for example `/path/to/drive/AI/meta-global-hackaton/runs/plant_export_v20261026_2355546_holdout10_seed42`. Build the API once and start it:

Set `PETSAFE_APP_ENV=development` to keep the Developer Inspector, model details, token usage, raw Meta response, API status control, and comparison page. Set `PETSAFE_APP_ENV=production` for the public-facing Muse experience: the comparison link and technical report are hidden, `/compare.html` redirects to the main page, API docs are disabled, and raw Meta response content, usage, and cache diagnostics are removed from responses. The user-facing plant, toxicity, source, follow-up, and safety information remains available. This presentation mode is **not access control**: inference endpoints remain callable, so add authentication and rate limiting before exposing the API publicly.

```bash
docker compose up --build -d api
```

If the API image already exists, apply the `web/` and `src/` bind mounts once with `docker compose up -d --force-recreate --no-deps api`. After that, edit files in `web/` and refresh the browser for UI-only changes. Restart with `docker compose restart api` after Python source changes so the process loads the new code. Recreate the container with `docker compose up -d --force-recreate --no-deps api` after changing `PETSAFE_APP_ENV` in `.env`; a restart alone does not reload Compose environment variables. Rebuild with `docker compose up --build -d api` when dependencies or the Dockerfile change. A 405 from `/api/flows/muse-stream` means an older API process is still serving requests; recreate or restart the container after applying the source mount.

The Muse web client is at `http://127.0.0.1:8000/`. In development, the comparison client is at `http://127.0.0.1:8000/compare.html` and the interactive API docs are at `http://127.0.0.1:8000/docs`. Compose persists Hugging Face and Torch model caches in a Docker volume. Stop the service with `docker compose down`. The model run directory is mounted read-only for the comparison flow, and credentials are passed at runtime rather than baked into the image.

The container currently targets CPU inference. The first build is large because it includes PyTorch, torchvision, and Transformers; first model use also downloads pretrained encoder weights into the cache volume.

## Evaluation

The current experiments focus on:

- Classification accuracy and Macro F1.
- Per-class precision and recall.
- Confusion matrices.
- Generalization across held-out images.
- Model disagreements and failure cases.

Future experiments will investigate multi-plant identification, segmentation, and toxicity-related classification errors.

Accuracy on a conventional classification benchmark is not sufficient to establish that a model is reliable for pet-safety applications.

## Planned Development

The intended application will combine:

1. **Muse Spark:** Candidate plant identification and visual reasoning.
2. **SAM 3.1:** Plant localization and segmentation, including mixed bouquets.
3. **Uncertainty handling:** Identifying ambiguous predictions and requesting additional visual evidence.
4. **Verified toxicity references:** Matching plant candidates to reliable veterinary information.
5. **Web application:** A user interface for uploading and investigating plant photographs.

These features are planned and should not be interpreted as completed functionality.

## Hackathon Contributions

This repository distinguishes work developed during the hackathon from existing datasets, models, and infrastructure.

### Developed During the Hackathon

- Reproducible dataset splitting and provenance tracking.
- Archive validation and exact-duplicate detection.
- Dataset exploration notebooks.
- Model training and comparison workflows.
- Evaluation notebooks and supporting documentation.

### Pre-existing or External Resources

- Original plant photographs and source archive.
- Pretrained Meta DINOv3 ViT-S/16 weights used by the active classifier. EfficientNet-B2 and SigLIP2 appear only in historical experiment records.
- PyTorch, torchvision, Hugging Face libraries, and other dependencies.
- Meta's models and inference services.
- Earlier classifier experiments used as background research.

The public project contains notebooks, documentation, and the inference API source. Supporting local preparation utilities, trained weights, and datasets are not published here.

## Dataset Rights

The source images originate from third-party materials collected from Wikimedia Commons and iNaturalist.

Source URLs and provenance metadata are retained where available, but a source URL does not establish permission to redistribute an image.

Raw photographs, dataset archives, model weights, and API credentials are excluded from the public repository.

## Disclaimer

PetSafe Plants is an experimental computer vision project.

Plant identifications may be incorrect, and the project is not a substitute for professional botanical identification or veterinary advice.

If an animal may have ingested a toxic plant, contact a veterinarian promptly rather than relying on an AI prediction.

---

**Meta Global AI Developer Hackathon 2026**  
Track: Indie Developer  
Theme: Build for Your World
