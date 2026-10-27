# PetSafe Plants

**AI-powered plant identification and pet toxicity detection using Meta's multimodal models.**

Built for the **Meta Global AI Developer Hackathon 2026** (October 26 – November 4), Indie Developer track.

## About the Project

PetSafe Plants aims to help pet owners identify potentially toxic plants from photographs.

The idea was inspired by a personal experience: my cat became ill after drinking water from a vase containing lilies. I wanted to explore how computer vision and multimodal AI could help pet owners recognize similar risks.

Plant identification is not always straightforward. Visually similar species, unfamiliar camera angles, poor lighting, and mixed bouquets can produce incorrect predictions.

The project explores whether combining specialized computer vision models with Meta's multimodal models can improve identification and help recognize uncertain predictions.

**Current stage:** Dataset preparation, model experimentation, and evaluation.

PetSafe Plants does not provide veterinary advice or a validated toxicity assessment. The app's 100-entry cat-toxicity reference is a curated source lookup; it does not confirm the visual identification or guarantee safety.

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
| EfficientNet-B2 | CNN classification baseline |
| Meta DINOv3 | Visual feature extraction and classification |
| SigLIP2 | Vision-language representation baseline |
| Meta Muse Spark | Multimodal identification comparison |
| Meta SAM 3.1 | Planned segmentation and multi-plant analysis |

The objective is not simply to maximize classification accuracy but to understand how the models behave on difficult photographs.

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
| `03_Train_Plant_Fetcher_Export.ipynb` | Feature extraction and training classifier heads |
| `04_Compare_Training_Runs.ipynb` | Comparison of training runs and evaluation metrics |
| `05_Test_Inference.ipynb` | Evaluation on the 180-image internal holdout (notebook section number 06) |
| `06_Compare_Muse_Spark.ipynb` | Optional comparison against Meta Muse Spark (notebook section number 05) |
| `07_Upload_Spark_SAM_Flow.ipynb` | Interactive image upload, Muse Spark identification, optional SAM 3.1 segmentation, and Spark on segmented crops |

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

DINOv3 requires approved Hugging Face access. Muse Spark experiments require a Meta Model API key.

API calls may incur charges.

## Inference API

The API serves the frozen-encoder classifier heads exported by notebook 03 and compares them with the Muse Spark → optional SAM 3.1 → Muse Spark flow. It does not train models. Set `PETSAFE_RUNS_DIR` to the experiment directory containing `efficientnet_b2/`, `dinov3/`, and `siglip2/`; each selected model needs `best_head.pt`, `config.json`, and `class_mapping.json` from notebook 03. Model weights and the dataset remain outside this repository.

After each visual identification, the API performs a deterministic local lookup in `data/toxicity/plant_toxicity.json` and returns the status, taxonomic scope, and source links alongside the prediction. The 100 plant labels are a curated selection of commonly encountered house, garden, and bouquet plants, not a statistically ranked global list. Spark does not generate toxicity claims. The table is a limited reference, not a diagnosis or a guarantee of safety; unknown or unmatched identities remain unknown. Set `PETSAFE_TOXICITY_DB` to use a different copy. Docker includes this JSON file in the API image.

Install the optional server dependencies and configure credentials and paths in the server environment:

```bash
uv sync --extra api
export PETSAFE_RUNS_DIR="/path/to/plant_export_v20261026_2355546_holdout10_seed42"
export MODEL_API_KEY="your-meta-model-api-key"
# Optional; defaults to 1200 completion tokens for structured Spark responses.
export PETSAFE_SPARK_MAX_COMPLETION_TOKENS="1200"
# Optional, required when the DINOv3 model artifacts are selected:
export HF_TOKEN="your-hugging-face-token"
uv run uvicorn petsafe.api.app:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/` for the web client. It is a static HTML/CSS/JavaScript page served by FastAPI, so it needs no Node.js build step.

`PETSAFE_RUNS_DIR` should point to the run root, with model subdirectories directly beneath it. For example, the notebook writes `.../runs/plant_export_v20261026_2355546_holdout10_seed42/efficientnet_b2/best_head.pt`. Make the notebook's Google Drive run folder available to the API process by mounting that drive or copying the run artifacts to a local folder. The default model is EfficientNet-B2. Choose `dinov3` or `siglip2` with the `model_name` form field when those artifacts are available. DINOv3 also requires accepted Hugging Face model access and `HF_TOKEN`.

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /healthz` | API status and whether trained artifacts are discoverable |
| `GET /api/models` | Supported model names and those with artifacts installed |
| `POST /api/flows/trained` | Left-side trained model prediction and local toxicity-reference lookup; multipart fields `image_file` and optional `model_name` |
| `POST /api/flows/muse-spark` | Muse Spark JSON assessment and local toxicity-reference lookup on the uploaded image |
| `POST /api/flows/compare` | Run both sides together, conditionally call SAM, reassess each segment with Muse Spark, and look up each returned identity |

Open `http://127.0.0.1:8000/docs` for the interactive API schema. Example comparison request:

```bash
curl -X POST http://127.0.0.1:8000/api/flows/compare \
  -F "image_file=@plant.jpg" \
  -F "model_name=efficientnet_b2" \
  -F "confidence_threshold=0.55"
```

The comparison endpoint requests SAM when Muse Spark estimates multiple flowers, reports low identification or flower-count confidence, recommends segmentation, or the trained model probability falls below `confidence_threshold`. Use `force_sam=true` or `skip_sam=true` to override that choice, and set `sam_prompt=plant` to segment whole plants instead of flowers. The response includes the trained model's top five classes, Spark's structured assessment, trigger reasons, and any SAM crops with Spark re-assessments. A toxicity lookup accompanies each returned identity and includes the reference links; it describes the database entry for a possible model match, not a confirmed plant identification. Classifier probabilities are not calibrated confidence scores. Spark's flower count is an estimate and is not ground truth.

The API accepts images up to 12 MB, normalizes the longest side to 2,048 pixels, and sends the normalized image to the selected Meta services. Keep the server private while it can make paid Model API calls; this initial API has no user authentication or rate limiting.

### Run with Docker Compose

Docker builds the API image and mounts saved model artifacts read-only. The image excludes datasets, notebooks, local environments, and model weights. Create a local Compose environment file, then set the artifact directory and API key there:

```bash
cp .env.example .env
```

Set `MODEL_API_KEY` and `PETSAFE_RUNS_HOST_DIR` in `.env`. The latter must point to the run root containing the model folders, for example `/path/to/drive/AI/meta-global-hackaton/runs/plant_export_v20261026_2355546_holdout10_seed42`. Set `HF_TOKEN` too if you plan to select DINOv3. Then build and start the API:

```bash
docker compose up --build
```

The web client is at `http://127.0.0.1:8000/`, and the interactive API docs are at `http://127.0.0.1:8000/docs`. Compose persists Hugging Face and Torch model caches in a Docker volume. Stop the service with `docker compose down`. The model run directory is mounted read-only, and credentials are passed at runtime rather than baked into the image.

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
- Pretrained weights for EfficientNet-B2, DINOv3, and SigLIP2.
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
