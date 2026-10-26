# PetSafe Plants

**AI-powered plant identification and pet toxicity detection using Meta's multimodal models.**

Built for the **Meta Global AI Developer Hackathon 2026** (October 26 – November 4), Indie Developer track.

## About the Project

PetSafe Plants aims to help pet owners identify potentially toxic plants from photographs.

The idea was inspired by a personal experience: my cat became ill after drinking water from a vase containing lilies. I wanted to explore how computer vision and multimodal AI could help pet owners recognize similar risks.

Plant identification is not always straightforward. Visually similar species, unfamiliar camera angles, poor lighting, and mixed bouquets can produce incorrect predictions.

The project explores whether combining specialized computer vision models with Meta's multimodal models can improve identification and help recognize uncertain predictions.

**Current stage:** Dataset preparation, model experimentation, and evaluation.

PetSafe Plants does not currently provide veterinary advice or a validated plant-toxicity assessment.

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

The repository currently focuses on Jupyter notebooks for reproducible ML experimentation.

| Notebook | Purpose |
|---|---|
| `01_Plant_export_exploration.ipynb` | Dataset exploration, image properties, duplicates, and split analysis |
| `02_Another_Dataset.ipynb` | Was removed before initial commit as some photos from dataset, could not be used. |
| `03_Train_Plant_Fetcher_Export.ipynb` | Feature extraction and training classifier heads |
| `04_Compare_Training_Runs.ipynb` | Comparison of training runs and evaluation metrics |
| `05_Test_Inference.ipynb` | Evaluation on the 180-image internal holdout (notebook section number 06) |
| `06_Compare_Muse_Spark.ipynb` | Optional comparison against Meta Muse Spark (notebook section number 05) |

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

The current public repository contains notebooks and documentation. Supporting local preparation utilities and datasets are not published here.

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