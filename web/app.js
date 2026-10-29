import {
  createInvestigationState,
  readPipelineEvents,
} from "./investigation.mjs";

const PRESENTATION_DELAY_MS = 1800;
// Motion is loaded only for decorative effects. Inference and controls still work
// when the CDN is unavailable (for example on an offline demo machine).
let motionAnimate = null;
import("https://cdn.jsdelivr.net/npm/motion@14.6.0/+esm")
  .then(({ animate }) => { motionAnimate = animate; })
  .catch(() => {});

const byId = (id) => document.getElementById(id);

const els = {
  workspace: document.querySelector(".workspace"),
  apiStatus: byId("api-status"),
  dropzone: byId("dropzone"),
  uploadEmpty: byId("upload-empty"),
  imageStage: byId("image-stage"),
  imageInput: byId("image-input"),
  chooseImage: byId("choose-image"),
  uploadTitle: byId("upload-title"),
  uploadSubtitle: byId("upload-subtitle"),
  preview: byId("image-preview"),
  flowerBoxes: byId("flower-boxes"),
  scanField: byId("scan-field"),
  questionLayer: byId("question-layer"),
  scanMessage: byId("scan-message"),
  workspaceStepLabel: byId("workspace-step-label"),
  removeImage: byId("remove-image"),
  model: byId("model-select"),
  samMode: byId("sam-mode"),
  samPrompt: byId("sam-prompt"),
  compare: byId("compare-button"),
  motionToggle: byId("motion-toggle"),
  error: byId("error-banner"),
  runStatus: byId("run-status"),
  scanProgress: byId("scan-progress"),
  progressTitle: byId("progress-title"),
  progressDetail: byId("progress-detail"),
  results: byId("results"),
  trainedResult: byId("trained-result"),
  sparkResult: byId("spark-result"),
  segmentationPanel: byId("segmentation-panel"),
  segmentationStatus: byId("segmentation-status"),
  segmentationReason: byId("segmentation-reason"),
  segments: byId("segments-grid"),
};

let selectedFile = null;
let previewUrl = null;
let visibleFlowerDetections = null;
const apiDefault = (() => {
  const { protocol, hostname, port, origin } = window.location;
  if (
    (hostname === "localhost" || hostname === "127.0.0.1") &&
    port === "3000"
  ) {
    return `${protocol}//${hostname}:8000`;
  }
  return origin;
})();
const apiInput = document.createElement("input");
apiInput.type = "hidden";
apiInput.id = "api-base-url";
apiInput.value = localStorage.getItem("petsafe-api-base") || apiDefault;
document.body.append(apiInput);

function apiUrl(path) {
  return `${apiInput.value.trim().replace(/\/+$/, "")}${path}`;
}

function setApiStatus(text, state) {
  els.apiStatus.className = `status-pill status-${state}`;
  els.apiStatus.replaceChildren();
  const dot = document.createElement("span");
  dot.className = "status-dot";
  els.apiStatus.append(dot, document.createTextNode(text));
}

async function loadModels() {
  try {
    const response = await fetch(apiUrl("/api/models"));
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.available) || data.available.length === 0) {
      setApiStatus("No model artifacts", "offline");
      return;
    }
    const previous = els.model.value;
    els.model.replaceChildren();
    const labels = {
      efficientnet_b2: "EfficientNet-B2",
      dinov3: "DINOv3",
      siglip2: "SigLIP2",
    };
    for (const name of data.available) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = labels[name] || name;
      els.model.append(option);
    }
    if (data.available.includes(previous)) els.model.value = previous;
    setApiStatus("API connected", "online");
  } catch (error) {
    setApiStatus("API unavailable", "offline");
    showError(
      `Could not reach the inference API at ${apiInput.value}. Start the API and refresh this page. ${error.message}`,
    );
  }
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function clearError() {
  els.error.textContent = "";
  els.error.hidden = true;
}

function setFile(file) {
  clearError();
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showError("The image must be 12 MB or smaller.");
    return;
  }
  archiveInvestigation();
  cancelInvestigation();
  resetReport();
  selectedFile = file;
  visibleFlowerDetections = null;
  els.flowerBoxes.replaceChildren();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.imageStage.classList.remove("acquired");
  els.preview.src = previewUrl;
  els.preview.hidden = false;
  els.uploadEmpty.hidden = true;
  els.imageStage.hidden = false;
  els.dropzone.classList.add("has-image");
  els.removeImage.hidden = false;
  els.uploadTitle.textContent = file.name;
  els.uploadSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · click to change image`;
  els.compare.disabled = false;
  els.scanMessage.textContent = "PHOTO READY";
  setInvestigationState("idle");
  requestAnimationFrame(() => {
    if (selectedFile !== file) return;
    els.imageStage.classList.add("acquired");
    compare();
  });
}

function clearFile() {
  archiveInvestigation();
  cancelInvestigation();
  resetReport();
  selectedFile = null;
  visibleFlowerDetections = null;
  els.imageInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  els.preview.removeAttribute("src");
  els.preview.hidden = true;
  els.flowerBoxes.replaceChildren();
  els.uploadEmpty.hidden = false;
  els.imageStage.hidden = true;
  els.dropzone.classList.remove("has-image", "is-scanning");
  els.removeImage.hidden = true;
  els.uploadTitle.textContent = "Drop a plant photo here";
  els.uploadSubtitle.replaceChildren(
    document.createTextNode("or "),
    els.chooseImage,
    document.createTextNode(" · JPG, PNG, WebP"),
  );
  els.chooseImage.hidden = false;
  els.compare.disabled = true;
  stopScanAnimation();
}

let scanAnimations = [];
function startScanAnimation() {
  els.dropzone.classList.add("is-scanning");
  if (scanAnimations.length || motionDisabled()) return;
  // Decorative search paths only. They do not imply model-detected locations.
  let seed = [...(selectedFile?.name || "plant")].reduce(
    (value, letter) => (Math.imul(value ^ letter.charCodeAt(0), 16777619) >>> 0),
    (selectedFile?.size || 42) >>> 0,
  );
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  const starts = [[0.08, 0.16], [0.69, 0.23], [0.36, 0.68]];
  [...els.scanField.children].forEach((circle, index) => {
    const spanX = Math.max(0, els.imageStage.clientWidth - circle.offsetWidth);
    const spanY = Math.max(0, els.imageStage.clientHeight - circle.offsetHeight);
    const initial = starts[index].map((value) =>
      Math.min(.9, Math.max(.06, value + (random() - .5) * .12)));
    const points = [
      initial,
      ...Array.from({ length: 4 }, () => [
        .06 + random() * .86, .08 + random() * .82,
      ]),
      initial,
    ];
    const frames = points.map(([x, y], step) => ({
      transform: `translate3d(${Math.round(x * spanX)}px, ${Math.round(y * spanY)}px, 0)`,
      opacity: step === 0 || step === points.length - 1 ? .5 : .85,
    }));
    scanAnimations.push(circle.animate(frames, {
      duration: 10500 + index * 1800,
      iterations: Infinity,
      easing: "ease-in-out",
      fill: "both",
    }));
  });
}
function stopScanAnimation() {
  els.dropzone.classList.remove("is-scanning");
  scanAnimations.forEach((animation) => animation.cancel());
  scanAnimations = [];
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function animalStatusRow(lookup) {
  const row = element("div", "animal-statuses");
  for (const [animal, record] of [["Cats", lookup], ["Dogs", lookup?.dog_toxicity]]) {
    const status = record?.toxicity_status;
    const potential = record?.potential_hazard;
    const tone = status === "toxic" || potential ? "hazard"
      : status === "non_toxic" ? "listed" : "unknown";
    const label = status === "toxic" ? "listed toxic"
      : potential ? "potential hazard"
      : status === "non_toxic" ? "listed non-toxic" : "unresolved";
    row.append(element("span", `animal-status ${tone}`, `${animal} · ${label}`));
  }
  return row;
}

function renderCacheNote(container, cacheHit) {
  if (cacheHit)
    container.append(
      element("span", "cache-note", "Cached result · this server process"),
    );
}

function renderToxicity(container, result, context) {
  const lookup = result?.toxicity;
  if (!lookup) return;
  const details = element("details", "evidence-details");
  const status = (animal) => animal?.toxicity_status === "toxic" || animal?.potential_hazard
    ? "Potential hazard" : animal?.toxicity_status === "non_toxic" ? "Listed non-toxic" : "Unresolved";
  details.append(element("summary", "", `Pet toxicity · Cats: ${status(lookup)} · Dogs: ${status(lookup.dog_toxicity)}`));
  const box = element(
    "div",
    `toxicity-lookup toxicity-${lookup.toxicity_status || "unknown"}`,
  );
  const labels = {
    toxic: "Listed as toxic to cats",
    non_toxic: "Listed as non-toxic to cats",
    unknown: lookup.potential_hazard
      ? "Potential hazard · identity scope unresolved"
      : "No confirmed toxicity match",
  };
  box.append(
    element(
      "p",
      "toxicity-heading",
      labels[lookup.toxicity_status] || labels.unknown,
    ),
  );
  if (lookup.scientific_name) {
    box.append(element("p", "toxicity-taxon", lookup.scientific_name));
  }
  const note = lookup.lookup_note || lookup.taxonomic_scope;
  if (note) box.append(element("p", "toxicity-note", note));
  if (context) box.append(element("p", "toxicity-context", context));
  const clinical = lookup.clinical_effects;
  if (clinical) {
    const effects = element("div", "clinical-effects");
    if (clinical.reported_signs?.length) {
      const scope = clinical.match_scope === "narrower_taxon"
        ? `Reported for ${clinical.source_taxon} only`
        : "Possible signs reported in cats";
      effects.append(element("p", "clinical-heading", scope));
      const signs = element("ul", "clinical-signs");
      clinical.reported_signs.forEach((sign) =>
        signs.append(element("li", "", sign)));
      effects.append(signs);
    }
    if (clinical.serious_effects?.length) {
      effects.append(element(
        "p", "clinical-serious",
        `Serious effects reported: ${clinical.serious_effects.join(", ")}.`,
      ));
    }
    if (clinical.qualifier)
      effects.append(element("p", "clinical-qualifier", clinical.qualifier));
    box.append(effects);
  }
  const links = element("div", "toxicity-sources");
  const seen = new Set();
  for (const evidence of lookup.evidence || []) {
    const url = evidence.source_url;
    if (!url || seen.has(url) || !url.startsWith("https://")) continue;
    seen.add(url);
    const link = element(
      "a",
      "",
      (evidence.source_id || "Source reference").replaceAll("_", " "),
    );
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    links.append(link);
  }
  if (clinical?.source_url?.startsWith("https://") && !seen.has(clinical.source_url)) {
    const link = element("a", "", "Clinical effects source");
    link.href = clinical.source_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    links.append(link);
    seen.add(clinical.source_url);
  }
  if (clinical?.taxonomy_source_url?.startsWith("https://") &&
      !seen.has(clinical.taxonomy_source_url)) {
    const link = element("a", "", "Taxonomy source");
    link.href = clinical.taxonomy_source_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    links.append(link);
  }
  if (links.childElementCount) box.append(links);
  details.append(box);
  const dog = lookup.dog_toxicity;
  if (dog) {
    const dogBox = element(
      "div",
      `toxicity-lookup toxicity-${dog.toxicity_status} dog-toxicity`,
    );
    const dogLabels = {
      toxic: "Listed as toxic to dogs",
      non_toxic: "Listed as non-toxic to dogs",
      unknown: "Dog toxicity unresolved",
    };
    dogBox.append(element("p", "toxicity-heading", dogLabels[dog.toxicity_status]));
    if (dog.lookup_note)
      dogBox.append(element("p", "toxicity-note", dog.lookup_note));
    const dogClinical = dog.clinical_effects;
    if (dogClinical?.reported_signs?.length || dogClinical?.serious_effects?.length) {
      const effects = element("div", "clinical-effects");
      const scope = dogClinical.match_scope === "narrower_taxon"
        ? `Reported for ${dogClinical.source_taxon} only`
        : "Possible signs reported in dogs";
      effects.append(element("p", "clinical-heading", scope));
      if (dogClinical.reported_signs?.length) {
        const signs = element("ul", "clinical-signs");
        dogClinical.reported_signs.forEach((sign) =>
          signs.append(element("li", "", sign)));
        effects.append(signs);
      }
      if (dogClinical.serious_effects?.length)
        effects.append(element(
          "p", "clinical-serious",
          `Serious effects reported: ${dogClinical.serious_effects.join(", ")}.`,
        ));
      if (dogClinical.qualifier)
        effects.append(element("p", "clinical-qualifier", dogClinical.qualifier));
      dogBox.append(effects);
    }
    const dogLinks = element("div", "toxicity-sources");
    const dogUrls = new Set();
    for (const [url, label] of [
      [dog.source_url, "Dog status source"],
      [dogClinical?.source_url, "Dog clinical effects source"],
      [dogClinical?.taxonomy_source_url, "Dog record taxonomy source"],
    ]) {
      if (!url?.startsWith("https://") || dogUrls.has(url)) continue;
      dogUrls.add(url);
      const link = element("a", "", label);
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      dogLinks.append(link);
    }
    if (dogLinks.childElementCount) dogBox.append(dogLinks);
    details.append(dogBox);
  }
  container.append(details);
}

function renderRawMetaResponse(container, result) {
  if (!result?.raw_meta_response) return;
  const details = element("details", "raw-response-details");
  details.append(element("summary", "", "Raw Meta API response"));
  const payload = element(
    "pre",
    "raw-response-json",
    JSON.stringify(result.raw_meta_response, null, 2),
  );
  details.append(payload);
  container.append(details);
}

function renderFlowerBoxes(detections, sourceWidth, sourceHeight) {
  visibleFlowerDetections = { detections, sourceWidth, sourceHeight };
  els.flowerBoxes.replaceChildren();
  if (!sourceWidth || !sourceHeight) return;
  const stage = els.imageStage.getBoundingClientRect();
  const scale = Math.min(
    stage.width / sourceWidth,
    stage.height / sourceHeight,
  );
  const offsetX = (stage.width - sourceWidth * scale) / 2;
  const offsetY = (stage.height - sourceHeight * scale) / 2;
  detections.forEach((item, index) => {
    const [left, top, right, bottom] = item.bounds;
    const box = element("div", "flower-box");
    box.style.left = `${offsetX + left * scale}px`;
    box.style.top = `${offsetY + top * scale}px`;
    box.style.width = `${(right - left) * scale}px`;
    box.style.height = `${(bottom - top) * scale}px`;
    box.append(element("span", "", `${index + 1} · ${item.predicted_class}`));
    els.flowerBoxes.append(box);
  });
}

function renderTrained(result, sourceWidth, sourceHeight) {
  els.trainedResult.replaceChildren();
  const primary = element("div", "prediction-primary");
  primary.append(
    element("strong", "", result.predicted_class),
    element("span", "confidence-value", percent(result.confidence)),
  );
  els.trainedResult.append(primary);
  els.trainedResult.append(
    element(
      "p",
      "confidence-caption",
      "Top class probability · not calibrated confidence",
    ),
  );
  els.trainedResult.append(animalStatusRow(result.toxicity));
  renderToxicity(
    els.trainedResult,
    result,
    "Reference for the classifier’s top class; the prediction may be wrong.",
  );
  const classifierDetails = element("details", "result-details");
  classifierDetails.append(element("summary", "", `Prediction details · ${result.detections?.length || 0} flower crops`));
  renderCacheNote(classifierDetails, result.cache_hit);
  const scores = element("div", "score-list");
  for (const item of result.top_predictions || []) {
    const row = element("div", "score-row");
    const name = element("span", "score-name", item.class_name);
    const track = element("span", "score-track");
    const fill = element("span", "score-fill");
    fill.style.width = `${Math.max(0, Math.min(100, Number(item.probability) * 100))}%`;
    track.append(fill);
    row.append(
      name,
      track,
      element("span", "score-percent", percent(item.probability)),
    );
    scores.append(row);
  }
  classifierDetails.append(scores);
  const heading = element("div", "detection-heading");
  heading.append(
    element(
      "strong",
      "",
      `Individual flower checks · ${result.detections?.length || 0}`,
    ),
  );
  classifierDetails.append(heading);
  if (result.detection_status === "failed") {
    classifierDetails.append(
      element(
        "p",
        "segment-notes",
        `Flower detection unavailable: ${result.detection_error || "unknown error"}`,
      ),
    );
  } else if (result.detection_status === "no_flowers") {
    classifierDetails.append(
      element(
        "p",
        "segment-notes",
        "No flower boxes passed the detector threshold. The full-image prediction is shown above.",
      ),
    );
  }
  const grid = element("div", "flower-crop-grid");
  for (const [index, item] of (result.detections || []).entries()) {
    const card = element("article", "flower-crop");
    const image = element("img", "flower-crop-image");
    image.src = item.crop_data_url;
    image.alt = `Detected flower ${index + 1}`;
    card.append(image);
    const details = element("div", "flower-crop-details");
    details.append(
      element(
        "span",
        "segment-label",
        `FLOWER ${index + 1} · DETECTION ${percent(item.detector_score)}`,
      ),
    );
    details.append(element("strong", "", item.predicted_class));
    details.append(
      element(
        "span",
        "confidence-value",
        `Classifier ${percent(item.confidence)}`,
      ),
    );
    renderToxicity(
      details,
      item,
      "Reference for this crop prediction; verify the plant identity.",
    );
    card.append(details);
    grid.append(card);
  }
  classifierDetails.append(grid);
  els.trainedResult.append(classifierDetails);
  renderFlowerBoxes(result.detections || [], sourceWidth, sourceHeight);
}

function renderSpark(result) {
  const overview = byId("scene-overview");
  overview.hidden = false;
  overview.replaceChildren(
    element(
      "strong",
      "",
      `${result.distinct_plant_count} distinct plant type${result.distinct_plant_count === 1 ? "" : "s"}`,
    ),
  );
  if (result.is_bouquet)
    overview.append(
      element("span", "", result.bouquet_type.replaceAll("_", " ")),
    );
  els.sparkResult.replaceChildren();
  if (result.plant_groups?.length)
    els.sparkResult.append(element("p", "scene-summary-line",
      "Visual identification is provisional. Check each plant and its sources."));
  for (const [index, group] of (result.plant_groups || []).entries()) {
    const card = element("details", "plant-group");
    card.id = `scene-group-${index}`;
    const heading = element("summary", "plant-summary");
    heading.append(
      element("span", "card-kicker", `PLANT TYPE ${group.id}`),
      element("strong", "", group.plant_identification),
      element("span", "plant-scientific", group.scientific_name || "Identity unresolved"),
      element("span", "mini-tag", `${group.identification_confidence} confidence`),
    );
    heading.append(animalStatusRow(group.toxicity));
    card.append(heading);
    if (group.visual_evidence?.length)
      card.append(
        element("p", "spark-evidence", group.visual_evidence.join(" · ")),
      );
    renderToxicity(
      card,
      group,
      "Reference for this visual hypothesis; verify the plant identity.",
    );
    if (group.needs_inspection)
      card.append(
        element("span", "decision-chip yes", "Closer inspection suggested"),
      );
    els.sparkResult.append(card);
  }
  if (!result.plant_groups?.length)
    els.sparkResult.append(
      element("p", "empty-state", "No distinct plant types were identified."),
    );
  if (result.another_photo_needed) {
    const link = element("a", "next-photo-link", "Another photo needed → See what to capture");
    link.href = "#followup-panel";
    els.sparkResult.append(link);
  }
  const technical = element("details", "result-details technical-details");
  technical.append(element("summary", "", "Model and segmentation details"));
  renderCacheNote(technical, result.cache_hit);
  renderSparkUsage(technical, result);
  renderRawMetaResponse(technical, result);
  const targetLabel = result.segmentation_target === "plant"
    ? "whole plants" : "flower heads";
  const recommendation = result.suggest_segmentation
    ? `SAM suggested · Spark target: ${targetLabel} · `
    : `SAM not suggested · If forced, use ${targetLabel} · `;
  const chip = element(
    "span",
    `decision-chip${result.suggest_segmentation ? " yes" : ""}`,
    recommendation +
      (result.segmentation_reason || "No specific reason returned."),
  );
  technical.append(chip);
  if (result.segmentation_target_reason)
    technical.append(element(
      "p", "spark-evidence", `Target choice: ${result.segmentation_target_reason}`,
    ));
  els.sparkResult.append(technical);
}

function renderSparkUsage(container, result) {
  const usage = result.api_usage;
  if (!usage) return;
  const box = element("div", "usage-box");
  box.append(
    element(
      "strong",
      "",
      result.cache_hit
        ? "Meta usage from the original request"
        : "Meta API token usage",
    ),
  );
  const cached =
    usage.cached_input_tokens == null
      ? "not reported"
      : usage.cached_input_tokens;
  const ratio =
    usage.cache_hit_ratio == null
      ? "not reported"
      : percent(usage.cache_hit_ratio);
  box.append(
    element(
      "span",
      "",
      `Input ${usage.input_tokens} · Cached input ${cached} · Output ${usage.output_tokens}`,
    ),
  );
  box.append(element("span", "", `Meta prompt cache ratio ${ratio}`));
  container.append(box);
}

function renderSegmentation(result) {
  els.segmentationPanel.hidden = false;
  els.segmentationStatus.className = `status-pill segmentation-status status-${result.status}`;
  els.segmentationStatus.textContent =
    {
      skipped: "Skipped",
      completed: `${result.segments.length} segment${result.segments.length === 1 ? "" : "s"}`,
      no_masks: "No masks found",
      failed: "Segmentation failed",
    }[result.status] || result.status;
  const reasons = Array.isArray(result.reason) ? result.reason.join(" ") : "";
  const target = result.prompt
    ? `SAM targeted ${result.prompt === "plant" ? "whole plants" : "flower heads"}.`
    : "";
  const counts = result.ran
    ? `${result.mask_count} mask(s) received · ${result.filtered_count} filtered · ${result.assessed_count} reassessed.`
    : "";
  els.segmentationReason.textContent = [target, reasons, counts, result.error || ""]
    .filter(Boolean)
    .join(" ");
  els.segments.replaceChildren();
  for (const segment of result.segments || []) {
    const card = element("article", "segment-card");
    const imageWrap = element("div", "segment-image-wrap");
    const image = element("img", "segment-image");
    image.src = segment.crop_data_url;
    image.alt = `SAM segment ${segment.object_id}`;
    imageWrap.append(image);
    const body = element("div", "segment-body");
    body.append(
      element("p", "segment-label", `SAM region ${segment.object_id}`),
    );
    for (const group of segment.assessment.plant_groups || []) {
      body.append(element("p", "segment-plant", group.plant_identification));
      body.append(element("p", "toxicity-taxon", group.scientific_name));
      body.append(
        element(
          "span",
          "mini-tag",
          `identity ${group.identification_confidence}`,
        ),
      );
      renderToxicity(
        body,
        group,
        "Reference for this crop’s visual hypothesis.",
      );
    }
    const tags = element("div", "segment-meta");
    tags.append(
      element(
        "span",
        "mini-tag",
        `${segment.assessment.distinct_plant_count} plant type(s)`,
      ),
      element(
        "span",
        `mini-tag${segment.in_trained_taxonomy ? "" : " outside"}`,
        segment.in_trained_taxonomy
          ? "in trained classes"
          : "outside trained classes",
      ),
    );
    body.append(tags);
    renderCacheNote(body, segment.assessment.cache_hit);
    renderSparkUsage(body, segment.assessment);
    renderRawMetaResponse(body, segment.assessment);
    card.append(imageWrap, body);
    els.segments.append(card);
  }
}

async function compare() {
  if (!selectedFile) return;
  cancelInvestigation(true);
  const request = ++requestGeneration;
  const controller = new AbortController();
  activeRequest = controller;
  resetReport();
  els.trainedResult.replaceChildren(
    element("p", "empty-state", "Awaiting the classifier result…"),
  );
  els.sparkResult.replaceChildren(
    element("p", "empty-state", "Awaiting the scene assessment…"),
  );
  els.flowerBoxes.replaceChildren();
  els.results.hidden = true;
  els.segmentationPanel.hidden = true;
  clearError();
  setInvestigationState("uploading");
  try {
    const form = new FormData();
    form.append("image_file", selectedFile, selectedFile.name);
    form.append("model_name", els.model.value);
    form.append("force_sam", String(els.samMode.value === "force"));
    form.append("skip_sam", String(els.samMode.value === "skip"));
    form.append("sam_prompt", els.samPrompt.value);
    const response = await fetch(apiUrl("/api/flows/compare-stream"), {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(
        typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail),
      );
    }
    for await (const { event, data } of readPipelineEvents(response.body)) {
      if (request !== requestGeneration) return;
      if (event === "completed" && !motionDisabled()) {
        // Show any new hazard immediately; only the final layout waits.
        sceneAssessments = [
          data.muse_spark_flow,
          ...data.segmentation.segments.map((item) => item.assessment),
        ];
        updateSafety();
        updateFollowup(data.muse_spark_flow);
        setInvestigationState("presenting");
        await new Promise((resolve, reject) => {
          const abort = () => {
            clearTimeout(timer);
            reject(new DOMException("Investigation cancelled", "AbortError"));
          };
          const timer = setTimeout(() => {
            controller.signal.removeEventListener("abort", abort);
            resolve();
          }, PRESENTATION_DELAY_MS);
          controller.signal.addEventListener("abort", abort, { once: true });
        });
        if (request !== requestGeneration) return;
      }
      handlePipelineEvent(event, data);
    }
    if (investigationState !== "completed")
      throw new Error(
        "The connection ended before the report was complete. Please retry.",
      );
  } catch (error) {
    if (request !== requestGeneration || error.name === "AbortError") return;
    controller.abort();
    setInvestigationState("error");
    showError(
      error.message || "The investigation could not finish. Please retry.",
    );
  } finally {
    if (request === requestGeneration) {
      activeRequest = null;
      els.compare.disabled = !selectedFile;
      els.compare.querySelector(".button-label").textContent =
        "Investigate again";
    }
  }
}

els.chooseImage.addEventListener("click", () => els.imageInput.click());
els.dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  if (event.target.closest(".image-stage")) return;
  els.imageInput.click();
});
els.imageInput.addEventListener("change", () =>
  setFile(els.imageInput.files[0]),
);
els.removeImage.addEventListener("click", clearFile);
els.compare.addEventListener("click", compare);
byId("scene-card").addEventListener("click", () => {
  els.sparkResult.scrollIntoView({ behavior: motionDisabled() ? "auto" : "smooth", block: "start" });
});
byId("clear-results").addEventListener("click", () => {
  cancelInvestigation();
  resetReport();
  clearError();
});
for (const eventName of ["dragenter", "dragover"]) {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("is-dragging");
  });
}
els.dropzone.addEventListener("drop", (event) =>
  setFile(event.dataTransfer.files[0]),
);
els.apiStatus.addEventListener("click", async () => {
  const value = window.prompt("Inference API base URL", apiInput.value);
  if (value === null || !value.trim()) return;
  apiInput.value = value.trim().replace(/\/+$/, "");
  localStorage.setItem("petsafe-api-base", apiInput.value);
  await loadModels();
});
els.apiStatus.title = "Click to change the API base URL";
window.addEventListener("beforeunload", () => {
  cancelInvestigation();
  historyUrls.forEach((url) => URL.revokeObjectURL(url));
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
window.addEventListener("resize", () => {
  if (els.dropzone.classList.contains("is-scanning")) {
    stopScanAnimation();
    startScanAnimation();
  }
  highlightSpecimen(null);
  clearSpecimenMotion();
  placeSpecimens();
  positionSceneCardLink();
  if (visibleFlowerDetections) {
    const { detections, sourceWidth, sourceHeight } = visibleFlowerDetections;
    renderFlowerBoxes(detections, sourceWidth, sourceHeight);
  }
});

// Pipeline states are entered only from upload actions or server events.
const machine = createInvestigationState();
let investigationState = machine.state;
let activeRequest = null;
let requestGeneration = 0;
let questionTimer = null;
let questionIndex = 0;
let specimenPreviewTimer = null;
let sceneAssessments = [];
let dismissedSafetyText = null;
let lastReport = null;
let sourceDimensions = { width: 1, height: 1 };
let inspectedSpecimen = null;
const specimenRecords = new Map();
const historyUrls = [];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const savedMotionChoice = localStorage.getItem("petsafe-motion-choice-v2");
let motionChoice = ["on", "off"].includes(savedMotionChoice) ? savedMotionChoice : "on";
function motionDisabled() {
  return motionChoice === "off";
}
function syncMotionControl() {
  const disabled = motionDisabled();
  document.documentElement.classList.toggle("motion-enabled", !disabled);
  document.documentElement.classList.toggle("motion-paused", disabled);
  els.motionToggle.value = motionChoice;
  if (disabled) {
    document.getAnimations().forEach((animation) => animation.cancel());
    scanAnimations = [];
    els.questionLayer.querySelectorAll(".thought-bubble").forEach((bubble) => {
      bubble.style.opacity = "1";
    });
  } else if (machine.busy) {
    startScanAnimation();
  }
}
const stateTitles = {
  idle: "Ready for a closer look",
  uploading: "Sending the photograph",
  analyzing_scene: "Reading the botanical scene",
  scene_analyzed: "Scene assessment received",
  segmenting: "Isolating plant regions",
  segments_ready: "Specimens selected",
  identifying_regions: "Investigating each specimen",
  presenting: "Arranging received findings",
  completed: "Investigation report ready",
  error: "Investigation interrupted",
};
const questions = [
  "How many distinct plant types are visible?",
  "Is this a mixed bouquet?",
  "Could another plant be partially hidden?",
  "Would closer inspection reveal more detail?",
];
const thoughtPositions = ["thought-one", "thought-two", "thought-three"];
let thoughtSerial = 0;
let layoutAnimation = null;
function centerPhotoWorkspace(state, keepCentered = false) {
  const centered = Boolean(selectedFile && (keepCentered || !["idle"].includes(state)));
  if (els.workspace.classList.contains("is-centered") === centered) return;
  layoutAnimation?.cancel();
  const before = els.dropzone.getBoundingClientRect();
  els.workspace.classList.toggle("is-centered", centered);
  const after = els.dropzone.getBoundingClientRect();
  if (motionDisabled() || !before.width || !after.width) return;
  const dx = before.left - after.left;
  const dy = before.top - after.top;
  const sx = before.width / after.width;
  const sy = before.height / after.height;
  const animation = els.dropzone.animate([
    { transform: `translate3d(${dx}px, ${dy}px, 0) scale(${sx}, ${sy})`, transformOrigin: "top left" },
    { transform: "translate3d(0, 0, 0) scale(1)", transformOrigin: "top left" },
  ], { duration: 740, easing: "cubic-bezier(.22,.85,.26,1)" });
  layoutAnimation = animation;
  animation.finished.catch(() => {}).finally(() => {
    if (layoutAnimation !== animation) return;
    layoutAnimation = null;
    placeSpecimens();
    highlightSpecimen(null);
  });
}
function animateThought(node, entering) {
  if (motionDisabled() || !node.isConnected) {
    if (entering) node.style.opacity = "1";
    return Promise.resolve();
  }
  const start = "translate3d(0,22px,0) scale(.58)";
  const overshoot = "translate3d(0,-4px,0) scale(1.07)";
  const rest = "translate3d(0,0,0) scale(1)";
  const exit = "translate3d(0,-12px,0) scale(.9)";
  if (motionAnimate) {
    try {
      const control = motionAnimate(node, {
        opacity: entering ? [0, 1, 1] : [1, 0],
        transform: entering ? [start, overshoot, rest] : [rest, exit],
      }, {
        duration: entering ? 0.62 : 0.25,
        ease: entering ? "easeOut" : "easeIn",
      });
      return Promise.resolve(control).then(() => {
        if (entering && node.isConnected) node.style.opacity = "1";
      }).catch(() => {
        if (entering && node.isConnected) node.style.opacity = "1";
      });
    } catch {
      // A failed CDN animation must not suppress the question in Chrome.
    }
  }
  const frames = entering
    ? [
        { opacity: 0, transform: start, offset: 0 },
        { opacity: 1, transform: overshoot, offset: 0.72 },
        { opacity: 1, transform: rest, offset: 1 },
      ]
    : [
        { opacity: 1, transform: rest },
        { opacity: 0, transform: exit },
      ];
  const animation = node.animate(frames, {
    duration: entering ? 620 : 250,
    easing: entering ? "cubic-bezier(.2,.8,.2,1)" : "ease-in",
    fill: "forwards",
  });
  return animation.finished.then(() => {
    if (entering && node.isConnected) node.style.opacity = "1";
    animation.cancel();
  }).catch(() => {
    if (entering && node.isConnected) node.style.opacity = "1";
  });
}
function showNextThought() {
  if (!els.dropzone.classList.contains("is-scanning")) return;
  const position = thoughtSerial % thoughtPositions.length;
  const old = els.questionLayer.querySelector(`[data-position="${position}"]`);
  if (old) animateThought(old, false).finally(() => old.remove());
  const bubble = element("span", `thought-bubble ${thoughtPositions[position]}`, questions[questionIndex]);
  bubble.dataset.position = String(position);
  bubble.style.opacity = motionDisabled() ? "1" : "0";
  els.questionLayer.append(bubble);
  animateThought(bubble, true);
  thoughtSerial += 1;
  questionIndex = (questionIndex + 1) % questions.length;
}
function clearThoughts() {
  clearInterval(questionTimer);
  questionTimer = null;
  els.questionLayer.replaceChildren();
  thoughtSerial = 0;
}
function setInvestigationState(state, keepCentered = false) {
  if (!Object.hasOwn(stateTitles, state)) return;
  machine.transition(state);
  investigationState = machine.state;
  document.body.dataset.investigation = state;
  centerPhotoWorkspace(state, keepCentered);
  els.workspaceStepLabel.textContent = state === "idle"
    ? "ADD A PHOTO"
    : state === "completed" ? "INVESTIGATION CANVAS · COMPLETE" : "LIVE INVESTIGATION";
  els.scanMessage.textContent = stateTitles[state].toUpperCase();
  els.progressTitle.textContent = stateTitles[state];
  const busy = machine.busy;
  els.scanProgress.hidden = !busy;
  els.compare.disabled = busy || !selectedFile;
  els.compare.classList.toggle("is-loading", busy);
  els.model.disabled = busy;
  els.samMode.disabled = busy;
  els.samPrompt.disabled = busy;
  els.compare.querySelector(".button-label").textContent = busy
    ? "Investigating…"
    : selectedFile ? "Run investigation again" : "Begin investigation";
  // Search circles appear immediately; no model result is implied by their positions.
  if (busy) startScanAnimation();
  else stopScanAnimation();
  els.progressDetail.textContent = {
    uploading: "Transferring your image. Visual search has begun; model findings will appear when returned.",
    analyzing_scene: "Looking for visual evidence in the photograph.",
    scene_analyzed: "Scene evidence is available below. The comparison is continuing.",
    segmenting: "SAM is finding masks. Regions will appear when they arrive.",
    segments_ready: "These are actual selected SAM crops.",
    identifying_regions: "Each specimen receives its own visual assessment and local reference lookup.",
    presenting: "The analysis is complete. Arranging your report.",
  }[state] || "";
  const showQuestions = busy && !["segments_ready", "identifying_regions", "presenting"].includes(state)
    && !els.workspace.classList.contains("has-specimens");
  if (showQuestions) {
    if (!questionTimer) {
      showNextThought();
      questionTimer = setInterval(showNextThought, 1650);
    }
  } else {
    clearThoughts();
  }
}
function cancelInvestigation(keepCentered = false) {
  requestGeneration += 1;
  activeRequest?.abort();
  activeRequest = null;
  clearThoughts();
  clearInterval(specimenPreviewTimer);
  specimenPreviewTimer = null;
  questionIndex = 0;
  stopScanAnimation();
  highlightSpecimen(null);
  clearSpecimenMotion();
  setInvestigationState("idle", keepCentered);
}
function clearSpecimenMotion() {
  document.querySelectorAll(".specimen-flight").forEach((node) => {
    node.getAnimations().forEach((animation) => animation.cancel());
    node.remove();
  });
}
function resetReport() {
  clearInterval(specimenPreviewTimer);
  specimenPreviewTimer = null;
  byId("scene-overview").hidden = true;
  sceneAssessments = [];
  byId("scene-card").hidden = true;
  byId("scene-card-link").classList.remove("visible");
  dismissedSafetyText = null;
  lastReport = null;
  specimenRecords.clear();
  byId("specimen-rail").replaceChildren();
  document.querySelector(".workspace").classList.remove("has-specimens");
  for (const id of ["specimen-workspace", "followup-panel", "safety-summary"])
    byId(id).hidden = true;
  els.results.hidden = true;
  visibleFlowerDetections = null;
  els.flowerBoxes.replaceChildren();
  byId("specimen-dialog").close();
}
function handlePipelineEvent(event, data) {
  if (event === "error") throw new Error(data.message);
  if (["scene_analyzed", "trained_ready", "completed"].includes(event))
    els.results.hidden = false;
  if (event === "analyzing_scene")
    sourceDimensions = { width: data.image_width, height: data.image_height };
  if (event === "scene_analyzed") {
    renderSpark(data);
    sceneAssessments = [data];
    updateSafety();
    updateFollowup(data);
  }
  if (event === "trained_ready")
    renderTrained(data, sourceDimensions.width, sourceDimensions.height);
  if (event === "segments_ready") {
    byId("scene-card").hidden = true;
    byId("scene-card-link").classList.remove("visible");
    clearThoughts();
    document
      .querySelector(".workspace")
      .classList.toggle("has-specimens", data.regions.length > 0);
    byId("specimen-workspace").hidden = !data.regions.length;
    byId("specimen-count").textContent =
      `${data.regions.length} selected / ${data.mask_count} masks`;
    if (visibleFlowerDetections) {
      const { detections, sourceWidth, sourceHeight } = visibleFlowerDetections;
      renderFlowerBoxes(detections, sourceWidth, sourceHeight);
    }
    data.regions.forEach(addSpecimen);
    if (data.regions.length) previewSpecimens(data.regions.map((region) => String(region.object_id)));
  }
  if (event === "region_assessed") {
    const item = specimenRecords.get(String(data.object_id));
    if (item) {
      item.data = data;
      const groups = data.assessment.plant_groups || [];
      item.caption.textContent =
        groups.map((group) => group.plant_identification).join(" · ") ||
        "Unresolved specimen";
      item.taxon.textContent = groups
        .map(
          (group) =>
            `${group.scientific_name} · ${group.identification_confidence} confidence`,
        )
        .join(" / ");
      item.button.classList.add("identified");
      const hazard = groups.some(
        (group) =>
          group.toxicity?.toxicity_status === "toxic" ||
          group.toxicity?.potential_hazard ||
          group.toxicity?.dog_toxicity?.toxicity_status === "toxic" ||
          group.toxicity?.dog_toxicity?.potential_hazard,
      );
      item.button.classList.toggle("hazard", hazard);
      const nonToxic =
        groups.length &&
        groups.every(
          (group) => group.toxicity?.toxicity_status === "non_toxic" &&
            group.toxicity?.dog_toxicity?.toxicity_status === "non_toxic",
        );
      item.button.classList.toggle("non-toxic", Boolean(nonToxic));
      item.status.textContent = hazard
        ? "Potential hazard"
        : nonToxic
          ? "Listed non-toxic"
          : "Unresolved · inspect";
      if (
        inspectedSpecimen === String(data.object_id) &&
        byId("specimen-dialog").open
      )
        inspectSpecimen(inspectedSpecimen);
      sceneAssessments.push(data.assessment);
      updateSafety();
      if (data.assessment.another_photo_needed) updateFollowup(data.assessment);
    }
  }
  if (event === "completed") {
    lastReport = data;
    renderTrained(data.trained_flow, data.image_width, data.image_height);
    renderSpark(data.muse_spark_flow);
    renderSegmentation(data.segmentation);
    if (!data.segmentation.segments?.length) showSceneCard(data.muse_spark_flow);
    sceneAssessments = [
      data.muse_spark_flow,
      ...Array.from(specimenRecords.values())
        .map((item) => item.data.assessment)
        .filter(Boolean),
    ];
    updateSafety();
    for (const item of specimenRecords.values()) {
      if (!item.data.assessment)
        item.status.textContent = "Assessment unavailable · see report";
    }
    if (data.muse_spark_flow.another_photo_needed)
      updateFollowup(data.muse_spark_flow);
  }
  if (Object.hasOwn(stateTitles, event)) setInvestigationState(event);
  if (event === "segmenting" && data.prompt)
    els.progressDetail.textContent =
      `SAM is isolating ${data.prompt === "plant" ? "whole plants" : "flower heads"}. Regions will appear when they arrive.`;
  if (
    event === "completed" &&
    inspectedSpecimen &&
    byId("specimen-dialog").open
  )
    inspectSpecimen(inspectedSpecimen);
}
function sourceBox(bounds) {
  const stage = { width: els.imageStage.clientWidth, height: els.imageStage.clientHeight };
  const scale = Math.min(
    stage.width / sourceDimensions.width,
    stage.height / sourceDimensions.height,
  );
  return {
    left:
      (stage.width - sourceDimensions.width * scale) / 2 + bounds[0] * scale,
    top:
      (stage.height - sourceDimensions.height * scale) / 2 + bounds[1] * scale,
    width: (bounds[2] - bounds[0]) * scale,
    height: (bounds[3] - bounds[1]) * scale,
  };
}
function positionSceneCardLink() {
  const card = byId("scene-card");
  const link = byId("scene-card-link");
  if (card.hidden) return;
  const stage = els.imageStage.getBoundingClientRect();
  const photo = els.preview.getBoundingClientRect();
  const target = card.getBoundingClientRect();
  link.setAttribute("width", els.imageStage.clientWidth);
  link.setAttribute("height", els.imageStage.clientHeight);
  const x = photo.left + photo.width / 2 - stage.left;
  const y = photo.top + photo.height / 2 - stage.top;
  const endX = target.left - stage.left;
  const endY = target.top + target.height / 2 - stage.top;
  const line = link.querySelector("line");
  line.setAttribute("x1", x);
  line.setAttribute("y1", y);
  line.setAttribute("x2", endX);
  line.setAttribute("y2", endY);
  link.querySelector("circle").setAttribute("cx", x);
  link.querySelector("circle").setAttribute("cy", y);
  link.classList.add("visible");
}
function showSceneCard(assessment) {
  const card = byId("scene-card");
  const groups = assessment.plant_groups || [];
  const names = groups.map((group) => group.plant_identification).filter(Boolean);
  card.replaceChildren(
    element("span", "eyebrow", "WHOLE PHOTO ASSESSMENT"),
    element("strong", "", names.length ? names.slice(0, 2).join(" · ") : "No plant identified"),
    element("small", "", `${assessment.distinct_plant_count ?? 0} distinct plant type(s) reported · View details below`),
  );
  card.hidden = false;
  requestAnimationFrame(positionSceneCardLink);
}
function clampSpecimen(button, left, top) {
  const width = els.imageStage.clientWidth;
  const height = els.imageStage.clientHeight;
  return {
    left: Math.max(5, Math.min(left, width - button.offsetWidth - 5)),
    top: Math.max(64, Math.min(top, height - button.offsetHeight - 72)),
  };
}
function placeSpecimens() {
  if (!els.imageStage.clientWidth) return;
  els.imageStage.classList.toggle(
    "compact-specimens",
    els.imageStage.clientHeight < 560,
  );
  const placed = [];
  [...specimenRecords.values()].forEach((item, index) => {
    if (item.manual) {
      const position = clampSpecimen(item.button,
        parseFloat(item.button.style.left) || 0,
        parseFloat(item.button.style.top) || 0);
      item.button.style.left = `${position.left}px`;
      item.button.style.top = `${position.top}px`;
      placed.push({ ...position, width: item.button.offsetWidth, height: item.button.offsetHeight });
      return;
    }
    const box = sourceBox(item.data.bounds);
    const width = item.button.offsetWidth;
    const height = item.button.offsetHeight;
    const centerX = box.left + box.width / 2;
    const centerY = box.top + box.height / 2;
    const gap = 14;
    const candidates = [
      [box.left + box.width + gap, centerY - height / 2],
      [box.left - width - gap, centerY - height / 2],
      [centerX - width / 2, box.top + box.height + gap],
      [centerX - width / 2, box.top - height - gap],
      [centerX + gap, centerY + gap],
      [centerX - width - gap, centerY - height - gap],
    ];
    const overlap = (a, b) => Math.max(0, Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left))
      * Math.max(0, Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top));
    const position = candidates.map(([left, top]) => {
      const p = clampSpecimen(item.button, left, top);
      const rect = { ...p, width, height };
      const distance = Math.hypot(p.left + width / 2 - centerX, p.top + height / 2 - centerY);
      const score = distance + overlap(rect, box) * 4 + placed.reduce((sum, other) => sum + overlap(rect, other) * 8, 0);
      return { ...p, score };
    }).sort((a, b) => a.score - b.score)[0];
    item.button.style.left = `${position.left}px`;
    item.button.style.top = `${position.top}px`;
    placed.push({ ...position, width, height });
  });
}
function enableSpecimenDrag(id, item) {
  const { button } = item;
  let pointer = null;
  button.setAttribute("aria-describedby", "specimen-hint");
  button.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    pointer = {
      id: event.pointerId, x: event.clientX, y: event.clientY,
      left: parseFloat(button.style.left) || 0,
      top: parseFloat(button.style.top) || 0,
      moved: false,
    };
    button.setPointerCapture(event.pointerId);
  });
  button.addEventListener("pointermove", (event) => {
    if (!pointer || event.pointerId !== pointer.id) return;
    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    const threshold = event.pointerType === "touch" ? 10 : 5;
    if (!pointer.moved && Math.hypot(dx, dy) < threshold) return;
    pointer.moved = true;
    event.preventDefault();
    button.classList.add("is-dragging");
    item.manual = true;
    const position = clampSpecimen(button, pointer.left + dx, pointer.top + dy);
    button.style.left = `${position.left}px`;
    button.style.top = `${position.top}px`;
    highlightSpecimen(id);
  });
  const endDrag = (event) => {
    if (!pointer || event.pointerId !== pointer.id) return;
    item.suppressClick = pointer.moved;
    if (pointer.moved) setTimeout(() => { item.suppressClick = false; }, 0);
    pointer = null;
    button.classList.remove("is-dragging");
    if (button.hasPointerCapture(event.pointerId)) button.releasePointerCapture(event.pointerId);
  };
  button.addEventListener("pointerup", endDrag);
  button.addEventListener("pointercancel", endDrag);
  button.addEventListener("keydown", (event) => {
    const steps = { ArrowLeft: [-20, 0], ArrowRight: [20, 0],
      ArrowUp: [0, -20], ArrowDown: [0, 20] };
    const step = steps[event.key];
    if (!step) return;
    event.preventDefault();
    item.manual = true;
    const position = clampSpecimen(button,
      (parseFloat(button.style.left) || 0) + step[0],
      (parseFloat(button.style.top) || 0) + step[1]);
    button.style.left = `${position.left}px`;
    button.style.top = `${position.top}px`;
    highlightSpecimen(id);
  });
}
function addSpecimen(data) {
  const id = String(data.object_id);
  if (specimenRecords.has(id)) return;
  const button = element("button", "floating-specimen");
  button.type = "button";
  button.style.setProperty("--float-delay", `${-specimenRecords.size * 1.3}s`);
  const image = element("img", "specimen-image");
  image.src = data.crop_data_url;
  image.alt = `SAM specimen ${id}`;
  image.draggable = false;
  const caption = element("strong", "", `Specimen ${id}`);
  const taxon = element("span", "specimen-taxon");
  const status = element("small", "", "Awaiting regional assessment");
  button.append(
    element("span", "specimen-id", `SPECIMEN / ${id}`),
    image,
    caption,
    taxon,
    status,
  );
  byId("specimen-rail").append(button);
  const item = { data, button, image, caption, taxon, status, manual: false, suppressClick: false };
  image.addEventListener("load", () => {
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 160 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    item.maskAlpha = {
      width: canvas.width,
      height: canvas.height,
      values: Uint8Array.from({ length: canvas.width * canvas.height }, (_, i) => pixels[i * 4 + 3]),
    };
  }, { once: true });
  specimenRecords.set(id, item);
  placeSpecimens();
  enableSpecimenDrag(id, item);
  button.addEventListener("pointerenter", () => {
    clearInterval(specimenPreviewTimer);
    specimenPreviewTimer = null;
    highlightSpecimen(id);
  });
  button.addEventListener("pointerleave", () => {
    if (!button.classList.contains("is-dragging")) highlightSpecimen(null);
  });
  button.addEventListener("focus", () => {
    clearInterval(specimenPreviewTimer);
    specimenPreviewTimer = null;
    highlightSpecimen(id);
  });
  button.addEventListener("blur", () => highlightSpecimen(null));
  button.addEventListener("click", () => {
    if (item.suppressClick) { item.suppressClick = false; return; }
    inspectSpecimen(id);
  });
  image.addEventListener(
    "load",
    () => {
      if (motionDisabled() || !image.isConnected) return;
      const start = sourceBox(data.bounds);
      const stage = els.imageStage.getBoundingClientRect();
      const ratioX = stage.width / els.imageStage.clientWidth;
      const ratioY = stage.height / els.imageStage.clientHeight;
      const end = image.getBoundingClientRect();
      // A fixed clone escapes the scrollable tray while the real specimen stays accessible.
      const flight = image.cloneNode();
      flight.className = "specimen-flight";
      flight.alt = "";
      flight.setAttribute("aria-hidden", "true");
      Object.assign(flight.style, {
        left: `${stage.left + start.left * ratioX}px`,
        top: `${stage.top + start.top * ratioY}px`,
        width: `${start.width * ratioX}px`,
        height: `${start.height * ratioY}px`,
      });
      document.body.append(flight);
      const dx =
        end.left + end.width / 2 - stage.left - (start.left + start.width / 2) * ratioX;
      const dy =
        end.top + end.height / 2 - stage.top - (start.top + start.height / 2) * ratioY;
      const scale = Math.min(
        end.width / (start.width * ratioX),
        end.height / (start.height * ratioY),
      );
      const frames = Array.from({ length: 41 }, (_, index) => {
        const t = index / 40;
        const spring =
          index === 40
            ? 1
            : 1 -
              Math.exp(-8 * t) * (Math.cos(10 * t) + 0.8 * Math.sin(10 * t));
        return {
          offset: t,
          opacity: 0.5 + 0.5 * Math.min(1, t * 5),
          transform: `translate(${dx * spring}px, ${dy * spring}px) scale(${1 + (scale - 1) * spring})`,
        };
      });
      image.style.opacity = "0";
      const animation = flight.animate(frames, {
        duration: 950,
        fill: "forwards",
      });
      animation.finished
        .catch(() => {})
        .finally(() => {
          image.style.opacity = "";
          flight.remove();
        });
    },
    { once: true },
  );
}
function highlightSpecimen(id) {
  const overlay = byId("source-region");
  const connection = byId("specimen-connection");
  if (!overlay || !connection) return;
  overlay.replaceChildren();
  connection.style.display = "none";
  for (const [key, record] of specimenRecords) {
    record.button.classList.toggle("is-linked", key === id);
  }
  const item = specimenRecords.get(id);
  if (!item) return;
  const box = sourceBox(item.data.bounds);
  const image = element("img", "source-mask");
  image.src = item.data.crop_data_url;
  Object.assign(image.style, {
    left: `${box.left}px`,
    top: `${box.top}px`,
    width: `${box.width}px`,
    height: `${box.height}px`,
  });
  overlay.append(image);
  const bounds = element("span", "source-bounds");
  Object.assign(bounds.style, {
    left: `${box.left}px`, top: `${box.top}px`,
    width: `${box.width}px`, height: `${box.height}px`,
  });
  overlay.append(bounds);
  const stage = els.imageStage.getBoundingClientRect();
  const ratioX = stage.width / els.imageStage.clientWidth;
  const ratioY = stage.height / els.imageStage.clientHeight;
  const target = item.button.getBoundingClientRect();
  const line = connection.querySelector("line");
  line.setAttribute("x1", stage.left + (box.left + box.width / 2) * ratioX);
  line.setAttribute("y1", stage.top + (box.top + box.height / 2) * ratioY);
  line.setAttribute("x2", target.left + target.width / 2);
  line.setAttribute("y2", target.top + 20);
  connection.style.display = "block";
}
function hoverPhotoSpecimen(event) {
  if (event.pointerType === "touch" || els.imageStage.querySelector(".floating-specimen.is-dragging")) return;
  if (event.target.closest(".floating-specimen, .scene-card, .remove-button")) return;
  const stage = els.imageStage.getBoundingClientRect();
  const x = (event.clientX - stage.left) * els.imageStage.clientWidth / stage.width;
  const y = (event.clientY - stage.top) * els.imageStage.clientHeight / stage.height;
  const matches = [...specimenRecords].filter(([, item]) => {
    const box = sourceBox(item.data.bounds);
    if (x < box.left || x > box.left + box.width || y < box.top || y > box.top + box.height) return false;
    if (!item.maskAlpha) return true;
    const pixelX = Math.min(item.maskAlpha.width - 1, Math.floor((x - box.left) / box.width * item.maskAlpha.width));
    const pixelY = Math.min(item.maskAlpha.height - 1, Math.floor((y - box.top) / box.height * item.maskAlpha.height));
    return item.maskAlpha.values[pixelY * item.maskAlpha.width + pixelX] > 32;
  });
  matches.sort((a, b) => {
    const one = sourceBox(a[1].data.bounds);
    const two = sourceBox(b[1].data.bounds);
    return one.width * one.height - two.width * two.height;
  });
  const id = matches[0]?.[0] || null;
  if (id && specimenRecords.get(id).button.classList.contains("is-linked")) return;
  if (!id && !els.imageStage.querySelector(".floating-specimen.is-linked")) return;
  clearInterval(specimenPreviewTimer);
  specimenPreviewTimer = null;
  highlightSpecimen(id);
}
els.imageStage.addEventListener("pointermove", hoverPhotoSpecimen);
els.imageStage.addEventListener("pointerleave", () => {
  if (!els.imageStage.querySelector(".floating-specimen.is-dragging")) highlightSpecimen(null);
});
function previewSpecimens(ids) {
  clearInterval(specimenPreviewTimer);
  let index = 0;
  highlightSpecimen(ids[index++]);
  specimenPreviewTimer = setInterval(() => {
    if (index >= ids.length) {
      clearInterval(specimenPreviewTimer);
      specimenPreviewTimer = null;
      highlightSpecimen(null);
      return;
    }
    highlightSpecimen(ids[index++]);
  }, 1100);
}
function inspectSpecimen(id) {
  const item = specimenRecords.get(id);
  if (!item) return;
  inspectedSpecimen = id;
  const panel = byId("specimen-detail");
  panel.replaceChildren(element("p", "eyebrow", `SPECIMEN / ${id}`));
  const image = element("img", "inspection-image");
  image.src = item.data.crop_data_url;
  image.alt = `Selected SAM specimen ${id}`;
  panel.append(image);
  const assessment = item.data.assessment;
  if (!assessment)
    panel.append(
      element(
        "p",
        "",
        machine.busy
          ? "Regional assessment is still running."
          : "No regional assessment was returned. Review the retained scene report.",
      ),
    );
  for (const group of assessment?.plant_groups || []) {
    panel.append(
      element("h2", "", group.plant_identification),
      element(
        "p",
        "",
        `${group.scientific_name} · ${group.identification_confidence} identification confidence`,
      ),
    );
    renderToxicity(panel, group, "Visual identity remains a hypothesis.");
  }
  if (assessment?.another_photo_needed)
    panel.append(element("p", "", assessment.photo_instructions));
  if (!byId("specimen-dialog").open) byId("specimen-dialog").showModal();
}
function updateSafety() {
  highlightSpecimen(null);
  const groups = sceneAssessments.flatMap((item) => item.plant_groups || []);
  const hazards = groups.filter(
    (item) =>
      item.toxicity?.toxicity_status === "toxic" ||
      item.toxicity?.potential_hazard ||
      item.toxicity?.dog_toxicity?.toxicity_status === "toxic" ||
      item.toxicity?.dog_toxicity?.potential_hazard,
  );
  const unresolved =
    sceneAssessments.some((item) => item.another_photo_needed) ||
    groups.some(
      (item) =>
        item.identification_confidence === "low" ||
        !item.toxicity ||
        item.toxicity.toxicity_status === "unknown" ||
        !item.toxicity.dog_toxicity ||
        item.toxicity.dog_toxicity.toxicity_status === "unknown",
    );
  const banner = byId("safety-summary");
  banner.className = hazards.length ? "hazard-summary" : "unresolved-summary";
  const names = [...new Set(hazards.map((item) => item.plant_identification))];
  const message = hazards.length
    ? `Potential hazard identified: ${names.join(", ")}. Review the linked toxicity references and verify identity.${unresolved ? " Some plants remain unresolved." : ""}`
    : "Investigation incomplete: some identities or toxicity matches remain unresolved.";
  banner.hidden = (!hazards.length && !unresolved) || dismissedSafetyText === message;
  banner.replaceChildren(document.createTextNode(message));
  if (!banner.hidden) {
    const dismiss = element("button", "safety-dismiss", "×");
    dismiss.type = "button";
    dismiss.setAttribute("aria-label", "Dismiss investigation warning");
    dismiss.addEventListener("click", () => {
      dismissedSafetyText = message;
      banner.hidden = true;
    });
    banner.append(dismiss);
  }
}
function updateFollowup(assessment) {
  if (!assessment.another_photo_needed) return;
  byId("followup-panel").hidden = false;
  const instructions = [
    ...new Set(
      [...sceneAssessments, assessment]
        .filter((item) => item.another_photo_needed)
        .map((item) => item.photo_instructions || item.photo_reason)
        .filter(Boolean),
    ),
  ];
  byId("followup-instructions").textContent =
    instructions.join(" ") ||
    "More photographic evidence is needed to resolve this investigation.";
}
function archiveInvestigation() {
  if ((!lastReport && !sceneAssessments.length) || !selectedFile) return;
  const entry = element("details", "history-entry");
  entry.append(
    element("summary", "", `Previous photograph · ${selectedFile.name}`),
  );
  const url = URL.createObjectURL(selectedFile);
  historyUrls.push(url);
  const image = element("img", "history-image");
  image.src = url;
  image.alt = "Previous investigation photograph";
  const report = els.results.cloneNode(true);
  report.hidden = false;
  report.querySelector("#clear-results")?.remove();
  report.removeAttribute("id");
  report.querySelectorAll("[id]").forEach((node) => node.removeAttribute("id"));
  entry.append(image);
  if (!lastReport)
    entry.append(
      element("p", "", "Partial report — this investigation was interrupted."),
    );
  entry.append(report);
  byId("investigation-history").prepend(entry);
  lastReport = null;
  sceneAssessments = [];
  els.results.hidden = true;
}
byId("followup-button").addEventListener("click", () => els.imageInput.click());
byId("close-specimen").addEventListener("click", () =>
  byId("specimen-dialog").close(),
);
byId("specimen-dialog").addEventListener("close", () => {
  inspectedSpecimen = null;
});
window.addEventListener(
  "scroll",
  () => {
    highlightSpecimen(null);
    clearSpecimenMotion();
  },
  { passive: true },
);
byId("specimen-workspace").addEventListener(
  "scroll",
  () => highlightSpecimen(null),
  { passive: true },
);
els.motionToggle.addEventListener("change", () => {
  motionChoice = els.motionToggle.value;
  localStorage.setItem("petsafe-motion-choice-v2", motionChoice);
  syncMotionControl();
});
reducedMotion.addEventListener("change", syncMotionControl);
syncMotionControl();
setInvestigationState("idle");
loadModels();
