import {
  createInvestigationState,
  largestSpecimenIds,
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
const museOnly = document.body.dataset.flow === "muse";

const els = {
  pageShell: document.querySelector(".page-shell"),
  workspace: document.querySelector(".workspace"),
  apiStatus: byId("api-status"),
  dropzone: byId("dropzone"),
  sampleGallery: byId("sample-gallery"),
  uploadEmpty: byId("upload-empty"),
  imageStage: byId("image-stage"),
  imageInput: byId("image-input"),
  chooseImage: byId("choose-image"),
  uploadTitle: byId("upload-title"),
  uploadSubtitle: byId("upload-subtitle"),
  preview: byId("image-preview"),
  backdrop: byId("image-backdrop"),
  scanField: byId("scan-field"),
  questionLayer: byId("question-layer"),
  scanMessage: byId("scan-message"),
  photoGuidance: byId("photo-guidance"),
  photoGuidanceHeading: byId("photo-guidance-heading"),
  photoGuidanceCount: byId("photo-guidance-count"),
  photoGuidanceInstructions: byId("photo-guidance-instructions"),
  photoGuidanceWhy: byId("photo-guidance-why"),
  photoGuidanceReason: byId("photo-guidance-reason"),
  photoGuidancePending: byId("photo-guidance-pending"),
  photoGuidanceButton: byId("photo-guidance-button"),
  photoGuidanceRestore: byId("photo-guidance-restore"),
  workspaceStepLabel: byId("workspace-step-label"),
  expandCanvas: byId("expand-canvas"),
  fullscreenCanvas: byId("fullscreen-canvas"),
  removeImage: byId("remove-image"),
  model: byId("model-select"),
  samMode: byId("sam-mode"),
  samPrompt: byId("sam-prompt"),
  compare: byId("compare-button"),
  rerunPhoto: byId("rerun-photo"),
  newPlant: byId("new-plant-button"),
  dialogNewPlant: byId("dialog-new-plant"),
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
let maxSeparatePlants = 8;
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
    if (museOnly) {
      const response = await fetch(apiUrl("/healthz"));
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setApiStatus("API connected", "online");
      return;
    }
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
      dinov3: "Meta DINOv3 ViT-S/16",
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
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.imageStage.classList.remove("acquired");
  els.preview.src = previewUrl;
  els.backdrop.src = previewUrl;
  els.backdrop.hidden = false;
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
  if (document.fullscreenElement === els.pageShell) document.exitFullscreen();
  setCanvasExpanded(false);
  archiveInvestigation();
  cancelInvestigation();
  resetReport();
  selectedFile = null;
  els.imageInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  els.preview.removeAttribute("src");
  els.backdrop.removeAttribute("src");
  els.backdrop.hidden = true;
  els.preview.hidden = true;
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
  syncInvestigationActions();
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

function animalStatusRow(lookup, onSelect = null, tagName = "div") {
  const row = element(tagName, "animal-statuses");
  for (const [animal, record] of [["Cats", lookup], ["Dogs", lookup?.dog_toxicity]]) {
    const status = record?.toxicity_status;
    const potential = record?.potential_hazard;
    const tone = status === "toxic" || potential ? "hazard"
      : status === "non_toxic" ? "listed" : "unknown";
    const label = status === "toxic" ? "listed toxic"
      : potential ? "potential hazard"
      : status === "non_toxic" ? "listed non-toxic" : "unresolved";
    const badge = element(onSelect ? "button" : "span",
      `animal-status ${tone}`, `${animal} · ${label}`);
    if (onSelect) {
      badge.type = "button";
      badge.addEventListener("click", (event) => onSelect(animal, badge, event));
    }
    row.append(badge);
  }
  return row;
}

function renderCacheNote(container, cacheHit) {
  if (cacheHit)
    container.append(
      element("span", "cache-note", "Cached result · this server process"),
    );
}

function renderToxicity(container, result, context, summaryLabel = null, splitAnimals = false) {
  const lookup = result?.toxicity;
  if (!lookup) {
    if (!splitAnimals) return;
    const cats = element("details", "evidence-details inspection-animal-evidence");
    cats.append(element("summary", "", "Cats · reference details"),
      element("p", "toxicity-note", "No matched cat toxicity reference for this identification."));
    const dogs = element("details", "evidence-details inspection-animal-evidence");
    dogs.append(element("summary", "", "Dogs · reference details"),
      element("p", "toxicity-note", "No matched dog toxicity reference for this identification."));
    container.append(cats, dogs);
    return { cats, dogs };
  }
  const details = element("details", "evidence-details");
  const status = (animal) => animal?.toxicity_status === "toxic" || animal?.potential_hazard
    ? "Potential hazard" : animal?.toxicity_status === "non_toxic" ? "Listed non-toxic" : "Unresolved";
  details.append(element("summary", "", summaryLabel ||
    `Pet toxicity · Cats: ${status(lookup)} · Dogs: ${status(lookup.dog_toxicity)}`));
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
  let dogBox = null;
  if (dog) {
    dogBox = element(
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
  if (splitAnimals) {
    const cats = element("details", "evidence-details inspection-animal-evidence");
    cats.append(element("summary", "", "Cats · symptoms and sources"), box);
    const dogs = element("details", "evidence-details inspection-animal-evidence");
    dogs.append(element("summary", "", "Dogs · symptoms and sources"));
    dogs.append(dogBox || element("p", "toxicity-note",
      "No matched dog toxicity reference for this identification."));
    container.append(cats, dogs);
    return { cats, dogs };
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

function renderTrained(result) {
  els.trainedResult.replaceChildren();
  const primary = element("div", "prediction-primary");
  primary.append(
    element("strong", "", result.predicted_class),
    element("span", "confidence-value", percent(result.confidence)),
  );
  els.trainedResult.append(primary);
  els.trainedResult.append(element("p", "confidence-caption",
    "Top class probability · not calibrated confidence"));
  els.trainedResult.append(animalStatusRow(result.toxicity));
  renderToxicity(els.trainedResult, result,
    "Reference for the classifier’s top class; the prediction may be wrong.");
  const classifierDetails = element("details", "result-details");
  classifierDetails.append(element("summary", "", "Top class probabilities"));
  renderCacheNote(classifierDetails, result.cache_hit);
  const scores = element("div", "score-list");
  for (const item of result.top_predictions || []) {
    const row = element("div", "score-row");
    const name = element("span", "score-name", item.class_name);
    const track = element("span", "score-track");
    const fill = element("span", "score-fill");
    fill.style.width = `${Math.max(0, Math.min(100, Number(item.probability) * 100))}%`;
    track.append(fill);
    row.append(name, track, element("span", "score-percent", percent(item.probability)));
    scores.append(row);
  }
  classifierDetails.append(scores);
  els.trainedResult.append(classifierDetails);
}

function renderSpark(result) {
  const openGroups = new Map([...els.sparkResult.querySelectorAll(".plant-group")].map((card) => [
    card.id,
    {
      open: card.open,
      cats: card.querySelector(".scene-cat-evidence")?.open,
      dogs: card.querySelector(".scene-dog-evidence")?.open,
    },
  ]));
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
    card.append(heading);
    if (group.visual_evidence?.length)
      card.append(
        element("p", "spark-evidence", group.visual_evidence.join(" · ")),
      );
    const evidence = renderToxicity(
      card,
      group,
      "Reference for this visual hypothesis; verify the plant identity.",
      null,
      true,
    );
    evidence.cats.classList.add("scene-cat-evidence");
    evidence.dogs.classList.add("scene-dog-evidence");
    const badges = animalStatusRow(group.toxicity, (animal, badge, event) => {
      event.preventDefault();
      event.stopPropagation();
      card.open = true;
      const selected = animal === "Cats" ? evidence.cats : evidence.dogs;
      const other = animal === "Cats" ? evidence.dogs : evidence.cats;
      other.open = false;
      selected.open = true;
      selected.scrollIntoView({ block: "nearest" });
    }, "span");
    for (const [badge, target] of [[badges.children[0], evidence.cats], [badges.children[1], evidence.dogs]]) {
      target.id = `scene-evidence-${index}-${target === evidence.cats ? "cats" : "dogs"}`;
      badge.setAttribute("aria-controls", target.id);
      badge.setAttribute("aria-expanded", String(target.open));
      target.addEventListener("toggle", () => badge.setAttribute("aria-expanded", String(target.open)));
    }
    heading.append(badges);
    const previous = openGroups.get(card.id);
    if (previous) {
      card.open = previous.open;
      evidence.cats.open = Boolean(previous.cats);
      evidence.dogs.open = Boolean(previous.dogs);
      badges.children[0].setAttribute("aria-expanded", String(evidence.cats.open));
      badges.children[1].setAttribute("aria-expanded", String(evidence.dogs.open));
    }
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
    link.href = "#photo-guidance";
    els.sparkResult.append(link);
  }
  const technical = element("details", "result-details technical-details");
  technical.append(element("summary", "", "Developer Inspector · model details"));
  technical.append(element("p", "spark-evidence",
    `Scene count · ${result.distinct_plant_count ?? "?"} distinct plant types · ${result.separate_plant_count ?? "missing"} separate whole plants. Closer-photo rule: more than ${maxSeparatePlants} separate plants.`));
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
  for (const [index, segment] of (result.segments || []).entries()) {
    const displayIndex = index + 1;
    const card = element("article", "segment-card");
    const imageWrap = element("div", "segment-image-wrap");
    const image = element("img", "segment-image");
    image.src = segment.crop_data_url;
    image.alt = `SAM specimen ${displayIndex}`;
    imageWrap.append(image);
    const body = element("div", "segment-body");
    body.append(
      element("p", "segment-label", `Specimen ${displayIndex}`),
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
    );
    if (!museOnly) tags.append(element(
      "span",
      `mini-tag${segment.in_trained_taxonomy ? "" : " outside"}`,
      segment.in_trained_taxonomy ? "in trained classes" : "outside trained classes",
    ));
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
  questionOrder = shuffledQuestionOrder();
  const request = ++requestGeneration;
  const controller = new AbortController();
  activeRequest = controller;
  resetReport();
  if (!museOnly) els.trainedResult.replaceChildren(
    element("p", "empty-state", "Awaiting the classifier result…"),
  );
  els.sparkResult.replaceChildren(
    element("p", "empty-state", "Awaiting the scene assessment…"),
  );
  els.results.hidden = true;
  els.segmentationPanel.hidden = true;
  clearError();
  setInvestigationState("uploading");
  try {
    const form = new FormData();
    form.append("image_file", selectedFile, selectedFile.name);
    if (!museOnly) form.append("model_name", els.model.value);
    form.append("force_sam", String(els.samMode.value === "force"));
    form.append("skip_sam", String(els.samMode.value === "skip"));
    form.append("sam_prompt", els.samPrompt.value);
    const response = await fetch(apiUrl(museOnly ? "/api/flows/muse-stream" : "/api/flows/compare-stream"), {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!response.ok) {
      if (museOnly && response.status === 405)
        throw new Error("The running API does not have the Muse investigation route yet. Recreate the API container with: docker compose up -d --force-recreate --no-deps api");
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
        const followupRequest = sceneAssessments.find((item) => item.another_photo_needed);
        if (followupRequest) updateFollowup(followupRequest);
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
      syncInvestigationActions();
    }
  }
}

function chooseNewPlant() {
  els.imageInput.value = "";
  els.imageInput.click();
}
els.chooseImage.addEventListener("click", () => els.imageInput.click());
els.sampleGallery?.addEventListener("click", async (event) => {
  const choice = event.target.closest("button[data-sample]");
  if (!choice || !els.sampleGallery.contains(choice)) return;
  const choices = els.sampleGallery.querySelectorAll("button[data-sample]");
  choices.forEach((button) => { button.disabled = true; });
  choice.classList.add("is-loading");
  clearError();
  try {
    const response = await fetch(choice.dataset.sample);
    if (!response.ok) throw new Error(`Sample image unavailable (${response.status}).`);
    const blob = await response.blob();
    const name = choice.dataset.sample.split("/").at(-1);
    setFile(new File([blob], name, { type: blob.type || "image/png" }));
  } catch (error) {
    showError(error.message || "Could not load the sample image.");
  } finally {
    choices.forEach((button) => { button.disabled = false; });
    choice.classList.remove("is-loading");
  }
});
els.dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  if (event.target.closest(".image-stage")) return;
  els.imageInput.click();
});
els.imageInput.addEventListener("change", () =>
  setFile(els.imageInput.files[0]),
);
els.removeImage.addEventListener("click", clearFile);
els.compare.addEventListener("click", () => {
  if (investigationState === "completed") chooseNewPlant();
  else compare();
});
els.rerunPhoto.addEventListener("click", compare);
els.newPlant.addEventListener("click", chooseNewPlant);
els.photoGuidanceButton.addEventListener("click", chooseNewPlant);
byId("photo-guidance-minimize").addEventListener("click", () => {
  followupMinimized = true;
  els.photoGuidance.hidden = true;
  els.photoGuidanceRestore.hidden = false;
  syncFollowupProgress();
  els.photoGuidanceRestore.focus();
});
els.photoGuidanceRestore.addEventListener("click", () => {
  followupMinimized = false;
  summaryPanelOpen = false;
  els.photoGuidanceRestore.hidden = true;
  els.photoGuidance.hidden = false;
  syncFollowupProgress();
  byId("photo-guidance-minimize").focus();
});
byId("photo-guidance-summary").addEventListener("click", showSummaryPanel);
byId("scene-summary-action").addEventListener("click", showSummaryPanel);
byId("scene-card-restore").addEventListener("click", showSummaryPanel);
els.dialogNewPlant.addEventListener("click", () => {
  byId("specimen-dialog").close();
  chooseNewPlant();
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
function refreshCanvasGeometry() {
  if (els.dropzone.classList.contains("is-scanning")) {
    stopScanAnimation();
    startScanAnimation();
  }
  highlightSpecimen(null);
  clearSpecimenMotion();
  placeSpecimens();
  positionSceneCardLink();
}
function setCanvasExpanded(expanded) {
  if (els.workspace.classList.contains("is-expanded") === expanded) return;
  els.workspace.classList.toggle("is-expanded", expanded);
  els.expandCanvas.setAttribute("aria-pressed", String(expanded));
  els.expandCanvas.textContent = expanded ? "Restore canvas" : "Expand canvas";
  if (expanded) els.workspace.scrollIntoView({ block: "start" });
  requestAnimationFrame(refreshCanvasGeometry);
}
els.expandCanvas.addEventListener("click", () =>
  setCanvasExpanded(!els.workspace.classList.contains("is-expanded")));
els.fullscreenCanvas.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement === els.pageShell) await document.exitFullscreen();
    else await els.pageShell.requestFullscreen();
  } catch (error) {
    showError(`Could not enter full screen: ${error.message}`);
  }
});
document.addEventListener("fullscreenchange", () => {
  const active = document.fullscreenElement === els.pageShell;
  els.fullscreenCanvas.setAttribute("aria-pressed", String(active));
  els.fullscreenCanvas.textContent = active ? "Exit full screen" : "Full screen";
  requestAnimationFrame(refreshCanvasGeometry);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && els.workspace.classList.contains("is-expanded") &&
      !byId("specimen-dialog").open) setCanvasExpanded(false);
});
window.addEventListener("resize", refreshCanvasGeometry);

// Pipeline states are entered only from upload actions or server events.
const machine = createInvestigationState();
let investigationState = machine.state;
let activeRequest = null;
let requestGeneration = 0;
let questionTimer = null;
let questionIndex = 0;
const toxicityOpinionRequests = new Set();
let specimenPreviewTimer = null;
let sceneAssessments = [];
let summaryPanelOpen = false;
let sceneCardMinimized = false;
let followupMinimized = false;
let followupAnnounced = false;
let samResponded = false;
let selectedRegionCount = 0;
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
function shuffledQuestionOrder(previousLast) {
  const order = [...questions];
  for (let index = order.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [order[index], order[randomIndex]] = [order[randomIndex], order[index]];
  }
  if (order.length > 1 && order[0] === previousLast)
    [order[0], order[1]] = [order[1], order[0]];
  return order;
}
let questionOrder = [...questions];
const thoughtPositions = ["thought-one", "thought-two", "thought-three"];
const guidedThoughtPositions = ["thought-guided-one", "thought-guided-two", "thought-guided-three"];
const QUESTION_INTERVAL_MS = 4000;
const GUIDED_QUESTION_INTERVAL_MS = 5200;
let thoughtSerial = 0;
let questionMode = null;
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
  const guided = questionMode === "guided";
  const positions = guided
    ? guidedThoughtPositions.slice(0, window.matchMedia("(max-width: 600px)").matches ? 2 : 3)
    : thoughtPositions;
  const position = thoughtSerial % positions.length;
  const old = els.questionLayer.querySelector(`[data-position="${position}"]`);
  if (old) {
    if (guided) old.remove();
    else animateThought(old, false).finally(() => old.remove());
  }
  const bubble = element("span", `thought-bubble ${positions[position]}`, questionOrder[questionIndex]);
  bubble.dataset.position = String(position);
  bubble.style.opacity = motionDisabled() ? "1" : "0";
  els.questionLayer.append(bubble);
  animateThought(bubble, true);
  thoughtSerial += 1;
  questionIndex += 1;
  if (questionIndex === questionOrder.length) {
    questionOrder = shuffledQuestionOrder(questionOrder.at(-1));
    questionIndex = 0;
  }
}
function clearThoughts() {
  clearInterval(questionTimer);
  questionTimer = null;
  els.questionLayer.replaceChildren();
  thoughtSerial = 0;
  questionMode = null;
}
function syncInvestigationActions() {
  const busy = machine.busy;
  els.compare.disabled = busy || !selectedFile;
  els.compare.classList.toggle("is-loading", busy);
  els.rerunPhoto.disabled = busy || !selectedFile;
  els.expandCanvas.disabled = !selectedFile;
  els.fullscreenCanvas.disabled = !selectedFile || !document.fullscreenEnabled;
  els.newPlant.hidden = investigationState !== "completed";
  els.dialogNewPlant.hidden = investigationState !== "completed";
  els.compare.querySelector(".button-label").textContent = busy
    ? "Investigating…"
    : investigationState === "completed" ? "Investigate another plant"
      : investigationState === "error" ? "Retry this photo"
        : selectedFile ? "Investigate this photo" : "Begin investigation";
}
const followupProgressNote = "More findings from this photo may help with your next shot.";
function syncFollowupProgress() {
  const ongoing = machine.busy && investigationState !== "presenting";
  els.photoGuidancePending.hidden = !ongoing || els.photoGuidance.hidden;
  els.photoGuidanceRestore.textContent = ongoing
    ? "Next photo · scan continues ↗" : "Next photo requested ↗";
  if (ongoing && !els.photoGuidance.hidden &&
      !els.progressDetail.textContent.includes(followupProgressNote))
    els.progressDetail.textContent = [els.progressDetail.textContent, followupProgressNote]
      .filter(Boolean).join(" ");
}
function syncRegionProgress() {
  const panel = byId("region-progress");
  const visible = machine.busy && sceneAssessments.length > 0 &&
    !summaryPanelOpen && els.photoGuidance.hidden && els.photoGuidanceRestore.hidden;
  panel.hidden = !visible;
  if (!visible) return;
  const selected = Math.max(selectedRegionCount, specimenRecords.size);
  const assessed = [...specimenRecords.values()].filter((item) => item.data?.assessment).length;
  byId("region-progress-title").textContent = investigationState === "presenting"
    ? "Preparing the final findings"
    : assessed
    ? `${assessed} of ${selected} specimens checked`
    : specimenRecords.size
      ? `${specimenRecords.size} specimen${specimenRecords.size === 1 ? "" : "s"} found`
      : investigationState === "segmenting" ? "Isolating plant regions" : "Scene assessment received";
  byId("region-progress-detail").textContent = investigationState === "presenting"
    ? "All available checks have returned. The investigation report is being prepared."
    : assessed
    ? "Muse Spark results are arriving for selected regions. The report will open when all checks finish."
    : investigationState === "segmenting"
      ? "SAM is finding useful regions. The report will open after regional checks finish."
      : investigationState === "segments_ready"
        ? "Regions are selected. Muse Spark will assess them next."
        : investigationState === "identifying_regions"
          ? "Muse Spark is checking the selected regions. The report will open when checks finish."
          : "Checking whether this photo needs closer inspection. The report will open when checks finish.";
}
function syncEarlyHazard() {
  const potentialHazard = sceneAssessments.some((assessment) =>
    (assessment.plant_groups || []).some(inspectionHazard));
  for (const id of ["region-progress-hazard", "photo-guidance-hazard"]) {
    const warning = byId(id);
    warning.hidden = !potentialHazard || investigationState === "completed";
    warning.textContent = potentialHazard
      ? "Possible pet hazard reported. Keep pets away until the plant identity is verified."
      : "";
  }
}
function syncFollowupSummaryAction() {
  const available = followupAnnounced &&
    (samResponded || investigationState === "completed");
  const button = byId("photo-guidance-summary");
  button.disabled = !available;
  button.textContent = available
    ? "View toxicity summary ↗"
    : "Summary available when scan is ready";
}
function setInvestigationState(state, keepCentered = false) {
  if (!Object.hasOwn(stateTitles, state)) return;
  machine.transition(state);
  investigationState = machine.state;
  document.body.dataset.investigation = state;
  centerPhotoWorkspace(state, keepCentered);
  const needsAnotherPhoto = sceneAssessments.some((item) => item.another_photo_needed);
  els.workspaceStepLabel.textContent = state === "idle"
    ? "ADD A PHOTO"
    : state === "completed" ? (needsAnotherPhoto
      ? "NEXT PHOTO NEEDED" : "INVESTIGATION CANVAS · COMPLETE") : "LIVE INVESTIGATION";
  els.scanMessage.textContent = state === "completed" && needsAnotherPhoto
    ? "CLOSER PHOTO REQUESTED" : stateTitles[state].toUpperCase();
  els.progressTitle.textContent = stateTitles[state];
  const busy = machine.busy;
  byId("scene-summary-action").disabled = state !== "completed";
  syncFollowupSummaryAction();
  els.scanProgress.hidden = !busy;
  syncInvestigationActions();
  if (els.model) els.model.disabled = busy;
  els.samMode.disabled = busy;
  els.samPrompt.disabled = busy;
  // Search circles appear immediately; no model result is implied by their positions.
  if (busy) startScanAnimation();
  else stopScanAnimation();
  els.progressDetail.textContent = {
    uploading: "Transferring your image. Visual search has begun; model findings will appear when returned.",
    analyzing_scene: "Looking for visual evidence in the photograph.",
    scene_analyzed: museOnly
      ? "Scene assessment received. Checking whether regions need inspection."
      : "Scene assessment received. The comparison is continuing.",
    segmenting: "SAM is finding masks. Regions will appear when they arrive.",
    segments_ready: "These are actual selected SAM crops.",
    identifying_regions: "Each specimen receives its own visual assessment and local reference lookup.",
    presenting: "The analysis is complete. Arranging your report.",
  }[state] || "";
  byId("followup-panel").hidden = state !== "completed" || !needsAnotherPhoto;
  syncFollowupProgress();
  syncRegionProgress();
  syncEarlyHazard();
  const showQuestions = busy && !["segments_ready", "identifying_regions", "presenting"].includes(state)
    && !els.workspace.classList.contains("has-specimens") && els.photoGuidance.hidden;
  if (showQuestions) {
    const nextMode = byId("region-progress").hidden ? "free" : "guided";
    if (!questionTimer || questionMode !== nextMode) {
      clearThoughts();
      questionMode = nextMode;
      showNextThought();
      if (nextMode === "guided") showNextThought();
      questionTimer = setInterval(showNextThought,
        nextMode === "guided" ? GUIDED_QUESTION_INTERVAL_MS : QUESTION_INTERVAL_MS);
    }
  } else {
    clearThoughts();
  }
}
function cancelInvestigation(keepCentered = false) {
  toxicityOpinionRequests.forEach((controller) => controller.abort());
  toxicityOpinionRequests.clear();
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
  resetOverview();
  sceneAssessments = [];
  summaryPanelOpen = false;
  sceneCardMinimized = false;
  followupMinimized = false;
  followupAnnounced = false;
  samResponded = false;
  selectedRegionCount = 0;
  byId("region-progress").hidden = true;
  byId("scene-card").hidden = true;
  byId("scene-card-restore").hidden = true;
  byId("scene-card-link").classList.remove("visible");
  lastReport = null;
  specimenRecords.clear();
  byId("specimen-rail").replaceChildren();
  document.querySelector(".workspace").classList.remove("has-specimens");
  for (const id of ["specimen-workspace", "followup-panel", "safety-summary", "photo-guidance"])
    byId(id).hidden = true;
  els.photoGuidanceRestore.hidden = true;
  byId("scene-summary-action").disabled = true;
  els.photoGuidanceWhy.open = false;
  els.photoGuidancePending.hidden = true;
  els.results.hidden = true;
  byId("specimen-dialog").close();
}
function handlePipelineEvent(event, data) {
  if (event === "error") throw new Error(data.message);
  if (event === "completed")
    els.results.hidden = false;
  if (event === "analyzing_scene")
    sourceDimensions = { width: data.image_width, height: data.image_height };
  if (event === "scene_analyzed") {
    if (!Number.isInteger(data.separate_plant_count))
      throw new Error("The running API uses an older Muse Spark response without a separate-plant count. Restart the API with: docker compose restart api, then retry this photo.");
    if (!Number.isInteger(data.max_separate_plants) || data.max_separate_plants < 1)
      throw new Error("The running API does not report the configured separate-plant limit. Restart the API with: docker compose up -d --force-recreate --no-deps api, then retry this photo.");
    maxSeparatePlants = data.max_separate_plants;
    renderSpark(data);
    sceneAssessments = [data];
    updateSafety();
    updateFollowup(data);
  }
  if (event === "trained_ready" && !museOnly)
    renderTrained(data);
  if (event === "specimen_ready") {
    selectedRegionCount = Math.max(selectedRegionCount, data.selected_count || 0);
    clearThoughts();
    document.querySelector(".workspace").classList.add("has-specimens");
    byId("specimen-workspace").hidden = false;
    byId("specimen-count").textContent =
      `${data.index} of ${data.selected_count} selected / ${data.mask_count} masks`;
    addSpecimen(data.region);
  }
  if (event === "segments_ready") {
    samResponded = true;
    selectedRegionCount = data.regions.length;
    clearThoughts();
    document
      .querySelector(".workspace")
      .classList.toggle("has-specimens", data.regions.length > 0);
    byId("specimen-workspace").hidden = !data.regions.length;
    data.regions.forEach(addSpecimen);
    const featuredIds = largestSpecimenIds(data.regions);
    featuredIds.forEach((id) => reopenSpecimenCard(id, { focus: false }));
    updateSpecimenCardCount();
    if (featuredIds.length) previewSpecimens(featuredIds);
  }
  if (event === "region_assessed") {
    const item = specimenRecords.get(String(data.object_id));
    if (item) {
      item.data = data;
      const groups = data.assessment.plant_groups || [];
      const primary = groups.find((group) =>
        group.plant_identification && !/^(unknown|unidentified|uncertain)/i.test(group.plant_identification)) || groups[0];
      const additional = Math.max(0, groups.length - 1);
      item.caption.textContent = primary?.plant_identification || "Unresolved specimen";
      item.taxon.textContent = primary?.scientific_name &&
        !/^(unknown|unidentified|uncertain)/i.test(primary.scientific_name)
        ? primary.scientific_name : "";
      item.button.classList.add("identified");
      const hazard = groups.some(
        (group) =>
          group.toxicity?.toxicity_status === "toxic" ||
          group.toxicity?.potential_hazard ||
          group.toxicity?.dog_toxicity?.toxicity_status === "toxic" ||
          group.toxicity?.dog_toxicity?.potential_hazard,
      );
      const primaryHazard = primary?.toxicity?.toxicity_status === "toxic" ||
        primary?.toxicity?.potential_hazard ||
        primary?.toxicity?.dog_toxicity?.toxicity_status === "toxic" ||
        primary?.toxicity?.dog_toxicity?.potential_hazard;
      item.button.classList.toggle("hazard", hazard);
      item.reopenButton.classList.toggle("hazard", hazard);
      item.reopenButton.setAttribute("aria-label",
        `Open specimen ${item.displayIndex}: ${item.caption.textContent}. ${hazard ? "Potential pet hazard." : "Regional assessment available."}`);
      if (hazard && item.closed && !item.userClosed)
        reopenSpecimenCard(String(data.object_id), { focus: false });
      const nonToxic =
        groups.length &&
        groups.every(
          (group) => group.toxicity?.toxicity_status === "non_toxic" &&
            group.toxicity?.dog_toxicity?.toxicity_status === "non_toxic",
        );
      item.button.classList.toggle("non-toxic", Boolean(nonToxic));
      const status = hazard
        ? (primaryHazard ? "Potential hazard" : "Hazard among groups")
        : nonToxic
          ? "Listed non-toxic"
          : "Unresolved · inspect";
      item.status.textContent = additional ? `${status} · +${additional} groups` : status;
      item.button.setAttribute("aria-label",
        `Inspect specimen ${item.displayIndex}: ${item.caption.textContent}. ${item.status.textContent}.`);
      placeSpecimens();
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
    if (!museOnly) renderTrained(data.trained_flow);
    if (!sceneAssessments.length) renderSpark(data.muse_spark_flow);
    renderSegmentation(data.segmentation);
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
    const followupRequest = sceneAssessments.find((item) => item.another_photo_needed);
    if (followupRequest) updateFollowup(followupRequest);
  }
  if (Object.hasOwn(stateTitles, event)) setInvestigationState(event);
  if (event === "completed" && (summaryPanelOpen ||
      els.photoGuidance.hidden && els.photoGuidanceRestore.hidden))
    showSceneCard(data.muse_spark_flow);
  if (event === "segmenting" && data.prompt)
    els.progressDetail.textContent =
      `SAM is isolating ${data.prompt === "plant" ? "whole plants" : "flower heads"}. Regions will appear when they arrive.`;
  syncFollowupProgress();
  syncRegionProgress();
  syncEarlyHazard();
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
  const groups = [...(assessment.plant_groups || [])];
  const hazards = groups.filter(inspectionHazard).length;
  const isUnresolved = (group) => !group.toxicity ||
    group.toxicity.toxicity_status === "unknown" ||
    !group.toxicity.dog_toxicity ||
    group.toxicity.dog_toxicity.toxicity_status === "unknown" ||
    group.identification_confidence === "low" || group.needs_inspection;
  const unresolved = groups.filter(isUnresolved).length;
  card.dataset.tone = hazards ? "hazard" : unresolved ? "unknown" : "reference";
  const list = element("ul", "scene-card-list");
  list.tabIndex = 0;
  list.setAttribute("aria-label", "Reported plant types and cat and dog toxicity statuses");
  groups.sort((a, b) => Number(inspectionHazard(b)) - Number(inspectionHazard(a)) ||
    Number(isUnresolved(b)) - Number(isUnresolved(a)));
  for (const group of groups) {
    const row = element("li", `scene-card-row${inspectionHazard(group) ? " hazard" : isUnresolved(group) ? " unknown" : ""}`);
    const identity = element("div", "scene-card-identity");
    identity.append(
      element("span", "scene-card-name", group.plant_identification || "Unidentified plant"),
      element("span", "scene-card-taxon", group.scientific_name || "Scientific identity unresolved"),
    );
    row.append(identity, animalStatusRow(group.toxicity));
    list.append(row);
  }
  if (!groups.length)
    list.append(element("li", "scene-card-empty", "No plant type identified; cat and dog toxicity remain unresolved."));
  const actions = element("div", "scene-card-actions");
  const minimize = element("button", "scene-card-minimize", "Minimize −");
  minimize.type = "button";
  minimize.setAttribute("aria-label", "Minimize whole-photo toxicity summary");
  minimize.addEventListener("click", () => {
    sceneCardMinimized = true;
    card.hidden = true;
    byId("scene-card-link").classList.remove("visible");
    byId("scene-card-restore").hidden = false;
    byId("scene-card-restore").focus();
  });
  const details = element("button", "scene-card-details", "See more details ↗");
  details.type = "button";
  details.addEventListener("click", () => showInspection(assessment, els.preview.src,
    "Original plant photograph", "WHOLE PHOTO / MUSE SPARK"));
  actions.append(details);
  const hasFollowup = sceneAssessments.some((item) => item.another_photo_needed);
  if (hasFollowup || !byId("specimen-workspace").hidden && machine.busy) {
    const back = element("button", "scene-card-back", hasFollowup ? "← Next photo" : "← Continue inspection");
    back.type = "button";
    back.addEventListener("click", () => {
      summaryPanelOpen = false;
      sceneCardMinimized = false;
      card.hidden = true;
      byId("scene-card-restore").hidden = true;
      byId("scene-card-link").classList.remove("visible");
      els.photoGuidance.hidden = followupMinimized;
      els.photoGuidanceRestore.hidden = !hasFollowup || !followupMinimized;
      if (!hasFollowup) els.photoGuidance.hidden = true;
      syncFollowupProgress();
      syncRegionProgress();
    });
    actions.prepend(back);
  }
  card.replaceChildren(
    element("span", "eyebrow", "WHOLE PHOTO / TOXICITY SUMMARY"),
    minimize,
    element("strong", "", hazards
      ? `${hazards} potential hazard match${hazards === 1 ? "" : "es"}`
      : unresolved ? `${unresolved} unresolved plant match${unresolved === 1 ? "" : "es"}`
        : groups.length ? "Cat and dog references ready" : "No plant identified"),
    element("small", "", `${groups.length} reported plant type${groups.length === 1 ? "" : "s"} · ${unresolved} unresolved · ${machine.busy ? "whole-photo findings; regional checks continue" : "visual matches only"}`),
    list,
    actions,
  );
  card.hidden = sceneCardMinimized;
  byId("scene-card-restore").hidden = !sceneCardMinimized;
  if (sceneCardMinimized) byId("scene-card-link").classList.remove("visible");
  else requestAnimationFrame(positionSceneCardLink);
}
function showSummaryPanel() {
  if (investigationState !== "completed" && !(followupAnnounced && samResponded)) return;
  const assessment = lastReport?.muse_spark_flow || sceneAssessments[0];
  if (!assessment) return;
  summaryPanelOpen = true;
  sceneCardMinimized = false;
  els.photoGuidance.hidden = true;
  els.photoGuidanceRestore.hidden = true;
  byId("region-progress").hidden = true;
  syncFollowupProgress();
  showSceneCard(assessment);
}
function clampSpecimen(button, left, top) {
  const width = els.imageStage.clientWidth;
  const height = els.imageStage.clientHeight;
  return {
    left: Math.max(5, Math.min(left, width - button.offsetWidth - 5)),
    top: Math.max(5, Math.min(top, height - button.offsetHeight - 5)),
  };
}
function positionSpecimenControls(item) {
  const card = item.button;
  const left = parseFloat(card.style.left) || 0;
  const top = parseFloat(card.style.top) || 0;
  item.closeButton.style.left = `${left + card.offsetWidth - 29}px`;
  item.closeButton.style.top = `${top + 7}px`;
  const box = sourceBox(item.data.bounds);
  item.reopenButton.style.left = `${box.left + box.width / 2}px`;
  item.reopenButton.style.top = `${box.top + box.height / 2}px`;
}
function placeSpecimens() {
  if (!els.imageStage.clientWidth) return;
  const stageWidth = els.imageStage.clientWidth;
  const stageHeight = els.imageStage.clientHeight;
  els.imageStage.classList.toggle(
    "compact-specimens",
    els.imageStage.clientHeight < 560,
  );
  const placed = [];
  [...specimenRecords.values()].forEach((item, index) => {
    if (item.closed) {
      positionSpecimenControls(item);
      return;
    }
    if (item.manual) {
      const position = clampSpecimen(item.button,
        parseFloat(item.button.style.left) || 0,
        parseFloat(item.button.style.top) || 0);
      item.button.style.left = `${position.left}px`;
      item.button.style.top = `${position.top}px`;
      positionSpecimenControls(item);
      placed.push({ ...position, width: item.button.offsetWidth, height: item.button.offsetHeight });
      return;
    }
    if (item.positioned && item.stageWidth === stageWidth &&
        item.stageHeight === stageHeight) {
      const position = clampSpecimen(item.button,
        parseFloat(item.button.style.left) || 0,
        parseFloat(item.button.style.top) || 0);
      placed.push({ ...position, width: item.button.offsetWidth,
        height: item.button.offsetHeight });
      positionSpecimenControls(item);
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
    for (let top = 5; top <= stageHeight - height - 5; top += Math.max(48, height / 2)) {
      for (let left = 5; left <= stageWidth - width - 5; left += Math.max(48, width / 2))
        candidates.push([left, top]);
    }
    const overlap = (a, b) => Math.max(0, Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left))
      * Math.max(0, Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top));
    const scored = candidates.map(([left, top]) => {
      const p = clampSpecimen(item.button, left, top);
      const rect = { ...p, width, height };
      const distance = Math.hypot(p.left + width / 2 - centerX, p.top + height / 2 - centerY);
      const cardOverlap = placed.reduce((sum, other) => sum + overlap(rect, other), 0);
      return { ...p, cardOverlap, score: distance + overlap(rect, box) * 4 + cardOverlap * 1000 };
    });
    const clear = scored.filter((candidate) => candidate.cardOverlap === 0);
    const position = (clear.length ? clear : scored).sort((a, b) => a.score - b.score)[0];
    item.button.style.left = `${position.left}px`;
    item.button.style.top = `${position.top}px`;
    item.positioned = true;
    item.stageWidth = stageWidth;
    item.stageHeight = stageHeight;
    positionSpecimenControls(item);
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
    positionSpecimenControls(item);
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
    positionSpecimenControls(item);
    highlightSpecimen(id);
  });
}
function updateSpecimenCardCount() {
  if (!samResponded) return;
  const open = [...specimenRecords.values()].filter((item) => !item.closed).length;
  const marked = specimenRecords.size - open;
  byId("specimen-count").textContent = marked
    ? `${open} open · click ${marked} markers`
    : `${open} open`;
}
function closeSpecimenCard(id) {
  const item = specimenRecords.get(id);
  if (!item) return;
  clearInterval(specimenPreviewTimer);
  specimenPreviewTimer = null;
  item.closed = true;
  item.userClosed = true;
  item.button.hidden = true;
  item.closeButton.hidden = true;
  item.reopenButton.hidden = false;
  highlightSpecimen(null);
  placeSpecimens();
  updateSpecimenCardCount();
  item.reopenButton.focus({ preventScroll: true });
}
function reopenSpecimenCard(id, { focus = true } = {}) {
  const item = specimenRecords.get(id);
  if (!item?.closed) return;
  item.closed = false;
  if (focus) item.userClosed = false;
  item.manual = false;
  item.positioned = false;
  item.button.hidden = false;
  item.closeButton.hidden = false;
  item.reopenButton.hidden = true;
  placeSpecimens();
  updateSpecimenCardCount();
  if (focus) {
    highlightSpecimen(id);
    item.button.focus();
  }
}
function addSpecimen(data) {
  const id = String(data.object_id);
  if (specimenRecords.has(id)) return;
  const displayIndex = specimenRecords.size + 1;
  const button = element("button", "floating-specimen");
  button.type = "button";
  button.id = `specimen-${id}`;
  button.style.setProperty("--float-delay", `${-specimenRecords.size * 1.3}s`);
  const image = element("img", "specimen-image");
  image.src = data.crop_data_url;
  image.alt = `SAM specimen ${displayIndex}`;
  image.draggable = false;
  const caption = element("strong", "", `Specimen ${displayIndex}`);
  const taxon = element("span", "specimen-taxon");
  const status = element("small", "", "Awaiting regional assessment");
  button.append(
    element("span", "specimen-id", `SPECIMEN / ${displayIndex}`),
    image,
    caption,
    taxon,
    status,
  );
  const closeButton = element("button", "specimen-close", "×");
  closeButton.type = "button";
  closeButton.setAttribute("aria-label", `Close specimen ${displayIndex} card`);
  const reopenButton = element("button", "specimen-reopen", `+${displayIndex}`);
  reopenButton.type = "button";
  reopenButton.hidden = false;
  reopenButton.setAttribute("aria-label", `Reopen specimen ${displayIndex} card`);
  byId("specimen-rail").append(button, closeButton, reopenButton);
  const item = { data, displayIndex, button, closeButton, reopenButton, image, caption, taxon,
    status, manual: false, closed: true, userClosed: false, suppressClick: false };
  button.hidden = true;
  closeButton.hidden = true;
  closeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    closeSpecimenCard(id);
  });
  reopenButton.addEventListener("click", (event) => {
    event.stopPropagation();
    reopenSpecimenCard(id);
  });
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
      if (motionDisabled() || !image.isConnected || item.closed) return;
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
  if (item.closed) return;
  const stage = els.imageStage.getBoundingClientRect();
  const ratioX = stage.width / els.imageStage.clientWidth;
  const ratioY = stage.height / els.imageStage.clientHeight;
  const target = item.button.getBoundingClientRect();
  connection.setAttribute("viewBox", `0 0 ${els.imageStage.clientWidth} ${els.imageStage.clientHeight}`);
  connection.setAttribute("preserveAspectRatio", "none");
  const startX = box.left + box.width / 2;
  const startY = box.top + box.height / 2;
  const endX = (target.left + target.width / 2 - stage.left) / ratioX;
  const endY = (target.top + 20 - stage.top) / ratioY;
  connection.querySelectorAll("line").forEach((line) => {
    line.setAttribute("x1", startX);
    line.setAttribute("y1", startY);
    line.setAttribute("x2", endX);
    line.setAttribute("y2", endY);
  });
  const origin = connection.querySelector(".connection-origin");
  origin.setAttribute("cx", startX);
  origin.setAttribute("cy", startY);
  const end = connection.querySelector(".connection-end");
  end.setAttribute("cx", endX);
  end.setAttribute("cy", endY);
  connection.style.display = "block";
}
function specimenAtPhotoPoint(event, closedOnly = false) {
  const stage = els.imageStage.getBoundingClientRect();
  const x = (event.clientX - stage.left) * els.imageStage.clientWidth / stage.width;
  const y = (event.clientY - stage.top) * els.imageStage.clientHeight / stage.height;
  const matches = [...specimenRecords].filter(([, item]) => {
    if (closedOnly && !item.closed) return false;
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
  return matches[0]?.[0] || null;
}
function hoverPhotoSpecimen(event) {
  if (event.pointerType === "touch" || els.imageStage.querySelector(".floating-specimen.is-dragging")) return;
  if (event.target.closest(".floating-specimen, .specimen-close, .specimen-reopen, .scene-card, .remove-button")) return;
  const id = specimenAtPhotoPoint(event);
  if (id && specimenRecords.get(id).button.classList.contains("is-linked")) return;
  if (!id && !els.imageStage.querySelector(".floating-specimen.is-linked")) return;
  clearInterval(specimenPreviewTimer);
  specimenPreviewTimer = null;
  highlightSpecimen(id);
}
els.imageStage.addEventListener("pointermove", hoverPhotoSpecimen);
els.imageStage.addEventListener("click", (event) => {
  if (event.target.closest(".floating-specimen, .specimen-close, .specimen-reopen, .scene-card, .remove-button")) return;
  const id = specimenAtPhotoPoint(event, true);
  if (id) reopenSpecimenCard(id);
});
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
function inspectionHazard(group) {
  const lookup = group.toxicity;
  return lookup?.toxicity_status === "toxic" || lookup?.potential_hazard ||
    lookup?.dog_toxicity?.toxicity_status === "toxic" || lookup?.dog_toxicity?.potential_hazard;
}
function addToxicityOpinion(details, group, animal, imageUrl) {
  const wrapper = element("div", "toxicity-opinion");
  const button = element("button", "toxicity-opinion-button", `Ask Muse Spark about ${animal} ↗`);
  button.type = "button";
  const answer = element("div", "toxicity-opinion-answer");
  answer.hidden = true;
  wrapper.append(button, element("p", "toxicity-opinion-disclaimer",
    "AI opinion only. This response is not verified against a toxicity database or source references and cannot establish pet safety."), answer);
  button.addEventListener("click", async () => {
    const controller = new AbortController();
    toxicityOpinionRequests.add(controller);
    button.disabled = true;
    button.textContent = "Asking Muse Spark…";
    answer.hidden = false;
    answer.replaceChildren(element("p", "", "Checking for a possible concern…"));
    try {
      const image = await fetch(imageUrl, { signal: controller.signal }).then((response) => response.blob());
      const form = new FormData();
      form.append("image_file", image, "inspection-image.jpg");
      form.append("plant_name", (group.scientific_name || group.plant_identification || "Unidentified plant").slice(0, 160));
      form.append("animal", animal);
      const response = await fetch(apiUrl("/api/flows/muse-toxicity-opinion"), {
        method: "POST", body: form, signal: controller.signal,
      });
      const result = await response.json();
      if ([404, 405].includes(response.status)) throw new Error("Restart the API to enable this question.");
      if (!response.ok) throw new Error(result.detail || `API returned ${response.status}`);
      answer.replaceChildren(
        element("strong", "", result.risk_signal === "possible_concern" ? "Possible concern · unverified" : "Unknown · unverified"),
        element("p", "", result.explanation),
        element("p", "toxicity-opinion-uncertainty", result.uncertainty),
      );
      button.textContent = `Ask Muse Spark about ${animal} again ↗`;
    } catch (error) {
      if (error.name === "AbortError") return;
      answer.replaceChildren(element("p", "toxicity-opinion-error", `Could not get an AI opinion: ${error.message}`));
      button.textContent = `Retry Muse Spark for ${animal} ↗`;
    } finally {
      toxicityOpinionRequests.delete(controller);
      button.disabled = false;
    }
  });
  details.append(wrapper);
}
function renderInspectionGroup(group, primary = false, tieCount = 1, imageUrl = "") {
  const card = element("section", primary ? "inspection-primary" : "inspection-group");
  if (primary) card.append(element("p", "inspection-kicker",
    `HIGHEST REPORTED CONFIDENCE · ${group.identification_confidence.toUpperCase()}${tieCount > 1 ? ` · TIED WITH ${tieCount - 1} OTHER GROUP${tieCount === 2 ? "" : "S"}` : ""}`));
  card.append(
    element("h2", "", group.plant_identification || "Unresolved plant"),
    element("p", "inspection-taxon", group.scientific_name || "Scientific identity unresolved"),
  );
  if (!primary) card.append(element("p", "inspection-confidence",
    `${group.identification_confidence} identification confidence`));
  const evidence = renderToxicity(card, group,
    "Reference applies only if this visual identification is correct.",
    null, true);
  if (!group.toxicity || group.toxicity.toxicity_status === "unknown")
    addToxicityOpinion(evidence.cats, group, "cats", imageUrl);
  if (!group.toxicity?.dog_toxicity || group.toxicity.dog_toxicity.toxicity_status === "unknown")
    addToxicityOpinion(evidence.dogs, group, "dogs", imageUrl);
  const badges = animalStatusRow(group.toxicity, (animal) => {
    const target = animal === "Cats" ? evidence.cats : evidence.dogs;
    const other = animal === "Cats" ? evidence.dogs : evidence.cats;
    other.open = false;
    target.open = true;
    target.scrollIntoView({ block: "nearest" });
  });
  const both = element("button", "inspection-both-toggle", "Show both cat and dog details");
  both.type = "button";
  const syncBoth = () => {
    const open = evidence.cats.open && evidence.dogs.open;
    both.textContent = open ? "Hide both cat and dog details" : "Show both cat and dog details";
    both.setAttribute("aria-expanded", String(open));
  };
  for (const [index, target] of [evidence.cats, evidence.dogs].entries()) {
    const badge = badges.children[index];
    target.id = `inspection-evidence-${++inspectionEvidenceSerial}`;
    badge.setAttribute("aria-controls", target.id);
    target.addEventListener("toggle", () => {
      badge.setAttribute("aria-expanded", String(target.open));
      syncBoth();
    });
    badge.setAttribute("aria-expanded", "false");
  }
  both.setAttribute("aria-controls", `${evidence.cats.id} ${evidence.dogs.id}`);
  both.setAttribute("aria-expanded", "false");
  both.addEventListener("click", () => {
    const open = !(evidence.cats.open && evidence.dogs.open);
    evidence.cats.open = open;
    evidence.dogs.open = open;
    syncBoth();
    if (open) evidence.cats.scrollIntoView({ block: "nearest" });
  });
  card.insertBefore(badges, evidence.cats);
  card.insertBefore(both, evidence.cats);
  return card;
}
let inspectionEvidenceSerial = 0;
function showInspection(assessment, imageUrl, imageAlt, heading, pendingMessage) {
  toxicityOpinionRequests.forEach((controller) => controller.abort());
  toxicityOpinionRequests.clear();
  inspectionEvidenceSerial = 0;
  const panel = byId("specimen-detail");
  const wholePhoto = heading === "WHOLE PHOTO / MUSE SPARK";
  byId("specimen-dialog").dataset.view = wholePhoto ? "scene-detail" : "specimen";
  panel.replaceChildren(element("p", "eyebrow", heading));
  if (wholePhoto) {
    const back = element("button", "scene-summary-back", "← Toxicity summary panel");
    back.type = "button";
    back.addEventListener("click", () => {
      byId("specimen-dialog").close();
      showSummaryPanel();
    });
    panel.append(back);
  }
  const top = element("div", "inspection-top");
  const image = element("img", "inspection-image");
  image.src = imageUrl;
  image.alt = imageAlt;
  const findings = element("div", "inspection-findings");
  top.append(image, findings);
  panel.append(top);
  if (!assessment) {
    findings.append(element("p", "inspection-empty", pendingMessage));
  } else {
    const confidence = { high: 3, medium: 2, low: 1 };
    const groups = [...(assessment.plant_groups || [])].sort((a, b) =>
      (confidence[b.identification_confidence] || 0) -
      (confidence[a.identification_confidence] || 0));
    if (groups.length) {
      const highest = groups[0].identification_confidence;
      const tieCount = groups.filter((group) => group.identification_confidence === highest).length;
      findings.append(renderInspectionGroup(groups[0], true, tieCount, imageUrl));
      findings.append(element("p", "inspection-caveat",
        "Visual identification is provisional; toxicity status depends on the plant match."));
    } else {
      findings.append(element("p", "inspection-empty",
        "No plant identity was resolved from this image."));
    }
    const others = groups.slice(1);
    if (others.length) {
      const otherHazards = others.filter(inspectionHazard).length;
      const more = element("details", "inspection-more");
      more.append(element("summary", "",
        `Other reported plant groups (${others.length})${otherHazards ? ` · ${otherHazards} potential hazard${otherHazards === 1 ? "" : "s"}` : ""}`));
      others.forEach((group) => more.append(renderInspectionGroup(group, false, 1, imageUrl)));
      panel.append(more);
    }
    if (assessment.another_photo_needed) {
      const followup = element("section", "inspection-followup");
      followup.append(element("h3", "", "A closer photo would help"));
      if (assessment.photo_reason)
        followup.append(element("p", "inspection-followup-reason", assessment.photo_reason));
      followup.append(element("p", "", assessment.photo_instructions ||
        "Photograph one plant more closely, including its leaves and stem."));
      const upload = element("button", "inspection-followup-button", "Upload a follow-up photo ↗");
      upload.type = "button";
      upload.addEventListener("click", () => {
        byId("specimen-dialog").close();
        chooseNewPlant();
      });
      followup.append(upload);
      panel.append(followup);
    }
  }
  if (!byId("specimen-dialog").open) byId("specimen-dialog").showModal();
}
function inspectSpecimen(id) {
  const item = specimenRecords.get(id);
  if (!item) return;
  inspectedSpecimen = id;
  showInspection(item.data.assessment, item.data.crop_data_url,
    `Selected SAM specimen ${item.displayIndex}`, `SPECIMEN / ${item.displayIndex}`,
    machine.busy ? "Regional assessment is still running." :
      "No regional assessment was returned. Review the retained scene report.");
}
function summaryMetric(value, label, tone = "") {
  const metric = element("span", `overview-metric ${tone}`);
  metric.append(element("strong", "", String(value)), element("span", "", label));
  return metric;
}
function resetOverview() {
  byId("scene-overview").replaceChildren(
    summaryMetric("—", "plant types"),
    summaryMetric("—", "potential hazards"),
    summaryMetric("—", "unresolved identities"),
  );
}
function updateSafety() {
  const records = new Map();
  sceneAssessments.forEach((assessment, assessmentIndex) => {
    const specimenId = assessmentIndex === 0 ? null :
      [...specimenRecords].find(([, item]) => item.data.assessment === assessment)?.[0];
    (assessment.plant_groups || []).forEach((group, groupIndex) => {
      const scientific = (group.scientific_name || "").trim();
      const name = group.plant_identification || scientific || "Unidentified plant";
      const key = (/^(unknown|unidentified|uncertain|not identified)/i.test(scientific)
        ? name : scientific || name).trim().toLocaleLowerCase();
      const lookup = group.toxicity;
      const hazard = lookup?.toxicity_status === "toxic" || lookup?.potential_hazard ||
        lookup?.dog_toxicity?.toxicity_status === "toxic" || lookup?.dog_toxicity?.potential_hazard;
      const unresolvedIdentity = group.needs_inspection || group.identification_confidence === "low" ||
        !scientific || /^(unknown|unidentified|uncertain|not identified)/i.test(scientific);
      const unresolvedToxicity = !lookup || lookup.toxicity_status === "unknown" ||
        !lookup.dog_toxicity || lookup.dog_toxicity.toxicity_status === "unknown";
      const href = specimenId ? `#specimen-${specimenId}` :
        assessmentIndex === 0 ? `#scene-group-${groupIndex}` : null;
      const prior = records.get(key);
      records.set(key, {
        name: prior?.name || name,
        href: prior?.href || href,
        hazard: Boolean(prior?.hazard || hazard),
        unresolvedIdentity: Boolean(prior?.unresolvedIdentity || unresolvedIdentity),
        unresolvedToxicity: Boolean(prior?.unresolvedToxicity || unresolvedToxicity),
      });
    });
  });
  const entries = [...records.values()];
  const hazards = entries.filter((item) => item.hazard);
  const unresolved = entries.filter((item) => item.unresolvedIdentity);
  const unknownToxicity = entries.filter((item) => item.unresolvedToxicity);
  const needsPhoto = sceneAssessments.some((item) => item.another_photo_needed);
  const unresolvedCount = Math.max(unresolved.length, needsPhoto ? 1 : 0);
  const reportedCount = sceneAssessments[0]?.distinct_plant_count ?? entries.length;
  byId("scene-overview").replaceChildren(
    summaryMetric(reportedCount, "plant types"),
    summaryMetric(hazards.length, "potential hazards", hazards.length ? "hazard" : ""),
    summaryMetric(unresolvedCount, "unresolved identities", unresolvedCount ? "unresolved" : ""),
  );
  const banner = byId("safety-summary");
  banner.hidden = !hazards.length && !unresolvedCount && !unknownToxicity.length;
  if (banner.hidden) return;
  banner.className = `investigation-alert ${hazards.length ? "has-hazard" : "has-unresolved"}`;
  banner.replaceChildren(
    element("strong", "", hazards.length
      ? `${hazards.length} potentially hazardous plant type${hazards.length === 1 ? "" : "s"}`
      : "Investigation incomplete"),
    element("p", "", hazards.length
      ? "Review each possible match and its linked cat and dog toxicity sources."
      : "One or more plant identities or toxicity matches need verification."),
  );
  const links = element("div", "summary-links");
  for (const record of hazards) {
    const link = element("a", "hazard-link", record.name);
    link.href = record.href || "#spark-result";
    link.addEventListener("click", (event) => {
      const target = document.getElementById(link.hash.slice(1));
      if (target?.tagName === "DETAILS") target.open = true;
      if (target?.classList.contains("floating-specimen")) {
        event.preventDefault();
        target.click();
      }
    });
    links.append(link);
  }
  if (links.childElementCount) banner.append(links);
  if (unresolvedCount) banner.append(element("p", "unresolved-note",
    `${unresolvedCount} unresolved identification${unresolvedCount === 1 ? "" : "s"}${unresolved.length ? `: ${unresolved.map((item) => item.name).join(", ")}.` : ". Another photo was requested."}`));
  if (unknownToxicity.length) banner.append(element("p", "unresolved-note",
    `${unknownToxicity.length} plant type${unknownToxicity.length === 1 ? " has" : "s have"} unresolved cat or dog toxicity references.`));
}
function updateFollowup(assessment) {
  if (!assessment.another_photo_needed) return;
  followupAnnounced = true;
  syncFollowupSummaryAction();
  const requests = [...new Set([...sceneAssessments, assessment])]
    .filter((item) => item?.another_photo_needed);
  const crowdedScene = requests.some((item) => item.separate_plant_count > maxSeparatePlants);
  const instructions = [...new Set(requests
    .map((item) => item.photo_instructions)
    .filter(Boolean))];
  const reasons = [...new Set(requests.map((item) => item.photo_reason).filter(Boolean))];
  els.photoGuidanceHeading.textContent = crowdedScene
    ? "One plant at a time." : "Show this plant clearly.";
  els.photoGuidanceCount.hidden = !crowdedScene && requests.length === 1;
  els.photoGuidanceCount.textContent = crowdedScene
    ? `${requests.find((item) => item.separate_plant_count > maxSeparatePlants).separate_plant_count} separate plants reported in this photo`
    : requests.length > 1 ? "Several plant views need another photo."
      : "";
  els.photoGuidanceInstructions.replaceChildren(
    ...(instructions.length ? instructions : [
      "Take a closer photo of one plant with its leaves and stem visible.",
    ]).map((instruction) => element("li", "", instruction)),
  );
  els.photoGuidanceInstructions.classList.toggle("single-instruction", instructions.length <= 1);
  els.photoGuidanceWhy.hidden = reasons.length === 0;
  els.photoGuidanceReason.textContent = reasons.join(" ");
  els.photoGuidance.hidden = summaryPanelOpen || followupMinimized;
  els.photoGuidanceRestore.hidden = summaryPanelOpen || !followupMinimized;
  byId("region-progress").hidden = true;
  syncFollowupProgress();
  if (summaryPanelOpen) showSceneCard(sceneAssessments[0] || assessment);
  else {
    byId("scene-card").hidden = true;
    byId("scene-card-link").classList.remove("visible");
  }
  byId("followup-panel").hidden = investigationState !== "completed";
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
  report.querySelector(".report-actions")?.remove();
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
byId("followup-button").addEventListener("click", chooseNewPlant);
byId("close-specimen").addEventListener("click", () =>
  byId("specimen-dialog").close(),
);
byId("specimen-dialog").addEventListener("close", () => {
  toxicityOpinionRequests.forEach((controller) => controller.abort());
  toxicityOpinionRequests.clear();
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
resetOverview();
setInvestigationState("idle");
loadModels();
