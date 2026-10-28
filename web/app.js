const byId = (id) => document.getElementById(id);

const els = {
  apiStatus: byId('api-status'),
  dropzone: byId('dropzone'),
  uploadEmpty: byId('upload-empty'),
  imageStage: byId('image-stage'),
  imageInput: byId('image-input'),
  chooseImage: byId('choose-image'),
  uploadTitle: byId('upload-title'),
  uploadSubtitle: byId('upload-subtitle'),
  preview: byId('image-preview'),
  scanCanvas: byId('scan-overlay'),
  flowerBoxes: byId('flower-boxes'),
  scanMessage: byId('scan-message'),
  removeImage: byId('remove-image'),
  model: byId('model-select'),
  samMode: byId('sam-mode'),
  samPrompt: byId('sam-prompt'),
  compare: byId('compare-button'),
  error: byId('error-banner'),
  runStatus: byId('run-status'),
  scanProgress: byId('scan-progress'),
  progressTitle: byId('progress-title'),
  progressDetail: byId('progress-detail'),
  results: byId('results'),
  trainedResult: byId('trained-result'),
  sparkResult: byId('spark-result'),
  segmentationPanel: byId('segmentation-panel'),
  segmentationStatus: byId('segmentation-status'),
  segmentationReason: byId('segmentation-reason'),
  segments: byId('segments-grid'),
};

let selectedFile = null;
let previewUrl = null;
let scanTimer = null;
let scanFrame = null;
let scanStartedAt = 0;
let scanResizeObserver = null;
let scanRegions = [];
let visibleFlowerDetections = null;
const scanStages = [
  ['Finding visual details…', 'Checking the photo for useful plant features.'],
  ['Comparing plant patterns…', 'The classifier and Muse Spark are reviewing the image.'],
  ['Checking plant regions…', 'Muse Spark may request SAM for a closer look.'],
];
const apiDefault = (() => {
  const { protocol, hostname, port, origin } = window.location;
  if ((hostname === 'localhost' || hostname === '127.0.0.1') && port === '3000') {
    return `${protocol}//${hostname}:8000`;
  }
  return origin;
})();
const apiInput = document.createElement('input');
apiInput.type = 'hidden';
apiInput.id = 'api-base-url';
apiInput.value = localStorage.getItem('petsafe-api-base') || apiDefault;
document.body.append(apiInput);

function apiUrl(path) {
  return `${apiInput.value.trim().replace(/\/+$/, '')}${path}`;
}

function setApiStatus(text, state) {
  els.apiStatus.className = `status-pill status-${state}`;
  els.apiStatus.replaceChildren();
  const dot = document.createElement('span');
  dot.className = 'status-dot';
  els.apiStatus.append(dot, document.createTextNode(text));
}

async function loadModels() {
  try {
    const response = await fetch(apiUrl('/api/models'));
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.available) || data.available.length === 0) {
      setApiStatus('No model artifacts', 'offline');
      return;
    }
    const previous = els.model.value;
    els.model.replaceChildren();
    const labels = {
      efficientnet_b2: 'EfficientNet-B2',
      dinov3: 'DINOv3',
      siglip2: 'SigLIP2',
    };
    for (const name of data.available) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = labels[name] || name;
      els.model.append(option);
    }
    if (data.available.includes(previous)) els.model.value = previous;
    setApiStatus('API connected', 'online');
  } catch (error) {
    setApiStatus('API unavailable', 'offline');
    showError(`Could not reach the inference API at ${apiInput.value}. Start the API and refresh this page. ${error.message}`);
  }
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function clearError() {
  els.error.textContent = '';
  els.error.hidden = true;
}

function setFile(file) {
  clearError();
  if (!file) return;
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showError('Choose a JPEG, PNG, or WebP image.');
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showError('The image must be 12 MB or smaller.');
    return;
  }
  selectedFile = file;
  visibleFlowerDetections = null;
  els.flowerBoxes.replaceChildren();
  scanRegions = [];
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.preview.onload = detectImageRegions;
  els.preview.src = previewUrl;
  els.preview.hidden = false;
  els.uploadEmpty.hidden = true;
  els.imageStage.hidden = false;
  els.dropzone.classList.add('has-image');
  els.removeImage.hidden = false;
  els.uploadTitle.textContent = file.name;
  els.uploadSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · click to change image`;
  els.compare.disabled = false;
  els.scanMessage.textContent = 'PHOTO READY';
  requestAnimationFrame(sizeScanCanvas);
}

function clearFile() {
  selectedFile = null;
  visibleFlowerDetections = null;
  els.imageInput.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  els.preview.removeAttribute('src');
  els.preview.hidden = true;
  els.flowerBoxes.replaceChildren();
  els.uploadEmpty.hidden = false;
  els.imageStage.hidden = true;
  els.dropzone.classList.remove('has-image', 'is-scanning');
  els.removeImage.hidden = true;
  els.uploadTitle.textContent = 'Drop a plant photo here';
  els.uploadSubtitle.replaceChildren(
    document.createTextNode('or '),
    els.chooseImage,
    document.createTextNode(' · JPG, PNG, WebP'),
  );
  els.chooseImage.hidden = false;
  els.compare.disabled = true;
  stopScanAnimation();
}

function sizeScanCanvas() {
  if (!els.imageStage || els.imageStage.hidden) return;
  const rect = els.imageStage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  els.scanCanvas.width = Math.max(1, Math.round(rect.width * ratio));
  els.scanCanvas.height = Math.max(1, Math.round(rect.height * ratio));
  els.scanCanvas.style.width = `${rect.width}px`;
  els.scanCanvas.style.height = `${rect.height}px`;
  const context = els.scanCanvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (!scanResizeObserver) {
    scanResizeObserver = new ResizeObserver(() => sizeScanCanvas());
    scanResizeObserver.observe(els.imageStage);
  }
}

function detectImageRegions() {
  const sourceWidth = els.preview.naturalWidth;
  const sourceHeight = els.preview.naturalHeight;
  if (!sourceWidth || !sourceHeight) return;
  const sampleSize = 64;
  const sample = document.createElement('canvas');
  sample.width = sampleSize;
  sample.height = sampleSize;
  const context = sample.getContext('2d', { willReadFrequently: true });
  const sampleScale = Math.min(sampleSize / sourceWidth, sampleSize / sourceHeight);
  const sampleWidth = sourceWidth * sampleScale;
  const sampleHeight = sourceHeight * sampleScale;
  const sampleOffsetX = (sampleSize - sampleWidth) / 2;
  const sampleOffsetY = (sampleSize - sampleHeight) / 2;
  context.drawImage(els.preview, sampleOffsetX, sampleOffsetY, sampleWidth, sampleHeight);
  const { data } = context.getImageData(0, 0, sampleSize, sampleSize);
  const active = new Uint8Array(sampleSize * sampleSize);
  for (let i = 0; i < active.length; i += 1) {
    const offset = i * 4;
    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];
    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    const vivid = max > 70 && max - min > 48;
    const leafTone = green > 55 && green > red * 1.12 && green > blue * 0.92;
    active[i] = vivid || leafTone ? 1 : 0;
  }
  const visited = new Uint8Array(active.length);
  const regions = [];
  for (let index = 0; index < active.length; index += 1) {
    if (!active[index] || visited[index]) continue;
    const queue = [index];
    visited[index] = 1;
    let minX = sampleSize, minY = sampleSize, maxX = 0, maxY = 0, area = 0;
    while (queue.length) {
      const point = queue.pop();
      const x = point % sampleSize;
      const y = Math.floor(point / sampleSize);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      area += 1;
      for (const neighbor of [x > 0 ? point - 1 : -1, x < sampleSize - 1 ? point + 1 : -1,
        y > 0 ? point - sampleSize : -1, y < sampleSize - 1 ? point + sampleSize : -1]) {
        if (neighbor >= 0 && active[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          queue.push(neighbor);
        }
      }
    }
    if (area >= 7) regions.push({
      x: minX / sampleSize, y: minY / sampleSize,
      width: (maxX - minX + 1) / sampleSize, height: (maxY - minY + 1) / sampleSize,
      area,
    });
  }
  scanRegions = regions.sort((a, b) => b.area - a.area).slice(0, 5)
    .map(({ x, y, width, height }) => ({
      x: Math.max(0, (x - sampleOffsetX) / sampleWidth),
      y: Math.max(0, (y - sampleOffsetY) / sampleHeight),
      width: Math.min(1, width / sampleWidth),
      height: Math.min(1, height / sampleHeight),
    }));
}

function drawScanHighlights(timestamp) {
  if (!selectedFile || els.imageStage.hidden || !els.dropzone.classList.contains('is-scanning')) return;
  const context = els.scanCanvas.getContext('2d');
  const ratio = window.devicePixelRatio || 1;
  const width = els.scanCanvas.width / ratio;
  const height = els.scanCanvas.height / ratio;
  const elapsed = (timestamp - scanStartedAt) / 1000;
  context.clearRect(0, 0, width, height);
  // Color-based focus marks communicate progress; they are not model output.
  const imageWidth = els.preview.naturalWidth || 1;
  const imageHeight = els.preview.naturalHeight || 1;
  const scale = Math.min(width / imageWidth, height / imageHeight);
  const drawnWidth = imageWidth * scale;
  const drawnHeight = imageHeight * scale;
  const offsetX = (width - drawnWidth) / 2;
  const offsetY = (height - drawnHeight) / 2;
  const marks = (scanRegions.length ? scanRegions : [
    { x: 0.30, y: 0.35, width: 0.18, height: 0.24 },
    { x: 0.64, y: 0.46, width: 0.21, height: 0.27 },
    { x: 0.47, y: 0.72, width: 0.15, height: 0.19 },
  ]).map((region, index) => ({
    x: offsetX + (region.x + region.width / 2) * drawnWidth,
    y: offsetY + (region.y + region.height / 2) * drawnHeight,
    rx: Math.min(width * 0.28, Math.max(18, region.width * drawnWidth * 0.58)),
    ry: Math.min(height * 0.29, Math.max(18, region.height * drawnHeight * 0.58)),
    phase: index * 1.7,
  }));
  marks.forEach((mark, index) => {
    const pulse = 0.5 + 0.5 * Math.sin(elapsed * 2.2 + mark.phase);
    const x = mark.x;
    const y = mark.y;
    context.save();
    context.translate(x, y);
    context.rotate(Math.sin(elapsed * 0.5 + mark.phase) * 0.08);
    context.strokeStyle = `rgba(203, 255, 138, ${0.20 + pulse * 0.48})`;
    context.lineWidth = 1.4;
    context.shadowColor = 'rgba(168, 255, 105, .75)';
    context.shadowBlur = 11 + pulse * 10;
    context.beginPath();
    context.ellipse(0, 0, mark.rx * (0.94 + pulse * 0.08), mark.ry * (0.94 + pulse * 0.08), 0, 0, Math.PI * 2);
    context.stroke();
    context.strokeStyle = `rgba(238, 255, 213, ${0.33 + pulse * 0.45})`;
    context.shadowBlur = 4;
    context.beginPath();
    context.moveTo(-mark.rx - 9, -mark.ry + 15);
    context.lineTo(-mark.rx - 9, -mark.ry - 9);
    context.lineTo(-mark.rx + 15, -mark.ry - 9);
    context.moveTo(mark.rx + 9, mark.ry - 15);
    context.lineTo(mark.rx + 9, mark.ry + 9);
    context.lineTo(mark.rx - 15, mark.ry + 9);
    context.stroke();
    context.globalAlpha = 0.18 + pulse * 0.10;
    context.fillStyle = index % 2 ? '#caff91' : '#f0ffb2';
    context.fill();
    context.restore();
  });
  scanFrame = requestAnimationFrame(drawScanHighlights);
}

function startScanAnimation() {
  stopScanAnimation();
  sizeScanCanvas();
  scanStartedAt = performance.now();
  els.dropzone.classList.add('is-scanning');
  els.scanMessage.textContent = 'ANALYZING IMAGE';
  els.scanProgress.hidden = false;
  els.progressTitle.textContent = scanStages[0][0];
  els.progressDetail.textContent = scanStages[0][1];
  let stage = 0;
  scanTimer = window.setInterval(() => {
    stage = Math.min(stage + 1, scanStages.length - 1);
    els.progressTitle.textContent = scanStages[stage][0];
    els.progressDetail.textContent = scanStages[stage][1];
  }, 4500);
  scanFrame = requestAnimationFrame(drawScanHighlights);
}

function stopScanAnimation() {
  if (scanTimer) window.clearInterval(scanTimer);
  if (scanFrame) cancelAnimationFrame(scanFrame);
  scanTimer = null;
  scanFrame = null;
  if (els.scanCanvas) {
    const context = els.scanCanvas.getContext('2d');
    context.clearRect(0, 0, els.scanCanvas.width, els.scanCanvas.height);
  }
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

function renderCacheNote(container, cacheHit) {
  if (cacheHit) container.append(element('span', 'cache-note', 'Cached result · this server process'));
}

function renderToxicity(container, result, context) {
  const lookup = result?.toxicity;
  if (!lookup) return;
  const box = element('div', `toxicity-lookup toxicity-${lookup.toxicity_status || 'unknown'}`);
  const labels = {
    toxic: 'Listed as toxic to cats',
    non_toxic: 'Listed as non-toxic to cats',
    unknown: lookup.potential_hazard
      ? 'Potential hazard · identity scope unresolved'
      : 'No confirmed toxicity match',
  };
  box.append(element('p', 'toxicity-heading', labels[lookup.toxicity_status] || labels.unknown));
  if (lookup.scientific_name) {
    box.append(element('p', 'toxicity-taxon', lookup.scientific_name));
  }
  const note = lookup.lookup_note || lookup.taxonomic_scope;
  if (note) box.append(element('p', 'toxicity-note', note));
  if (context) box.append(element('p', 'toxicity-context', context));
  const links = element('div', 'toxicity-sources');
  const seen = new Set();
  for (const evidence of lookup.evidence || []) {
    const url = evidence.source_url;
    if (!url || seen.has(url) || !url.startsWith('https://')) continue;
    seen.add(url);
    const link = element('a', '', evidence.source_id.replaceAll('_', ' '));
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    links.append(link);
  }
  if (links.childElementCount) box.append(links);
  container.append(box);
}

function renderRawMetaResponse(container, result) {
  if (!result?.raw_meta_response) return;
  const details = element('details', 'raw-response-details');
  details.append(element('summary', '', 'Raw Meta API response'));
  const payload = element('pre', 'raw-response-json', JSON.stringify(result.raw_meta_response, null, 2));
  details.append(payload);
  container.append(details);
}

function renderFlowerBoxes(detections, sourceWidth, sourceHeight) {
  visibleFlowerDetections = { detections, sourceWidth, sourceHeight };
  els.flowerBoxes.replaceChildren();
  if (!sourceWidth || !sourceHeight) return;
  const stage = els.imageStage.getBoundingClientRect();
  const scale = Math.min(stage.width / sourceWidth, stage.height / sourceHeight);
  const offsetX = (stage.width - sourceWidth * scale) / 2;
  const offsetY = (stage.height - sourceHeight * scale) / 2;
  detections.forEach((item, index) => {
    const [left, top, right, bottom] = item.bounds;
    const box = element('div', 'flower-box');
    box.style.left = `${offsetX + left * scale}px`;
    box.style.top = `${offsetY + top * scale}px`;
    box.style.width = `${(right - left) * scale}px`;
    box.style.height = `${(bottom - top) * scale}px`;
    box.append(element('span', '', `${index + 1} · ${item.predicted_class}`));
    els.flowerBoxes.append(box);
  });
}

function renderTrained(result, sourceWidth, sourceHeight) {
  els.trainedResult.replaceChildren();
  const primary = element('div', 'prediction-primary');
  primary.append(
    element('strong', '', result.predicted_class),
    element('span', 'confidence-value', percent(result.confidence)),
  );
  els.trainedResult.append(primary);
  els.trainedResult.append(element('p', 'confidence-caption', 'Top class probability · not calibrated confidence'));
  renderCacheNote(els.trainedResult, result.cache_hit);
  renderToxicity(els.trainedResult, result, 'Reference for the classifier’s top class; the prediction may be wrong.');
  const scores = element('div', 'score-list');
  for (const item of result.top_predictions || []) {
    const row = element('div', 'score-row');
    const name = element('span', 'score-name', item.class_name);
    const track = element('span', 'score-track');
    const fill = element('span', 'score-fill');
    fill.style.width = `${Math.max(0, Math.min(100, Number(item.probability) * 100))}%`;
    track.append(fill);
    row.append(name, track, element('span', 'score-percent', percent(item.probability)));
    scores.append(row);
  }
  els.trainedResult.append(scores);
  const heading = element('div', 'detection-heading');
  heading.append(element('strong', '', `Individual flower checks · ${result.detections?.length || 0}`));
  els.trainedResult.append(heading);
  if (result.detection_status === 'failed') {
    els.trainedResult.append(element('p', 'segment-notes', `Flower detection unavailable: ${result.detection_error || 'unknown error'}`));
  } else if (result.detection_status === 'no_flowers') {
    els.trainedResult.append(element('p', 'segment-notes', 'No flower boxes passed the detector threshold. The full-image prediction is shown above.'));
  }
  const grid = element('div', 'flower-crop-grid');
  for (const [index, item] of (result.detections || []).entries()) {
    const card = element('article', 'flower-crop');
    const image = element('img', 'flower-crop-image');
    image.src = item.crop_data_url;
    image.alt = `Detected flower ${index + 1}`;
    card.append(image);
    const details = element('div', 'flower-crop-details');
    details.append(element('span', 'segment-label', `FLOWER ${index + 1} · DETECTION ${percent(item.detector_score)}`));
    details.append(element('strong', '', item.predicted_class));
    details.append(element('span', 'confidence-value', `Classifier ${percent(item.confidence)}`));
    renderToxicity(details, item, 'Reference for this crop prediction; verify the plant identity.');
    card.append(details);
    grid.append(card);
  }
  els.trainedResult.append(grid);
  renderFlowerBoxes(result.detections || [], sourceWidth, sourceHeight);
}

function renderSpark(result) {
  els.sparkResult.replaceChildren();
  const summary = element('div', 'spark-summary');
  const countTile = element('div', 'summary-tile');
  countTile.append(
    element('span', 'summary-label', 'Distinct plant types'),
    element('span', 'summary-value', result.distinct_plant_count),
  );
  const confidenceTile = element('div', 'summary-tile');
  confidenceTile.append(
    element('span', 'summary-label', 'Arrangement'),
    element('span', 'summary-value', result.bouquet_type.replaceAll('_', ' ')),
  );
  summary.append(countTile, confidenceTile);
  els.sparkResult.append(summary);
  renderCacheNote(els.sparkResult, result.cache_hit);
  for (const group of result.plant_groups || []) {
    const card = element('div', 'plant-group');
    card.append(element('p', 'card-kicker', `PLANT TYPE ${group.id} · ${group.identification_confidence} confidence`));
    card.append(element('p', 'spark-identification', group.plant_identification));
    if (group.scientific_name && group.scientific_name.toLowerCase() !== 'unknown') {
      card.append(element('p', 'toxicity-taxon', group.scientific_name));
    }
    if (group.visual_evidence?.length) card.append(element('p', 'spark-evidence', group.visual_evidence.join(' · ')));
    renderToxicity(card, group, 'Reference for this visual hypothesis; verify the plant identity.');
    if (group.needs_inspection) card.append(element('span', 'decision-chip yes', 'Closer inspection suggested'));
    els.sparkResult.append(card);
  }
  if (!result.plant_groups?.length) els.sparkResult.append(element('p', 'empty-state', 'No distinct plant types were identified.'));
  if (result.another_photo_needed) {
    const guidance = element('div', 'photo-guidance');
    guidance.append(element('strong', '', 'Another photo could help'));
    guidance.append(element('p', '', result.photo_instructions));
    if (result.photo_reason) guidance.append(element('p', 'spark-evidence', result.photo_reason));
    els.sparkResult.append(guidance);
  }
  renderSparkUsage(els.sparkResult, result);
  renderRawMetaResponse(els.sparkResult, result);
  const recommendation = result.suggest_segmentation ? 'SAM suggested · ' : 'SAM not suggested · ';
  const chip = element('span', `decision-chip${result.suggest_segmentation ? ' yes' : ''}`, recommendation + (result.segmentation_reason || 'No specific reason returned.'));
  els.sparkResult.append(chip);
}

function renderSparkUsage(container, result) {
  const usage = result.api_usage;
  if (!usage) return;
  const box = element('div', 'usage-box');
  box.append(element('strong', '', result.cache_hit ? 'Meta usage from the original request' : 'Meta API token usage'));
  const cached = usage.cached_input_tokens == null ? 'not reported' : usage.cached_input_tokens;
  const ratio = usage.cache_hit_ratio == null ? 'not reported' : percent(usage.cache_hit_ratio);
  box.append(element('span', '', `Input ${usage.input_tokens} · Cached input ${cached} · Output ${usage.output_tokens}`));
  box.append(element('span', '', `Meta prompt cache ratio ${ratio}`));
  container.append(box);
}

function renderSegmentation(result) {
  els.segmentationPanel.hidden = false;
  els.segmentationStatus.className = `status-pill segmentation-status status-${result.status}`;
  els.segmentationStatus.textContent = {
    skipped: 'Skipped',
    completed: `${result.segments.length} segment${result.segments.length === 1 ? '' : 's'}`,
    no_masks: 'No masks found',
    failed: 'Segmentation failed',
  }[result.status] || result.status;
  const reasons = Array.isArray(result.reason) ? result.reason.join(' ') : '';
  const counts = result.ran
    ? `${result.mask_count} mask(s) received · ${result.filtered_count} filtered · ${result.assessed_count} reassessed.`
    : '';
  els.segmentationReason.textContent = [reasons, counts, result.error || ''].filter(Boolean).join(' ');
  els.segments.replaceChildren();
  for (const segment of result.segments || []) {
    const card = element('article', 'segment-card');
    const imageWrap = element('div', 'segment-image-wrap');
    const image = element('img', 'segment-image');
    image.src = segment.crop_data_url;
    image.alt = `SAM segment ${segment.object_id}`;
    imageWrap.append(image);
    const body = element('div', 'segment-body');
    body.append(element('p', 'segment-label', `SAM region ${segment.object_id}`));
    for (const group of segment.assessment.plant_groups || []) {
      body.append(element('p', 'segment-plant', group.plant_identification));
      body.append(element('span', 'mini-tag', `identity ${group.identification_confidence}`));
      renderToxicity(body, group, 'Reference for this crop’s visual hypothesis.');
    }
    const tags = element('div', 'segment-meta');
    tags.append(
      element('span', 'mini-tag', `${segment.assessment.distinct_plant_count} plant type(s)`),
      element('span', `mini-tag${segment.in_trained_taxonomy ? '' : ' outside'}`,
        segment.in_trained_taxonomy ? 'in trained classes' : 'outside trained classes'),
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
  clearError();
  els.compare.disabled = true;
  els.compare.classList.add('is-loading');
  els.compare.querySelector('.button-label').textContent = 'Checking…';
  els.runStatus.hidden = true;
  startScanAnimation();
  els.results.hidden = false;
  els.segmentationPanel.hidden = true;
  try {
    const form = new FormData();
    form.append('image_file', selectedFile, selectedFile.name);
    form.append('model_name', els.model.value);
    form.append('force_sam', String(els.samMode.value === 'force'));
    form.append('skip_sam', String(els.samMode.value === 'skip'));
    form.append('sam_prompt', els.samPrompt.value);
    const response = await fetch(apiUrl('/api/flows/compare'), { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail || payload);
      throw new Error(detail || `Request failed (${response.status})`);
    }
    renderTrained(payload.trained_flow, payload.image_width, payload.image_height);
    renderSpark(payload.muse_spark_flow);
    renderSegmentation(payload.segmentation);
    els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    showError(error.message || 'The comparison failed. Check the API and try again.');
  } finally {
    stopScanAnimation();
    els.dropzone.classList.remove('is-scanning');
    els.scanMessage.textContent = 'SCAN COMPLETE';
    els.scanProgress.hidden = true;
    els.compare.disabled = !selectedFile;
    els.compare.classList.remove('is-loading');
    els.compare.querySelector('.button-label').textContent = 'Check this plant';
  }
}

els.chooseImage.addEventListener('click', () => els.imageInput.click());
els.dropzone.addEventListener('click', (event) => {
  if (event.target.closest('button')) return;
  if (event.target.closest('.image-stage')) return;
  els.imageInput.click();
});
els.imageInput.addEventListener('change', () => setFile(els.imageInput.files[0]));
els.removeImage.addEventListener('click', clearFile);
els.compare.addEventListener('click', compare);
byId('clear-results').addEventListener('click', () => {
  els.results.hidden = true;
  visibleFlowerDetections = null;
  els.flowerBoxes.replaceChildren();
  clearError();
});
for (const eventName of ['dragenter', 'dragover']) {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add('is-dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove('is-dragging');
  });
}
els.dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));
els.apiStatus.addEventListener('click', async () => {
  const value = window.prompt('Inference API base URL', apiInput.value);
  if (value === null || !value.trim()) return;
  apiInput.value = value.trim().replace(/\/+$/, '');
  localStorage.setItem('petsafe-api-base', apiInput.value);
  await loadModels();
});
els.apiStatus.title = 'Click to change the API base URL';
window.addEventListener('beforeunload', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
window.addEventListener('resize', () => {
  if (visibleFlowerDetections) {
    const { detections, sourceWidth, sourceHeight } = visibleFlowerDetections;
    renderFlowerBoxes(detections, sourceWidth, sourceHeight);
  }
});

loadModels();
