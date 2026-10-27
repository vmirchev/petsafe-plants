const byId = (id) => document.getElementById(id);

const els = {
  apiStatus: byId('api-status'),
  dropzone: byId('dropzone'),
  imageInput: byId('image-input'),
  chooseImage: byId('choose-image'),
  uploadTitle: byId('upload-title'),
  uploadSubtitle: byId('upload-subtitle'),
  preview: byId('image-preview'),
  removeImage: byId('remove-image'),
  model: byId('model-select'),
  threshold: byId('confidence-threshold'),
  thresholdValue: byId('threshold-value'),
  samMode: byId('sam-mode'),
  samPrompt: byId('sam-prompt'),
  compare: byId('compare-button'),
  error: byId('error-banner'),
  runStatus: byId('run-status'),
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
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.preview.src = previewUrl;
  els.preview.hidden = false;
  els.removeImage.hidden = false;
  els.uploadTitle.textContent = file.name;
  els.uploadSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · click to change image`;
  els.compare.disabled = false;
}

function clearFile() {
  selectedFile = null;
  els.imageInput.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  els.preview.removeAttribute('src');
  els.preview.hidden = true;
  els.removeImage.hidden = true;
  els.uploadTitle.textContent = 'Drop a plant photo here';
  els.uploadSubtitle.replaceChildren(
    document.createTextNode('or '),
    els.chooseImage,
    document.createTextNode(' · JPG, PNG, WebP'),
  );
  els.chooseImage.hidden = false;
  els.compare.disabled = true;
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

function renderTrained(result) {
  els.trainedResult.replaceChildren();
  const primary = element('div', 'prediction-primary');
  primary.append(
    element('strong', '', result.predicted_class),
    element('span', 'confidence-value', percent(result.confidence)),
  );
  els.trainedResult.append(primary);
  els.trainedResult.append(element('p', 'confidence-caption', 'Top class probability · not calibrated confidence'));
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
}

function renderSpark(result) {
  els.sparkResult.replaceChildren();
  const summary = element('div', 'spark-summary');
  const countTile = element('div', 'summary-tile');
  countTile.append(
    element('span', 'summary-label', 'Flower estimate'),
    element('span', 'summary-value', `${result.flower_count_estimate} · ${result.flower_count_confidence} confidence`),
  );
  const confidenceTile = element('div', 'summary-tile');
  confidenceTile.append(
    element('span', 'summary-label', 'Identity confidence'),
    element('span', 'summary-value', result.identification_confidence),
  );
  summary.append(countTile, confidenceTile);
  els.sparkResult.append(summary);
  els.sparkResult.append(element('p', 'spark-identification', result.plant_identification));
  renderToxicity(els.sparkResult, result, 'Reference lookup for Spark’s visual hypothesis; verify the plant identity.');
  renderRawMetaResponse(els.sparkResult, result);
  if (result.flower_count_notes) els.sparkResult.append(element('p', 'spark-evidence', result.flower_count_notes));
  if (Array.isArray(result.visual_evidence) && result.visual_evidence.length) {
    els.sparkResult.append(element('p', 'spark-evidence', result.visual_evidence.join(' · ')));
  }
  const recommendation = result.suggest_segmentation ? 'SAM suggested · ' : 'SAM not suggested · ';
  const chip = element('span', `decision-chip${result.suggest_segmentation ? ' yes' : ''}`, recommendation + (result.segmentation_reason || 'No specific reason returned.'));
  els.sparkResult.append(chip);
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
  els.segmentationReason.textContent = result.error ? `${reasons} ${result.error}` : reasons;
  els.segments.replaceChildren();
  for (const segment of result.segments || []) {
    const card = element('article', 'segment-card');
    const imageWrap = element('div', 'segment-image-wrap');
    const image = element('img', 'segment-image');
    image.src = segment.crop_data_url;
    image.alt = `SAM segment ${segment.object_id}`;
    imageWrap.append(image);
    const body = element('div', 'segment-body');
    body.append(
      element('p', 'segment-label', `SAM region ${segment.object_id}`),
      element('p', 'segment-plant', segment.assessment.plant_identification),
    );
    const tags = element('div', 'segment-meta');
    tags.append(
      element('span', 'mini-tag', `identity ${segment.assessment.identification_confidence}`),
      element('span', 'mini-tag', `${segment.assessment.flower_count_estimate} flower(s)`),
      element('span', `mini-tag${segment.in_trained_taxonomy ? '' : ' outside'}`,
        segment.in_trained_taxonomy ? 'in trained classes' : 'outside trained classes'),
    );
    body.append(tags);
    renderToxicity(body, segment.assessment,
      `Reference lookup for this segment’s visual hypothesis (${segment.assessment.identification_confidence} confidence).`);
    renderRawMetaResponse(body, segment.assessment);
    if (segment.assessment.flower_count_notes) {
      body.append(element('p', 'segment-notes', segment.assessment.flower_count_notes));
    }
    card.append(imageWrap, body);
    els.segments.append(card);
  }
}

async function compare() {
  if (!selectedFile) return;
  clearError();
  els.compare.disabled = true;
  els.compare.querySelector('span:first-child').textContent = 'Working…';
  els.runStatus.hidden = false;
  els.results.hidden = false;
  els.segmentationPanel.hidden = true;
  try {
    const form = new FormData();
    form.append('image_file', selectedFile, selectedFile.name);
    form.append('model_name', els.model.value);
    form.append('confidence_threshold', els.threshold.value);
    form.append('force_sam', String(els.samMode.value === 'force'));
    form.append('skip_sam', String(els.samMode.value === 'skip'));
    form.append('sam_prompt', els.samPrompt.value);
    const response = await fetch(apiUrl('/api/flows/compare'), { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail || payload);
      throw new Error(detail || `Request failed (${response.status})`);
    }
    renderTrained(payload.trained_flow);
    renderSpark(payload.muse_spark_flow);
    renderSegmentation(payload.segmentation);
    els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    showError(error.message || 'The comparison failed. Check the API and try again.');
  } finally {
    els.runStatus.hidden = true;
    els.compare.disabled = !selectedFile;
    els.compare.querySelector('span:first-child').textContent = 'Compare flows';
  }
}

els.chooseImage.addEventListener('click', () => els.imageInput.click());
els.dropzone.addEventListener('click', (event) => {
  if (event.target.closest('button')) return;
  els.imageInput.click();
});
els.imageInput.addEventListener('change', () => setFile(els.imageInput.files[0]));
els.removeImage.addEventListener('click', clearFile);
els.compare.addEventListener('click', compare);
els.threshold.addEventListener('input', () => {
  els.thresholdValue.textContent = Number(els.threshold.value).toFixed(2);
});
byId('clear-results').addEventListener('click', () => {
  els.results.hidden = true;
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

loadModels();
