import { AutoModel, AutoProcessor, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const btnIcon = document.getElementById('btn-icon');
const btnText = document.getElementById('btn-text');
const modelSelect = document.getElementById('model-select');
const toggleDetect = document.getElementById('toggle-detect');
const togglePose = document.getElementById('toggle-pose');
const thresholdInput = document.getElementById('threshold');
const thresholdValueEl = document.getElementById('threshold-value');
const fpsEl = document.getElementById('fps');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// State
let detectModel = null;
let poseModel = null;
let processor = null;
let isRunning = false;
let isProcessing = false;
let threshold = 0.5;
let enableDetect = true;
let enablePose = true;
let animationId = null;

// Offscreen canvas for frame capture
const offscreen = document.createElement('canvas');
const offscreenCtx = offscreen.getContext('2d');

// Constants
const COLORS = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6', '#ef4444', '#10b981', '#3b82f6'];
const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16]
];
const POSE_THRESHOLD = 0.0001;

// UI Helpers
const setStatus = (text, type = 'default') => {
  statusText.textContent = text;
  statusDot.className = 'status-dot ' + type;
};

const showLoader = (text) => {
  loaderText.textContent = text;
  loader.classList.add('visible');
};

const hideLoader = () => loader.classList.remove('visible');

// Model Loading
async function loadModels(modelId) {
  try {
    if (isRunning) stopCamera(true);
    while (isProcessing) await new Promise(r => setTimeout(r, 50));

    if (detectModel) await detectModel.dispose();
    if (poseModel) await poseModel.dispose();
    detectModel = poseModel = processor = null;
    startBtn.disabled = true;

    const poseModelId = modelId.replace('-ONNX', '-pose-ONNX');
    const progressCallback = (label) => (info) => {
      if (info.status === 'progress' && info.file.endsWith('.onnx')) {
        showLoader(`${label} (${Math.round((info.loaded / info.total) * 100)}%)`);
      }
    };

    setStatus('Loading...', 'loading');
    showLoader('Loading detection model...');
    detectModel = await AutoModel.from_pretrained(modelId, {
      device: 'webgpu',
      dtype: 'fp16',
      progress_callback: progressCallback('Loading detection model')
    });

    showLoader('Loading pose model...');
    poseModel = await AutoModel.from_pretrained(poseModelId, {
      device: 'webgpu',
      dtype: 'fp32',
      progress_callback: progressCallback('Loading pose model')
    });

    showLoader('Loading processor...');
    processor = await AutoProcessor.from_pretrained(modelId);

    setStatus('Ready', 'ready');
    hideLoader();
    startBtn.disabled = false;
    startCamera();
  } catch (error) {
    console.error('Model loading failed:', error);
    setStatus('Error', 'error');
    showLoader('Failed: ' + error.message);
  }
}

// Camera Control
async function startCamera() {
  try {
    showLoader('Accessing camera...');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 640 } },
      audio: false
    });

    video.srcObject = stream;
    video.onloadedmetadata = () => {
      canvas.width = offscreen.width = video.videoWidth;
      canvas.height = offscreen.height = video.videoHeight;

      isRunning = true;
      btnIcon.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>';
      btnText.textContent = 'Stop Camera';
      startBtn.classList.add('running');

      hideLoader();
      setStatus('Running', 'running');
      loop();
    };
  } catch (error) {
    console.error('Camera error:', error);
    setStatus('Camera Error', 'error');
    showLoader('Camera access denied');
  }
}

function stopCamera(keepProcessingFlag = false) {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  isRunning = false;
  if (!keepProcessingFlag) isProcessing = false;

  btnIcon.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4l15 8-15 8V4z"/></svg>';
  btnText.textContent = 'Start Camera';
  startBtn.classList.remove('running');
  canvas.width = canvas.width; // Clear canvas
  setStatus('Ready', 'ready');
  fpsEl.textContent = '0';
}

// Detection Loop
function loop() {
  if (!isRunning) return;

  if (detectModel && poseModel && processor && !isProcessing) {
    isProcessing = true;
    const startTime = performance.now();
    detect()
      .then(() => fpsEl.textContent = Math.round(1000 / (performance.now() - startTime)))
      .finally(() => isProcessing = false);
  }

  if (isRunning) animationId = requestAnimationFrame(loop);
}

async function detect() {
  offscreenCtx.drawImage(video, 0, 0);
  const image = RawImage.fromCanvas(offscreen);
  const inputs = await processor(image);

  const promises = [];
  if (enableDetect) promises.push(detectModel(inputs));
  if (enablePose) promises.push(poseModel(inputs));

  const results = await Promise.all(promises);
  let idx = 0;
  const detectOutput = enableDetect ? results[idx++] : null;
  const poseOutput = enablePose ? results[idx++] : null;

  const detections = [];

  if (detectOutput) {
    const scores = detectOutput.logits.sigmoid().data;
    const boxes = detectOutput.pred_boxes.data;
    const id2label = detectModel.config.id2label;

    for (let i = 0; i < 300; i++) {
      let maxScore = 0, maxClass = 0;
      for (let j = 0; j < 80; j++) {
        const score = scores[i * 80 + j];
        if (score > maxScore) { maxScore = score; maxClass = j; }
      }
      if (maxScore >= threshold) {
        const [cx, cy, w, h] = [boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
        detections.push({
          type: 'object',
          box: [(cx - w / 2) * canvas.width, (cy - h / 2) * canvas.height, w * canvas.width, h * canvas.height],
          score: maxScore,
          classId: maxClass,
          label: id2label[maxClass] || `Class ${maxClass}`
        });
      }
    }
  }

  if (poseOutput) {
    const data = Object.values(poseOutput)[0].data;
    for (let i = 0; i < 300; i++) {
      const offset = i * 57;
      const score = data[offset + 4];
      if (score >= threshold) {
        const keypoints = [];
        for (let k = 0; k < 17; k++) {
          const kIdx = offset + 6 + k * 3;
          keypoints.push({ x: data[kIdx] * canvas.width, y: data[kIdx + 1] * canvas.height, c: data[kIdx + 2] });
        }
        detections.push({
          type: 'pose',
          box: [data[offset] * canvas.width, data[offset + 1] * canvas.height,
                (data[offset + 2] - data[offset]) * canvas.width, (data[offset + 3] - data[offset + 1]) * canvas.height],
          score,
          keypoints
        });
      }
    }
  }

  if (isRunning) draw(detections);
}

// Drawing
function draw(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const det of detections.filter(d => d.type === 'object')) {
    const [x, y, w, h] = det.box;
    const color = COLORS[det.classId % COLORS.length];
    const label = `${det.label} ${Math.round(det.score * 100)}%`;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.font = 'bold 12px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y > 18 ? y - 18 : y, tw + 8, 18);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 4, y > 18 ? y - 5 : y + 13);
  }

  for (const det of detections.filter(d => d.type === 'pose')) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#22d3ee';
    for (const [i, j] of SKELETON) {
      const a = det.keypoints[i], b = det.keypoints[j];
      if (a?.c >= POSE_THRESHOLD && b?.c >= POSE_THRESHOLD) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    for (const kp of det.keypoints) {
      if (kp.c < POSE_THRESHOLD) continue;
      ctx.fillStyle = '#6366f1';
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

// Event Listeners
startBtn.addEventListener('click', () => isRunning ? stopCamera() : startCamera());
thresholdInput.addEventListener('input', (e) => {
  threshold = e.target.value / 100;
  thresholdValueEl.textContent = `${e.target.value}%`;
});
toggleDetect.addEventListener('change', (e) => enableDetect = e.target.checked);
togglePose.addEventListener('change', (e) => enablePose = e.target.checked);
modelSelect.addEventListener('change', (e) => loadModels(e.target.value));

// Initialize
loadModels(modelSelect.value);
