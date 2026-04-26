/* ─── State ──────────────────────────────────────────────── */
let stream       = null;
let uploadedFile = null;
let liveInterval = null;
let isProcessing = false;

/* ─── Tab switching ──────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
  if (name !== 'camera') stopLive();
}

/* ═══════════════════ CAMERA ════════════════════════════════ */

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    const video = document.getElementById('video');
    video.srcObject = stream;

    await new Promise((resolve, reject) => {
      video.onloadedmetadata = () => video.play().then(resolve).catch(reject);
      video.onerror = reject;
    });
    await new Promise(r => setTimeout(r, 500));

    // size the overlay canvas to match video
    resizeOverlay();
    window.addEventListener('resize', resizeOverlay);

    document.getElementById('camera-overlay').classList.add('hidden');
    document.getElementById('btn-start-camera').style.display = 'none';
    document.getElementById('btn-stop-camera').style.display  = '';
    document.getElementById('live-status').style.display      = '';
    clearNames('camera-names');

    startLive();
  } catch (err) {
    alert('Cannot access camera: ' + err.message);
  }
}

function stopCamera() {
  stopLive();
  window.removeEventListener('resize', resizeOverlay);

  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }

  const video = document.getElementById('video');
  video.srcObject = null;

  // clear overlay
  const overlay = document.getElementById('canvas-overlay');
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);

  document.getElementById('camera-overlay').classList.remove('hidden');
  document.getElementById('btn-start-camera').style.display = '';
  document.getElementById('btn-stop-camera').style.display  = 'none';
  document.getElementById('live-status').style.display      = 'none';
  clearNames('camera-names');
}

/* make overlay canvas always cover the video element */
function resizeOverlay() {
  const video   = document.getElementById('video');
  const overlay = document.getElementById('canvas-overlay');
  overlay.width  = video.clientWidth;
  overlay.height = video.clientHeight;
}

/* ── Live loop ─────────────────────────────────────────────── */
const LIVE_INTERVAL_MS = 800;

function startLive() {
  if (liveInterval) return;
  liveInterval = setInterval(sendFrame, LIVE_INTERVAL_MS);
}

function stopLive() {
  clearInterval(liveInterval);
  liveInterval = null;
  isProcessing = false;
}

async function sendFrame() {
  if (isProcessing) return;

  let dataUrl;
  try { dataUrl = captureFrame(); } catch (e) { return; }

  isProcessing = true;
  setLiveStatus('recognizing');

  try {
    const res  = await fetch('/api/recognize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();

    if (data.error) { setLiveStatus('error'); return; }

    drawOverlay(data.persons);
    renderNames('camera-names', data.persons);
    setLiveStatus('live');

  } catch (e) {
    setLiveStatus('error');
  } finally {
    isProcessing = false;
  }
}

function captureFrame() {
  const video  = document.getElementById('video');
  const canvas = document.getElementById('canvas-capture');

  if (!video.srcObject || video.readyState < 2) throw new Error('not ready');
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) throw new Error('no dimensions');

  canvas.width  = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.85);
}

/* draw bounding boxes + names on the transparent overlay canvas */
function drawOverlay(persons) {
  const video   = document.getElementById('video');
  const overlay = document.getElementById('canvas-overlay');
  const ctx     = overlay.getContext('2d');

  const dispW = overlay.width;
  const dispH = overlay.height;
  const vidW  = video.videoWidth  || dispW;
  const vidH  = video.videoHeight || dispH;

  // scale factor: server coords are in video pixels, overlay is in CSS pixels
  const scaleX = dispW / vidW;
  const scaleY = dispH / vidH;

  ctx.clearRect(0, 0, dispW, dispH);

  const COLORS = [
    '#34d399','#818cf8','#fb923c','#f87171',
    '#38bdf8','#c084fc','#facc15','#2dd4bf'
  ];
  const personColor = {};
  let colorIdx = 0;

  (persons || []).forEach(p => {
    if (!personColor[p.name]) {
      personColor[p.name] = COLORS[colorIdx++ % COLORS.length];
    }
    const color = personColor[p.name];
    const { top, right, bottom, left } = p.location;

    const x = left   * scaleX;
    const y = top    * scaleY;
    const w = (right - left)   * scaleX;
    const h = (bottom - top)   * scaleY;

    // box
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.strokeRect(x, y, w, h);

    // label background
    const label    = p.name === 'Unknown' ? '❓ Unknown' : p.name;
    const fontSize = Math.max(14, dispW * 0.018);
    ctx.font       = `bold ${fontSize}px sans-serif`;
    const textW    = ctx.measureText(label).width;
    const padX = 8, padY = 6;
    const boxH = fontSize + padY * 2;

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x - 1, y - boxH - 2, textW + padX * 2, boxH, 6);
    ctx.fill();

    // label text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + padX - 1, y - padY - 2);
  });
}

function setLiveStatus(state) {
  const el = document.getElementById('live-status');
  if (!el) return;
  const map = {
    live:        '🟢 Live — Scanning…',
    recognizing: '🔵 Recognizing…',
    error:       '🔴 Error — retrying…',
  };
  el.textContent = map[state] || '';
}

/* ═══════════════════ UPLOAD ════════════════════════════════ */

function handleDragOver(e)  { e.preventDefault(); document.getElementById('upload-zone').classList.add('drag-over'); }
function handleDragLeave(e) { document.getElementById('upload-zone').classList.remove('drag-over'); }
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadUploadedFile(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) loadUploadedFile(file);
}

function loadUploadedFile(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('upload-result-img').src = '';
    document.getElementById('preview-wrapper').style.display = 'flex';
    document.getElementById('upload-names').style.display    = 'none';
    document.getElementById('btn-recognize-upload').disabled = false;
    clearNames('upload-names');
  };
  reader.readAsDataURL(file);
}

async function recognizeFromUpload() {
  if (!uploadedFile) return;
  const loading = document.getElementById('upload-loading');
  const btn     = document.getElementById('btn-recognize-upload');
  const names   = document.getElementById('upload-names');

  btn.disabled = true;
  loading.style.display = 'flex';
  names.style.display   = 'none';
  clearNames('upload-names');

  try {
    const dataUrl = await fileToDataUrl(uploadedFile);
    const res     = await fetch('/api/recognize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    document.getElementById('upload-result-img').src = data.image;
    renderNames('upload-names', data.persons);
    names.style.display = 'flex';
  } catch (e) {
    showError(e.message);
  } finally {
    loading.style.display = 'none';
    btn.disabled = false;
  }
}

function resetUpload() {
  uploadedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('preview-wrapper').style.display = 'none';
  document.getElementById('upload-names').style.display    = 'none';
  document.getElementById('btn-recognize-upload').disabled = true;
  clearNames('upload-names');
}

/* ─── Helpers ────────────────────────────────────────────── */
function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload  = e => resolve(e.target.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function renderNames(containerId, persons) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!persons || persons.length === 0) {
    container.innerHTML = '<p class="no-face">No faces detected.</p>';
    return;
  }
  persons.forEach(p => {
    const chip = document.createElement('span');
    chip.className = `name-chip ${p.name === 'Unknown' ? 'unknown' : 'known'}`;
    chip.textContent = p.name === 'Unknown' ? '❓ Unknown' : '✅ ' + p.name;
    container.appendChild(chip);
  });
}

function clearNames(containerId) {
  document.getElementById(containerId).innerHTML = '';
}

function showError(msg) { alert('Error: ' + msg); }