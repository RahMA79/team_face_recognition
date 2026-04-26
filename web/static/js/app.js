/* ─── State ──────────────────────────────────────────────────── */
let stream = null;
let uploadedFile = null;

/* ─── Tab switching ──────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
}

/* ═══════════════════ CAMERA ════════════════════════════════════ */

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    const video = document.getElementById('video');
    video.srcObject = stream;
    await video.play();

    document.getElementById('camera-overlay').classList.add('hidden');
    document.getElementById('btn-start-camera').style.display = 'none';
    document.getElementById('btn-stop-camera').style.display  = '';
    document.getElementById('btn-recognize-camera').disabled  = false;
    document.getElementById('camera-result').style.display    = 'none';
    clearNames('camera-names');
  } catch (err) {
    alert('Cannot access camera: ' + err.message);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  const video = document.getElementById('video');
  video.srcObject = null;

  document.getElementById('camera-overlay').classList.remove('hidden');
  document.getElementById('btn-start-camera').style.display  = '';
  document.getElementById('btn-stop-camera').style.display   = 'none';
  document.getElementById('btn-recognize-camera').disabled   = true;
  document.getElementById('camera-result').style.display     = 'none';
  clearNames('camera-names');
}

function captureFrame() {
  const video  = document.getElementById('video');
  const canvas = document.getElementById('canvas-camera');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.92);
}

async function recognizeFromCamera() {
  const loading = document.getElementById('camera-loading');
  const result  = document.getElementById('camera-result');
  const btn     = document.getElementById('btn-recognize-camera');

  const dataUrl = captureFrame();
  btn.disabled  = true;
  loading.style.display = 'flex';
  result.style.display  = 'none';
  clearNames('camera-names');

  try {
    const res  = await fetch('/api/recognize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();

    if (data.error) { showError(data.error); return; }

    document.getElementById('camera-result-img').src = data.image;
    result.style.display = '';
    renderNames('camera-names', data.persons);
  } catch (e) {
    showError(e.message);
  } finally {
    loading.style.display = 'none';
    btn.disabled = false;
  }
}

/* ═══════════════════ UPLOAD ════════════════════════════════════ */

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

/* ─── Helpers ────────────────────────────────────────────────── */

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
    container.innerHTML = '<p class="no-face">No faces detected in this image.</p>';
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

function showError(msg) {
  alert('Error: ' + msg);
}
