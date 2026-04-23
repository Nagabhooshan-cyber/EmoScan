/* ═══════════════════════════════════════════════════════════════════════════
   EmoScan — script.js
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────
const API_URL        = '/predict';
const STATUS_URL     = '/status';
const FRAME_INTERVAL = 80;      // ms — draw loop runs fast for smooth boxes
const CANVAS_QUALITY = 0.75;

const EMOTION_COLORS = {
  Angry:    '#ff3c3c',
  Disgust:  '#a0e040',
  Fear:     '#b060ff',
  Happy:    '#ffe040',
  Sad:      '#4090ff',
  Surprise: '#ff9040',
  Neutral:  '#80c0ff',
};
const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

// ── DOM refs ──────────────────────────────────────────────────────────────
const btnScan          = document.getElementById('btnScan');
const btnStop          = document.getElementById('btnStop');
const video            = document.getElementById('video');
const overlay          = document.getElementById('overlay');
const arenaPlaceholder = document.getElementById('arenaPlaceholder');
const videoWrapper     = document.getElementById('videoWrapper');
const scanLine         = document.getElementById('scanLine');
const statsBar         = document.getElementById('statsBar');
const emotionPanel     = document.getElementById('emotionPanel');
const emotionBarsEl    = document.getElementById('emotionBars');
const statusDot        = document.getElementById('statusDot');
const statusText       = document.getElementById('statusText');
const toast            = document.getElementById('toast');
const statFaces        = document.getElementById('statFaces');
const statEmotion      = document.getElementById('statEmotion');
const statConf         = document.getElementById('statConf');
const statFps          = document.getElementById('statFps');

// ── State ─────────────────────────────────────────────────────────────────
let stream       = null;
let animFrameId  = null;   // requestAnimationFrame handle
let capturing    = false;
let fetchPending = false;  // prevent overlapping fetches
let lastFaces    = [];     // cache — redrawn every rAF tick so no blinking
let fpsCounter   = { frames: 0, last: performance.now() };
let lastSendTime = 0;

const captureCanvas = document.createElement('canvas');
const captureCtx    = captureCanvas.getContext('2d');

// ── Backend status ────────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const res  = await fetch(STATUS_URL);
    const data = await res.json();
    if (data.model_loaded) {
      setStatus('ready', 'Model ready');
    } else {
      setStatus('warn', 'Demo mode (no model)');
    }
  } catch {
    setStatus('error', 'Backend offline');
    showToast('⚠ Cannot reach backend. Make sure Flask is running on port 5000.');
  }
}

function setStatus(type, text) {
  statusDot.className    = 'status-dot ' + type;
  statusText.textContent = text;
}

let toastTimer = null;
function showToast(msg, duration = 4000) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), duration);
}

// ── Camera ────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await video.play();

    arenaPlaceholder.style.display = 'none';
    videoWrapper.style.display     = 'flex';
    scanLine.style.display         = 'block';
    btnScan.style.display          = 'none';
    btnStop.style.display          = 'flex';
    statsBar.style.display         = 'flex';
    emotionPanel.style.display     = 'block';

    buildEmotionBars();
    setStatus('running', 'Scanning…');
    capturing = true;
    animFrameId = requestAnimationFrame(renderLoop);

  } catch (err) {
    handleCameraError(err);
  }
}

function stopCamera() {
  capturing    = false;
  fetchPending = false;
  lastFaces    = [];
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;

  arenaPlaceholder.style.display = 'flex';
  videoWrapper.style.display     = 'none';
  scanLine.style.display         = 'none';
  btnStop.style.display          = 'none';
  btnScan.style.display          = 'flex';
  statsBar.style.display         = 'none';
  emotionPanel.style.display     = 'none';

  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  setStatus('ready', 'Model ready');
}

function handleCameraError(err) {
  let msg = '⚠ Could not access camera. ';
  if (err.name === 'NotAllowedError')       msg += 'Permission was denied.';
  else if (err.name === 'NotFoundError')    msg += 'No camera found.';
  else if (err.name === 'NotReadableError') msg += 'Camera already in use.';
  else msg += err.message;
  showToast(msg, 6000);
  setStatus('error', 'Camera error');
}

// ── Render loop (requestAnimationFrame) ───────────────────────────────────
// Runs every display frame (~60fps). Always redraws lastFaces so boxes
// never disappear between backend responses. Sends to backend at most
// every FRAME_INTERVAL ms and only when no fetch is in flight.
function renderLoop(now) {
  if (!capturing) return;

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (vw && vh) {
    overlay.width  = vw;
    overlay.height = vh;

    // Always draw the last known faces — this is what stops blinking
    drawFaces(lastFaces, vw, vh);

    // Send a new frame to backend only when ready and enough time has passed
    if (!fetchPending && (now - lastSendTime) >= FRAME_INTERVAL) {
      lastSendTime = now;
      captureCanvas.width  = vw;
      captureCanvas.height = vh;
      captureCtx.drawImage(video, 0, 0, vw, vh);
      const dataURL = captureCanvas.toDataURL('image/jpeg', CANVAS_QUALITY);

      fetchPending = true;
      fetch(API_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image: dataURL }),
      })
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then(data => { lastFaces = data.faces || []; updateStats(lastFaces); })
        .catch(e => console.warn('[EmoScan]', e))
        .finally(() => { fetchPending = false; });
    }

    // FPS
    fpsCounter.frames++;
    if (now - fpsCounter.last >= 1000) {
      statFps.textContent = fpsCounter.frames;
      fpsCounter.frames   = 0;
      fpsCounter.last     = now;
    }
  }

  animFrameId = requestAnimationFrame(renderLoop);
}

// ── Draw faces on canvas ──────────────────────────────────────────────────
// The <video> is CSS-mirrored with scaleX(-1).
// The <canvas> is NOT mirrored in CSS.
// Strategy:
//   1. Mirror face X coords manually: mx = vw - x - w
//   2. Draw box & accents in canvas space (un-mirrored)
//   3. For text only: flip the context, translate, draw — text reads correctly
function drawFaces(faces, vw, vh) {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, vw, vh);

  faces.forEach(({ x, y, w, h, emotion, confidence }) => {
    const color = EMOTION_COLORS[emotion] || '#00e5ff';
    const pct   = Math.round(confidence * 100);

    // Mirrored left edge of the face box
    const mx = vw - x - w;

    // ── Bounding box ───────────────────────────────────────────────────
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 10;
    ctx.strokeRect(mx, y, w, h);
    ctx.shadowBlur  = 0;
    ctx.restore();

    // ── Corner accents ─────────────────────────────────────────────────
    drawCorners(ctx, mx, y, w, h, color, 14);

    // ── Confidence bar (bottom of box) ─────────────────────────────────
    const barH = 4;
    const barY = y + h - barH - 6;
    ctx.fillStyle = 'rgba(0,0,0,.5)';
    ctx.fillRect(mx + 6, barY, w - 12, barH);
    ctx.fillStyle = color;
    ctx.fillRect(mx + 6, barY, (w - 12) * confidence, barH);

    // ── Label (text must read left-to-right despite mirror) ────────────
    const label    = `${emotion}  ${pct}%`;
    const fontSize = Math.max(12, Math.round(w * 0.12));
    ctx.font = `bold ${fontSize}px "Space Mono", monospace`;
    const textW  = ctx.measureText(label).width;
    const padH   = fontSize + 10;
    // Place label above box; if not enough room place it below
    const labelY = y > padH + 2 ? y - padH : y + h;

    // Draw pill background and text in normal (un-flipped) canvas space.
    // The mirrored left edge of the box is mx, so we place the pill starting at mx.
    const pillW = textW + 16;

    ctx.save();

    // Pill background
    ctx.fillStyle = color + 'dd';
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(mx, labelY, pillW, padH, 4);
    } else {
      ctx.rect(mx, labelY, pillW, padH);
    }
    ctx.fill();

    // Text — drawn in normal orientation so it reads left-to-right
    ctx.fillStyle    = '#090909';
    ctx.textBaseline = 'top';
    ctx.fillText(label, mx + 8, labelY + 5);

    ctx.restore();
  });
}

function drawCorners(ctx, x, y, w, h, color, s) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 3;
  // top-left
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + s, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + s); ctx.stroke();
  // top-right
  ctx.beginPath(); ctx.moveTo(x+w, y); ctx.lineTo(x+w-s, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x+w, y); ctx.lineTo(x+w, y+s); ctx.stroke();
  // bottom-left
  ctx.beginPath(); ctx.moveTo(x, y+h); ctx.lineTo(x+s, y+h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y+h); ctx.lineTo(x, y+h-s); ctx.stroke();
  // bottom-right
  ctx.beginPath(); ctx.moveTo(x+w, y+h); ctx.lineTo(x+w-s, y+h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x+w, y+h); ctx.lineTo(x+w, y+h-s); ctx.stroke();
  ctx.restore();
}

// ── Stats ─────────────────────────────────────────────────────────────────
function updateStats(faces) {
  statFaces.textContent = faces.length;
  if (!faces.length) {
    statEmotion.textContent = '—';
    statConf.textContent    = '—';
    resetEmotionBars();
    return;
  }
  const best = faces.reduce((a, b) => a.confidence > b.confidence ? a : b);
  statEmotion.textContent = best.emotion;
  statEmotion.style.color = EMOTION_COLORS[best.emotion] || 'var(--accent)';
  statConf.textContent    = Math.round(best.confidence * 100) + '%';
  if (best.all_probs) updateEmotionBars(best.all_probs);
}

// ── Emotion bars ──────────────────────────────────────────────────────────
function buildEmotionBars() {
  emotionBarsEl.innerHTML = '';
  EMOTIONS.forEach(em => {
    const row = document.createElement('div');
    row.className = 'emotion-row';
    row.innerHTML = `
      <span class="emotion-name">${em}</span>
      <div class="bar-track">
        <div class="bar-fill" id="bar-${em}" style="width:0%;background:${EMOTION_COLORS[em]}"></div>
      </div>
      <span class="bar-pct" id="pct-${em}">0%</span>`;
    emotionBarsEl.appendChild(row);
  });
}

function updateEmotionBars(probs) {
  EMOTIONS.forEach(em => {
    const val = (probs[em] || 0) * 100;
    const bar = document.getElementById(`bar-${em}`);
    const pct = document.getElementById(`pct-${em}`);
    if (bar) bar.style.width = val.toFixed(1) + '%';
    if (pct) pct.textContent = val.toFixed(0) + '%';
  });
}

function resetEmotionBars() {
  EMOTIONS.forEach(em => {
    const bar = document.getElementById(`bar-${em}`);
    const pct = document.getElementById(`pct-${em}`);
    if (bar) bar.style.width = '0%';
    if (pct) pct.textContent = '0%';
  });
}

// ── Event listeners ───────────────────────────────────────────────────────
btnScan.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
video.addEventListener('loadedmetadata', () => {
  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;
});

// ── Boot ──────────────────────────────────────────────────────────────────
checkBackend();