/**
 * Kokoro TTS — Frontend Logic
 *
 * API flow (Gradio SSE v3):
 *  1. POST /gradio_api/upload  → returns [{path, url, orig_name}]
 *  2. POST /gradio_api/queue/join  → {"event_id": "..."}
 *  3. GET  /gradio_api/queue/data?session_hash=<uuid>  (SSE stream)
 *     Events:
 *       process_starts   → queued
 *       process_generating / heartbeat → in-progress
 *       process_completed → {output: {data: [audioFileData, txtFileData, statusStr]}}
 *       process_error    → error
 */

const BASE = 'https://audio8899-kokorov2.hf.space';

/* ── DOM refs ──────────────────────────────────────────────── */
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const dzDefault     = document.getElementById('dz-default');
const dzFile        = document.getElementById('dz-file');
const dzFileName    = document.getElementById('dz-file-name');
const dzFileSize    = document.getElementById('dz-file-size');
const dzFileIcon    = document.getElementById('dz-file-icon');
const dzRemove      = document.getElementById('dz-remove');
const convertBtn    = document.getElementById('convert-btn');

const progressSec   = document.getElementById('progress-section');
const progressBar   = document.getElementById('progress-bar');
const progressBarW  = document.getElementById('progress-bar-wrap');
const progressStatus= document.getElementById('progress-status');

const resultsSec    = document.getElementById('results-section');
const audioPlayer   = document.getElementById('audio-player');
const downloadAudio = document.getElementById('download-audio');
const downloadTxt   = document.getElementById('download-txt');
const resetBtn      = document.getElementById('reset-btn');
const resultsStatusText = document.getElementById('results-status-text');

const errorSec      = document.getElementById('error-section');
const errorMsg      = document.getElementById('error-msg');
const errorResetBtn = document.getElementById('error-reset-btn');

/* ── State ─────────────────────────────────────────────────── */
let selectedFile = null;
let currentSSE   = null;

/* ── File icon map ─────────────────────────────────────────── */
const FILE_ICONS = {
  'txt':  '📄',
  'pdf':  '📑',
  'png':  '🖼️',
  'jpg':  '🖼️',
  'jpeg': '🖼️',
};

function getExt(name) { return (name.split('.').pop() || '').toLowerCase(); }

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/* ── UUID helper ───────────────────────────────────────────── */
function uuid() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
      });
}

/* ── UI helpers ─────────────────────────────────────────────── */
function showFile(file) {
  selectedFile = file;
  dzDefault.hidden = true;
  dzFile.hidden = false;
  dzFileName.textContent = file.name;
  dzFileSize.textContent = formatBytes(file.size);
  dzFileIcon.textContent = FILE_ICONS[getExt(file.name)] || '📄';
  dropZone.classList.add('has-file');
  convertBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  dzDefault.hidden = false;
  dzFile.hidden = true;
  dropZone.classList.remove('has-file', 'drag-over');
  convertBtn.disabled = true;
}

function resetUI() {
  if (currentSSE) { currentSSE.close(); currentSSE = null; }
  clearFile();
  progressSec.hidden  = true;
  resultsSec.hidden   = true;
  errorSec.hidden     = true;
  convertBtn.classList.remove('loading');
  convertBtn.disabled = true;
  setProgress(0, 'Uploading file');
  setStep('upload', 'pending');
  setStep('extract', 'pending');
  setStep('synth', 'pending');
  setStep('done', 'pending');
  setLines(false, false, false);
  audioPlayer.src = '';
}

function showError(msg) {
  progressSec.hidden  = true;
  resultsSec.hidden   = true;
  errorSec.hidden     = false;
  errorMsg.textContent = msg;
  convertBtn.classList.remove('loading');
  convertBtn.disabled = false;
}

/* ── Progress / steps ──────────────────────────────────────── */
function setProgress(pct, label) {
  progressBar.style.width = `${Math.min(pct, 100)}%`;
  progressBarW.setAttribute('aria-valuenow', pct);
  if (label) progressStatus.textContent = label;
}

// state: 'pending' | 'active' | 'done'
function setStep(name, state) {
  const el = document.getElementById(`step-${name}`);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (state === 'active') el.classList.add('active');
  if (state === 'done')   el.classList.add('done');
}

function setLines(l1, l2, l3) {
  const lines = document.querySelectorAll('.step-line');
  if (lines[0]) lines[0].classList.toggle('active', l1);
  if (lines[1]) lines[1].classList.toggle('active', l2);
  if (lines[2]) lines[2].classList.toggle('active', l3);
}

/* ── Drag and drop ──────────────────────────────────────────── */
dropZone.addEventListener('dragenter', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', e => {
  if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file) showFile(file);
});

dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) showFile(fileInput.files[0]);
});

dzRemove.addEventListener('click', e => {
  e.stopPropagation();
  e.preventDefault();
  clearFile();
});

/* ── Reset buttons ─────────────────────────────────────────── */
resetBtn.addEventListener('click', resetUI);
errorResetBtn.addEventListener('click', resetUI);

/* ── Main conversion flow ────────────────────────────────────── */
convertBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  await runConversion(selectedFile);
});

async function runConversion(file) {
  // UI: loading state
  convertBtn.classList.add('loading');
  convertBtn.disabled = true;
  progressSec.hidden  = false;
  resultsSec.hidden   = true;
  errorSec.hidden     = true;

  setStep('upload', 'active');
  setProgress(5, 'Uploading file…');

  try {
    /* ── Step 1: Upload file ── */
    const uploadedPath = await uploadFile(file);
    setStep('upload', 'done');
    setStep('extract', 'active');
    setLines(true, false, false);
    setProgress(25, 'Extracting text…');

    /* ── Step 2: Queue job ── */
    const sessionHash = uuid();
    const eventId = await queueJob(uploadedPath, file.name, sessionHash);
    setProgress(35, 'Job queued — waiting for model…');

    /* ── Step 3: Stream results ── */
    await streamResults(sessionHash, eventId);

  } catch (err) {
    console.error(err);
    showError(err.message || 'An unexpected error occurred. Please try again.');
  }
}

/* ── Step 1: Upload ────────────────────────────────────────── */
async function uploadFile(file) {
  const fd = new FormData();
  fd.append('files', file, file.name);

  const res = await fetch(`${BASE}/gradio_api/upload`, {
    method: 'POST',
    body: fd,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Upload failed (${res.status}): ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  // Returns array of paths, e.g. ["tmp/xyz/filename.pdf"]
  if (!Array.isArray(data) || !data[0]) {
    throw new Error('Unexpected upload response from server.');
  }
  return typeof data[0] === 'string' ? data[0] : data[0].path;
}

/* ── Step 2: Queue job ─────────────────────────────────────── */
async function queueJob(serverPath, origName, sessionHash) {
  const payload = {
    data: [{
      path: serverPath,
      orig_name: origName,
      meta: { _type: 'gradio.FileData' }
    }],
    fn_index: 0,
    session_hash: sessionHash,
    trigger_id: null,
    event_data: null,
  };

  const res = await fetch(`${BASE}/gradio_api/queue/join`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Queue join failed (${res.status}): ${text.slice(0, 200)}`);
  }

  const json = await res.json();
  return json.event_id;
}

/* ── Step 3: SSE stream ────────────────────────────────────── */
function streamResults(sessionHash, eventId) {
  return new Promise((resolve, reject) => {
    const url = `${BASE}/gradio_api/queue/data?session_hash=${encodeURIComponent(sessionHash)}`;
    const sse  = new EventSource(url);
    currentSSE = sse;

    let synthStarted = false;

    sse.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }

      const type = msg.msg;

      if (type === 'send_hash') {
        // Nothing needed
        return;
      }

      if (type === 'estimation') {
        const rank = msg.rank ?? 0;
        if (rank > 0) {
          setProgress(38, `Queue position: ${rank}…`);
        }
        return;
      }

      if (type === 'process_starts') {
        setStep('extract', 'done');
        setStep('synth', 'active');
        setLines(true, true, false);
        setProgress(50, 'Synthesizing speech…');
        synthStarted = true;
        return;
      }

      if (type === 'heartbeat' || type === 'process_generating') {
        if (!synthStarted) {
          setStep('synth', 'active');
          synthStarted = true;
        }
        // Animate progress between 50 and 88
        const curr = parseFloat(progressBar.style.width || '50');
        if (curr < 88) setProgress(Math.min(curr + 4, 88), 'Synthesizing speech…');
        return;
      }

      if (type === 'process_completed') {
        sse.close();
        currentSSE = null;

        const output = msg.output;
        if (output?.error) {
          reject(new Error(output.error));
          return;
        }

        const data = output?.data;
        if (!Array.isArray(data) || data.length < 3) {
          reject(new Error('Unexpected response structure from model.'));
          return;
        }

        setStep('synth', 'done');
        setStep('done', 'done');
        setLines(true, true, true);
        setProgress(100, 'Complete!');

        handleCompletion(data[0], data[1], data[2]);
        resolve();
        return;
      }

      if (type === 'process_error' || type === 'error') {
        sse.close();
        currentSSE = null;
        reject(new Error(msg.output?.error || 'Processing error on the server.'));
        return;
      }
    };

    sse.onerror = (e) => {
      sse.close();
      currentSSE = null;
      reject(new Error('Connection to model server lost. The space may be waking up — please try again in a moment.'));
    };

    // Safety timeout: 10 minutes (CPU can be slow)
    const timeout = setTimeout(() => {
      sse.close();
      currentSSE = null;
      reject(new Error('Request timed out after 10 minutes. Large files may take longer — try a smaller file or try again.'));
    }, 10 * 60 * 1000);

    // Clear timeout on completion
    const origOnMessage = sse.onmessage;
    sse.onmessage = (e) => {
      origOnMessage(e);
      // Check if we resolved/rejected — if sse is closed, clear timeout
      if (sse.readyState === EventSource.CLOSED) clearTimeout(timeout);
    };
  });
}

/* ── Handle results ────────────────────────────────────────── */
function handleCompletion(audioData, txtData, statusText) {
  convertBtn.classList.remove('loading');
  progressSec.hidden = true;
  resultsSec.hidden  = false;

  resultsStatusText.textContent = typeof statusText === 'string'
    ? statusText
    : 'Your document has been converted to speech.';

  // Audio
  if (audioData?.url) {
    audioPlayer.src = audioData.url;
    downloadAudio.href = audioData.url;
    downloadAudio.download = audioData.orig_name || 'kokoro_speech.mp3';
  } else if (audioData?.path) {
    const audioUrl = `${BASE}/gradio_api/file=${encodeURIComponent(audioData.path)}`;
    audioPlayer.src = audioUrl;
    downloadAudio.href = audioUrl;
    downloadAudio.download = audioData.orig_name || 'kokoro_speech.mp3';
  }

  // Text file
  if (txtData?.url) {
    downloadTxt.href = txtData.url;
    downloadTxt.download = txtData.orig_name || 'processed_text.txt';
  } else if (txtData?.path) {
    const txtUrl = `${BASE}/gradio_api/file=${encodeURIComponent(txtData.path)}`;
    downloadTxt.href = txtUrl;
    downloadTxt.download = txtData.orig_name || 'processed_text.txt';
  } else {
    // Hide txt download if not available
    downloadTxt.hidden = true;
  }

  // Smooth scroll to results
  resultsSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
