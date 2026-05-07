/**
 * Kokoro TTS — app.js
 * Gradio SSE v3 API client + upload timing UI
 */

const BASE = 'https://audio8899-kokorov2.hf.space';

/* ── DOM refs ── */
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const dzIdle        = document.getElementById('dz-idle');
const dzDragover    = document.getElementById('dz-dragover');
const dzFile        = document.getElementById('dz-file');
const dzFileEmoji   = document.getElementById('dz-file-emoji');
const dzFileName    = document.getElementById('dz-file-name');
const dzFileSize    = document.getElementById('dz-file-size');
const dzFileType    = document.getElementById('dz-file-type');
const dzRemove      = document.getElementById('dz-remove');
const convertBtn    = document.getElementById('convert-btn');

const phaseUpload   = document.getElementById('phase-upload');
const phaseProgress = document.getElementById('phase-progress');
const phaseResults  = document.getElementById('phase-results');
const phaseError    = document.getElementById('phase-error');

const procLabel     = document.getElementById('proc-label');
const progressFill  = document.getElementById('progress-fill');
const progressGlow  = document.getElementById('progress-glow');
const progressPct   = document.getElementById('progress-pct');

const stepUpload    = document.getElementById('step-upload');
const stepExtract   = document.getElementById('step-extract');
const stepSynth     = document.getElementById('step-synth');
const stepDone      = document.getElementById('step-done');
const conn1         = document.getElementById('conn-1');
const conn2         = document.getElementById('conn-2');
const conn3         = document.getElementById('conn-3');

const audioPlayer   = document.getElementById('audio-player');
const downloadAudio = document.getElementById('download-audio');
const downloadTxt   = document.getElementById('download-txt');
const resetBtn      = document.getElementById('reset-btn');
const errorResetBtn = document.getElementById('error-reset-btn');
const errorMsg      = document.getElementById('error-msg');
const resultStatusText = document.getElementById('result-status-text');

/* Custom player */
const apcPlay     = document.getElementById('apc-play');
const playIcon    = document.getElementById('play-icon');
const pauseIcon   = document.getElementById('pause-icon');
const apcSeek     = document.getElementById('apc-seek');
const apcCurrent  = document.getElementById('apc-current');
const apcDuration = document.getElementById('apc-duration');
const apcVol      = document.getElementById('apc-vol');
const vizIdle     = document.getElementById('viz-idle');

/* ── State ── */
let selectedFile = null;
let currentSSE   = null;
let audioCtx     = null;
let analyser     = null;
let vizRaf       = null;
let uploadStartTime = null;

/* ── Helpers ── */
const FILE_ICONS = { txt:'📄', pdf:'📑', png:'🖼️', jpg:'🖼️', jpeg:'🖼️' };
const FILE_LABELS = { txt:'TXT', pdf:'PDF', png:'PNG', jpg:'JPG', jpeg:'JPEG' };

function getExt(n) { return (n.split('.').pop() || '').toLowerCase(); }

function formatBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(2)} MB`;
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2,'0')}`;
}

function generateSessionHash() {
  return Math.random().toString(36).substring(2, 12);
}

/* ── Phase switcher ── */
function showPhase(name) {
  phaseUpload.hidden   = name !== 'upload';
  phaseProgress.hidden = name !== 'progress';
  phaseResults.hidden  = name !== 'results';
  phaseError.hidden    = name !== 'error';
}

/* ── Progress bar ── */
function setProgress(pct, label) {
  const v = Math.min(Math.max(pct, 0), 100);
  progressFill.style.width = `${v}%`;
  progressGlow.style.left  = `${v}%`;
  progressPct.textContent  = `${Math.round(v)}%`;
  if (label) procLabel.textContent = label;
}

/* ── Step state: 'idle' | 'active' | 'done' ── */
function setStep(el, state) {
  if (!el) return;
  el.classList.remove('active','done');
  if (state === 'active') el.classList.add('active');
  if (state === 'done')   el.classList.add('done');
}
function setConn(el, active) {
  if (el) el.classList.toggle('active', active);
}

/* ── File selected UI ── */
function showFile(file) {
  selectedFile = file;
  const ext = getExt(file.name);
  dzIdle.hidden     = true;
  dzDragover.hidden = true;
  dzFile.hidden     = false;
  dzFileEmoji.textContent = FILE_ICONS[ext] || '📄';
  dzFileName.textContent  = file.name;
  dzFileSize.textContent  = formatBytes(file.size);
  dzFileType.textContent  = FILE_LABELS[ext] || ext.toUpperCase();
  dropZone.classList.add('has-file');
  convertBtn.disabled = false;
  // Animate the file bar fill
  const bar = dzFile.querySelector('.dz-file-bar-fill');
  if (bar) { bar.style.width = '0'; requestAnimationFrame(() => { bar.style.width = '100%'; }); }
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  dzIdle.hidden     = false;
  dzDragover.hidden = true;
  dzFile.hidden     = true;
  dropZone.classList.remove('has-file','drag-over');
  convertBtn.disabled = true;
}

function resetUI() {
  if (currentSSE) { currentSSE.close(); currentSSE = null; }
  stopVisualizer();
  clearFile();
  showPhase('upload');
  convertBtn.classList.remove('loading');
  setProgress(0, 'Starting…');
  [stepUpload, stepExtract, stepSynth, stepDone].forEach(s => setStep(s,'idle'));
  [conn1, conn2, conn3].forEach(c => setConn(c, false));
  audioPlayer.src = '';
  audioPlayer.pause();
  uploadStartTime = null;
}

function showError(msg) {
  console.error('[Kokoro Error]', msg);
  showPhase('error');
  errorMsg.textContent = msg;
  convertBtn.classList.remove('loading');
}

/* ── Drag & drop ── */
dropZone.addEventListener('dragenter', e => {
  e.preventDefault();
  dzIdle.hidden = true; dzDragover.hidden = false;
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); });
dropZone.addEventListener('dragleave', e => {
  if (!dropZone.contains(e.relatedTarget)) {
    dzIdle.hidden = false; dzDragover.hidden = true;
    dropZone.classList.remove('drag-over');
  }
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer?.files?.[0];
  if (f) showFile(f);
  else { dzIdle.hidden = false; dzDragover.hidden = true; }
});
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) showFile(fileInput.files[0]); });
dzRemove.addEventListener('click', e => { e.stopPropagation(); e.preventDefault(); clearFile(); });

/* ── Reset ── */
resetBtn.addEventListener('click', resetUI);
errorResetBtn.addEventListener('click', resetUI);

/* ── Convert ── */
convertBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  await runConversion(selectedFile);
});

/* ══════════════════════════════════════════════
   MAIN CONVERSION FLOW
═══════════════════════════════════════════════ */
async function runConversion(file) {
  convertBtn.classList.add('loading');
  convertBtn.disabled = true;
  showPhase('progress');

  setStep(stepUpload, 'active');
  setProgress(3, 'Waking up space…');

  try {
    /* ── 0. Wake up check ── */
    // A quick fetch to ensure the space is alive
    await fetch(`${BASE}/config`).catch(() => {
        // If it fails, wait a bit and retry once
        return new Promise(r => setTimeout(r, 2000)).then(() => fetch(`${BASE}/config`));
    }).catch(() => {
        throw new Error('Hugging Face Space is currently sleeping or unavailable. Please try again in 30 seconds.');
    });

    /* ── 1. UPLOAD with timing ── */
    uploadStartTime = performance.now();
    const uploadData = await uploadFile(file);
    const uploadMs   = performance.now() - uploadStartTime;
    const uploadSec  = (uploadMs / 1000).toFixed(1);

    setStep(stepUpload, 'done');
    setConn(conn1, true);
    setStep(stepExtract, 'active');
    setProgress(28, `Uploaded in ${uploadSec}s — Extracting text…`);

    /* ── 2. QUEUE ── */
    const sessionHash = generateSessionHash();
    console.log('[Kokoro] Starting session:', sessionHash);
    await queueJob(uploadData, sessionHash);
    setProgress(38, 'Job queued — waiting for model…');

    /* ── 3. STREAM ── */
    await streamResults(sessionHash);

  } catch (err) {
    console.error(err);
    showError(err.message || 'An unexpected error occurred. Please try again.');
  }
}

/* ── Upload (with XHR for real progress) ── */
function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd  = new FormData();
    fd.append('files', file, file.name);

    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        const pct = (e.loaded / e.total) * 25; // 0-25%
        const elapsed = ((performance.now() - uploadStartTime) / 1000).toFixed(1);
        setProgress(pct, `Uploading… ${Math.round(e.loaded/1024)}KB / ${Math.round(e.total/1024)}KB · ${elapsed}s`);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        return reject(new Error(`Upload failed (${xhr.status}). The space might be waking up.`));
      }
      try {
        const data = JSON.parse(xhr.responseText);
        if (!Array.isArray(data) || data.length === 0) throw new Error('Bad upload response');
        // Return the first path/object
        resolve(data[0]);
      } catch (e) { reject(e); }
    });

    xhr.addEventListener('error', () => reject(new Error('Network error during upload. Check your internet or the HF Space status.')));
    xhr.open('POST', `${BASE}/upload`);
    xhr.send(fd);
  });
}


/* ── Queue job ── */
async function queueJob(uploadData, sessionHash) {
  // Ensure we have a valid FileData object
  let fileData;
  if (typeof uploadData === 'string') {
    fileData = { path: uploadData, meta: { _type: 'gradio.FileData' } };
  } else if (uploadData && typeof uploadData === 'object') {
    fileData = uploadData;
    // Ensure meta is present
    if (!fileData.meta) fileData.meta = { _type: 'gradio.FileData' };
  } else {
    throw new Error('Invalid upload data received from server.');
  }

  const payload = {
    data: [fileData],
    fn_index: 0,
    api_name: 'run_pipeline', // Using api_name is more robust in Gradio 4
    session_hash: sessionHash,
    trigger_id: null,
    event_data: null,
  };

  console.log('[Kokoro] Joining queue with payload:', payload);

  const res = await fetch(`${BASE}/queue/join`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => 'No error detail');
    console.error('[Kokoro] Queue join failed:', res.status, errText);
    throw new Error(`Queue join failed (${res.status}). The server might be waking up or overloaded.`);
  }

  const json = await res.json();
  console.log('[Kokoro] Queue join success, event_id:', json.event_id);
  return json.event_id;
}

/* ── SSE stream ── */
function streamResults(sessionHash) {
  return new Promise((resolve, reject) => {
    const sseUrl = `${BASE}/queue/data?session_hash=${encodeURIComponent(sessionHash)}`;
    console.log('[Kokoro] Opening SSE connection:', sseUrl);
    const sse = new EventSource(sseUrl);
    currentSSE = sse;
    let synthStarted = false;

    const timeout = setTimeout(() => {
      sse.close(); currentSSE = null;
      reject(new Error('Timed out after 10 minutes. The model is taking too long to respond.'));
    }, 10 * 60 * 1000);

    sse.onmessage = e => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }
      const type = msg.msg;

      if (type === 'estimation') {
        const rank = msg.rank ?? 0;
        if (rank > 0) setProgress(40, `Queue position ${rank} — please wait…`);
        return;
      }
      if (type === 'process_starts') {
        setStep(stepExtract, 'done');
        setConn(conn1, true); // Ensure previous line is active
        setConn(conn2, true);
        setStep(stepSynth, 'active');
        setProgress(52, 'Synthesizing speech…');
        synthStarted = true;
        return;
      }
      if (type === 'heartbeat' || type === 'process_generating') {
        if (!synthStarted) { 
          setStep(stepExtract, 'done');
          setStep(stepSynth,'active'); 
          setConn(conn2, true);
          synthStarted = true; 
        }
        const cur = parseFloat(progressFill.style.width || '52');
        if (cur < 88) setProgress(Math.min(cur + 2, 88), 'Synthesizing speech…');
        return;
      }
      if (type === 'process_completed') {
        clearTimeout(timeout); sse.close(); currentSSE = null;
        const out = msg.output;
        if (out?.error) return reject(new Error(out.error));
        const data = out?.data;
        if (!Array.isArray(data) || data.length < 3) return reject(new Error('Unexpected response structure from model.'));
        setStep(stepSynth,'done'); setConn(conn3,true); setStep(stepDone,'done');
        setProgress(100, 'Done!');
        setTimeout(() => handleCompletion(data[0], data[1], data[2]), 600);
        resolve();
        return;
      }
      if (type === 'process_error' || type === 'error') {
        clearTimeout(timeout); sse.close(); currentSSE = null;
        reject(new Error(msg.output?.error || 'Server processing error. This can happen with very long texts.'));
      }
    };

    sse.onerror = () => {
      clearTimeout(timeout); sse.close(); currentSSE = null;
      reject(new Error('Connection lost. The space may be waking up or resetting. Please try again.'));
    };
  });
}

/* ── Completion ── */
function handleCompletion(audioData, txtData, statusText) {
  convertBtn.classList.remove('loading');
  showPhase('results');

  /* Status text — include total time */
  const totalSec = uploadStartTime
    ? ((performance.now() - uploadStartTime) / 1000).toFixed(0)
    : null;
  resultStatusText.textContent = totalSec
    ? `Completed in ${totalSec}s · ${typeof statusText === 'string' ? statusText : 'Your audio is ready'}`
    : (typeof statusText === 'string' ? statusText : 'Your audio is ready');

  /* Set audio src */
  let audioUrl = null;
  if (audioData?.url) audioUrl = audioData.url;
  else if (audioData?.path) audioUrl = `${BASE}/file=${encodeURIComponent(audioData.path)}`;

  if (audioUrl) {
    audioPlayer.src = audioUrl;
    downloadAudio.href = audioUrl;
    downloadAudio.download = audioData?.orig_name || 'kokoro_speech.mp3';
  }

  /* Text file */
  if (txtData?.url) {
    downloadTxt.href = txtData.url;
    downloadTxt.download = txtData?.orig_name || 'processed_text.txt';
  } else if (txtData?.path) {
    const u = `${BASE}/file=${encodeURIComponent(txtData.path)}`;
    downloadTxt.href = u;
    downloadTxt.download = txtData?.orig_name || 'processed_text.txt';
  } else {
    downloadTxt.style.display = 'none';
  }

  confetti();
  initPlayer();
  phaseResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ══════════════════════════════════════════════
   CUSTOM AUDIO PLAYER
═══════════════════════════════════════════════ */
function initPlayer() {
  audioPlayer.addEventListener('loadedmetadata', () => {
    apcDuration.textContent = formatTime(audioPlayer.duration);
    apcSeek.max = audioPlayer.duration;
  });
  audioPlayer.addEventListener('timeupdate', () => {
    apcCurrent.textContent = formatTime(audioPlayer.currentTime);
    apcSeek.value = audioPlayer.currentTime;
  });
  audioPlayer.addEventListener('ended', () => {
    playIcon.style.display = ''; pauseIcon.style.display = 'none';
    vizIdle.style.display = 'flex';
    stopVisualizer();
  });
}

apcPlay.addEventListener('click', () => {
  if (audioPlayer.paused) {
    audioPlayer.play();
    playIcon.style.display = 'none'; pauseIcon.style.display = '';
    vizIdle.style.display = 'none';
    startVisualizer();
  } else {
    audioPlayer.pause();
    playIcon.style.display = ''; pauseIcon.style.display = 'none';
    stopVisualizer();
  }
});

apcSeek.addEventListener('input', () => { audioPlayer.currentTime = apcSeek.value; });
apcVol.addEventListener('click', () => { audioPlayer.muted = !audioPlayer.muted; apcVol.style.opacity = audioPlayer.muted ? '0.4' : '1'; });

/* ── Canvas visualizer ── */
function startVisualizer() {
  const canvas = document.getElementById('visualizer');
  if (!canvas) return;
  if (!audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = audioCtx.createMediaElementSource(audioPlayer);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      analyser.connect(audioCtx.destination);
    } catch { return; }
  }
  const ctx  = canvas.getContext('2d');
  const buf  = new Uint8Array(analyser.frequencyBinCount);
  const W    = canvas.width;
  const H    = canvas.height;
  const draw = () => {
    vizRaf = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(buf);
    ctx.clearRect(0, 0, W, H);
    const barW = W / buf.length * 2.5;
    let x = 0;
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#c084fc');
    grad.addColorStop(1, '#38bdf8');
    buf.forEach(v => {
      const h = (v / 255) * H;
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x, H - h, barW - 2, h, 3);
      ctx.fill();
      x += barW + 1;
    });
  };
  draw();
}

function stopVisualizer() {
  if (vizRaf) { cancelAnimationFrame(vizRaf); vizRaf = null; }
  const canvas = document.getElementById('visualizer');
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

/* ── Size canvas on load ── */
window.addEventListener('load', () => {
  const canvas = document.getElementById('visualizer');
  if (canvas) { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; }
});

/* ══════════════════════════════════════════════
   CONFETTI
═══════════════════════════════════════════════ */
function confetti() {
  const container = document.getElementById('confetti-container');
  if (!container) return;
  container.innerHTML = '';
  const colors = ['#c084fc','#38bdf8','#34d399','#fbbf24','#f472b6','#818cf8'];
  for (let i = 0; i < 60; i++) {
    const p = document.createElement('div');
    p.className = 'confetti-piece';
    p.style.cssText = `
      left:${Math.random()*100}%;
      background:${colors[Math.floor(Math.random()*colors.length)]};
      width:${6+Math.random()*6}px;
      height:${6+Math.random()*6}px;
      border-radius:${Math.random()>0.5?'50%':'2px'};
      animation-delay:${Math.random()*0.6}s;
      animation-duration:${0.8+Math.random()*0.8}s;
    `;
    container.appendChild(p);
  }
  setTimeout(() => { if (container) container.innerHTML = ''; }, 2500);
}

/* ══════════════════════════════════════════════
   BACKGROUND CANVAS PARTICLES
═══════════════════════════════════════════════ */
(function initBgCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      r: 1 + Math.random() * 2,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      a: 0.1 + Math.random() * 0.4,
      hue: Math.random() > 0.5 ? 270 : 200,
    };
  }

  function init() { resize(); particles = Array.from({length: 80}, mkParticle); }

  function tick() {
    requestAnimationFrame(tick);
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 70%, 70%, ${p.a})`;
      ctx.fill();
    });
  }

  window.addEventListener('resize', resize);
  init(); tick();
})();

/* ══════════════════════════════════════════════
   TYPEWRITER
═══════════════════════════════════════════════ */
(function typewriter() {
  const el = document.getElementById('typewriter-text');
  const cursor = document.querySelector('.typewriter-cursor');
  if (!el) return;
  const phrases = ['Into Spoken Audio', 'Into Natural Speech', 'Into an Audiobook'];
  let pi = 0, ci = 0, deleting = false;

  function tick() {
    const phrase = phrases[pi];
    if (!deleting) {
      el.textContent = phrase.slice(0, ++ci);
      if (ci === phrase.length) { deleting = true; setTimeout(tick, 2200); return; }
    } else {
      el.textContent = phrase.slice(0, --ci);
      if (ci === 0) { deleting = false; pi = (pi + 1) % phrases.length; }
    }
    setTimeout(tick, deleting ? 45 : 80);
  }
  setTimeout(tick, 800);
  setInterval(() => { if (cursor) cursor.style.opacity = cursor.style.opacity === '0' ? '1' : '0'; }, 500);
})();

/* ══════════════════════════════════════════════
   COUNTER ANIMATION
═══════════════════════════════════════════════ */
(function animateCounters() {
  const els = document.querySelectorAll('.stat-number[data-target]');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const target = parseInt(el.dataset.target, 10);
      let cur = 0;
      const step = Math.ceil(target / 40);
      const t = setInterval(() => {
        cur = Math.min(cur + step, target);
        el.textContent = cur;
        if (cur >= target) clearInterval(t);
      }, 30);
      obs.unobserve(el);
    });
  }, { threshold: 0.5 });
  els.forEach(el => obs.observe(el));
})();

/* ══════════════════════════════════════════════
   CARD TILT
═══════════════════════════════════════════════ */
(function cardTilt() {
  const card = document.getElementById('converter-card');
  if (!card || window.matchMedia('(hover:none)').matches) return;
  card.addEventListener('mousemove', e => {
    const r = card.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width  - 0.5;
    const y = (e.clientY - r.top)  / r.height - 0.5;
    card.style.transform = `perspective(800px) rotateY(${x*6}deg) rotateX(${-y*6}deg) translateZ(8px)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = 'perspective(800px) rotateY(0deg) rotateX(0deg) translateZ(0)';
  });
})();
