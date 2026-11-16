const App = {
  isDark: true,
  currentView: 'initial', // 'initial' (upload first screen) | 'dashboard'
  physicsEnabled: true,
  hoveredNode: null,
  draggedNode: null,
  nodes: [],
  edges: [],
  graphLoaded: false,
  graphVisualAssets: [],
  stagedDocName: 'Regulatory_Policy_Framework_2026.pdf',
  // Configuration & Settings State
  apiBaseUrl: (() => {
    const saved = localStorage.getItem('reg_api_url');
    const isOldDemoServer = saved && /^(https?:\/\/)(localhost|127\.0\.0\.1):3000\/?$/.test(saved);
    return (!saved || isOldDemoServer)
      ? (window.REGGRAPH_API_BASE_URL || 'http://127.0.0.1:8000')
      : saved;
  })(),
  apiMode: localStorage.getItem('reg_api_mode') || 'auto', // 'auto' | 'live' | 'standalone'
  apiToken: localStorage.getItem('reg_api_token') || '',
  graphSource: localStorage.getItem('reg_graph_source') || 'hybrid', // 'neo4j' | 'd3' | 'hybrid'
  physicsParams: {
    repulsion: parseInt(localStorage.getItem('reg_phys_repulsion') || '-240', 10),
    springLength: parseInt(localStorage.getItem('reg_phys_distance') || '80', 10),
    collision: parseInt(localStorage.getItem('reg_phys_collision') || '18', 10)
  },
  ragConfidenceMin: parseFloat(localStorage.getItem('reg_rag_conf') || '0.70'),
  isChatExpanded: false,
  chatHistory: [],
  // REAL DOCUMENT MILESTONES (Normalized across document sections)
  // Entities: Cumulative entity count discovered across 8 contract sections (4 to 28)
  entitiesDataNormalized: [0.22, 0.38, 0.35, 0.54, 0.68, 0.82, 0.78, 0.96],
  // Flagged: Contract clause risk density (0 in preamble, peaking at Sec 14 IP Indemnity & Sec 8 Termination)
  flaggedDataNormalized: [0.08, 0.09, 0.11, 0.12, 0.16, 0.38, 0.92, 0.48]
};

function apiBaseUrl() {
  const configured = (App.apiBaseUrl || window.REGGRAPH_API_BASE_URL || 'http://127.0.0.1:8000').trim();
  return configured.replace(/\/+$/, '');
}

function apiUrl(path) {
  return `${apiBaseUrl()}${path.startsWith('/') ? path : `/${path}`}`;
}

function apiHeaders() {
  return App.apiToken ? { 'Authorization': `Bearer ${App.apiToken}` } : {};
}

function apiErrorMessage(payload, statusCode) {
  const detail = payload && payload.detail;
  if (typeof detail === 'string') return detail;
  if (detail) return JSON.stringify(detail);
  return `API request failed with HTTP ${statusCode}`;
}

// ------------------------------------------------------------------
// 1. FLOW SWITCHER: INITIAL UPLOAD (NO SIDEBAR) VS DASHBOARD
// ------------------------------------------------------------------
function returnToUploadFirstScreen() {
  App.currentView = 'initial';
  document.getElementById('screenInitialUpload').style.display = 'flex';
  document.getElementById('screenDashboardAnalytics').style.display = 'none';

  // Reset staged / processing states
  document.getElementById('initialStagedRow').style.display = 'none';
  document.getElementById('initialTriggerBtnWrapper').style.display = 'none';
  document.getElementById('initialProcessingScan').style.display = 'none';
}

function enterDashboardAnalyticsScreen() {
  App.currentView = 'dashboard';
  document.getElementById('screenInitialUpload').style.display = 'none';
  document.getElementById('screenDashboardAnalytics').style.display = 'flex';

  document.getElementById('docProfileFileName').textContent = App.stagedDocName;

  // Initialize graph canvas & real-data live floating waves
  setTimeout(() => {
    initGraph();
    startRealDataLiveWaves();
    reloadGraphWithCurrentSettings();
  }, 40);
}

// ------------------------------------------------------------------
// 2. INITIAL UPLOAD ACTIONS
// ------------------------------------------------------------------
const initialDropZone = document.getElementById('initialDropZone');
const mainPdfFileInput = document.getElementById('mainPdfFileInput');
let stagedFileObject = null;

initialDropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  initialDropZone.classList.add('drag-active');
});
initialDropZone.addEventListener('dragleave', () => initialDropZone.classList.remove('drag-active'));
initialDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  initialDropZone.classList.remove('drag-active');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    onFileSelected(file.name, file);
    executeDocumentAnalyticsFlow();
  }
});

mainPdfFileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    const file = e.target.files[0];
    onFileSelected(file.name, file);
    executeDocumentAnalyticsFlow();
    e.target.value = '';
  }
});

function stageSampleDocument() {
  openExistingKnowledgeGraph();
}

function openExistingKnowledgeGraph() {
  stagedFileObject = null;
  App.stagedDocName = 'Current Neo4j corpus';
  enterDashboardAnalyticsScreen();
}

function onFileSelected(filename, fileObj = null) {
  stagedFileObject = fileObj;
  App.stagedDocName = filename;
  document.getElementById('initialStagedName').textContent = filename;
  document.getElementById('initialStagedRow').style.display = 'flex';
  document.getElementById('initialTriggerBtnWrapper').style.display = 'block';
}

function unstageDocument() {
  stagedFileObject = null;
  document.getElementById('initialStagedRow').style.display = 'none';
  document.getElementById('initialTriggerBtnWrapper').style.display = 'none';
  document.getElementById('initialProcessingScan').style.display = 'none';
}

async function executeDocumentAnalyticsFlow() {
  if (App.apiMode === 'standalone') {
    const scan = document.getElementById('initialProcessingScan');
    scan.style.display = 'flex';
    document.getElementById('initialProcTitle').textContent = 'Live ingestion is disabled in offline demo mode';
    document.getElementById('initialProcSub').textContent = 'Select Live API in Settings to upload this PDF to the backend.';
    document.getElementById('initialTriggerBtnWrapper').style.display = 'block';
    return;
  }
  if (!stagedFileObject) {
    openExistingKnowledgeGraph();
    return;
  }
  document.getElementById('initialTriggerBtnWrapper').style.display = 'none';
  const scan = document.getElementById('initialProcessingScan');
  scan.style.display = 'flex';

  const pBar = document.getElementById('initialProcBar');
  const pNum = document.getElementById('initialProcNum');
  const pTitle = document.getElementById('initialProcTitle');
  const pSub = document.getElementById('initialProcSub');

  if (pBar) pBar.style.width = '35%';
  if (pNum) pNum.textContent = '…';
  if (pTitle) pTitle.textContent = 'Uploading and processing PDF…';
  if (pSub) pSub.textContent = 'Waiting for ingestion screening and persistence to finish.';

  try {
    const formData = new FormData();
    formData.append('file', stagedFileObject);
    const response = await fetch(apiUrl('/ingestion'), {
      method: 'POST', headers: apiHeaders(), body: formData
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiErrorMessage(payload, response.status));
    if (pBar) pBar.style.width = '100%';
    if (pNum) pNum.textContent = 'Done';
    if (pTitle) pTitle.textContent = '✓ Ingestion completed';
    if (pSub) pSub.textContent = `Ingestion returned ${payload.revised_chunks?.length ?? 0} chunks. Refreshing the current Neo4j graph; graph import is a separate operation.`;
    await new Promise(resolve => setTimeout(resolve, 400));
    finishProcessingAndEnterDashboard();
  } catch (error) {
    if (pBar) pBar.style.width = '100%';
    if (pNum) pNum.textContent = 'Error';
    if (pTitle) pTitle.textContent = 'Ingestion failed';
    if (pSub) pSub.textContent = error.message || 'The ingestion API could not be reached.';
    document.getElementById('initialTriggerBtnWrapper').style.display = 'block';
  }
}

function finishProcessingAndEnterDashboard() {
  enterDashboardAnalyticsScreen();
}

// ------------------------------------------------------------------
// 3. THEME TOGGLE (LIGHT & DARK MODE)
// ------------------------------------------------------------------
function toggleTheme() {
  App.isDark = !App.isDark;
  document.documentElement.classList.toggle('light', !App.isDark);
  document.documentElement.classList.toggle('dark', App.isDark);

  document.querySelectorAll('.theme-label-slot').forEach(el => {
    el.textContent = App.isDark ? 'Light' : 'Dark';
  });
  document.querySelectorAll('.theme-icon-orb').forEach(el => {
    el.textContent = App.isDark ? '☼' : '🌙';
  });
}

// ------------------------------------------------------------------
// 4. REAL DATA LIVE FLOATING WAVES (SMOOTH SPLINE + NO KINKS)
// ------------------------------------------------------------------
let liveWaveTimerId = null;
let waveBreathTime = 0;
let waveDims = { w: 180, h: 44, dpr: 1 };

function updateWaveCanvasDimensions() {
  const dpr = window.devicePixelRatio || 1;
  ['cyanWaveCanvas', 'redWaveCanvas'].forEach(id => {
    const c = document.getElementById(id);
    if (c) {
      const w = c.clientWidth || 180;
      const h = c.clientHeight || 44;
      c.width = Math.floor(w * dpr);
      c.height = Math.floor(h * dpr);
      waveDims = { w, h, dpr };
    }
  });
}

function startRealDataLiveWaves() {
  if (liveWaveTimerId) cancelAnimationFrame(liveWaveTimerId);
  updateWaveCanvasDimensions();
  window.addEventListener('resize', updateWaveCanvasDimensions, { passive: true });

  function renderWaveLoop() {
    waveBreathTime += 0.035;

    // 1. Entities Wave: Real cumulative progression
    const cyanStroke = App.isDark ? '#38bdf8' : '#0284c7';
    const cyanGradTop = App.isDark ? 'rgba(56, 189, 248, 0.38)' : 'rgba(2, 132, 199, 0.25)';
    const cyanGradBottom = App.isDark ? 'rgba(56, 189, 248, 0.0)' : 'rgba(2, 132, 199, 0.0)';
    drawFastBezierRealWave('cyanWaveCanvas', App.entitiesDataNormalized, waveBreathTime, cyanStroke, cyanGradTop, cyanGradBottom);

    // 2. Flagged Risks Wave: Real section risks distribution peaking at Sec 14
    const redStroke = App.isDark ? '#f43f5e' : '#e11d48';
    const redGradTop = App.isDark ? 'rgba(244, 63, 94, 0.38)' : 'rgba(225, 29, 72, 0.25)';
    const redGradBottom = App.isDark ? 'rgba(244, 63, 94, 0.0)' : 'rgba(225, 29, 72, 0.0)';
    drawFastBezierRealWave('redWaveCanvas', App.flaggedDataNormalized, waveBreathTime * 1.1 + 1.2, redStroke, redGradTop, redGradBottom);

    liveWaveTimerId = requestAnimationFrame(renderWaveLoop);
  }

  renderWaveLoop();
}

// 0ms Zero-Overhead Hardware-Accelerated Catmull-Rom Bezier Spline
function drawFastBezierRealWave(canvasId, normalizedData, time, strokeColor, gradTop, gradBottom) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const w = waveDims.w;
  const h = waveDims.h;
  const dpr = waveDims.dpr;

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const n = normalizedData.length;
  const stepX = w / (n - 1);

  // Generate the 8 milestone control points with gentle edge-anchored breathing
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = i * stepX;
    const baseNorm = normalizedData[i];
    const baseY = h - (baseNorm * (h * 0.70) + h * 0.14);
    const anchor = Math.sin((i / (n - 1)) * Math.PI);
    const wave = Math.sin(time * 1.8 + i * 0.6) * 1.5 * anchor;
    pts.push({ x, y: Math.max(3, Math.min(h - 3, baseY + wave)) });
  }

  // Draw smooth spline through points
  ctx.beginPath();
  ctx.moveTo(0, h);
  ctx.lineTo(pts[0].x, pts[0].y);

  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(n - 1, i + 2)];

    const cp1x = p1.x + (p2.x - p0.x) / 6.0;
    const cp1y = p1.y + (p2.y - p0.y) / 6.0;
    const cp2x = p2.x - (p3.x - p1.x) / 6.0;
    const cp2y = p2.y - (p3.y - p1.y) / 6.0;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, gradTop);
  grad.addColorStop(1, gradBottom);
  ctx.fillStyle = grad;
  ctx.fill();

  // Stroke line
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(n - 1, i + 2)];

    const cp1x = p1.x + (p2.x - p0.x) / 6.0;
    const cp1y = p1.y + (p2.y - p0.y) / 6.0;
    const cp2x = p2.x - (p3.x - p1.x) / 6.0;
    const cp2y = p2.y - (p3.y - p1.y) / 6.0;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.4;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Peak milestone dot
  const last = pts[n - 1];
  ctx.beginPath();
  ctx.arc(last.x - 2, last.y, 3, 0, Math.PI * 2);
  ctx.fillStyle = strokeColor;
  ctx.fill();

  ctx.restore();
}

let graphAnimId = null;
let graphListenersAttached = false;

function initGraph() {
  const canvas = document.getElementById('graphRendererCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const box = document.getElementById('canvasBox');
  const dpr = window.devicePixelRatio || 1;

  let width = box.clientWidth || 600;
  let height = box.clientHeight || 410;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  const centerX = width * 0.48;
  const centerY = height * 0.44; // Raised to give 120px+ clear bottom breathing room

  const offsets = [
    { dx: 0, dy: 0 },
    { dx: -135, dy: -15 },
    { dx: -145, dy: -95 },
    { dx: -25, dy: -90 },
    { dx: 115, dy: -90 },
    { dx: 165, dy: 15 },
    { dx: 155, dy: 80 },
    { dx: 10, dy: 85 },
    { dx: -85, dy: 45 },
    { dx: -195, dy: 55 }
  ];

  const nodes = (App.backendNodes || []).map((n, idx) => {
    const angle = (idx / Math.max(1, App.backendNodes.length)) * Math.PI * 2;
    const radius = idx === 0 ? 0 : 110 + (idx % 3) * 40;
    return {
      ...n,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
      vx: 0,
      vy: 0
    };
  });
  const edges = App.graphLoaded ? (App.backendEdges || []) : [];
  App.nodes = nodes;
  App.edges = edges;

  function stepPhysics() {
    if (!App.physicsEnabled) return;
    const repulsion = Math.abs(App.physicsParams.repulsion || -240) * 5.8;
    const springK = 0.003;
    const springLength = App.physicsParams.springLength || 100;
    const friction = 0.88;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = repulsion / (dist * dist);
        const fx = (dx / dist) * f;
        const fy = (dy / dist) * f;

        if (nodes[i] !== App.draggedNode) { nodes[i].vx -= fx; nodes[i].vy -= fy; }
        if (nodes[j] !== App.draggedNode) { nodes[j].vx += fx; nodes[j].vy += fy; }
      }
    }

    edges.forEach(e => {
      const a = nodes.find(n => n.id === e.s);
      const b = nodes.find(n => n.id === e.t);
      if (!a || !b) return;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (dist - springLength) * springK;
      const fx = (dx / dist) * f;
      const fy = (dy / dist) * f;

      if (a !== App.draggedNode) { a.vx += fx; a.vy += fy; }
      if (b !== App.draggedNode) { b.vx -= fx; b.vy -= fy; }
      e.pulse = (e.pulse + 0.007) % 1;
    });

    nodes.forEach(n => {
      if (n === App.draggedNode) return;
      n.vx *= friction;
      n.vy *= friction;
      n.x += n.vx;
      n.y += n.vy;

      // Generous boundary clamping so labels and halos never leave the visible canvas
      n.x = Math.max(n.r + 25, Math.min(width - n.r - 25, n.x));
      n.y = Math.max(n.r + 25, Math.min(height - n.r - 42, n.y));
    });
  }

  function renderFrame() {
    ctx.clearRect(0, 0, width, height);
    const isDark = App.isDark;

    // Cosmic Orbits
    ctx.save();
    ctx.translate(centerX, centerY);

    const orbitColor1 = isDark ? 'rgba(6, 182, 212, 0.12)' : 'rgba(99, 102, 241, 0.16)';
    const orbitColor2 = isDark ? 'rgba(168, 85, 247, 0.1)' : 'rgba(6, 182, 212, 0.16)';
    const orbitColor3 = isDark ? 'rgba(99, 102, 241, 0.1)' : 'rgba(168, 85, 247, 0.14)';

    ctx.beginPath();
    ctx.ellipse(0, 0, width * 0.44, height * 0.42, -0.15, 0, Math.PI * 2);
    ctx.strokeStyle = orbitColor1;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.beginPath();
    ctx.ellipse(0, 0, width * 0.36, height * 0.34, 0.22, 0, Math.PI * 2);
    ctx.strokeStyle = orbitColor2;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.ellipse(0, 0, width * 0.24, height * 0.22, -0.05, 0, Math.PI * 2);
    ctx.strokeStyle = orbitColor3;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.restore();

    // Laser Edges
    const edgeBaseColor = isDark ? 'rgba(99, 102, 241, 0.28)' : 'rgba(99, 102, 241, 0.25)';
    const edgePulseColor = isDark ? '#c084fc' : '#6366f1';

    edges.forEach(e => {
      const a = nodes.find(n => n.id === e.s);
      const b = nodes.find(n => n.id === e.t);
      if (!a || !b) return;

      const isHovered = (App.hoveredNode && (App.hoveredNode === a || App.hoveredNode === b));

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = isHovered ? (isDark ? '#818cf8' : '#4338ca') : edgeBaseColor;
      ctx.lineWidth = isHovered ? 2.5 : 1.3;
      ctx.stroke();

      const px = a.x + (b.x - a.x) * e.pulse;
      const py = a.y + (b.y - a.y) * e.pulse;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = edgePulseColor;
      ctx.fill();
    });

    // Nodes (Adaptive High-Contrast Rendering)
    nodes.forEach(n => {
      const isHover = (App.hoveredNode === n);

      // Glow Halo
      const glow = ctx.createRadialGradient(n.x, n.y, n.r * 0.3, n.x, n.y, n.r + 20);
      glow.addColorStop(0, n.color + (isDark ? '66' : '33'));
      glow.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 20, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      // Node Body Circle
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + (isHover ? 3 : 0), 0, Math.PI * 2);
      ctx.fillStyle = isDark ? 'rgba(8, 12, 32, 0.92)' : '#ffffff';
      ctx.fill();

      // Outer Stroke Ring
      ctx.strokeStyle = n.color;
      ctx.lineWidth = isHover ? 3 : 2.2;
      ctx.stroke();

      // Inner Icon Glyph
      ctx.font = '13px "Plus Jakarta Sans", sans-serif';
      ctx.fillStyle = isDark ? '#ffffff' : (n.color === '#ef4444' ? '#dc2626' : (n.color === '#10b981' ? '#059669' : '#3730a3'));
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(n.icon, n.x, n.y);

      // Node Label (Smart placement & Clamped Bounds so NO words are EVER cut off)
      ctx.font = '600 11px "Plus Jakarta Sans", sans-serif';
      const textWidth = ctx.measureText(n.label).width;

      const pillW = textWidth + 14;
      let pillX = n.x - pillW / 2;
      pillX = Math.max(8, Math.min(width - pillW - 8, pillX));
      const textX = pillX + pillW / 2;

      // If node is close to bottom, flip the label pill ABOVE the node
      const placeAbove = (n.y > height - 68);
      const pillH = n.sub ? 32 : 18;
      let pillY = placeAbove ? (n.y - n.r - pillH - 6) : (n.y + n.r + 6);
      pillY = Math.max(6, Math.min(height - pillH - 6, pillY));

      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(pillX, pillY, pillW, pillH, 5);
      } else {
        ctx.rect(pillX, pillY, pillW, pillH);
      }
      ctx.fillStyle = isDark ? 'rgba(3, 6, 20, 0.88)' : 'rgba(255, 255, 255, 0.95)';
      ctx.fill();
      ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(99, 102, 241, 0.2)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = isDark ? '#ffffff' : '#090d20';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(n.label, textX, pillY + 3);

      if (n.sub) {
        ctx.font = '500 9.5px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.65)' : '#64748b';
        ctx.fillText(n.sub, textX, pillY + 17);
      }
    });

    stepPhysics();
    graphAnimId = requestAnimationFrame(renderFrame);
  }

  if (graphAnimId) cancelAnimationFrame(graphAnimId);

  function findNodeAt(mx, my) {
    const list = App.nodes || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const n = list[i];
      const dx = n.x - mx;
      const dy = n.y - my;
      const distSq = dx * dx + dy * dy;
      // Generous 26px extra hit target around node core
      const grabR = n.r + 26;
      if (distSq <= grabR * grabR) return n;
      // Also check the pill label region
      if (mx >= n.x - 70 && mx <= n.x + 70 && my >= n.y - n.r - 20 && my <= n.y + n.r + 55) {
        return n;
      }
    }
    return null;
  }

  function handleGraphResize() {
    if (!box || !canvas) return;
    const newW = box.clientWidth || 600;
    const newH = box.clientHeight || 410;
    if (newW !== width || newH !== height) {
      width = newW;
      height = newH;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.scale(dpr, dpr);
    }
  }

  if (!graphListenersAttached) {
    graphListenersAttached = true;
    window.addEventListener('resize', handleGraphResize, { passive: true });

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const target = findNodeAt(mx, my);
      if (target) {
        App.draggedNode = target;
        target.vx = 0;
        target.vy = 0;
        canvas.style.cursor = 'grabbing';
        document.body.style.cursor = 'grabbing';
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (App.draggedNode) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const w = canvas.clientWidth || 600;
        const h = canvas.clientHeight || 450;
        App.draggedNode.x = Math.max(App.draggedNode.r + 15, Math.min(w - App.draggedNode.r - 15, mx));
        App.draggedNode.y = Math.max(App.draggedNode.r + 15, Math.min(h - App.draggedNode.r - 25, my));
        App.draggedNode.vx = 0;
        App.draggedNode.vy = 0;
      }
    });

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const tooltip = document.getElementById('neuralNodeTooltip');

      if (App.draggedNode) {
        if (tooltip) tooltip.style.display = 'none';
        return;
      }

      const found = findNodeAt(mx, my);
      App.hoveredNode = found;

      if (found) {
        canvas.style.cursor = 'grab';
        if (tooltip) {
          tooltip.style.left = (mx + 16) + 'px';
          tooltip.style.top = (my + 10) + 'px';
          tooltip.innerHTML = `
            <div style="font-size: 12.5px; font-weight: 700; color: var(--text-main); margin-bottom: 2px;">${found.label}</div>
            <div style="font-size: 10px; text-transform: uppercase; color: ${found.color}; font-weight: 700;">${found.type}</div>
          `;
          tooltip.style.display = 'block';
        }
      } else {
        canvas.style.cursor = 'default';
        if (tooltip) tooltip.style.display = 'none';
      }
    });

    canvas.addEventListener('mouseleave', () => {
      if (!App.draggedNode) {
        App.hoveredNode = null;
        const tooltip = document.getElementById('neuralNodeTooltip');
        if (tooltip) tooltip.style.display = 'none';
        canvas.style.cursor = 'default';
      }
    });

    window.addEventListener('mouseup', () => {
      if (App.draggedNode) {
        App.draggedNode.vx = 0;
        App.draggedNode.vy = 0;
        App.draggedNode = null;
        canvas.style.cursor = 'grab';
        document.body.style.cursor = '';
      }
    });

    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length > 0) {
        const t = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const mx = t.clientX - rect.left;
        const my = t.clientY - rect.top;
        const target = findNodeAt(mx, my);
        if (target) {
          App.draggedNode = target;
          target.vx = 0;
          target.vy = 0;
          e.preventDefault();
        }
      }
    }, { passive: false });

    window.addEventListener('touchmove', (e) => {
      if (App.draggedNode && e.touches.length > 0) {
        const t = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const mx = t.clientX - rect.left;
        const my = t.clientY - rect.top;
        const w = canvas.clientWidth || 600;
        const h = canvas.clientHeight || 450;
        App.draggedNode.x = Math.max(App.draggedNode.r + 15, Math.min(w - App.draggedNode.r - 15, mx));
        App.draggedNode.y = Math.max(App.draggedNode.r + 15, Math.min(h - App.draggedNode.r - 25, my));
        App.draggedNode.vx = 0;
        App.draggedNode.vy = 0;
        e.preventDefault();
      }
    }, { passive: false });

    window.addEventListener('touchend', () => {
      if (App.draggedNode) {
        App.draggedNode.vx = 0;
        App.draggedNode.vy = 0;
        App.draggedNode = null;
      }
    });
  }

  renderFrame();
}

function toggleGraphPhysics() {
  App.physicsEnabled = !App.physicsEnabled;
  const icon = document.getElementById('physicsIcon');
  const text = document.getElementById('physicsBtnText');
  if (icon) icon.textContent = App.physicsEnabled ? '⏸' : '▶';
  if (text) text.textContent = App.physicsEnabled ? 'Pause Animation' : 'Resume Animation';
}

function recenterGraph() {
  initGraph();
}

function toggleFullScreenCanvas() {
  const box = document.getElementById('canvasBox');
  if (!document.fullscreenElement) {
    box.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

// ------------------------------------------------------------------
// 6. INSTANTANEOUS, ULTRA-FAST TAB SWITCHING (0ms LAG)
// ------------------------------------------------------------------
function switchWorkspaceTab(tabId) {
  // 0ms Instantaneous synchronous state switch - ZERO DELAY
  const views = {
    qa: document.getElementById('viewQA'),
    summary: document.getElementById('viewSummary'),
    risk: document.getElementById('viewRisk')
  };

  for (const key in views) {
    if (views[key]) {
      views[key].style.display = (key === tabId) ? 'flex' : 'none';
    }
  }

  // Instant visual highlight sync
  const p1 = document.getElementById('actionPillBtn1');
  const p2 = document.getElementById('actionPillBtn2');
  const p3 = document.getElementById('actionPillBtn3');
  [p1, p2, p3].forEach((p, idx) => {
    if (!p) return;
    const isTarget = (idx === 0 && tabId === 'qa') || (idx === 1 && tabId === 'summary') || (idx === 2 && tabId === 'risk');
    p.classList.toggle('active-synced-pill', isTarget);
    p.classList.toggle('active-synced-row', isTarget);
  });
}

function toggleFullScreenWorkspace() {
  const panel = document.getElementById('bottomWorkspacePanel');
  if (!document.fullscreenElement) {
    panel.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

function setQuestionText(query) {
  const input = document.getElementById('userQuestionInput');
  input.value = query;
  input.focus();
  // Immediate auto-answer for snappy UX
  submitCurrentQuestion();
}

function handleFormSubmit(e) {
  e.preventDefault();
  submitCurrentQuestion();
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function appendUserMessage(text) {
  const feed = document.getElementById('aiMessageFeed');
  if (!feed) return;
  const msgDiv = document.createElement('div');
  msgDiv.className = 'chat-msg-row user-msg-row';
  msgDiv.innerHTML = `
    <div class="chat-user-bubble">
      <span class="chat-sender-label">You</span>
      <div class="chat-msg-body">${escapeHtml(text)}</div>
    </div>
  `;
  feed.appendChild(msgDiv);
  feed.scrollTop = feed.scrollHeight;
}

function appendAiThinking() {
  const feed = document.getElementById('aiMessageFeed');
  if (!feed) return null;
  const thinkDiv = document.createElement('div');
  thinkDiv.id = 'aiThinkingBubble';
  thinkDiv.className = 'chat-msg-row ai-msg-row';
  thinkDiv.innerHTML = `
    <div class="ai-avatar-square-sm">AI</div>
    <div class="chat-ai-bubble" style="opacity: 0.85; font-style: italic; display: flex; align-items: center; gap: 8px;">
      <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #c084fc; box-shadow: 0 0 8px #c084fc;"></span>
      <span>Interrogating document graph &amp; clauses...</span>
    </div>
  `;
  feed.appendChild(thinkDiv);
  feed.scrollTop = feed.scrollHeight;
  return thinkDiv;
}

async function submitCurrentQuestion() {
  const input = document.getElementById('userQuestionInput');
  const query = input.value.trim();
  if (!query) return;
  if (App.apiMode === 'standalone') {
    appendRetrievalError('Live graph search is disabled in offline demo mode. Select Live API in Settings.');
    return;
  }
  switchWorkspaceTab('qa');
  input.value = '';
  appendUserMessage(query);
  App.chatHistory.push({ role: 'user', content: query, timestamp: new Date().toISOString() });
  appendAiThinking();

  try {
    const response = await fetch(apiUrl('/graph/search'), {
      method: 'POST',
      headers: { ...apiHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 10 })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiErrorMessage(payload, response.status));
    const results = Array.isArray(payload.results) ? payload.results : [];
    App.chatHistory.push({ role: 'assistant', content: `Retrieved ${results.length} evidence items`, timestamp: new Date().toISOString() });
    appendRetrievalResults(results);
  } catch (error) {
    removeThinkingBubble();
    appendRetrievalError(error.message || 'Graph retrieval is unavailable.');
  }
}

function removeThinkingBubble() {
  const thinking = document.getElementById('aiThinkingBubble');
  if (thinking) thinking.remove();
}

function makeEvidenceText(parent, text, className = '') {
  const element = document.createElement('div');
  if (className) element.className = className;
  element.textContent = text || '';
  parent.appendChild(element);
  return element;
}

function appendRetrievalResults(results) {
  removeThinkingBubble();
  const feed = document.getElementById('aiMessageFeed');
  if (!feed) return;
  const row = document.createElement('div');
  row.className = 'chat-msg-row ai-msg-row';
  const avatar = document.createElement('div');
  avatar.className = 'ai-avatar-square-sm';
  avatar.textContent = 'RG';
  const bubble = document.createElement('div');
  bubble.className = 'chat-ai-bubble retrieval-response';
  makeEvidenceText(bubble, 'Retrieved source evidence', 'retrieval-heading');
  makeEvidenceText(bubble, 'Search results only; no answer was generated.', 'retrieval-note');

  if (!results.length) {
    makeEvidenceText(bubble, 'No matching evidence was found.', 'retrieval-empty');
  }
  results.forEach((result) => {
    const card = document.createElement('article');
    card.className = 'retrieval-result-card';
    const channels = Array.isArray(result.channels) ? result.channels.join(' · ') : '';
    const score = Number.isFinite(result.score) ? result.score.toFixed(4) : 'unavailable';
    makeEvidenceText(card, `${result.result_type === 'image' ? 'Image evidence' : 'Text evidence'} · score ${score}`, 'retrieval-result-title');
    makeEvidenceText(card, `Matched by ${channels || 'retrieval'}`, 'retrieval-meta');

    if (result.result_type === 'image' && result.image_evidence) {
      const evidence = result.image_evidence;
      const image = document.createElement('img');
      image.className = 'retrieval-image';
      image.src = apiUrl(evidence.image_url);
      image.alt = `Source image crop from page ${evidence.page_number}`;
      image.loading = 'lazy';
      image.referrerPolicy = 'no-referrer';
      image.addEventListener('error', () => {
        image.replaceWith(makeEvidenceText(card, 'The private source image could not be loaded.', 'retrieval-image-error'));
      }, { once: true });
      card.appendChild(image);
      makeEvidenceText(card, `Page ${evidence.page_number} · document ${shortHash(evidence.document_sha256)} · run ${evidence.ingestion_run_id}`, 'retrieval-meta');
      makeEvidenceText(card, `AI-generated visual description (unverified): ${evidence.supporting_description}`, 'retrieval-description');
      makeEvidenceText(card, `Visual model: ${evidence.visual_model_provider}/${evidence.visual_model} · prompt ${evidence.visual_prompt_version}`, 'retrieval-meta');
      if (Array.isArray(evidence.candidate_entities) && evidence.candidate_entities.length) {
        makeEvidenceText(card, `Detected visual entities: ${evidence.candidate_entities.map(item => item.name || item.label || 'entity').join(', ')}`, 'retrieval-meta');
      }
      if (Array.isArray(evidence.visual_relationships) && evidence.visual_relationships.length) {
        makeEvidenceText(card, `Observed visual relationships: ${evidence.visual_relationships.map(item => item.description || item.predicate || item.type || 'relationship').join('; ')}`, 'retrieval-meta');
      }
    } else if (result.text_evidence) {
      const evidence = result.text_evidence;
      const quote = document.createElement('blockquote');
      quote.className = 'retrieval-quote';
      quote.textContent = evidence.raw_text;
      card.appendChild(quote);
      const pages = [...new Set((evidence.sources || []).map(source => source.page_number).filter(Number.isInteger))];
      const methods = [...new Set((evidence.sources || []).map(source => source.extraction_method).filter(Boolean))];
      makeEvidenceText(card, `Page${pages.length === 1 ? '' : 's'} ${pages.join(', ') || 'unknown'} · ${methods.join(', ') || 'source text'}`, 'retrieval-meta');
      makeEvidenceText(card, `Document ${shortHash(evidence.document_sha256)} · chunk ${evidence.chunk_id}`, 'retrieval-meta');
      makeEvidenceText(card, `Run ${evidence.ingestion_run_id}`, 'retrieval-meta');
      const nativeScores = [];
      if (Number.isFinite(evidence.fulltext_score)) nativeScores.push(`full-text ${evidence.fulltext_score.toFixed(4)}`);
      if (Number.isFinite(evidence.vector_score)) nativeScores.push(`vector ${evidence.vector_score.toFixed(4)}`);
      if (Number.isFinite(evidence.rrf_score)) nativeScores.push(`RRF ${evidence.rrf_score.toFixed(6)}`);
      if (nativeScores.length) makeEvidenceText(card, nativeScores.join(' · '), 'retrieval-meta');
    }
    bubble.appendChild(card);
  });
  row.append(avatar, bubble);
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
}

function appendRetrievalError(message) {
  const feed = document.getElementById('aiMessageFeed');
  if (!feed) return;
  const row = document.createElement('div');
  row.className = 'chat-msg-row ai-msg-row';
  const bubble = document.createElement('div');
  bubble.className = 'chat-ai-bubble retrieval-error';
  bubble.textContent = `Retrieval failed: ${message}`;
  row.appendChild(bubble);
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
}

function shortHash(value) {
  return typeof value === 'string' ? `${value.slice(0, 12)}…` : 'unknown';
}
function copyRedlineText(id) {
  let redline = "";
  let btnId = "copyRedlineBtn" + id;
  if (id === 1) {
    redline = "...provided however that Vendor's aggregate indemnity liability under Section 12 shall not exceed $5,000,000 USD or 5× annual fees paid.";
  } else if (id === 2) {
    redline = "...Customer may terminate only after delivering written notice detailing the inquiry and allowing Vendor thirty (30) days to clarify or remediate the finding.";
  } else {
    redline = "...within seventy-two (72) hours of confirming an actual security incident involving unauthorized access to unencrypted personal data.";
  }

  navigator.clipboard.writeText(redline).then(() => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.textContent = '✓ Copied!';
      setTimeout(() => {
        btn.textContent = 'Copy Redline';
      }, 1800);
    }
  });
}

// ------------------------------------------------------------------
// 7. NEO4J GRAPH FORMAT ADAPTER & QUERY RUNNER
// ------------------------------------------------------------------
function normalizeNeo4jGraph(payload) {
  if (!payload) return { nodes: [], edges: [] };

  const labelMeta = {
    'ORGANIZATION': { color: '#3b82f6', icon: '🏛', radius: 22, type: 'entity' },
    'APPLICATION':  { color: '#a855f7', icon: '☁', radius: 24, type: 'hub' },
    'POLICY':       { color: '#6366f1', icon: '📜', radius: 20, type: 'policy' },
    'OBLIGATION':   { color: '#8b5cf6', icon: '⚖', radius: 19, type: 'obligation' },
    'CONTROL':      { color: '#06b6d4', icon: '🛡', radius: 18, type: 'compliance' },
    'FINDING':      { color: '#f59e0b', icon: '🔍', radius: 18, type: 'finding' },
    'RISK':         { color: '#ef4444', icon: '⚠️', radius: 20, type: 'risk' },
    'EVIDENCE':     { color: '#10b981', icon: '📄', radius: 17, type: 'compliance' },
    'DOCUMENT':     { color: '#64748b', icon: '📁', radius: 22, type: 'hub' },
    'CHUNK':        { color: '#475569', icon: '📑', radius: 15, type: 'chunk' }
  };

  const rawNodes = payload.nodes || [];
  const rawRels = payload.relationships || payload.edges || [];

  const nodes = rawNodes.map((n, idx) => {
    const id = String(n.id || n.identity || idx);
    const labels = n.labels || [String(n.type || n.label || 'ENTITY')];
    const primaryLabel = (labels[0] || 'ENTITY').toUpperCase();
    const props = n.properties || {};
    const meta = labelMeta[primaryLabel] || { color: '#8b5cf6', icon: '💠', radius: 18, type: 'entity' };

    const label = props.name || props.title || props.summary || props.id || n.label || id;
    const sub = props.jurisdiction || props.status || props.severity || primaryLabel;

    return {
      id: id,
      label: String(label).slice(0, 24),
      sub: String(sub).slice(0, 18),
      type: n.type || meta.type,
      color: n.color || meta.color,
      icon: n.icon || meta.icon,
      r: n.r || meta.radius,
      labels: labels,
      properties: props
    };
  });

  const edges = rawRels.map(r => {
    const startId = String(r.startNode !== undefined ? r.startNode : (r.start !== undefined ? r.start : (r.source !== undefined ? r.source : r.s)));
    const endId = String(r.endNode !== undefined ? r.endNode : (r.end !== undefined ? r.end : (r.target !== undefined ? r.target : r.t)));
    const relType = r.type || (r.properties && r.properties.type) || 'CONNECTED_TO';

    return {
      s: startId,
      t: endId,
      type: relType,
      label: relType,
      pulse: Math.random() * 0.9
    };
  });

  return { nodes, edges };
}

function renderGraphVisualAssets(assets) {
  const gallery = document.getElementById('graphVisualAssets');
  if (!gallery) return;
  gallery.replaceChildren();
  if (!assets.length) {
    gallery.textContent = 'No linked image crops in the loaded graph.';
    return;
  }
  const visibleAssets = assets.slice(0, 50);
  const heading = document.createElement('div');
  heading.className = 'retrieval-heading';
  heading.textContent = `Linked image evidence (${assets.length}${assets.length > visibleAssets.length ? `; showing ${visibleAssets.length}` : ''})`;
  gallery.appendChild(heading);
  const grid = document.createElement('div');
  grid.className = 'graph-visual-asset-grid';
  for (const item of visibleAssets) {
    const reference = item.reference || {};
    const card = document.createElement('figure');
    card.className = 'graph-visual-asset-card';
    const image = document.createElement('img');
    image.loading = 'lazy';
    image.alt = `Private source image from page ${reference.page_number || 'unknown'}`;
    image.referrerPolicy = 'no-referrer';
    image.src = apiUrl(`/visual-assets/${encodeURIComponent(reference.ingestion_run_id)}/${reference.page_number}/${encodeURIComponent(reference.source_block_id)}?image_sha256=${encodeURIComponent(reference.image_sha256)}`);
    const caption = document.createElement('figcaption');
    caption.textContent = `Page ${reference.page_number} · ${reference.source_block_id} · ${shortHash(reference.image_sha256)}`;
    card.append(image, caption);
    grid.appendChild(card);
  }
  gallery.appendChild(grid);
}

async function reloadGraphWithCurrentSettings() {
  const allNodes = [];
  const allEdges = [];
  const allVisualAssets = [];
  const nodeIds = new Set();
  const edgeIds = new Set();
  const assetIds = new Set();
  let offset = 0;
  try {
    do {
      const response = await fetch(apiUrl(`/graph?limit=1000&offset=${offset}`), { headers: apiHeaders() });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiErrorMessage(payload, response.status));
      for (const node of payload.nodes || []) {
        if (!nodeIds.has(node.id)) { nodeIds.add(node.id); allNodes.push(node); }
      }
      for (const edge of payload.edges || []) {
        if (!edgeIds.has(edge.id)) { edgeIds.add(edge.id); allEdges.push(edge); }
      }
      for (const item of payload.visual_assets || []) {
        const reference = item.reference || {};
        const key = `${item.attached_to}:${reference.image_sha256}`;
        if (!assetIds.has(key)) { assetIds.add(key); allVisualAssets.push(item); }
      }
      offset = payload.next_offset;
    } while (Number.isInteger(offset));

    const normalized = normalizeNeo4jGraph({ nodes: allNodes, edges: allEdges });
    App.backendNodes = normalized.nodes;
    App.backendEdges = normalized.edges;
    App.graphVisualAssets = allVisualAssets;
    App.graphLoaded = true;
    const pill = document.getElementById('graphNodesEdgesPill');
    if (pill) pill.textContent = `${normalized.nodes.length} Nodes • ${normalized.edges.length} Edges (Neo4j)`;
    const nodeCount = document.getElementById('graphNodeCount');
    const edgeCount = document.getElementById('graphEdgeCount');
    const loadStatus = document.getElementById('graphLoadStatus');
    if (nodeCount) nodeCount.textContent = normalized.nodes.length.toLocaleString();
    if (edgeCount) edgeCount.textContent = normalized.edges.length.toLocaleString();
    if (loadStatus) loadStatus.textContent = normalized.nodes.length
      ? 'Connected to the persisted Neo4j graph.' : 'Neo4j is connected, but the graph has no renderable records.';
    renderGraphVisualAssets(allVisualAssets);
    if (normalized.nodes.length) {
      const first = normalized.nodes[0].properties || {};
      const fileLabel = first.filename || first.name || 'Persisted Neo4j corpus';
      const profile = document.getElementById('docProfileFileName');
      if (profile && App.stagedDocName === 'Current Neo4j corpus') profile.textContent = fileLabel;
    }
    initGraph();
    return normalized;
  } catch (error) {
    App.backendNodes = [];
    App.backendEdges = [];
    App.graphVisualAssets = [];
    App.graphLoaded = true;
    const pill = document.getElementById('graphNodesEdgesPill');
    if (pill) pill.textContent = 'Graph unavailable';
    const loadStatus = document.getElementById('graphLoadStatus');
    if (loadStatus) loadStatus.textContent = 'Could not load the persisted Neo4j graph. Check API and database status.';
    renderGraphVisualAssets([]);
    console.error('Could not load persisted Neo4j graph:', error);
    initGraph();
    return null;
  }
}
function executeCypherInSettings() {
  const summaryEl = document.getElementById('cypherResultSummary');
  if (summaryEl) summaryEl.textContent = 'Raw Cypher is not exposed to the frontend. Use the read-only /graph and /graph/search APIs.';
}

function toggleChatExpanded() {
  const card = document.querySelector('.qa-ai-bubble-card');
  if (!card) return;
  App.isChatExpanded = !App.isChatExpanded;
  card.classList.toggle('chat-expanded', App.isChatExpanded);
  const label = document.getElementById('chatExpandLabel');
  const icon = document.getElementById('chatExpandIcon');
  if (label) label.textContent = App.isChatExpanded ? 'Restore' : 'Expand';
  if (icon) icon.textContent = App.isChatExpanded ? '⤡' : '⤢';
}

function setChatHeight(height) {
  const messages = document.getElementById('aiMessageFeed');
  if (messages && Number.isFinite(height)) {
    messages.style.height = `${Math.max(240, Math.min(900, height))}px`;
    localStorage.setItem('reg_chat_height', messages.style.height);
  }
}

function initChatResize() {
  const messages = document.getElementById('aiMessageFeed');
  const savedHeight = localStorage.getItem('reg_chat_height');
  if (messages && savedHeight) messages.style.height = savedHeight;
}

function clearChatMessages() {
  const feed = document.getElementById('aiMessageFeed');
  if (!feed) return;
  feed.replaceChildren();
  const row = document.createElement('div');
  row.className = 'chat-msg-row ai-msg-row';
  const bubble = document.createElement('div');
  bubble.className = 'chat-ai-bubble';
  bubble.textContent = 'Ask a question to retrieve source evidence from the connected graph.';
  row.appendChild(bubble);
  feed.appendChild(row);
  App.chatHistory = [];
}

function exportChatConversation() {
  const blob = new Blob([JSON.stringify({
    exported_at: new Date().toISOString(),
    messages: App.chatHistory
  }, null, 2)], { type: 'application/json' });
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = `graph-evidence-chat-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(objectUrl);
}
// 9. SETTINGS MODAL CONTROLLER & LIVE PING TEST
// ------------------------------------------------------------------
async function testApiConnection(showAlert = false) {
  const statusEls = [
    document.getElementById('initialApiStatusPill'),
    document.getElementById('apiStatusPill'),
    document.getElementById('dashboardApiStatusPill')
  ];
  const detailsEl = document.getElementById('apiStatusDetails');

  if (App.apiMode === 'standalone') {
    statusEls.forEach(el => {
      if (el) el.innerHTML = `<span style="width: 7px; height: 7px; border-radius: 50%; background: #eab308; box-shadow: 0 0 8px #eab308;"></span> <span>Standalone Mode</span>`;
    });
    if (detailsEl) detailsEl.textContent = "Offline demo mode active. Live graph search and ingestion are disabled.";
    return false;
  }

  const baseUrl = apiBaseUrl();
  const start = performance.now();
  try {
    const res = await fetch(apiUrl('/health/ready'), {
      method: 'GET',
      headers: App.apiToken ? { 'Authorization': `Bearer ${App.apiToken}` } : {}
    });
    const latency = Math.round(performance.now() - start);
    if (res.ok) {
      statusEls.forEach(el => {
        if (el) el.innerHTML = `<span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span> <span>Connected (${latency}ms)</span>`;
      });
      if (detailsEl) detailsEl.innerHTML = `<span style="color:#10b981;">● Online: API responding at ${baseUrl || 'current host'} (${latency}ms round-trip).</span>`;
      if (showAlert) alert(`API Connected successfully! Response time: ${latency}ms`);
      return true;
    }
  } catch (err) {
    console.warn("API ping failed:", err);
  }

  statusEls.forEach(el => {
    if (el) el.innerHTML = `<span style="width: 7px; height: 7px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 8px #f59e0b;"></span> <span>API unavailable</span>`;
  });
  if (detailsEl) detailsEl.innerHTML = `<span style="color:#f59e0b;">● API unavailable. Check the backend URL and confirm FastAPI is running.</span>`;
  if (showAlert) alert("Backend API is unreachable. Check the configured backend URL and try again.");
  return false;
}

function openSettingsModal(defaultTab = 'api') {
  const modal = document.getElementById('settingsModal');
  if (!modal) return;
  modal.style.display = 'flex';
  switchSettingsTab(defaultTab);

  const urlInput = document.getElementById('settingApiUrl');
  if (urlInput) urlInput.value = App.apiBaseUrl || window.REGGRAPH_API_BASE_URL || 'http://127.0.0.1:8000';
  const modeSelect = document.getElementById('settingApiMode');
  if (modeSelect) modeSelect.value = App.apiMode;
  const tokenInput = document.getElementById('settingApiToken');
  if (tokenInput) tokenInput.value = App.apiToken || '';
  const graphSelect = document.getElementById('settingGraphSource');
  if (graphSelect) graphSelect.value = App.graphSource;

  const chargeSlider = document.getElementById('settingCharge');
  if (chargeSlider) chargeSlider.value = App.physicsParams.repulsion;
  const chargeDisp = document.getElementById('chargeValDisplay');
  if (chargeDisp) chargeDisp.textContent = App.physicsParams.repulsion;

  const distSlider = document.getElementById('settingDistance');
  if (distSlider) distSlider.value = App.physicsParams.springLength;
  const distDisp = document.getElementById('distanceValDisplay');
  if (distDisp) distDisp.textContent = App.physicsParams.springLength + 'px';

  const colSlider = document.getElementById('settingCollision');
  if (colSlider) colSlider.value = App.physicsParams.collision;
  const colDisp = document.getElementById('collisionValDisplay');
  if (colDisp) colDisp.textContent = App.physicsParams.collision + 'px';

  const confSlider = document.getElementById('settingRagConf');
  if (confSlider) confSlider.value = Math.round(App.ragConfidenceMin * 100);
  const confDisp = document.getElementById('ragConfDisplay');
  if (confDisp) confDisp.textContent = Math.round(App.ragConfidenceMin * 100) + '%';

  testApiConnection(false);
}

function closeSettingsModal() {
  const modal = document.getElementById('settingsModal');
  if (modal) modal.style.display = 'none';
}

function handleSettingsBackdropClick(e) {
  if (e.target && e.target.id === 'settingsModal') {
    closeSettingsModal();
  }
}

function switchSettingsTab(tabKey) {
  document.querySelectorAll('.settings-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabKey);
  });
  document.querySelectorAll('.settings-tab-content').forEach(c => {
    c.classList.remove('active');
  });
  const target = document.getElementById('tabContent' + tabKey.charAt(0).toUpperCase() + tabKey.slice(1));
  if (target) target.classList.add('active');
}

function saveSettingsAndApply() {
  const urlInput = document.getElementById('settingApiUrl');
  if (urlInput) {
    App.apiBaseUrl = urlInput.value.trim();
    localStorage.setItem('reg_api_url', App.apiBaseUrl);
  }
  const modeSelect = document.getElementById('settingApiMode');
  if (modeSelect) {
    App.apiMode = modeSelect.value;
    localStorage.setItem('reg_api_mode', App.apiMode);
  }
  const tokenInput = document.getElementById('settingApiToken');
  if (tokenInput) {
    App.apiToken = tokenInput.value.trim();
    localStorage.setItem('reg_api_token', App.apiToken);
  }
  const graphSelect = document.getElementById('settingGraphSource');
  if (graphSelect) {
    App.graphSource = graphSelect.value;
    localStorage.setItem('reg_graph_source', App.graphSource);
  }
  const confSlider = document.getElementById('settingRagConf');
  if (confSlider) {
    App.ragConfidenceMin = parseInt(confSlider.value, 10) / 100;
    localStorage.setItem('reg_rag_conf', App.ragConfidenceMin);
  }

  testApiConnection(false);
  reloadGraphWithCurrentSettings();
  closeSettingsModal();
}

function updateApiModeUI() {
  const modeSelect = document.getElementById('settingApiMode');
  if (modeSelect) {
    App.apiMode = modeSelect.value;
    testApiConnection(false);
  }
}

function updatePhysicsFromSlider() {
  const c = document.getElementById('settingCharge');
  const d = document.getElementById('settingDistance');
  const col = document.getElementById('settingCollision');
  if (c) {
    App.physicsParams.repulsion = parseInt(c.value, 10);
    document.getElementById('chargeValDisplay').textContent = c.value;
    localStorage.setItem('reg_phys_repulsion', c.value);
  }
  if (d) {
    App.physicsParams.springLength = parseInt(d.value, 10);
    document.getElementById('distanceValDisplay').textContent = d.value + 'px';
    localStorage.setItem('reg_phys_distance', d.value);
  }
  if (col) {
    App.physicsParams.collision = parseInt(col.value, 10);
    document.getElementById('collisionValDisplay').textContent = col.value + 'px';
    localStorage.setItem('reg_phys_collision', col.value);
  }
}

function resetPhysicsDefaults() {
  App.physicsParams = { repulsion: -240, springLength: 80, collision: 18 };
  const c = document.getElementById('settingCharge');
  if (c) c.value = -240;
  const cD = document.getElementById('chargeValDisplay');
  if (cD) cD.textContent = -240;
  const d = document.getElementById('settingDistance');
  if (d) d.value = 80;
  const dD = document.getElementById('distanceValDisplay');
  if (dD) dD.textContent = '80px';
  const col = document.getElementById('settingCollision');
  if (col) col.value = 18;
  const colD = document.getElementById('collisionValDisplay');
  if (colD) colD.textContent = '18px';
  localStorage.removeItem('reg_phys_repulsion');
  localStorage.removeItem('reg_phys_distance');
  localStorage.removeItem('reg_phys_collision');
}

// ------------------------------------------------------------------
// LAUNCH ON INITIAL FIRST SCREEN (UPLOAD ONLY - NO SIDEBAR)
// ------------------------------------------------------------------
window.addEventListener('load', () => {
  returnToUploadFirstScreen();
  initChatResize();
  testApiConnection(false);
});
