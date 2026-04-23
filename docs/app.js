// Tool-silhouette SVG — client-side OpenCV.js pipeline.
// Mirrors the Python segmenter: border detect -> perspective warp -> Otsu -> morph -> contours -> SVG.

const $ = (id) => document.getElementById(id);

const els = {
  loading: $('loading'),
  ui: $('ui'),
  file: $('file'),
  status: $('status'),
  overlay: $('overlay'),
  binary: $('binary'),
  download: $('downloadBtn'),
  downloadJpg: $('downloadJpgBtn'),
  showOriginal: $('showOriginal'),
  gridW: $('gridW'),
  gridH: $('gridH'),
  pxPerMm: $('pxPerMm'),
  blur: $('blur'),
  thrOff: $('thrOff'),
  morphOpen: $('morphOpen'),
  morphClose: $('morphClose'),
  invert: $('invert'),
  autoOrient: $('autoOrient'),
  minArea: $('minArea'),
  edgeMargin: $('edgeMargin'),
};

// Live-update <output> next to each range slider.
document.querySelectorAll('input[type=range]').forEach((r) => {
  const out = r.parentElement.querySelector('output');
  if (out) { out.textContent = r.value; r.addEventListener('input', () => (out.textContent = r.value)); }
});

let sourceMat = null;      // cv.Mat of uploaded image
let lastSvg = null;        // cached SVG string for download
let ready = false;

window.addEventListener('opencv-ready', () => {
  ready = true;
  els.loading.hidden = true;
  els.ui.hidden = false;
  setStatus('Upload a backlit top-down photo to begin.');
});

function setStatus(msg) { els.status.textContent = msg; }

els.file.addEventListener('change', (e) => loadFile(e.target.files[0]));

const drop = document.querySelector('.drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('drag')));
drop.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

async function loadFile(file) {
  if (!ready || !file) return;
  const bitmap = await createImageBitmap(file);
  const cnv = document.createElement('canvas');
  cnv.width = bitmap.width;
  cnv.height = bitmap.height;
  cnv.getContext('2d').drawImage(bitmap, 0, 0);
  if (sourceMat) sourceMat.delete();
  sourceMat = cv.imread(cnv);
  process();
}

// Re-run pipeline on any control change.
['input', 'change'].forEach((ev) =>
  ['aside', 'section'].forEach((sel) => document.querySelector(sel)?.addEventListener(ev, () => sourceMat && process()))
);

els.download.addEventListener('click', () => {
  if (!lastSvg) return;
  const blob = new Blob([lastSvg], { type: 'image/svg+xml' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tool_outlines.svg';
  a.click();
  URL.revokeObjectURL(a.href);
});

els.downloadJpg.addEventListener('click', () => {
  // JPG of the overlay canvas. JPEG has no alpha -> flatten onto white first.
  const src = els.overlay;
  if (!src.width || !src.height) return;
  const flat = document.createElement('canvas');
  flat.width = src.width;
  flat.height = src.height;
  const ctx = flat.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, flat.width, flat.height);
  ctx.drawImage(src, 0, 0);
  flat.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tool_outlines.jpg';
    a.click();
    URL.revokeObjectURL(a.href);
  }, 'image/jpeg', 0.92);
});

function odd(n) { n = n | 0; return Math.max(1, n % 2 === 0 ? n + 1 : n); }

// Detect the outer grid as a 4-corner quad. Returns cv.Mat (4x1 CV_32FC2) or null.
// Caller must delete the returned Mat.
function detectGridQuad(gray) {
  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
  const thr = new cv.Mat();
  cv.threshold(blurred, thr, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(thr, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const imgArea = gray.rows * gray.cols;
  let bestPts = null;
  let bestArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const area = cv.contourArea(c);
    if (area < 0.2 * imgArea) { c.delete(); continue; }
    const peri = cv.arcLength(c, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(c, approx, 0.02 * peri, true);
    if (approx.rows === 4 && area > bestArea) {
      if (bestPts) bestPts = null;
      const pts = [];
      for (let j = 0; j < 4; j++) pts.push([approx.data32S[j * 2], approx.data32S[j * 2 + 1]]);
      bestPts = pts;
      bestArea = area;
    }
    approx.delete();
    c.delete();
  }
  blurred.delete(); thr.delete(); contours.delete(); hierarchy.delete();
  if (!bestPts) return null;

  // Order TL, TR, BR, BL by sum/diff.
  const sums = bestPts.map(([x, y]) => x + y);
  const diffs = bestPts.map(([x, y]) => x - y);
  const tl = bestPts[sums.indexOf(Math.min(...sums))];
  const br = bestPts[sums.indexOf(Math.max(...sums))];
  const tr = bestPts[diffs.indexOf(Math.max(...diffs))];
  const bl = bestPts[diffs.indexOf(Math.min(...diffs))];
  return cv.matFromArray(4, 1, cv.CV_32FC2, [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]);
}

function dist(quad, aIdx, bIdx) {
  const ax = quad.data32F[aIdx * 2], ay = quad.data32F[aIdx * 2 + 1];
  const bx = quad.data32F[bIdx * 2], by = quad.data32F[bIdx * 2 + 1];
  return Math.hypot(ax - bx, ay - by);
}

function process() {
  if (!sourceMat) return;
  const p = {
    gridW_mm: parseFloat(els.gridW.value) * 25.4,
    gridH_mm: parseFloat(els.gridH.value) * 25.4,
    pxPerMm: parseFloat(els.pxPerMm.value),
    blur: parseInt(els.blur.value),
    thrOff: parseInt(els.thrOff.value),
    morphOpen: parseInt(els.morphOpen.value),
    morphClose: parseInt(els.morphClose.value),
    invert: els.invert.checked,
    autoOrient: els.autoOrient.checked,
    minArea: parseInt(els.minArea.value),
    edgeMargin: parseInt(els.edgeMargin.value),
    showOriginal: els.showOriginal.checked,
  };

  let outW = Math.round(p.gridW_mm * p.pxPerMm);
  let outH = Math.round(p.gridH_mm * p.pxPerMm);

  const gray = new cv.Mat();
  cv.cvtColor(sourceMat, gray, cv.COLOR_RGBA2GRAY);

  const quad = detectGridQuad(gray);
  let borderFound = false;

  const warped = new cv.Mat();
  if (quad) {
    borderFound = true;
    if (p.autoOrient) {
      const topW = dist(quad, 0, 1);
      const leftH = dist(quad, 0, 3);
      const quadPortrait = leftH > topW;
      const targetPortrait = outH > outW;
      if (quadPortrait !== targetPortrait) { [outW, outH] = [outH, outW]; }
    }
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, outW - 1, 0, outW - 1, outH - 1, 0, outH - 1]);
    const M = cv.getPerspectiveTransform(quad, dst);
    cv.warpPerspective(sourceMat, warped, M, new cv.Size(outW, outH));
    quad.delete(); dst.delete(); M.delete();
  } else {
    cv.resize(sourceMat, warped, new cv.Size(outW, outH), 0, 0, cv.INTER_AREA);
  }

  const wgray = new cv.Mat();
  cv.cvtColor(warped, wgray, cv.COLOR_RGBA2GRAY);
  const wblur = new cv.Mat();
  const k = odd(p.blur);
  cv.GaussianBlur(wgray, wblur, new cv.Size(k, k), 0);

  // Otsu to get optimal threshold, then re-threshold with offset + desired polarity.
  const tmp = new cv.Mat();
  const otsuVal = cv.threshold(wblur, tmp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  tmp.delete();
  const thrVal = Math.max(0, Math.min(255, otsuVal + p.thrOff));
  const binary = new cv.Mat();
  cv.threshold(wblur, binary, thrVal, 255, p.invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY);

  if (p.morphOpen > 0) {
    const ks = odd(p.morphOpen);
    const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ks, ks));
    cv.morphologyEx(binary, binary, cv.MORPH_OPEN, kern);
    kern.delete();
  }
  if (p.morphClose > 0) {
    const ks = odd(p.morphClose);
    const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ks, ks));
    cv.morphologyEx(binary, binary, cv.MORPH_CLOSE, kern);
    kern.delete();
  }

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

  // Filter contours and collect point arrays for SVG + overlay.
  const kept = [];
  const m = p.edgeMargin;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const area = cv.contourArea(c);
    if (area < p.minArea) { c.delete(); continue; }
    const r = cv.boundingRect(c);
    if (r.x <= m || r.y <= m || (r.x + r.width) >= (outW - m) || (r.y + r.height) >= (outH - m)) {
      c.delete(); continue;
    }
    const pts = [];
    for (let j = 0; j < c.rows; j++) pts.push([c.data32S[j * 2], c.data32S[j * 2 + 1]]);
    kept.push(pts);
    c.delete();
  }

  // Render overlay: warped photo (or blank) + red contour outlines.
  const overlay = new cv.Mat();
  if (p.showOriginal) warped.copyTo(overlay);
  else overlay.create(outH, outW, cv.CV_8UC4), overlay.setTo(new cv.Scalar(255, 255, 255, 255));
  const keptVec = new cv.MatVector();
  for (const pts of kept) {
    const flat = new Int32Array(pts.length * 2);
    for (let j = 0; j < pts.length; j++) { flat[j * 2] = pts[j][0]; flat[j * 2 + 1] = pts[j][1]; }
    keptVec.push_back(cv.matFromArray(pts.length, 1, cv.CV_32SC2, Array.from(flat)));
  }
  cv.drawContours(overlay, keptVec, -1, new cv.Scalar(255, 0, 0, 255), 2);

  cv.imshow('overlay', overlay);
  cv.imshow('binary', binary);

  lastSvg = buildSvg(kept, outW, outH, p.gridW_mm, p.gridH_mm);
  els.download.disabled = kept.length === 0;
  els.downloadJpg.disabled = false;
  setStatus(`${kept.length} object(s) kept. ${borderFound ? 'Border auto-detected.' : 'Border NOT detected — using full frame.'}`);

  // Cleanup
  gray.delete(); warped.delete(); wgray.delete(); wblur.delete();
  binary.delete(); contours.delete(); hierarchy.delete(); overlay.delete(); keptVec.delete();
}

function buildSvg(contours, warpW, warpH, gridW_mm, gridH_mm) {
  const sx = gridW_mm / warpW;
  const sy = gridH_mm / warpH;
  const polys = contours.map((pts) => {
    const coords = pts.map(([x, y]) => `${(x * sx).toFixed(3)},${(y * sy).toFixed(3)}`).join(' ');
    return `  <polygon points="${coords}" />`;
  }).join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${gridW_mm}mm" height="${gridH_mm}mm" viewBox="0 0 ${gridW_mm} ${gridH_mm}">
  <rect x="0" y="0" width="${gridW_mm}" height="${gridH_mm}" fill="none" stroke="black" stroke-width="0.5"/>
  <g fill="none" stroke="black" stroke-width="0.3">
${polys}
  </g>
</svg>`;
}
