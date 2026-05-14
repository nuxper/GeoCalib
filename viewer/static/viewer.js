let images = [];
let viewer = null;
let current = 0;
let corrected = true;
let manualRoll = 0, manualPitch = 0;

function poseData() {
  if (!corrected) return { poseRoll: 0, posePitch: 0 };
  return { poseRoll: manualRoll, posePitch: manualPitch };
}

function updateSliders() {
  document.getElementById('sl-roll').value  = manualRoll;
  document.getElementById('sl-pitch').value = manualPitch;
  document.getElementById('val-roll').textContent  = (manualRoll  >= 0 ? '+' : '') + manualRoll.toFixed(1)  + '°';
  document.getElementById('val-pitch').textContent = (manualPitch >= 0 ? '+' : '') + manualPitch.toFixed(1) + '°';
}

function inlierTag(ratio) {
  if (ratio == null) return '<span class="tag-na">inliers N/A</span>';
  const pct = (ratio * 100).toFixed(0) + '%';
  const cls = ratio >= 0.6 ? 'tag-ok' : ratio >= 0.4 ? 'tag-warn' : 'tag-bad';
  return `<span class="${cls}">inliers ${pct}</span>`;
}

function updateToggleButton() {
  const btn = document.getElementById('btn-toggle');
  if (corrected) { btn.textContent = 'Correction ON';  btn.className = 'on'; }
  else           { btn.textContent = 'Correction OFF'; btn.className = 'off'; }
}

function applyPose(yaw = 0, pitch = 0, zoom = null) {
  const img = images[current];
  viewer.setPanorama(img.url, {
    panoData: poseData(),
    transition: false,
    showLoader: false,
  }).then(() => {
    viewer.rotate({ yaw, pitch });
    if (zoom !== null) viewer.zoom(zoom);
  });
}

function reapplyPose() {
  const pos  = viewer.getPosition();
  const zoom = viewer.getZoomLevel();
  viewer.setPanorama(images[current].url, {
    panoData: poseData(),
    transition: false,
    showLoader: false,
  }).then(() => {
    viewer.rotate({ yaw: pos.yaw, pitch: pos.pitch });
    viewer.zoom(zoom);
  });
}

function inlierClass(ratio) {
  if (ratio == null) return '';
  return ratio >= 0.6 ? 'ok' : ratio >= 0.4 ? 'warn' : 'bad';
}

function buildSidebar() {
  const sidebar = document.getElementById('sidebar');
  images.forEach((img, i) => {
    const el = document.createElement('div');
    el.className = 'si ' + inlierClass(img.inlier_ratio);
    el.textContent = img.name;
    el.title = img.name;
    el.addEventListener('click', () => navigate(i));
    sidebar.appendChild(el);
  });
}

function updateSidebar() {
  const items = document.querySelectorAll('.si');
  items.forEach((el, i) => el.classList.toggle('active', i === current));
  items[current]?.scrollIntoView({ block: 'nearest' });
}

function navigate(index) {
  const zoom = viewer.getZoomLevel();
  current = ((index % images.length) + images.length) % images.length;
  const img = images[current];
  manualRoll  = img.roll  ?? 0;
  manualPitch = img.pitch ?? 0;
  updateSliders();
  applyPose(0, 0, zoom);
  document.getElementById('filename').textContent = img.name;
  document.getElementById('inlier').innerHTML = inlierTag(img.inlier_ratio);
  document.getElementById('counter').textContent = `${current + 1} / ${images.length}`;
  updateSidebar();
}

function toggleCorrection() {
  const pos = viewer.getPosition();
  const zoom = viewer.getZoomLevel();
  const pitchDelta = manualPitch * Math.PI / 180;
  corrected = !corrected;
  updateToggleButton();
  applyPose(pos.yaw, pos.pitch + (corrected ? pitchDelta : -pitchDelta), zoom);
}

const STEP = 0.5;
document.querySelectorAll('.btn-step').forEach(btn => {
  btn.addEventListener('click', () => {
    const delta = parseFloat(btn.dataset.dir) * STEP;
    if (btn.dataset.axis === 'roll') {
      manualRoll  = Math.max(-45, Math.min(45, Math.round((manualRoll  + delta) * 10) / 10));
    } else {
      manualPitch = Math.max(-45, Math.min(45, Math.round((manualPitch + delta) * 10) / 10));
    }
    updateSliders();
    reapplyPose();
  });
});

document.getElementById('btn-prev').onclick   = () => navigate(current - 1);
document.getElementById('btn-next').onclick   = () => navigate(current + 1);
document.getElementById('btn-toggle').onclick = () => toggleCorrection();

document.getElementById('sl-roll').oninput = e => {
  manualRoll = parseFloat(e.target.value);
  document.getElementById('val-roll').textContent = (manualRoll >= 0 ? '+' : '') + manualRoll.toFixed(1) + '°';
  reapplyPose();
};
document.getElementById('sl-pitch').oninput = e => {
  manualPitch = parseFloat(e.target.value);
  document.getElementById('val-pitch').textContent = (manualPitch >= 0 ? '+' : '') + manualPitch.toFixed(1) + '°';
  reapplyPose();
};
document.getElementById('btn-reset').onclick = () => {
  const img = images[current];
  manualRoll  = img.roll  ?? 0;
  manualPitch = img.pitch ?? 0;
  updateSliders();
  reapplyPose();
};

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft'  || e.key === 'a') navigate(current - 1);
  if (e.key === 'ArrowRight' || e.key === 'd') navigate(current + 1);
  if (e.key === ' ' || e.key === 'c')          { e.preventDefault(); toggleCorrection(); }
  if (e.key === 'Home') navigate(0);
  if (e.key === 'End')  navigate(images.length - 1);
});

const start = parseInt(new URLSearchParams(window.location.search).get('start') ?? '0');

fetch('/api/images')
  .then(r => r.json())
  .then(data => {
    images = data;
    viewer = new PhotoSphereViewer.Viewer({
      container: document.getElementById('viewer'),
      panorama: images[0].url,
      panoData: poseData(),
      defaultPitch: 0,
      defaultYaw: 0,
      navbar: false,
      loadingImg: null,
      touchmoveTwoFingers: false,
    });
    buildSidebar();
    navigate(start);
  });
