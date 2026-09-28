/* SAMUDRA — investigator view.
 *
 * Presentation only. Every value shown comes from the SAMUDRA API, which serves
 * the pipeline's artifacts (artifacts/<incident>/). This file computes no
 * score, no drift and no forecast. The only arithmetic it does is display
 * arithmetic on those outputs — a posterior as a percentage, 1/area_ratio as
 * "within N×", the displacement between two polygons the pipeline produced —
 * and each of those is labelled where it appears.
 *
 * Forecasts snap to the computed +24/48/72 h horizons and are never
 * interpolated: a morphing polygon would be a second drift model in a browser.
 */
(function () {
'use strict';

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const HR = 3600, D2R = Math.PI / 180;
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = x => x != null && !Number.isNaN(Number(x));
const fmt = (x, n = 1) => num(x) ? Number(x).toFixed(n) : '—';
const pct = (x, n = 1) => num(x) ? (x * 100).toFixed(n) : '—';
const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const pad = n => String(n).padStart(2, '0');
const dt = v => (typeof v === 'number' ? new Date(v * 1000) : new Date(v));
const fmtLong = v => { const d = dt(v); return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()} · ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`; };
const fmtHM = v => { const d = dt(v); return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}Z`; };
const fmtDM = v => { const d = dt(v); return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}Z`; };
const ll = (lat, lon) => `${fmt(Math.abs(lat), 3)}°${lat >= 0 ? 'N' : 'S'} ${fmt(Math.abs(lon), 3)}°${lon >= 0 ? 'E' : 'W'}`;
const C = { slick:'#ff8a3d', origin:'#f2c94c', sel:'#ff4d4f', cand:'#6ea8fe', traffic:'#6b7785', fc:'#4cc38a', cg:'#ff9478' };

if (!/^https?:$/.test(location.protocol)) {
  $('#boot').innerHTML = 'This page is served by the SAMUDRA API.<br><br>Run <code>python -m samudra.api --port 8000</code><br>then open <b>http://127.0.0.1:8000/investigate</b>';
  throw new Error('investigator view must be served by the API');
}

async function jget(u) {
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${u} → ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}
let toastT = null;
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, 7000);
}

/* =================================================================== state */
const S = {
  inc: null, tracks: null, cfg: null, alert: null, scene: null,
  TR: new Map(), sus: new Map(), sel: null,
  T0: 0, TMIN: 0, TMAX: 0, T: 0, split: 0.42,
  playing: false, playTo: null, playDur: 24, lastFrame: 0,
  view: 'analysis', satOverlay: false, sim: null,
  replays: {}, fcShown: null, coastKey: null,
};

/* ===================================================================== map */
const map = L.map('map', { zoomControl: false, minZoom: 3, maxZoom: 13, fadeAnimation: false })
  .setView([16.1, 69.7], 7);
map.attributionControl.setPrefix(false);
L.control.zoom({ position: 'topright' }).addTo(map);
L.control.scale({ imperial: false, position: 'bottomright' }).addTo(map);
[['scene',250],['india',300],['grid',320],['coast',330],['labels',360],['traffic',400],
 ['envelope',410],['forecast',420],['slick',430],['sim',440],['selected',450],['ships',460]]
  .forEach(([n, z]) => { const p = map.createPane(n); p.style.zIndex = z; });
map.getPane('grid').style.pointerEvents = 'none';
map.getPane('labels').style.pointerEvents = 'none';
map.getPane('coast').style.pointerEvents = 'none';
const TRAFFIC_R = L.canvas({ pane: 'traffic', padding: 0.4 });
const SHIP_R = L.canvas({ pane: 'ships', padding: 0.4 });
const PART_R = L.canvas({ pane: 'sim', padding: 0.4 });

const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const ATTR = 'Tiles © Esri';
const BASES = {
  dark: L.tileLayer(ESRI + 'Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, attribution: ATTR }),
  ocean: L.tileLayer(ESRI + 'Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, maxNativeZoom: 10, attribution: ATTR + ', GEBCO, NOAA' }),
  imagery: L.tileLayer(ESRI + 'World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, attribution: ATTR + ', Maxar, Earthstar Geographics' }),
};
const LABELS = {
  dark: L.tileLayer(ESRI + 'Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, pane: 'labels' }),
  ocean: L.tileLayer(ESRI + 'Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, maxNativeZoom: 10, pane: 'labels' }),
  imagery: L.tileLayer(ESRI + 'Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', { maxZoom: 16, pane: 'labels' }),
};
let BASE = 'dark';
function applyBase() {
  Object.entries(BASES).forEach(([k, l]) => k === BASE ? l.addTo(map) : map.removeLayer(l));
  Object.entries(LABELS).forEach(([k, l]) => (k === BASE && $('#optLabels').checked) ? l.addTo(map) : map.removeLayer(l));
  drawIndia();
}

/* Context layers: coastline (offline vectors), graticule, India boundary. */
const G_coast = L.layerGroup().addTo(map);
const G_grid = L.layerGroup().addTo(map);
const G_india = L.layerGroup().addTo(map);
let INDIA = null;

async function loadCoast(bounds) {
  const [w, s, e, n] = bounds.map(v => Math.round(v));
  const key = [w, s, e, n].join(',');
  if (S.coastKey === key) return;
  S.coastKey = key; G_coast.clearLayers();
  try {
    const gj = await jget(`/api/basemap/coastline?min_lon=${w - 8}&min_lat=${s - 8}&max_lon=${e + 8}&max_lat=${n + 8}`);
    if (gj.features && gj.features.length)
      L.geoJSON(gj, { pane: 'coast', interactive: false, style: { color: '#56687a', weight: 1.1, opacity: .9, fill: false } }).addTo(G_coast);
  } catch (e) { /* no coastline on disk — draw no land rather than a fabricated one */ }
}
function drawIndia() {
  G_india.clearLayers();
  if (!INDIA || !$('#optIndia').checked) return;
  // Filled over the dark basemap at small scale so the tiles' own boundary is
  // covered (CLAUDE.md 10.3); outline only elsewhere, so imagery stays visible.
  const fill = BASE === 'dark' && map.getZoom() <= 7 ? .95 : 0;
  L.geoJSON(INDIA, { pane: 'india', interactive: false,
    style: { color: '#7d8590', weight: 1, opacity: .8, fillColor: '#1c2128', fillOpacity: fill } }).addTo(G_india);
}
function drawGrid() {
  G_grid.clearLayers();
  if (!$('#optGrid').checked) return;
  const z = map.getZoom(), b = map.getBounds();
  const step = z <= 4 ? 10 : z <= 5 ? 5 : z <= 6 ? 2 : z <= 7 ? 1 : z <= 9 ? 0.5 : z <= 10 ? 0.25 : 0.1;
  const st = { pane: 'grid', color: '#8795a3', weight: .6, opacity: .16, interactive: false };
  const dec = step < 1 ? (step < 0.25 ? 2 : 2) : 0;
  const lab = (v, h) => `${fmt(Math.abs(v), dec)}°${h}`;
  for (let x = Math.ceil(b.getWest() / step) * step; x <= b.getEast(); x += step) {
    L.polyline([[b.getSouth() - 1, x], [b.getNorth() + 1, x]], st).addTo(G_grid);
    L.marker([b.getNorth(), x], { pane: 'grid', interactive: false, icon: L.divIcon({ className: 'gridlab',
      html: lab(x, x >= 0 ? 'E' : 'W'), iconSize: [60, 12], iconAnchor: [-3, -40] }) }).addTo(G_grid);
  }
  for (let y = Math.ceil(b.getSouth() / step) * step; y <= b.getNorth(); y += step) {
    L.polyline([[y, b.getWest() - 1], [y, b.getEast() + 1]], st).addTo(G_grid);
    L.marker([y, b.getEast()], { pane: 'grid', interactive: false, icon: L.divIcon({ className: 'gridlab',
      html: lab(y, y >= 0 ? 'N' : 'S'), iconSize: [60, 12], iconAnchor: [52, 14] }) }).addTo(G_grid);
  }
}
map.on('moveend', drawGrid);
map.on('zoomend', drawIndia);

/* ========================================================== analysis layers */
const G = {};
['traffic','cands','selected','release','envelope','envStep','slick','fcPoly','fcCone',
 'sim','particles','ships','cg','scene','detections','simRun'].forEach(k => { G[k] = L.layerGroup(); });

const svgRect = (stroke, fill, fo, dash) => `<svg viewBox="0 0 22 12" aria-hidden="true"><rect x="2" y="1.5" width="18" height="9" rx="2" fill="${fill}" fill-opacity="${fo}" stroke="${stroke}" stroke-width="1.6"${dash ? ` stroke-dasharray="${dash}"` : ''}/></svg>`;
const svgLine = (c, w, o, arrow, dash) => `<svg viewBox="0 0 22 12" aria-hidden="true"><path d="M1 6h20" stroke="${c}" stroke-width="${w}" stroke-opacity="${o}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>${arrow ? `<path d="M10 2.5l4 3.5-4 3.5" fill="none" stroke="${c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>` : ''}</svg>`;
const svgDot = (c, ring) => `<svg viewBox="0 0 22 12" aria-hidden="true"><circle cx="11" cy="6" r="4" fill="${ring ? '#2a0f10' : c}" stroke="${c}" stroke-width="${ring ? 2 : 1}"/>${ring ? `<circle cx="11" cy="6" r="1.4" fill="${c}"/>` : ''}</svg>`;

const LEGEND = [
  { k: 'slick', label: 'Observed slick', sym: svgRect(C.slick, C.slick, .55), groups: ['slick'],
    tip: 'The oil-like surface anomaly seen in the SAR acquisition. SAR measures surface roughness, so this is an anomaly consistent with oil, not confirmed oil.' },
  { k: 'origin', label: 'Probable origin', sym: svgRect(C.origin, C.origin, .12, '3 2'), groups: ['envelope','envStep'],
    tip: 'Where the oil could have been before acquisition: the slick drifted backwards through the wind and current field, one polygon per hour. Vessels must pass through it at the matching time to be considered.' },
  { k: 'selected', label: 'Selected candidate track', sym: svgLine(C.sel, 2.6, 1, true), groups: ['selected'],
    tip: 'AIS track of the selected candidate. Bold segment: from two hours before its estimated release up to acquisition. Arrows show direction of travel; the faint dashed line is the rest of its track.' },
  { k: 'release', label: 'Estimated release point', sym: svgDot(C.sel, true), groups: ['release'],
    tip: 'Best-scoring release hypothesis for the selected vessel: the time and position along its own track from which a simulated discharge best reproduces the observed slick.' },
  { k: 'sim', label: 'Simulated discharge', sym: svgRect(C.sel, C.sel, .12, '3 2'), groups: ['sim','particles'],
    tip: 'Slick simulated from the selected vessel’s best release hypothesis, drifted forward to acquisition time by the same drift model. Compare its shape with the observed slick.' },
  { k: 'cands', label: 'Other candidates', sym: svgLine(C.cand, 1.8, .85), groups: ['cands'],
    tip: 'Other vessels that passed through the origin envelope at the matching time. Ranked candidates can be selected by clicking their track.' },
  { k: 'traffic', label: 'Other AIS traffic', sym: svgLine(C.traffic, 1, .7), groups: ['traffic'],
    tip: 'Vessels filtered out: never inside the origin envelope at the right time. Kept faint so the relevant traffic stands out.' },
  { k: 'ships', label: 'Vessel positions at time', sym: `<svg viewBox="0 0 22 12" aria-hidden="true"><circle cx="6" cy="6" r="3.2" fill="${C.cand}"/><circle cx="15" cy="6" r="2" fill="${C.traffic}"/></svg>`, groups: ['ships'],
    tip: 'Where each vessel was at the time shown on the timeline, interpolated between AIS reports.' },
  { k: 'forecast', label: 'Forecast slick', sym: svgRect(C.fc, C.fc, .4), groups: ['fcPoly'],
    tip: 'Forward drift of the observed slick at the selected horizon (+24, +48 or +72 h).' },
  { k: 'cone', label: 'Forecast uncertainty', sym: svgRect(C.fc, C.fc, .05, '2 2'), groups: ['fcCone'],
    tip: 'Uncertainty cone from the spread of forecast particles at that horizon.' },
  { k: 'cg', label: 'Coast Guard units', sym: svgDot(C.cg), groups: ['cg'], off: true,
    tip: 'Indian Coast Guard establishments. Units on the composed alert are ringed. Public locations to ~1 km.' },
];
const VIS = Object.fromEntries(LEGEND.map(l => [l.k, !l.off]));

function allowed(k) {
  if (S.view === 'sat' && !S.satOverlay) return k === 'slick';
  if (S.sim) return ['slick','origin','release','selected'].includes(k);
  return true;
}
function applyVis() {
  LEGEND.forEach(l => {
    const on = VIS[l.k] && allowed(l.k);
    l.groups.forEach(g => on ? G[g].addTo(map) : map.removeLayer(G[g]));
  });
  const sat = S.view === 'sat';
  [G.scene, G.detections].forEach(g => sat ? g.addTo(map) : map.removeLayer(g));
  S.sim ? G.simRun.addTo(map) : map.removeLayer(G.simRun);
}
function renderLegend() {
  $('#legendList').innerHTML = LEGEND.map(l => `<li><button data-k="${l.k}" aria-pressed="${VIS[l.k]}" data-tip="${esc(l.tip)}">${l.sym}<span class="lbl">${l.label}</span></button></li>`).join('');
  $$('#legendList button').forEach(b => b.onclick = async () => {
    const k = b.dataset.k; VIS[k] = !VIS[k]; b.setAttribute('aria-pressed', VIS[k]);
    if (k === 'cg' && VIS.cg) drawCG();
    applyVis();
  });
}
$('#legendFold').onclick = () => setLegend($('#legend').classList.contains('collapsed'));

/* ======================================================== geometry helpers */
function bearing(a, b) {
  const f1 = a[0] * D2R, f2 = b[0] * D2R, dl = (b[1] - a[1]) * D2R;
  const y = Math.sin(dl) * Math.cos(f2), x = Math.cos(f1) * Math.sin(f2) - Math.sin(f1) * Math.cos(f2) * Math.cos(dl);
  return (Math.atan2(y, x) / D2R + 360) % 360;
}
function hav(a, b) {
  const dLa = (b[0] - a[0]) * D2R, dLo = (b[1] - a[1]) * D2R;
  const h = Math.sin(dLa / 2) ** 2 + Math.cos(a[0] * D2R) * Math.cos(b[0] * D2R) * Math.sin(dLo / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}
function centroid(geom) {   // planar centroid of the largest ring, [lat, lon]
  const rings = geom.type === 'Polygon' ? [geom.coordinates[0]] : geom.coordinates.map(p => p[0]);
  let best = null, bestA = -1;
  rings.forEach(r => {
    let a = 0, cx = 0, cy = 0;
    for (let i = 0; i < r.length - 1; i++) {
      const [x0, y0] = r[i], [x1, y1] = r[i + 1], f = x0 * y1 - x1 * y0;
      a += f; cx += (x0 + x1) * f; cy += (y0 + y1) * f;
    }
    if (Math.abs(a) > bestA) { bestA = Math.abs(a); best = a ? [cy / (3 * a), cx / (3 * a)] : [r[0][1], r[0][0]]; }
  });
  return best;
}
const compass = d => ['N','NE','E','SE','S','SW','W','NW'][Math.round(d / 45) % 8];

/* Track helpers: features from /tracks carry coordinates [lon,lat] and `t`. */
function posAt(f, t) {
  const ts = f.properties.t, c = f.geometry.coordinates;
  if (!ts || ts.length < 2 || t < ts[0] || t > ts[ts.length - 1]) return null;
  let i = 1; while (i < ts.length - 1 && ts[i] < t) i++;
  const u = (t - ts[i - 1]) / Math.max(1e-6, ts[i] - ts[i - 1]);
  return [c[i - 1][1] + (c[i][1] - c[i - 1][1]) * u, c[i - 1][0] + (c[i][0] - c[i - 1][0]) * u];
}
function segment(f, a, b) {
  const ts = f.properties.t, c = f.geometry.coordinates, out = [];
  if (!ts) return c.map(p => [p[1], p[0]]);
  const pa = posAt(f, a); if (pa) out.push(pa);
  for (let i = 0; i < ts.length; i++) if (ts[i] > a && ts[i] < b) out.push([c[i][1], c[i][0]]);
  const pb = posAt(f, b); if (pb) out.push(pb);
  return out;
}
const lls = f => f.geometry.coordinates.map(p => [p[1], p[0]]);

/* ======================================================= evidence semantics */
/* Strength bands are DISPLAY bands over the model's own normalised term values
   (each in [0,1]); they are not thresholds the model uses. */
const band = v => !num(v) ? 'weak' : v >= 0.7 ? 'strong' : v >= 0.4 ? 'moderate' : 'weak';
const BAND_TXT = { strong: 'Strong match', moderate: 'Partial match', weak: 'Weak match' };
const MK = { strong: '✓', moderate: '~', weak: '✕', info: 'i' };
function terms(h) {
  return {
    iou: { v: h.iou, label: 'Slick overlap', value: `${pct(h.iou, 0)}%`,
      what: 'Share of the observed slick reproduced by a discharge simulated from this vessel',
      tech: `IoU = ${fmt(h.iou, 3)}` },
    centroid: { v: h.centroid_term, label: 'Centre proximity', value: `${fmt(h.centroid_offset_km, 1)} km`,
      what: 'Distance between the simulated slick’s centre and the observed slick’s centre',
      tech: `centroid_term = ${fmt(h.centroid_term, 3)}` },
    orientation: { v: h.orientation_term, label: 'Course alignment', value: `${fmt(h.orientation_delta_deg, 0)}°`,
      what: 'Angle between the slick’s long axis and the vessel’s course — a moving ship lays an elongated slick along its track',
      tech: `orientation_term = ${fmt(h.orientation_term, 3)}` },
    area: { v: h.area_ratio, label: 'Area similarity', value: `within ${fmt(1 / Math.max(h.area_ratio, 1e-6), 2)}×`,
      what: 'How close the simulated slick’s area is to the observed area',
      tech: `area_ratio (min/max) = ${fmt(h.area_ratio, 3)}` },
  };
}
/* Flags arrive as objects ({code, severity, detail, at, ...}) from the trust
   stage; older artifacts may carry plain strings. Normalise both. */
const flagCode = f => typeof f === 'string' ? f : (f && (f.code || f.type)) || '';
const flagDetail = f => typeof f === 'string' ? f : [flagCode(f), f && f.detail].filter(Boolean).join(' — ');
const codes = s => [...(s.trust_flags || []), ...(s.behaviour_flags || [])].map(flagCode);
/* trust/score.py multiplies the behaviour prior by ais_gap_in_aoi when an
   AIS_GAP trust flag is present, so a behaviour factor can move with no
   behaviour flag at all — say where it came from. */
function behaviourWhy(su) {
  const b = (su.behaviour_flags || []).map(flagDetail);
  const gap = (su.trust_flags || []).find(f => flagCode(f) === 'AIS_GAP');
  if (gap) b.push(`AIS went silent inside the area${typeof gap === 'object' && gap.detail ? ` (${gap.detail})` : ''}`);
  return b.length ? b.join('; ') : 'No anomalous behaviour flagged';
}
const factorTxt = v => Math.abs(v - 1) < 1e-9 ? { t: 'no effect', c: 'none' }
  : { t: `${v > 1 ? '+' : '−'}${fmt(Math.abs(v - 1) * 100, 0)}%`, c: v > 1 ? 'up' : 'down' };

/* ==================================================================== boot */
async function boot() {
  try {
    const list = (await jget('/api/incidents')).filter(i => i.ready);
    if (!list.length) {
      $('#boot').innerHTML = 'No analysed cases.<br><br>Run <code>python -m samudra.attribution --incident demo-001</code>';
      return;
    }
    $('#pick').innerHTML = list.map(i => `<option value="${esc(i.incident_id)}">${esc(i.incident_id)}${i.synthetic ? ' · simulation' : ''} · ${fmt(i.area_km2, 1)} km²</option>`).join('');
    $('#pick').onchange = () => load($('#pick').value);
    S.cfg = await jget('/api/config/scoring').catch(() => null);
    jget('/static/india_boundary.json').then(g => { INDIA = g; drawIndia(); }).catch(() => {});
    renderLegend(); applyBase();
    const want = new URLSearchParams(location.search).get('case');
    await load(list.some(i => i.incident_id === want) ? want : list[0].incident_id);
    $('#pick').value = S.inc.incident_id;
    $('#boot').remove();
  } catch (e) {
    $('#boot').innerHTML = `<div style="max-width:560px;letter-spacing:0">${esc(e.message)}</div>`;
  }
}

async function load(id) {
  stopPlay(); if (S.sim) exitSim(true);
  Object.values(G).forEach(g => g.clearLayers());
  S.fcShown = null; S.replays = {}; S.alert = null; S.scene = null; S.view = 'analysis'; S.satOverlay = false;
  setViewButtons();
  $('#satCard').hidden = true;
  setStatus('Loading', '');
  try {
    S.inc = await jget(`/api/incident/${encodeURIComponent(id)}`);
    S.tracks = await jget(`/api/incident/${encodeURIComponent(id)}/tracks`);
  } catch (e) { toast(e.message); setStatus('Load failed', 'partial'); return; }
  const inc = S.inc;
  history.replaceState(null, '', `?case=${encodeURIComponent(id)}${location.hash}`);

  S.TR = new Map(S.tracks.features.map(f => [f.properties.mmsi, f]));
  S.sus = new Map(inc.suspects.map(s => [s.mmsi, s]));
  S.sel = inc.suspects.length ? inc.suspects[0].mmsi : null;

  S.T0 = Date.parse(inc.acquisition_at) / 1000;
  const steps = inc.envelope_steps || [];
  const backH = steps.length ? Math.max(...steps.map(s => s.hours_back)) : 6;
  const rels = inc.suspects.map(s => Date.parse(s.best_hypothesis.release_at) / 1000);
  S.TMIN = Math.min(S.T0 - backH * HR, ...rels) - 0.5 * HR;
  S.horizons = [24, 48, 72].filter(h => (inc.forecasts || {})['forecast_' + h + 'h']);
  S.TMAX = S.horizons.length ? S.T0 + Math.max(...S.horizons) * HR : S.T0 + 0.5 * HR;
  S.split = S.horizons.length ? 0.42 : 0.92;

  drawSlick(); drawEnvelope(); drawTraffic(); drawCands(); buildShips(); drawSelected();
  renderAll();
  buildTimeline();
  applyVis();
  setTime(S.T0);
  fitIncident();
  if (inc.aoi_bounds) loadCoast(inc.aoi_bounds);

  // Non-fatal context: the page works without either.
  jget(`/api/incident/${encodeURIComponent(id)}/alert`).then(a => {
    if (S.inc !== inc) return;
    S.alert = a; renderSummary(); renderData(); if (VIS.cg) drawCG();
  }).catch(() => {});
  jget(`/api/incident/${encodeURIComponent(id)}/scene`).then(s => {
    if (S.inc !== inc) return;
    S.scene = s; renderPipeline(); renderData();
  }).catch(() => { S.scene = { available: false, reason: 'Scene endpoint unavailable.' }; });
}

function renderAll() {
  renderHeader(); renderSummary(); renderPipeline(); renderHero(); renderCands();
  renderFunnel(); renderEvidence(); renderForecast(); renderSlick(); renderData(); renderFoot();
}

/* ================================================================ drawing */
let slickLayer = null;
function drawSlick() {
  G.slick.clearLayers();
  const s = S.inc.observed_slick;
  L.geoJSON(s.geometry, { pane: 'slick', interactive: false, style: { color: '#05080b', weight: 5, opacity: .55, fill: false } }).addTo(G.slick);
  slickLayer = L.geoJSON(s.geometry, { pane: 'slick', style: { color: C.slick, weight: 2, fillColor: C.slick, fillOpacity: .5 } })
    .bindTooltip(`<b>Observed slick</b><br>${fmt(s.area_km2, 1)} km² · long axis ${fmt(s.major_axis_deg, 0)}°<br>SAR acquisition ${fmtLong(S.inc.acquisition_at)}`, { sticky: true })
    .addTo(G.slick);
}
function drawEnvelope() {
  G.envelope.clearLayers();
  const st = S.inc.envelope_steps || [];
  const hb = st.map(s => s.hours_back);
  L.geoJSON(S.inc.origin_envelope, { pane: 'envelope', style: { color: C.origin, weight: 1.6, opacity: .9, dashArray: '6,5', fillColor: C.origin, fillOpacity: .07 } })
    .bindTooltip(`<b>Probable origin envelope</b><br>Where the oil could have been ${hb.length ? `${fmt(Math.min(...hb), 0)}–${fmt(Math.max(...hb), 0)} h` : ''} before acquisition,<br>from reverse drift through the wind and current field.`, { sticky: true })
    .addTo(G.envelope);
}
function drawTraffic() {
  G.traffic.clearLayers();
  S.tracks.features.forEach(f => {
    const r = f.properties.role;
    if (r !== 'filtered' && r !== 'outside') return;
    const line = L.polyline(lls(f), { renderer: TRAFFIC_R, color: C.traffic, weight: r === 'filtered' ? .9 : .6,
      opacity: r === 'filtered' ? .3 : .12, interactive: r === 'filtered' });
    if (r === 'filtered') line.bindTooltip(`<b>${esc(f.properties.vessel_name || 'Unknown vessel')}</b> · MMSI ${f.properties.mmsi}<br>${esc(f.properties.vessel_type || '')}<br><span style="color:#aab5c1">Filtered — never inside the origin envelope at the matching time</span>`, { sticky: true });
    line.addTo(G.traffic);
  });
}
function drawCands() {
  G.cands.clearLayers();
  S.tracks.features.forEach(f => {
    const p = f.properties;
    if ((p.role !== 'suspect' && p.role !== 'envelope') || p.mmsi === S.sel) return;
    const su = S.sus.get(p.mmsi);
    const line = L.polyline(lls(f), { renderer: TRAFFIC_R, color: C.cand, weight: su ? 1.8 : 1.2, opacity: su ? .75 : .45 })
      .bindTooltip(`<b>${esc(p.vessel_name || 'Unknown vessel')}</b> · MMSI ${p.mmsi}<br>` +
        (su ? `Candidate #${su.rank} · attribution score ${pct(su.posterior)}%<br><i>Click to select</i>` : 'Inside origin envelope — not ranked'), { sticky: true });
    if (su) line.on('click', () => select(p.mmsi));
    line.addTo(G.cands);
  });
}
const chevron = b => L.divIcon({ className: 'arrowicon', iconSize: [14, 14], iconAnchor: [7, 7],
  html: `<svg width="14" height="14" viewBox="-7 -7 14 14" style="transform:rotate(${b}deg)"><path d="M-4 3 0-3 4 3" fill="none" stroke="#05080b" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round" opacity=".7"/><path d="M-4 3 0-3 4 3" fill="none" stroke="${C.sel}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>` });
const shipIcon = (b, c = C.sel) => L.divIcon({ className: 'shipicon', iconSize: [20, 20], iconAnchor: [10, 10],
  html: `<svg width="20" height="20" viewBox="-10 -10 20 20" style="transform:rotate(${b}deg)"><path d="M0-8.5 5.5 7 0 3.8-5.5 7z" fill="${c}" stroke="#fff" stroke-width="1.3" stroke-linejoin="round"/></svg>` });

let SEL_SHIP = null;
function drawSelected() {
  G.selected.clearLayers(); G.release.clearLayers(); G.sim.clearLayers();
  SEL_SHIP = null;
  const su = S.sus.get(S.sel), f = S.TR.get(S.sel);
  if (!su || !f) return;
  const b = su.best_hypothesis, rel = Date.parse(b.release_at) / 1000;
  const name = esc(su.vessel_name || su.mmsi);

  // Whole track, faint: context without dominating the map.
  L.polyline(lls(f), { pane: 'selected', color: C.sel, weight: 1.3, opacity: .32, dashArray: '2,6', interactive: false }).addTo(G.selected);
  // The relevant window: two hours before the estimated release up to acquisition.
  const seg = segment(f, rel - 2 * HR, S.T0);
  if (seg.length >= 2) {
    L.polyline(seg, { pane: 'selected', color: '#05080b', weight: 6.5, opacity: .5, interactive: false }).addTo(G.selected);
    L.polyline(seg, { pane: 'selected', color: C.sel, weight: 3, opacity: .95 })
      .bindTooltip(`<b>${name}</b> · track from ${fmtHM(rel - 2 * HR)} to acquisition ${fmtHM(S.T0)}`, { sticky: true })
      .addTo(G.selected);
    // Direction arrows, evenly spaced along the segment.
    const cum = [0]; for (let i = 1; i < seg.length; i++) cum.push(cum[i - 1] + hav(seg[i - 1], seg[i]));
    const total = cum[cum.length - 1], n = Math.max(2, Math.min(7, Math.round(total / 6)));
    for (let k = 1; k <= n; k++) {
      const target = total * k / (n + 1);
      let i = 1; while (i < seg.length - 1 && cum[i] < target) i++;
      const u = (target - cum[i - 1]) / Math.max(1e-9, cum[i] - cum[i - 1]);
      const p = [seg[i - 1][0] + (seg[i][0] - seg[i - 1][0]) * u, seg[i - 1][1] + (seg[i][1] - seg[i - 1][1]) * u];
      L.marker(p, { pane: 'selected', icon: chevron(bearing(seg[i - 1], seg[i])), interactive: false, keyboard: false }).addTo(G.selected);
    }
    // Hourly time marks, labelled on hover.
    for (let t = Math.ceil((rel - 2 * HR) / HR) * HR; t < S.T0; t += HR) {
      const p = posAt(f, t); if (!p) continue;
      L.circleMarker(p, { pane: 'selected', radius: 2.6, color: '#fff', weight: 1, fillColor: C.sel, fillOpacity: 1 })
        .bindTooltip(fmtDM(t), { direction: 'top', className: 'lbl' }).addTo(G.selected);
    }
  }
  // Position at acquisition.
  const pA = posAt(f, S.T0), pB = posAt(f, S.T0 - 600);
  if (pA) {
    L.marker(pA, { pane: 'selected', icon: shipIcon(pB ? bearing(pB, pA) : 0), keyboard: false })
      .bindTooltip(`${name} · ${fmtHM(S.T0)}`, { permanent: true, direction: 'right', offset: [10, 0], className: 'lbl sel' })
      .addTo(G.selected);
  }
  // Estimated release point.
  L.circleMarker([b.release_lat, b.release_lon], { pane: 'selected', radius: 7.5, color: C.sel, weight: 2.2, fillColor: '#2a0f10', fillOpacity: .95 })
    .addTo(G.release);
  L.circleMarker([b.release_lat, b.release_lon], { pane: 'selected', radius: 2.2, stroke: false, fillColor: C.sel, fillOpacity: 1 })
    .bindTooltip(`Est. release · ${fmtDM(rel)}`, { permanent: true, direction: 'left', offset: [-10, 0], className: 'lbl rel' })
    .addTo(G.release);
  // Simulated slick for the selected vessel (as the original view showed on select).
  if (b.simulated_geometry)
    L.geoJSON(b.simulated_geometry, { pane: 'sim', style: { color: C.sel, weight: 1.6, dashArray: '5,4', fillColor: C.sel, fillOpacity: .1 } })
      .bindTooltip(`<b>Simulated discharge — ${name}</b><br>Release ${fmtLong(b.release_at)}<br>Overlap with observed slick ${pct(b.iou, 0)}%`, { sticky: true })
      .addTo(G.sim);
}

/* Ship markers are built once per case and moved, never rebuilt per frame. */
let SHIPS = [];
function buildShips() {
  G.ships.clearLayers(); SHIPS = [];
  S.tracks.features.forEach(f => {
    const r = f.properties.role;
    if (r === 'outside') return;
    const cand = r === 'suspect' || r === 'envelope';
    const m = L.circleMarker([0, 0], { renderer: SHIP_R, radius: cand ? 3.6 : 1.9, color: cand ? '#0b1320' : C.traffic,
      weight: cand ? 1 : 0, fillColor: cand ? C.cand : C.traffic, fillOpacity: cand ? .95 : .55, interactive: cand });
    if (cand) m.bindTooltip(`${esc(f.properties.vessel_name || f.properties.mmsi)}`, { direction: 'top', className: 'lbl' });
    SHIPS.push({ f, m, cand, shown: false });
  });
}
let MOVER = null;
function drawShipsAt(t) {
  SHIPS.forEach(s => {
    const p = s.f.properties.mmsi === S.sel ? null : posAt(s.f, t);
    if (p) { s.m.setLatLng(p); if (!s.shown) { s.m.addTo(G.ships); s.shown = true; } }
    else if (s.shown) { G.ships.removeLayer(s.m); s.shown = false; }
  });
  // The selected vessel moves as a heading-aware marker when away from T0.
  const f = S.TR.get(S.sel);
  const p = f && Math.abs(t - S.T0) > 60 ? posAt(f, t) : null;
  if (p) {
    const q = posAt(f, t - 600) || p;
    if (!MOVER) MOVER = L.marker(p, { pane: 'ships', icon: shipIcon(bearing(q, p)), interactive: false, keyboard: false });
    MOVER.setLatLng(p); MOVER.setIcon(shipIcon(bearing(q, p)));
    MOVER.addTo(G.ships);
  } else if (MOVER) G.ships.removeLayer(MOVER);
}

function drawEnvStep(h) {
  G.envStep.clearLayers();
  const st = S.inc.envelope_steps || [];
  if (h >= -0.1 || !st.length) return;
  let best = st[0];
  for (const s of st) if (Math.abs(s.hours_back + h) < Math.abs(best.hours_back + h)) best = s;
  L.geoJSON(best.geometry, { pane: 'envelope', style: { color: C.origin, weight: 2.2, opacity: .95, fillColor: C.origin, fillOpacity: .16 } })
    .bindTooltip(`Reverse drift — probable oil position ${fmt(best.hours_back, 0)} h before acquisition (${fmtDM(best.time)})`, { sticky: true })
    .addTo(G.envStep);
}

let PARTS = [];
function drawParticles(t) {
  const R = S.replays[S.sel];
  if (!R || S.sim) { G.particles.clearLayers(); PARTS = []; return; }
  const rel = Date.parse(R.release_at) / 1000;
  if (t < rel - 60 || t > S.T0 + 60) { G.particles.clearLayers(); PARTS = []; return; }
  const want = (t - rel) / HR;
  let fr = R.frames[0];
  for (const x of R.frames) if (Math.abs(x.hours_after_release - want) < Math.abs(fr.hours_after_release - want)) fr = x;
  if (PARTS.length !== fr.lat.length) {
    G.particles.clearLayers();
    PARTS = fr.lat.map(() => L.circleMarker([0, 0], { renderer: PART_R, radius: 1.8, stroke: false, fillColor: C.sel, fillOpacity: .6, interactive: false }).addTo(G.particles));
  }
  for (let k = 0; k < PARTS.length; k++) PARTS[k].setLatLng([fr.lat[k], fr.lon[k]]);
}

function drawForecast(h) {
  if (S.fcShown === h) return;
  S.fcShown = h;
  G.fcPoly.clearLayers(); G.fcCone.clearLayers();
  if (!h) return;
  const fc = S.inc.forecasts['forecast_' + h + 'h'];
  const poly = fc.features.find(x => x.properties.kind === 'forecast');
  const cone = fc.features.find(x => x.properties.kind === 'uncertainty_cone');
  if (cone) L.geoJSON(cone, { pane: 'forecast', style: { color: C.fc, weight: 1, opacity: .6, dashArray: '3,4', fillColor: C.fc, fillOpacity: .05 } })
    .bindTooltip(`Uncertainty cone at +${h} h — ${fmt(cone.properties.area_km2, 0)} km²`, { sticky: true }).addTo(G.fcCone);
  if (poly) {
    const p = poly.properties;
    L.geoJSON(poly, { pane: 'forecast', style: { color: C.fc, weight: 2, fillColor: C.fc, fillOpacity: .3 } })
      .bindTooltip(`<b>Forecast +${h} h</b> · ${fmt(p.area_km2, 1)} km²<br>${p.coastline_intersects ? `Shore contact — ETA ${fmtLong(p.coastline_eta)}` : `No shore contact · ${fmt(p.distance_to_coast_km, 0)} km to coast`}`, { sticky: true })
      .addTo(G.fcPoly);
    const a = centroid(S.inc.observed_slick.geometry), b = centroid(poly.geometry);
    const d = hav(a, b), brg = bearing(a, b);
    L.polyline([a, b], { pane: 'forecast', color: C.fc, weight: 1.6, opacity: .9, dashArray: '4,4', interactive: false }).addTo(G.fcPoly);
    L.marker(b, { pane: 'forecast', icon: L.divIcon({ className: 'arrowicon', iconSize: [14, 14], iconAnchor: [7, 7],
      html: `<svg width="14" height="14" viewBox="-7 -7 14 14" style="transform:rotate(${brg}deg)"><path d="M-4 3 0-3 4 3" fill="none" stroke="${C.fc}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>` }), interactive: false })
      .bindTooltip(`+${h} h · ${fmt(d, 0)} km ${compass(brg)}`, { permanent: true, direction: 'right', offset: [8, 0], className: 'lbl' })
      .addTo(G.fcPoly);
  }
}

async function drawCG() {
  if (G.cg.getLayers().length) return;
  if (!S.alert) { try { S.alert = await jget(`/api/incident/${encodeURIComponent(S.inc.incident_id)}/alert`); } catch (e) { toast('Coast Guard layer unavailable: ' + e.message); return; } }
  (S.alert.stations ? S.alert.stations.features : []).forEach(f => {
    const p = f.properties, pt = [f.geometry.coordinates[1], f.geometry.coordinates[0]];
    L.circleMarker(pt, { pane: 'ships', radius: p.addressed ? 6 : 3, color: p.addressed ? C.cg : '#57606a', weight: p.addressed ? 2 : 1,
      fillColor: p.addressed ? C.cg : '#1b242e', fillOpacity: p.addressed ? .8 : .6 })
      .bindTooltip(`<b>${esc(p.name)}</b><br>${esc(String(p.role).replace('_', ' '))} · ${esc(p.region)}` + (p.addressed ? `<br>On the composed alert · ${fmt(p.distance_km, 0)} km` : ''), { direction: 'top' })
      .addTo(G.cg);
  });
}

/* ================================================================== header */
function setStatus(txt, cls) { const s = $('#status'); s.className = 'status ' + (cls || ''); s.querySelector('span').textContent = txt; s.title = txt; }
function datasetInfo() {
  const inc = S.inc;
  if ((inc.incident_id || '').startsWith('houston')) return { mode: 'Real AIS · simulated release', cls: 'mixed', ds: 'Real AIS traffic, simulated release' };
  if (inc.synthetic) return { mode: 'Simulation mode', cls: '', ds: 'Synthetic exercise' };
  return { mode: 'Operational data', cls: 'real', ds: 'Operational data' };
}
function renderHeader() {
  const d = datasetInfo(), m = $('#mode');
  m.hidden = false; m.textContent = d.mode; m.className = 'mode ' + d.cls;
  m.title = S.inc.synthetic ? 'Generated scenario with planted ground truth — not a real-world incident.' : '';
  $('#acq').textContent = fmtLong(S.inc.acquisition_at);
  const done = pipelineSteps().filter(s => s.state === 'done' || s.state === 'provided').length;
  setStatus(done === 6 ? 'Analysis complete' : `Partial analysis · ${done}/6 stages`, done === 6 ? 'ok' : 'partial');
}

function availability() {
  const inc = S.inc, a = inc.slick_age || {}, det = (inc.slick_source || {}).source === 'detected';
  return [
    { ok: det ? 'strong' : 'moderate', t: 'Slick detection', d: det ? 'Detected by the segmentation model' : 'Supplied by the scenario — detector not run' },
    { ok: S.tracks.features.length ? 'strong' : 'weak', t: 'AIS traffic', d: `${S.tracks.features.length} vessel tracks` },
    { ok: inc.env_summary ? 'strong' : 'weak', t: 'Wind & current field', d: inc.env_summary ? (inc.env_summary.wind_gate_pass ? 'Present · wind inside detection window' : 'Present · wind OUTSIDE detection window') : 'Missing' },
    { ok: (inc.envelope_steps || []).length ? 'strong' : 'weak', t: 'Reverse drift', d: `${(inc.envelope_steps || []).length} hourly steps` },
    { ok: S.horizons.length ? 'strong' : 'weak', t: 'Forecast', d: S.horizons.length ? S.horizons.map(h => '+' + h + ' h').join(' / ') : 'Not computed' },
    { ok: a.agrees_with_fay ? 'strong' : 'moderate', t: 'Age cross-check', d: a.agrees_with_fay ? 'Consistent' : 'Inconsistent — see details' },
  ];
}
function availabilitySummary() {
  const av = availability(), n = av.filter(x => x.ok === 'strong').length;
  return { n, total: av.length, label: n === av.length ? 'Complete' : n >= av.length - 2 ? 'Mostly complete' : 'Limited' };
}

function renderSummary() {
  const inc = S.inc, s = inc.observed_slick, a = inc.slick_age || {}, e = inc.env_summary || {}, f = inc.funnel;
  const det = (inc.slick_source || {}).source === 'detected';
  $('#headline').textContent = det ? 'Oil-like slick detected' : 'Oil-like slick under investigation';
  const bits = [`<span class="num">${ll(s.centroid_lat, s.centroid_lon)}</span>`];
  if (S.alert && S.alert.recipients && S.alert.recipients.length) {
    const r = [...S.alert.recipients].sort((x, y) => x.distance_km - y.distance_km)[0];
    bits.push(`nearest Coast Guard unit <b>${esc(r.name)}</b>, ${fmt(r.distance_km, 0)} km`);
  }
  bits.push(`${datasetInfo().ds}`);
  $('#loc').innerHTML = bits.join(' · ');
  const av = availabilitySummary();
  const kp = [
    ['Slick area', `${fmt(s.area_km2, 1)}<small>km²</small>`, 'Area of the observed slick polygon.'],
    ['Estimated age', `${fmt(a.best_hours, 1)}<small>h</small>`, `Time from the best release hypothesis to acquisition. Range across the top hypotheses: ${fmt(a.low_hours, 1)}–${fmt(a.high_hours, 1)} h.`],
    ['Candidates', `${f.ranked}<small>/ ${f.total_in_scene}</small>`, `${f.ranked} candidate vessels ranked out of ${f.total_in_scene} vessels reporting in the scene.`],
    ['Conditions', `${fmt(e.mean_wind_speed_ms, 1)}<small>m/s wind</small>`, `Wind ${fmt(e.mean_wind_speed_ms, 1)} m/s toward ${fmt(e.mean_wind_dir_deg, 0)}° · current ${fmt(e.mean_current_speed_ms, 2)} m/s toward ${fmt(e.mean_current_dir_deg, 0)}°. Mean over the scene at acquisition.`],
    ['Data availability', `${av.label}<small>${av.n}/${av.total}</small>`, 'Count of analysis inputs present for this case (see Data & provenance). This is an availability count, not a confidence score.'],
  ];
  $('#kpis').innerHTML = kp.map(k => `<div class="kpi"><dt>${k[0]}<button class="info" data-tip="${esc(k[2])}" aria-label="About ${k[0]}">i</button></dt><dd>${k[1]}</dd></div>`).join('');
}

function pipelineSteps() {
  const inc = S.inc, src = inc.slick_source || {}, f = inc.funnel, s = inc.observed_slick;
  const st = inc.envelope_steps || [];
  return [
    { k: 'detect', n: 'Detect', state: src.source === 'detected' ? 'done' : 'provided',
      d: src.source === 'detected' ? `Model-detected · confidence ${pct(src.confidence, 0)}%` : 'Scenario-supplied slick' },
    { k: 'char', n: 'Characterise', state: num(s.area_km2) ? 'done' : 'missing', d: `${fmt(s.area_km2, 1)} km² · axis ${fmt(s.major_axis_deg, 0)}°` },
    { k: 'trace', n: 'Trace', state: st.length ? 'done' : 'missing', d: st.length ? `${st.length} h reverse drift` : 'No reverse drift' },
    { k: 'filter', n: 'Filter', state: num(f.in_envelope) ? 'done' : 'missing', d: `${f.total_in_scene} → ${f.in_envelope} vessels` },
    { k: 'attr', n: 'Attribute', state: inc.suspects.length ? 'done' : 'missing', d: inc.suspects.length ? `${inc.suspects.length} candidates scored` : 'No candidate survived' },
    { k: 'fc', n: 'Forecast', state: S.horizons.length ? 'done' : 'missing', d: S.horizons.length ? S.horizons.map(h => '+' + h).join(' / ') + ' h' : 'Not computed' },
  ];
}
function renderPipeline() {
  const ic = { done: '✓', provided: '◇', missing: '✕' };
  const lbl = { done: 'completed', provided: 'provided by scenario, not computed', missing: 'not available' };
  $('#pipeline').innerHTML = pipelineSteps().map((s, i) => `<li class="step ${s.state}"><button data-step="${s.k}" title="${esc(s.d)}" aria-label="Step ${i + 1}, ${s.n}: ${lbl[s.state]}. ${esc(s.d)}">
    <span class="sh"><span class="ic" aria-hidden="true">${ic[s.state]}</span>${i + 1} ${s.n}</span><span class="sd">${esc(s.d)}</span></button></li>`).join('');
  $$('#pipeline button').forEach(b => b.onclick = () => stepAction(b.dataset.step));
}
function stepAction(k) {
  if (k === 'detect') return setView('sat');
  setView('analysis');
  if (k === 'char') { panelTo('#secSlick'); setTime(S.T0); fitIncident(); }
  if (k === 'trace') replayDrift();
  if (k === 'filter') panelTo('#secFunnel');
  if (k === 'attr') panelTo('#secTop');
  if (k === 'fc' && S.horizons.length) setHorizon(S.horizons[0]);
}
function panelTo(sel) {
  document.body.classList.remove('panel-closed');
  const el = $(sel); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ============================================================= top result */
function renderHero() {
  const inc = S.inc, el = $('#secTop');
  if (!inc.suspects.length) {
    el.innerHTML = `<div class="eyebrow" id="heroEyebrow">Potential source vessel</div>
      <h2 class="vname">No candidate vessel</h2>
      <p class="note" style="margin-top:8px">No vessel passed through the reverse-drift origin envelope at the matching time, so no attribution is offered. Widening the search until some vessel falls inside is how an attribution engine starts naming innocent ships, so it does not.</p>`;
    return;
  }
  const su = S.sus.get(S.sel), b = su.best_hypothesis, T = terms(b), n = inc.suspects.length;
  const top = inc.suspects[0], second = inc.suspects[1];
  const margin = su.rank === 1
    ? (second ? `<b>${fmt(su.posterior / second.posterior, 1)}×</b>the next candidate` : '<b>only</b>candidate')
    : `<b>${fmt(top.posterior / su.posterior, 1)}×</b>below #1`;
  const rows = [
    { mk: 'strong', t: 'Inside probable origin envelope', small: su.rationale && su.rationale[0] ? su.rationale[0] : 'Passed the space-and-time envelope filter.', v: 'Yes', vs: '' },
    ...['iou','centroid','orientation','area'].map(k => ({ mk: band(T[k].v), t: T[k].label, small: T[k].what, v: T[k].value, vs: BAND_TXT[band(T[k].v)] })),
  ];
  if (su.gap_coincidence) rows.push({ mk: 'info', t: 'AIS gap around estimated release', small: 'The vessel stopped reporting AIS near the estimated release time.', v: 'Yes', vs: '' });
  const flags = [...(su.trust_flags || []).map(x => `<span class="flag t" title="AIS integrity flag: ${esc(flagDetail(x))}">${esc(flagCode(x))}</span>`),
    ...(su.behaviour_flags || []).map(x => `<span class="flag b" title="Behaviour flag: ${esc(flagDetail(x))}">${esc(flagCode(x))}</span>`)];
  const shares = inc.suspects.map(s => `<i class="${s.mmsi === S.sel ? 'on' : ''}" style="width:${Math.max(1, s.posterior * 100)}%" data-m="${s.mmsi}" title="#${s.rank} ${esc(s.vessel_name || s.mmsi)} · ${pct(s.posterior)}%"></i>`).join('');
  const meta = [su.vessel_type || 'Type not reported', su.length_m ? `${fmt(su.length_m, 0)} m` : null, su.flag ? `Flag ${esc(su.flag)}` : null, `MMSI ${su.mmsi}`].filter(Boolean).join(' · ');
  el.innerHTML = `
    <div class="eyebrow" id="heroEyebrow">${su.rank === 1 ? 'Potential source vessel' : 'Candidate vessel'} · ranked #${su.rank} of ${n}</div>
    <h2 class="vname">${esc(su.vessel_name || 'Unknown vessel')}</h2>
    <div class="vmeta">${esc(meta)}</div>
    <div class="scorebox">
      <div class="score"><b class="num">${pct(su.posterior)}<small>%</small></b>
        <span>Attribution score <button class="info" aria-label="What the attribution score means" data-tip="Composite model score based on slick overlap, spatial proximity, course alignment, area similarity and contextual vessel factors. Scores are normalised across the ${n} candidates scored for this case, so they sum to 100%. It is a relative ranking score, not a probability of responsibility.">i</button></span></div>
      <div class="margin">${margin}</div>
    </div>
    <div class="shares" aria-hidden="true">${shares}</div>
    <div class="shares-l"><span>Share of attribution score across ${n} candidates</span><span>${pct(su.posterior, 0)}%</span></div>
    <div class="herobtns">
      <button class="btn" id="btnFull">View full evidence</button>
      <button class="btn primary" id="btnSim">
        <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="4" cy="11" r="2" fill="currentColor"/><path d="M6 10c3-1 4-4 8-6" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="2 2"/></svg>
        Simulate discharge</button>
    </div>
    <div class="sub-h" style="margin-top:14px">Key evidence</div>
    <ul class="evlist" style="margin-top:0">${rows.map(r => `<li><span class="mk ${r.mk}" role="img" aria-label="${r.mk === 'info' ? 'note' : (BAND_TXT[r.mk] || r.mk)}">${MK[r.mk]}</span>
      <span class="t">${esc(r.t)} <button class="info" data-tip="${esc(r.small)}" aria-label="About ${esc(r.t)}">i</button><small>${esc(r.small)}</small></span><span class="v">${esc(r.v)}${r.vs ? `<small>${esc(r.vs)}</small>` : ''}</span></li>`).join('')}</ul>
    ${flags.length ? `<div class="flags">${flags.join('')}</div>` : ''}
    <div class="origin">
      <div><span>Estimated release</span><b>${fmtLong(b.release_at)}</b><small>hypotheses sampled every 30 min along its track</small></div>
      <div><span>Release position</span><b>${ll(b.release_lat, b.release_lon)}</b><small>slick age at acquisition ${fmt(b.age_hours, 1)} h</small></div>
    </div>
    <div class="disc"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor"/><path d="M8 7v4M8 5v.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
      Model output is decision-support evidence and does not establish legal responsibility.</div>`;
  $('#btnFull').onclick = () => panelTo('#secEvidence');
  $('#btnSim').onclick = () => simulate();
  $$('#secTop .shares i').forEach(i => i.onclick = () => select(+i.dataset.m));
}

function renderCands() {
  const inc = S.inc;
  $('#candCount').textContent = inc.suspects.length ? `${inc.suspects.length} ranked` : '';
  $('#clist').innerHTML = inc.suspects.length ? inc.suspects.map(s => {
    const b = s.best_hypothesis;
    return `<button class="crow" data-m="${s.mmsi}" aria-pressed="${s.mmsi === S.sel}" aria-label="Candidate ${s.rank}, ${esc(s.vessel_name || s.mmsi)}, attribution score ${pct(s.posterior)} percent">
      <span class="rk">${s.rank}</span>
      <span style="min-width:0"><span class="cn" style="display:block">${esc(s.vessel_name || 'Unknown vessel')}</span>
        <span class="cm"><span><b>${pct(b.iou, 0)}%</b> overlap</span><span><b>${fmt(b.centroid_offset_km, 1)} km</b> centre</span><span><b>${fmt(b.orientation_delta_deg, 0)}°</b> course</span></span></span>
      <span class="cs"><b>${pct(s.posterior)}%</b><i><u style="width:${s.posterior * 100}%"></u></i></span></button>`;
  }).join('') : '<p class="note">No vessel was ranked for this case.</p>';
  $$('#clist .crow').forEach(b => b.onclick = () => select(+b.dataset.m));
  $('#btnCompare').disabled = inc.suspects.length < 2;
}

function select(mmsi) {
  if (!S.sus.has(mmsi)) return;
  if (S.sim) exitSim(true);
  S.sel = mmsi;
  drawCands(); drawSelected();
  renderHero(); renderCands(); renderEvidence();
  buildTimeline(); setTime(S.T);
  applyVis();
}

/* ================================================================= funnel */
function renderFunnel() {
  const f = S.inc.funnel, tot = Math.max(1, f.total_in_scene);
  const w = n => Math.max(4, n / tot * 100);
  const stage = (n, lbl, cls) => `<div class="fstage ${cls || ''}"><span class="n">${n}</span><div class="bar"><i style="width:${w(n)}%"></i><span>${lbl}</span></div></div>`;
  const op = t => `<div class="fop"><span class="a" aria-hidden="true">↓</span><span>${t}</span></div>`;
  const out = f.total_in_scene - f.in_envelope;
  $('#funnel').innerHTML = `<div class="funnel">
      ${stage(f.total_in_scene, 'Vessels reporting in scene')}
      ${op('Spatial + temporal filter — inside a reverse-drift envelope step at its time (±30 min)')}
      ${stage(f.in_envelope, 'Inside probable origin envelope', 'env')}
      ${op('Discharge simulated along each track, every 30 min')}
      ${stage(f.scored, 'Candidates scored')}
      ${op('Ranked by attribution score')}
      ${stage(f.ranked, 'Potential source vessels', 'fin')}
    </div>
    <div class="fsum"><span><b>${out}</b> of ${f.total_in_scene} vessels filtered out</span>
      <button class="linkbtn" id="btnWhy">Why were they filtered?</button></div>
    ${num(f.in_envelope_underway) && f.in_envelope_underway < f.in_envelope ? `<p class="note" style="margin-top:8px">${f.in_envelope - f.in_envelope_underway} vessel(s) in the envelope were not under way; they were kept but given lower scoring priority.</p>` : ''}`;
  $('#btnWhy').onclick = openFilter;
}
function openFilter() {
  const f = S.inc.funnel;
  const byRole = r => S.tracks.features.filter(x => x.properties.role === r);
  const outside = byRole('outside').length, filtered = byRole('filtered');
  const reasons = [
    [f.total_in_scene - f.in_envelope, 'Never inside the origin envelope at the right time',
      'The origin envelope is one polygon per hour before acquisition. Each vessel is tested at every hourly step, and 30 minutes either side of it; it must be inside that step’s polygon at that step’s time. Passing through the same water at a different time does not count.'],
    [f.in_envelope - f.scored, 'Not simulated — beyond the per-case scoring limit',
      'Vessels inside the envelope are ordered by how often they were inside it while under way; only the leading vessels are simulated.'],
    [f.scored - f.ranked, 'No valid discharge hypothesis', 'Scored, but no release along the track produced a hypothesis that could be ranked.'],
  ].filter(r => r[0] > 0);
  $('#filterBody').innerHTML = `
    <p class="note" style="margin:0 0 12px">These are the filters the attribution stage actually applied (<code>attribution/prune.py</code>), with the counts it recorded for this case.</p>
    <div class="reasons">${reasons.map(r => `<div class="reason"><b>${r[0]}</b><div><h4>${esc(r[1])}</h4><p>${esc(r[2])}</p></div></div>`).join('') || '<p class="note">No vessel was filtered for this case.</p>'}</div>
    ${outside ? `<p class="note" style="margin-top:12px">${outside} further track(s) in the AIS file never reported inside the scene during the search window and were not counted as in-scene traffic.</p>` : ''}
    ${num(f.in_envelope_underway) && f.in_envelope_underway < f.in_envelope ? `<p class="note">${f.in_envelope - f.in_envelope_underway} vessel(s) inside the envelope were not under way at the matching time. They were <b>not removed</b> — only given lower priority.</p>` : ''}
    ${filtered.length ? `<details class="tech"><summary>Show the ${filtered.length} filtered vessels</summary><div><table class="t"><thead><tr><th>Vessel</th><th>MMSI</th><th>Type</th></tr></thead><tbody>
      ${filtered.map(x => `<tr><td>${esc(x.properties.vessel_name || '—')}</td><td>${x.properties.mmsi}</td><td>${esc(x.properties.vessel_type || '—')}</td></tr>`).join('')}</tbody></table></div></details>` : ''}`;
  $('#dlgFilter').showModal();
}

/* =============================================================== evidence */
function renderEvidence() {
  const el = $('#secEvidence');
  const su = S.sus.get(S.sel);
  if (!su) { el.innerHTML = '<h2>Attribution evidence</h2><p class="note">No candidate to explain.</p>'; return; }
  const b = su.best_hypothesis, T = terms(b), W = S.cfg ? S.cfg.score_weights : null;
  const keys = ['iou','centroid','orientation','area'];
  const wtxt = k => W && num(W[k]) ? `${fmt(W[k] * 100, 0)}% of match score` : '';
  const rows = keys.map(k => {
    const t = T[k], bd = band(t.v);
    return `<div class="erow"><div class="eh"><span>${t.label}</span><b>${esc(t.value)}</b></div>
      <div class="ebar" role="img" aria-label="${t.label}: ${BAND_TXT[bd]}"><i class="${bd}" style="width:${Math.max(2, (t.v || 0) * 100)}%"></i></div>
      <div class="es"><span><span class="band ${bd}">${BAND_TXT[bd]}</span>${wtxt(k) ? ' · ' + wtxt(k) : ''}</span><code>${esc(t.tech)}</code></div></div>`;
  }).join('');
  const model = W ? keys.map(k => {
    const w = W[k] || 0, c = w * (T[k].v || 0);
    return `<div class="mrow"><span>${T[k].label}</span><span class="mb" title="Contributes ${fmt(c * 100, 1)} of a possible ${fmt(w * 100, 0)} points"><i style="width:${w * 100}%"></i><u style="width:${c * 100}%"></u></span><span class="w">${fmt(w * 100, 0)}%</span></div>`;
  }).join('') : '<p class="note">Scoring weights unavailable — the configuration endpoint did not respond.</p>';
  const cfg = S.cfg || {};
  const gapF = num(cfg.gap_coincidence) ? cfg.gap_coincidence : null;
  const factors = [
    ['Vessel-type factor', `${su.vessel_type || 'type not reported'} — tanker and bulk types weighted higher, fishing lower`, su.type_risk_prior],
    ['AIS trust factor', (su.trust_flags || []).length ? `Flags: ${su.trust_flags.map(flagDetail).join('; ')}` : 'No AIS integrity issues found', su.trust_prior],
    ['Behaviour factor', behaviourWhy(su), su.behaviour_prior],
    ['Origin-envelope position', 'Inside the envelope (edge positions are reduced)', su.proximity_prior],
    ['AIS gap at release', su.gap_coincidence ? 'Stopped reporting near the estimated release' : 'Reporting continuously around release', su.gap_coincidence ? gapF : 1],
    ['Prior confirmed offences', `${su.prior_offences || 0} on record`, 1 + (cfg.prior_offence_weight || 0) * (su.prior_offences || 0)],
  ];
  const fRows = factors.map(([n, d, v]) => {
    const ft = num(v) ? factorTxt(v) : { t: '—', c: 'none' };
    return `<div class="factor"><span>${n}<small>${esc(d)}</small></span><b class="${ft.c}" title="Technical prior multiplier ×${fmt(v, 2)}">${ft.t}</b></div>`;
  }).join('');
  const tmode = (cfg.score_terms || {}).orientation_mode === 'linear' ? `(1 − Δθ/${(cfg.score_terms || {}).orientation_span_deg || 90})` : 'cos(Δθ)';
  const decay = (cfg.score_terms || {}).centroid_decay_km;
  const formula = W ? `match  = ${fmt(W.iou, 2)}·IoU + ${fmt(W.centroid, 2)}·exp(−d/${fmt(decay, 0)} km) + ${fmt(W.orientation, 2)}·${tmode} + ${fmt(W.area, 2)}·area_ratio
       = ${fmt(W.iou, 2)}·${fmt(b.iou, 3)} + ${fmt(W.centroid, 2)}·${fmt(b.centroid_term, 3)} + ${fmt(W.orientation, 2)}·${fmt(b.orientation_term, 3)} + ${fmt(W.area, 2)}·${fmt(b.area_ratio, 3)}
       = ${fmt(b.score, 3)}
likelihood = softmax(match / T) across ${S.inc.suspects.length} candidates, T = ${fmt(cfg.temperature, 2)}  →  ${fmt(su.likelihood, 3)}
posterior  = likelihood × trust ${fmt(su.trust_prior, 2)} × behaviour ${fmt(su.behaviour_prior, 2)} × proximity ${fmt(su.proximity_prior, 2)}
             × type ${fmt(su.type_risk_prior, 2)} × gap ${su.gap_coincidence ? fmt(gapF, 2) : '1.00'} × (1 + ${fmt(cfg.prior_offence_weight, 1)}·${su.prior_offences || 0})
             → renormalised across candidates = ${fmt(su.posterior, 3)}` : `geometric score ${fmt(b.score, 3)} · likelihood ${fmt(su.likelihood, 3)} · posterior ${fmt(su.posterior, 3)}`;
  const kv = [
    ['Hypothesis', b.hypothesis_id], ['Release time', b.release_at], ['Release position', `${fmt(b.release_lat, 4)}, ${fmt(b.release_lon, 4)}`],
    ['Age at acquisition', `${fmt(b.age_hours, 2)} h`], ['IoU', fmt(b.iou, 4)], ['Centre offset', `${fmt(b.centroid_offset_km, 3)} km`],
    ['Centroid term', fmt(b.centroid_term, 4)], ['Orientation Δ', `${fmt(b.orientation_delta_deg, 2)}°`], ['Orientation term', fmt(b.orientation_term, 4)],
    ['Area ratio', fmt(b.area_ratio, 4)], ['Geometric score', fmt(su.geometric_score, 4)], ['Likelihood', fmt(su.likelihood, 4)],
    ['Posterior', fmt(su.posterior, 4)], ['AIS trust score', fmt(su.trust_score, 3)], ['Classification', su.classification || '—'],
    ['Trust flags', (su.trust_flags || []).map(flagDetail).join('; ') || 'none'], ['Behaviour flags', (su.behaviour_flags || []).map(flagDetail).join('; ') || 'none'],
  ];
  const hyps = su.top_hypotheses || [];
  el.innerHTML = `
    <h2>Why this vessel scored ${pct(su.posterior)}% <span class="r">${esc(su.vessel_name || su.mmsi)}</span></h2>
    <p class="note" style="margin:-4px 0 12px">A discharge was simulated from points along this vessel’s own track and drifted to acquisition time. These measure how well the best simulation matches what the satellite saw.</p>
    <div class="ebreak">${rows}</div>
    <div class="sub-h">Scoring model <button class="info" data-tip="Weights read from config/weights.yaml. Grey = the weight each term can contribute; blue = what this vessel actually earned. The weights are never tuned to make a case produce an expected answer.">i</button></div>
    <div class="model">${model}</div>
    <div class="chain"><span>Match score <b>${pct(su.geometric_score, 0)}%</b></span><span class="arr">→</span>
      <span>relative likelihood <b>${pct(su.likelihood, 1)}%</b></span><span class="arr">→</span>
      <span>× context factors</span><span class="arr">→</span><span>attribution score <b>${pct(su.posterior, 1)}%</b></span></div>
    <div class="sub-h">Contextual factors</div>
    <div class="factors">${fRows}</div>
    <details class="tech"><summary>View technical scoring details</summary><div>
      <code class="formula">${esc(formula)}</code>
      <dl class="kv">${kv.map(r => `<dt>${esc(r[0])}</dt><dd><code>${esc(r[1])}</code></dd>`).join('')}</dl>
      ${hyps.length ? `<div class="sub-h">Top release hypotheses for this vessel</div><table class="t"><thead><tr><th>Release</th><th>IoU</th><th>Centre km</th><th>Δθ</th><th>Area</th><th>Score</th></tr></thead><tbody>
        ${hyps.map(h => `<tr><td>${fmtDM(h.release_at)}</td><td>${fmt(h.iou, 3)}</td><td>${fmt(h.centroid_offset_km, 1)}</td><td>${fmt(h.orientation_delta_deg, 0)}°</td><td>${fmt(h.area_ratio, 2)}</td><td>${fmt(h.score, 3)}</td></tr>`).join('')}</tbody></table>` : ''}
      <div class="sub-h">Model rationale (as written to the dossier)</div>
      <ul class="rat">${(su.rationale || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
    </div></details>`;
}

/* ================================================================ compare */
function openCompare() {
  const su = S.inc.suspects;
  const best = (vals, dir) => { const v = vals.filter(num); if (!v.length) return null; return dir > 0 ? Math.max(...v) : Math.min(...v); };
  const rows = [
    ['grp', 'Result'],
    ['Attribution score', s => s.posterior, v => `${pct(v)}%`, 1],
    ['Match score (geometry only)', s => s.geometric_score, v => `${pct(v, 0)}%`, 1],
    ['grp', 'Evidence'],
    ['Slick overlap', s => s.best_hypothesis.iou, v => `${pct(v, 0)}%`, 1],
    ['Centre distance', s => s.best_hypothesis.centroid_offset_km, v => `${fmt(v, 1)} km`, -1],
    ['Course alignment Δ', s => s.best_hypothesis.orientation_delta_deg, v => `${fmt(v, 0)}°`, -1],
    ['Area similarity', s => s.best_hypothesis.area_ratio, v => `within ${fmt(1 / Math.max(v, 1e-6), 2)}×`, 1],
    ['Estimated release', s => s.best_hypothesis.release_at, v => fmtDM(v), 0],
    ['grp', 'Context'],
    ['Vessel type', s => s.vessel_type || '—', v => esc(v), 0],
    ['Vessel-type factor', s => s.type_risk_prior, v => factorTxt(v).t, 0],
    ['AIS trust factor', s => s.trust_prior, v => factorTxt(v).t, 0],
    ['Behaviour factor', s => s.behaviour_prior, v => factorTxt(v).t, 0],
    ['AIS gap at release', s => s.gap_coincidence ? 'yes' : 'no', v => v, 0],
    ['Flags', s => codes(s).join(', ') || '—', v => esc(v), 0],
  ];
  $('#compareBody').innerHTML = `<table class="cmp"><thead><tr><th>Evidence</th>${su.map(s => `<th class="${s.mmsi === S.sel ? 'sel' : ''}">#${s.rank} ${esc(s.vessel_name || s.mmsi)}<small>MMSI ${s.mmsi}</small></th>`).join('')}</tr></thead><tbody>
    ${rows.map(r => {
      if (r[0] === 'grp') return `<tr class="grp"><td colspan="${su.length + 1}">${r[1]}</td></tr>`;
      const vals = su.map(r[1]), bv = r[3] ? best(vals, r[3]) : null;
      return `<tr><td>${r[0]}</td>${vals.map(v => `<td class="${bv != null && v === bv && su.length > 1 ? 'best' : ''}">${r[2](v)}</td>`).join('')}</tr>`;
    }).join('')}</tbody></table>
    <p class="note" style="margin-top:10px">● marks the strongest value in each evidence row. All values are the pipeline’s own outputs for this case.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">${su.map(s => `<button class="btn sm" data-m="${s.mmsi}">Select #${s.rank} on map</button>`).join('')}</div>`;
  $$('#compareBody button[data-m]').forEach(b => b.onclick = () => { select(+b.dataset.m); $('#dlgCompare').close(); });
  $('#dlgCompare').showModal();
}
$('#btnCompare').onclick = openCompare;
$$('dialog [data-close]').forEach(b => b.onclick = () => b.closest('dialog').close());
$$('dialog').forEach(d => d.addEventListener('click', e => { if (e.target === d) d.close(); }));

/* =============================================================== forecast */
function fcProps(h) {
  const fc = (S.inc.forecasts || {})['forecast_' + h + 'h']; if (!fc) return null;
  const poly = fc.features.find(x => x.properties.kind === 'forecast');
  const cone = fc.features.find(x => x.properties.kind === 'uncertainty_cone');
  const a = centroid(S.inc.observed_slick.geometry), b = poly ? centroid(poly.geometry) : null;
  return { p: poly ? poly.properties : {}, cone: cone ? cone.properties : null, d: b ? hav(a, b) : null, brg: b ? bearing(a, b) : null };
}
function currentHorizon() {
  const h = (S.T - S.T0) / HR;
  if (h <= 0.1) return 0;
  return S.horizons.filter(x => x <= h + 0.01).pop() || 0;
}
function renderForecast() {
  const el = $('#forecast');
  if (!S.horizons.length) { el.innerHTML = '<p class="note">No forecast was computed for this case. Run <code>python -m samudra.impact.forecast</code>.</p>'; return; }
  const cur = currentHorizon(), obs = S.inc.observed_slick.area_km2;
  const cards = S.horizons.map(h => {
    const f = fcProps(h);
    return `<button class="fcard" data-h="${h}" aria-pressed="${cur === h}"><span class="h">+${h} H</span>
      <b>${fmt(f.p.area_km2, 0)} km²</b><span>${fmt(f.p.area_km2 / obs, 1)}× observed</span>
      <span>${num(f.d) ? `${fmt(f.d, 0)} km ${compass(f.brg)}` : '—'}</span>
      <span>${f.p.coastline_intersects ? '⚠ shore contact' : `${fmt(f.p.distance_to_coast_km, 0)} km to coast`}</span></button>`;
  }).join('');
  let det = '<p class="note" style="margin-top:10px">Select a horizon, or use the timeline under the map, to show the forecast slick on the map.</p>';
  if (cur) {
    const f = fcProps(cur), p = f.p;
    det = `<div class="fdetail"><dl class="kv">
      <dt>Valid at</dt><dd>${fmtLong(p.valid_at)}</dd>
      <dt>Predicted slick area</dt><dd>${fmt(p.area_km2, 1)} km² (${fmt(p.area_km2 / obs, 1)}× observed)</dd>
      <dt>Centre displacement <button class="info" data-tip="Distance and direction between the centre of the observed slick polygon and the centre of the forecast polygon. Computed from the two polygons the pipeline produced.">i</button></dt><dd>${fmt(f.d, 1)} km toward ${fmt(f.brg, 0)}° (${compass(f.brg)})</dd>
      <dt>Uncertainty cone</dt><dd>${f.cone ? fmt(f.cone.area_km2, 0) + ' km²' : '—'}</dd>
      <dt>Nearest coast</dt><dd>${fmt(p.distance_to_coast_km, 1)} km</dd></dl>
      ${p.coastline_intersects
        ? `<div class="shore hit"><b>Shore contact forecast.</b> ${fmt(p.affected_shoreline_km, 1)} km of shoreline affected · ETA ${fmtLong(p.coastline_eta)}</div>`
        : `<div class="shore clear">No shore contact by +${cur} h.</div>`}
      ${p.coastline_source ? `<p class="note" style="margin-top:6px">Coastline: ${esc(p.coastline_source)}</p>` : ''}</div>`;
  }
  el.innerHTML = `<div class="fcards">${cards}</div>${det}
    <p class="note" style="margin-top:8px">Forward drift of the observed slick using the same particle model as the hindcast. The map shows the computed horizons only and never interpolates between them.</p>`;
  $$('#forecast .fcard').forEach(b => b.onclick = () => setHorizon(+b.dataset.h));
}
function setHorizon(h) {
  stopPlay(); if (S.sim) exitSim(true);
  if (S.view === 'sat') setView('analysis');
  setTime(S.T0 + h * HR);
  if (!h) return fitIncident();
  const bb = L.geoJSON(S.inc.observed_slick.geometry).getBounds();
  const fc = S.inc.forecasts['forecast_' + h + 'h'];
  fc.features.forEach(x => bb.extend(L.geoJSON(x).getBounds()));
  fitTo(bb.pad(0.15));
}
$$('#horizons button').forEach(b => b.onclick = () => setHorizon(+b.dataset.h));

/* ======================================================= slick & data panels */
function renderSlick() {
  const inc = S.inc, s = inc.observed_slick, a = inc.slick_age || {}, e = inc.env_summary || {}, src = inc.slick_source || {};
  const gate = S.cfg && S.cfg.wind_gate_ms ? `${S.cfg.wind_gate_ms[0]}–${S.cfg.wind_gate_ms[1]} m/s` : 'configured window';
  $('#slick').innerHTML = `
    <dl class="kv" style="margin-top:0">
      <dt>Area</dt><dd>${fmt(s.area_km2, 1)} km²</dd>
      <dt>Perimeter</dt><dd>${fmt(s.perimeter_km, 1)} km</dd>
      ${num(s.major_axis_m) ? `<dt>Length × width</dt><dd>${fmt(s.major_axis_m / 1000, 1)} × ${fmt(s.minor_axis_m / 1000, 1)} km</dd>` : ''}
      <dt>Long-axis orientation</dt><dd>${fmt(s.major_axis_deg, 0)}°</dd>
      <dt>Elongation <button class="info" data-tip="Eccentricity: 0 = round, 1 = a line. Discharges from moving vessels are typically strongly elongated.">i</button></dt><dd>${fmt(s.eccentricity, 3)}</dd>
      <dt>Edge complexity <button class="info" data-tip="Perimeter relative to a circle of the same area. 1.0 = a circle.">i</button></dt><dd>${fmt(s.shape_complexity, 2)}</dd>
    </dl>
    <div class="sub-h">Slick age</div>
    <dl class="kv" style="margin-top:0">
      <dt>Estimated age</dt><dd>${fmt(a.best_hours, 1)} h</dd>
      <dt>Range (top hypotheses)</dt><dd>${fmt(a.low_hours, 1)}–${fmt(a.high_hours, 1)} h</dd>
      <dt>Age cross-check <button class="info" data-tip="Spreading-law (Fay) estimate of age from the slick’s area. It relies on an assumed discharge volume and oil type, so it is a consistency check, not independent confirmation.">i</button></dt>
      <dd>${fmt(a.fay_estimate_hours, 1)} h — ${a.agrees_with_fay ? 'consistent' : 'inconsistent'}${num(a.fay_ratio) ? ` (${fmt(a.fay_ratio, 2)}×)` : ''}</dd>
    </dl>
    <div class="sub-h">Conditions at acquisition</div>
    <dl class="kv" style="margin-top:0">
      <dt>Wind</dt><dd>${fmt(e.mean_wind_speed_ms, 1)} m/s toward ${fmt(e.mean_wind_dir_deg, 0)}°</dd>
      <dt>Surface current</dt><dd>${fmt(e.mean_current_speed_ms, 2)} m/s toward ${fmt(e.mean_current_dir_deg, 0)}°</dd>
      <dt>Detection wind window <button class="info" data-tip="SAR slick detection is only reliable between these wind speeds: below, calm water looks dark everywhere; above, waves scrub the slick signature.">i</button></dt><dd>${e.wind_gate_pass ? 'Inside' : 'OUTSIDE'} ${gate}</dd>
    </dl>
    ${src.source === 'detected' ? `<details class="tech"><summary>Detection details</summary><div><dl class="kv">
      <dt>Detected polygon</dt><dd><code>${esc(src.slick_id)}</code></dd>
      <dt>Fused confidence</dt><dd>${fmt(src.confidence, 3)}</dd>
      <dt>CNN oil probability</dt><dd>${fmt(src.cnn_oil_prob, 3)}</dd>
      <dt>Baseline anomaly z</dt><dd>${fmt(src.baseline_anomaly_z, 2)}</dd>
      <dt>Detected dark features</dt><dd>${src.candidates ?? '—'}</dd></dl></div></details>` : ''}`;
}
function renderData() {
  const inc = S.inc, src = inc.slick_source || {}, av = availability(), sm = availabilitySummary();
  const sc = S.scene;
  $('#data').innerHTML = `
    <div class="eh" style="display:flex;justify-content:space-between;margin-bottom:6px"><span class="note">Data availability</span><b>${sm.label} · ${sm.n}/${sm.total}</b></div>
    <ul class="avail">${av.map(x => `<li><span class="mk ${x.ok}" role="img" aria-label="${x.ok === 'strong' ? 'available' : 'partial'}">${x.ok === 'strong' ? '✓' : x.ok === 'moderate' ? '~' : '✕'}</span><span>${x.t}</span><small>${esc(x.d)}</small></li>`).join('')}</ul>
    <p class="note" style="margin-top:6px">A count of inputs present, not a confidence score.</p>
    <div class="sub-h">Provenance</div>
    <dl class="kv" style="margin-top:0">
      <dt>Dataset</dt><dd>${datasetInfo().ds}</dd>
      <dt>Slick source</dt><dd>${esc(src.source || '—')}${src.reason ? ` (${esc(src.reason)})` : ''}</dd>
      <dt>Satellite scene</dt><dd>${sc ? (sc.available ? (sc.synthetic ? 'Simulated SAR scene' : 'Stored scene') : 'Not stored') : '…'}</dd>
      <dt>Environment field</dt><dd>env.npz ${inc.synthetic ? '(scenario)' : '(supplied with case)'}</dd>
      <dt>Alert</dt><dd>${S.alert ? `${esc(S.alert.severity)} · composed, not transmitted` : '—'}</dd>
    </dl>
    <p class="note" style="margin-top:10px">Other views: <a href="/" style="color:var(--cand)">original dashboard</a> · <a href="/app" style="color:var(--cand)">timeline</a> · <a href="/ops" style="color:var(--cand)">operations console</a></p>`;
}
function renderFoot() {
  $('#pfoot').innerHTML = `Model output is decision-support evidence and does not establish legal responsibility. Candidates are associated with the event by spatio-temporal and geometric correlation.${S.inc.synthetic ? ' <b>This case is a synthetic exercise, not a real-world incident.</b>' : ''}<br>Values are read from <code>artifacts/${esc(S.inc.incident_id)}/</code> through the SAMUDRA API.`;
}

/* ================================================================ timeline */
function frac(t) {
  if (t <= S.T0) return S.split * (t - S.TMIN) / Math.max(1, S.T0 - S.TMIN);
  return S.split + (1 - S.split) * (t - S.T0) / Math.max(1, S.TMAX - S.T0);
}
function unfrac(u) {
  if (u <= S.split) return S.TMIN + u / S.split * (S.T0 - S.TMIN);
  return S.T0 + (u - S.split) / (1 - S.split) * (S.TMAX - S.T0);
}
function buildTimeline() {
  $('#axis').style.setProperty('--t0', (S.split * 100).toFixed(2) + '%');
  const marks = [];
  (S.inc.envelope_steps || []).forEach(s => marks.push({ t: S.T0 - s.hours_back * HR, cls: '' }));
  marks.push({ t: S.TMIN + 0.5 * HR, cls: '', lab: `−${fmt((S.T0 - S.TMIN) / HR - 0.5, 0)} h` });
  const su = S.sus.get(S.sel);
  if (su) marks.push({ t: Date.parse(su.best_hypothesis.release_at) / 1000, cls: 'rel', lab: 'est. release' });
  marks.push({ t: S.T0, cls: 'acq', lab: 'acquisition' });
  S.horizons.forEach(h => marks.push({ t: S.T0 + h * HR, cls: 'fc', lab: '+' + h + ' h' }));
  $('#marks').innerHTML = marks.map(m => {
    const p = frac(m.t) * 100, x = p.toFixed(2) + '%';
    const tf = p > 94 ? 'translateX(-100%)' : p < 5 ? 'translateX(0)' : 'translateX(-50%)';
    return `<span class="tick ${m.cls}" style="left:${x}"></span>` + (m.lab ? `<span class="tlab ${m.cls}" style="left:${x};transform:${tf}">${m.lab}</span>` : '');
  }).join('');
  $$('#horizons button').forEach(b => { const h = +b.dataset.h; b.disabled = h !== 0 && !S.horizons.includes(h); });
}
$('#range').addEventListener('input', e => { stopPlay(); if (S.sim) exitSim(true); setTime(unfrac(+e.target.value / 1000)); });

function setTime(t) {
  S.T = Math.max(S.TMIN, Math.min(S.TMAX, t));
  $('#range').value = Math.round(frac(S.T) * 1000);
  const h = (S.T - S.T0) / HR;
  $('#tnow').textContent = fmtLong(S.T);
  $('#trel').textContent = Math.abs(h) < 0.05 ? 'T0 · acquisition' : `${h > 0 ? 'T+' : 'T−'}${fmt(Math.abs(h), 1)} h`;
  const ph = $('#tphase');
  let sub = '';
  if (h < -0.1) { ph.textContent = 'Hindcast'; ph.className = 'phase back'; sub = 'Reverse drift — probable position of the oil before acquisition'; }
  else if (h <= 0.1) { ph.textContent = 'Observed'; ph.className = 'phase obs'; sub = 'SAR acquisition — observed slick'; }
  else {
    ph.textContent = 'Forecast'; ph.className = 'phase fwd';
    const c = S.horizons.filter(x => x <= h + 0.01).pop();
    sub = c ? `Showing computed +${c} h forecast` : `Next computed horizon: +${S.horizons[0]} h`;
  }
  $('#tsub').textContent = sub;
  const cur = currentHorizon();
  $$('#horizons button').forEach(b => b.setAttribute('aria-selected', String(h >= -0.1 && +b.dataset.h === cur)));
  if (slickLayer) slickLayer.setStyle(Math.abs(h) <= 0.1 ? { fillOpacity: .5, opacity: 1, weight: 2 } : { fillOpacity: h > 0 ? .12 : .22, opacity: .7, weight: 1.4 });
  drawEnvStep(h);
  drawParticles(S.T);
  drawShipsAt(S.T);
  drawForecast(cur);
  if (S._fcCard !== cur) { S._fcCard = cur; renderForecast(); }
}

function play(to, dur) {
  stopPlay();
  S.playTo = to; S.playDur = dur; S.playing = true; S.lastFrame = performance.now();
  $('#play').setAttribute('aria-label', 'Pause timeline');
  $('#play').innerHTML = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 2h2v8H3zM7 2h2v8H7z" fill="currentColor"/></svg>';
  requestAnimationFrame(tick);
}
function tick(now) {
  if (!S.playing) return;
  const dts = Math.max(0, (now - S.lastFrame) / 1000); S.lastFrame = now;
  const rate = (S.TMAX - S.TMIN) / S.playDur;
  const next = S.T + dts * rate;
  if (next >= S.playTo) { setTime(S.playTo); stopPlay(); return; }
  setTime(next);
  requestAnimationFrame(tick);
}
function stopPlay() {
  S.playing = false;
  $('#play').setAttribute('aria-label', 'Play timeline');
  $('#play').innerHTML = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7-4.5z" fill="currentColor"/></svg>';
  const b = $('#btnReplay');
  if (b.dataset.on) { delete b.dataset.on; b.querySelector('.lbl').textContent = 'Replay drift'; }
}
$('#play').onclick = async () => {
  if (S.playing) return stopPlay();
  if (S.sim) exitSim(true);
  await ensureReplay(S.sel);
  if (S.T >= S.TMAX - 1) setTime(S.TMIN);
  play(S.TMAX, 30);
};

async function ensureReplay(mmsi) {
  if (!mmsi || S.replays[mmsi]) return S.replays[mmsi];
  try {
    S.replays[mmsi] = await jget(`/api/incident/${encodeURIComponent(S.inc.incident_id)}/replay?mmsi=${mmsi}`);
  } catch (e) { toast('Drift replay unavailable: ' + e.message); }
  return S.replays[mmsi];
}
/* Replay drift: the reconstructed past up to acquisition — hindcast envelope,
   vessel movements and the selected candidate's simulated discharge. */
async function replayDrift() {
  const b = $('#btnReplay');
  if (b.dataset.on) { stopPlay(); setTime(S.T0); return; }
  if (S.sim) exitSim(true);
  if (S.view === 'sat') setView('analysis');
  b.querySelector('.lbl').textContent = 'Loading…';
  await ensureReplay(S.sel);
  if (!S.inc) return;
  setTime(S.TMIN); fitIncident();
  play(S.T0, 12 * (S.TMAX - S.TMIN) / Math.max(1, S.T0 - S.TMIN));
  b.dataset.on = '1'; b.querySelector('.lbl').textContent = 'Stop replay';
}
$('#btnReplay').onclick = replayDrift;

/* ====================================================== discharge simulation */
async function simulate() {
  const su = S.sus.get(S.sel); if (!su) return;
  stopPlay();
  if (S.view === 'sat') setView('analysis');
  const R = await ensureReplay(su.mmsi);
  if (!R) return;
  const b = su.best_hypothesis, f = S.TR.get(su.mmsi), rel = Date.parse(R.release_at) / 1000;
  S.sim = { mmsi: su.mmsi, run: S.sim ? S.sim.run : 0, legendWasOpen: S.sim ? S.sim.legendWasOpen : undefined };
  setTime(S.T0);
  G.simRun.clearLayers();
  applyVis();
  const bb = L.geoJSON(S.inc.observed_slick.geometry).getBounds().extend([b.release_lat, b.release_lon]);
  if (b.simulated_geometry) bb.extend(L.geoJSON(b.simulated_geometry).getBounds());
  const run = ++S.sim.run;
  // Clear the stage before framing it: fold the legend, close the tablet
  // drawer, show the card, THEN fit the slick into the space beside the card.
  const lg = $('#legend');
  if (S.sim.legendWasOpen === undefined) S.sim.legendWasOpen = !lg.classList.contains('collapsed');
  setLegend(false);
  if (window.innerWidth <= 1000 && window.innerWidth > 760) document.body.classList.add('panel-closed');
  const hud = $('#simHud'); hud.hidden = false;
  const name = esc(su.vessel_name || su.mmsi);
  const steps = ['Release point', 'Drift under wind & current', 'Compare with observed slick'];
  const renderHud = (k, extra) => {
    hud.innerHTML = `<div class="hh"><b>Discharge simulation · ${name}</b><button class="btn sm ghost" id="simX" aria-label="Exit simulation">Exit</button></div>
      <ol>${steps.map((s, i) => `<li class="${i < k ? 'done' : i === k ? 'on' : ''}">${i + 1}. ${s}</li>`).join('')}</ol>${extra || ''}`;
    $('#simX').onclick = () => exitSim();
  };
  renderHud(0, `<div class="note">Estimated release ${fmtLong(R.release_at)} at ${ll(b.release_lat, b.release_lon)}.</div>`);
  chromeOffsets();
  simFit(bb.pad(0.2));
  await wait(900); if (!S.sim || S.sim.run !== run) return;

  const parts = R.frames[0].lat.map(() => L.circleMarker([0, 0], { renderer: PART_R, radius: 1.9, stroke: false, fillColor: C.sel, fillOpacity: .65, interactive: false }).addTo(G.simRun));
  const ship = L.marker([b.release_lat, b.release_lon], { pane: 'ships', icon: shipIcon(0), interactive: false }).addTo(G.simRun);
  renderHud(1, `<div class="prog"><i id="simProg"></i></div><div class="note" id="simT"></div>`);
  const dur = 7000, t0 = performance.now(), nF = R.frames.length;
  await new Promise(res => {
    const step = now => {
      if (!S.sim || S.sim.run !== run) return res();
      const u = Math.max(0, Math.min(1, (now - t0) / dur)), fi = Math.max(0, Math.min(nF - 1, Math.floor(u * (nF - 1) + 1e-9)));
      const fr = R.frames[fi];
      for (let k = 0; k < parts.length; k++) parts[k].setLatLng([fr.lat[k], fr.lon[k]]);
      const tt = rel + fr.hours_after_release * HR, p = posAt(f, tt), q = posAt(f, tt - 600);
      if (p) { ship.setLatLng(p); if (q) ship.setIcon(shipIcon(bearing(q, p))); }
      const pg = $('#simProg'); if (pg) pg.style.width = (u * 100) + '%';
      const st = $('#simT'); if (st) st.textContent = `T+${fmt(fr.hours_after_release, 1)} h after release · ${fmtDM(tt)} — particles advected by the drift model`;
      u < 1 ? requestAnimationFrame(step) : res();
    };
    requestAnimationFrame(step);
  });
  if (!S.sim || S.sim.run !== run) return;
  if (b.simulated_geometry)
    L.geoJSON(b.simulated_geometry, { pane: 'sim', style: { color: C.sel, weight: 2, dashArray: '5,4', fillColor: C.sel, fillOpacity: .14 } }).addTo(G.simRun);
  const T = terms(b);
  renderHud(2, `<div class="cmp">
      <div><b>${T.iou.value}</b><span>Slick overlap</span></div>
      <div><b>${T.centroid.value}</b><span>Centre distance</span></div>
      <div><b>${T.orientation.value}</b><span>Course alignment Δ</span></div>
      <div><b>${fmt(1 / Math.max(b.area_ratio, 1e-6), 2)}×</b><span>Area within</span></div></div>
    <div class="keys"><span>${svgRect(C.slick, C.slick, .55)}Observed slick</span><span>${svgRect(C.sel, C.sel, .14, '3 2')}Simulated slick</span><span><svg viewBox="0 0 18 10" aria-hidden="true"><circle cx="5" cy="5" r="1.8" fill="${C.sel}"/><circle cx="11" cy="4" r="1.8" fill="${C.sel}"/></svg>Replay particles</span></div>
    <div class="note">Particles are a re-run of the same drift model for display; the dashed outline is the simulated slick scored during attribution.</div>
    <div style="display:flex;gap:8px;margin-top:10px"><button class="btn sm" id="simAgain">Replay simulation</button><button class="btn sm" id="simDone">Back to analysis</button></div>`);
  chromeOffsets();
  simFit(bb.pad(0.15));
  $('#simAgain').onclick = () => simulate();
  $('#simDone').onclick = () => exitSim();
}
const wait = ms => new Promise(r => setTimeout(r, ms));
/* Frame the simulation in the part of the map the card does not cover. On
   desktop/tablet the card is docked top-left, so reserve its width on the
   left; on phones it sits below the map and needs no reservation. */
function simFit(bb) {
  const hud = $('#simHud'), pd = padding();
  if (window.innerWidth > 760 && !hud.hidden) pd.paddingTopLeft = [hud.offsetWidth + 36, 60];
  fitWhenSized(() => map.fitBounds(bb, { maxZoom: 12, ...pd }));
}
function setLegend(open) {
  $('#legend').classList.toggle('collapsed', !open);
  $('#legendFold').textContent = open ? 'Hide' : 'Show';
  $('#legendFold').setAttribute('aria-expanded', open);
}
function exitSim(silent) {
  const reopen = S.sim && S.sim.legendWasOpen;
  if (reopen) setLegend(true);
  S.sim = null; G.simRun.clearLayers(); $('#simHud').hidden = true;
  applyVis();
  if (!silent) { setTime(S.T0); fitIncident(); }
}

/* ======================================================== satellite evidence */
function setViewButtons() {
  $('#vAnalysis').setAttribute('aria-pressed', S.view === 'analysis');
  $('#vSat').setAttribute('aria-pressed', S.view === 'sat');
}
$('#vAnalysis').onclick = () => setView('analysis');
$('#vSat').onclick = () => setView('sat');
async function setView(v) {
  if (v === S.view) return;
  stopPlay(); if (S.sim) exitSim(true);
  S.view = v; setViewButtons();
  const card = $('#satCard');
  if (v === 'analysis') { card.hidden = true; applyVis(); return; }
  if (!S.scene) {
    card.hidden = false; card.className = 'satcard'; card.innerHTML = '<p class="note">Loading satellite evidence…</p>';
    try { S.scene = await jget(`/api/incident/${encodeURIComponent(S.inc.incident_id)}/scene`); }
    catch (e) { S.scene = { available: false, reason: e.message }; }
    if (S.view !== 'sat') return;
  }
  drawScene(); applyVis();
}
function drawScene() {
  const sc = S.scene, card = $('#satCard'), src = S.inc.slick_source || {};
  G.scene.clearLayers(); G.detections.clearLayers();
  card.hidden = false;
  if (!sc.available) {
    card.className = 'satcard empty';
    card.innerHTML = `<h3>Satellite evidence layer unavailable for this scenario</h3>
      <p class="note">${esc(sc.reason || 'No scene raster is stored with this case.')}</p>
      <p class="note">Slick source: <b>${esc(src.source || 'unknown')}</b>${src.reason ? ` (${esc(src.reason)})` : ''}. No imagery is substituted.</p>
      <label class="note" style="display:inline-flex;gap:6px;align-items:center;margin-top:6px"><input type="checkbox" id="satOv"${S.satOverlay ? ' checked' : ''}> Show analysis overlay</label>`;
    $('#satOv').onchange = e => { S.satOverlay = e.target.checked; applyVis(); };
    return;
  }
  const img = L.imageOverlay(sc.image_url, sc.bounds, { pane: 'scene', opacity: 1, interactive: false }).addTo(G.scene);
  L.rectangle(sc.bounds, { pane: 'scene', color: '#9aa7b4', weight: 1, dashArray: '4,4', fill: false, interactive: false }).addTo(G.scene);
  const feats = (sc.detections && sc.detections.features) || [];
  feats.forEach(ft => {
    const p = ft.properties, chosen = p.slick_id === src.slick_id;
    L.geoJSON(ft, { pane: 'selected', style: chosen ? { color: C.slick, weight: 2.6, fill: false } : { color: '#e8eef4', weight: 1.4, dashArray: '4,3', fill: false } })
      .bindTooltip(`<b>${chosen ? 'Attributed slick' : 'Other detected dark feature'}</b> · ${esc(p.slick_id)}<br>${fmt(p.area_km2, 1)} km² · CNN oil probability ${pct(p.cnn_oil_prob, 0)}% · fused confidence ${pct(p.confidence, 0)}%` +
        (chosen ? '' : '<br><i>Not selected as this case’s observed slick</i>'), { sticky: true })
      .addTo(G.detections);
  });
  const meta = sc.detection_meta || {};
  card.className = 'satcard';
  card.innerHTML = `<h3>${sc.synthetic ? 'Simulated SAR scene' : 'SAR scene'}</h3>
    <p class="note">${esc(sc.source)}</p>
    <dl class="kv">
      <dt>Detections</dt><dd>${feats.length} dark feature${feats.length === 1 ? '' : 's'}</dd>
      <dt>Attributed slick</dt><dd>${esc(src.slick_id || '—')}</dd>
      ${num(src.confidence) ? `<dt>Fused confidence</dt><dd>${pct(src.confidence, 0)}%</dd>` : ''}
      ${meta.epoch != null ? `<dt>Model checkpoint</dt><dd>epoch ${meta.epoch}</dd>` : ''}
    </dl>
    <div class="keys" style="display:flex;gap:12px;font-size:11px;margin-top:8px;flex-wrap:wrap">
      <span>${svgLine(C.slick, 2.4, 1)} attributed</span><span>${svgLine('#e8eef4', 1.4, 1, false, '3 2')} other features</span></div>
    <p class="note" style="margin-top:8px">SAR sees surface roughness. Oil, biogenic films and low-wind patches all look dark, so every detection is an oil-like surface anomaly, not confirmed oil.</p>
    <label class="note" style="display:flex;gap:6px;align-items:center;margin-top:6px"><input type="checkbox" id="satOv"${S.satOverlay ? ' checked' : ''}> Show analysis overlay</label>
    <label class="note" style="display:flex;gap:6px;align-items:center;margin-top:4px">Scene opacity <input type="range" id="satOp" min="20" max="100" value="100" style="flex:1"></label>`;
  $('#satOv').onchange = e => { S.satOverlay = e.target.checked; applyVis(); };
  $('#satOp').oninput = e => img.setOpacity(+e.target.value / 100);
  const bb = L.geoJSON(S.inc.observed_slick.geometry).getBounds();
  feats.forEach(ft => bb.extend(L.geoJSON(ft).getBounds()));
  fitTo(bb.pad(0.2));
}

/* ================================================================ dossier */
$('#btnDossier').onclick = async () => {
  const b = $('#btnDossier'), l = b.querySelector('.lbl'), was = l.textContent;
  b.disabled = true; l.textContent = 'Generating — re-hashing artifacts…';
  try {
    const r = await fetch(`/api/incident/${encodeURIComponent(S.inc.incident_id)}/dossier`);
    if (!r.ok) throw new Error(`dossier → ${r.status} ${(await r.text()).slice(0, 160)}`);
    const blob = await r.blob(), url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `dossier_${S.inc.incident_id}.pdf`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    l.textContent = '✓ Dossier downloaded'; setTimeout(() => { l.textContent = was; }, 2600);
  } catch (e) { toast(e.message); l.textContent = was; }
  finally { b.disabled = false; }
};

/* ============================================================ chrome, misc */
function padding() {
  const tb = $('#timebar'), narrow = window.innerWidth <= 760;
  const drawer = window.innerWidth <= 1000 && !narrow && !document.body.classList.contains('panel-closed') ? $('#panel').offsetWidth : 0;
  const mw = map.getContainer().clientWidth - drawer;
  const left = narrow ? 20 : Math.min(240, Math.max(20, mw * 0.22));
  return { paddingTopLeft: [left, 60], paddingBottomRight: [30 + drawer, narrow ? 20 : tb.offsetHeight + 34] };
}
function fitTo(bb) { fitWhenSized(() => map.fitBounds(bb, { maxZoom: 11, ...padding() })); }
function fitIncident() {
  const bb = L.geoJSON(S.inc.origin_envelope).getBounds();
  bb.extend(L.geoJSON(S.inc.observed_slick.geometry).getBounds());
  const su = S.sus.get(S.sel);
  if (su) bb.extend([su.best_hypothesis.release_lat, su.best_hypothesis.release_lon]);
  fitTo(bb.pad(0.08));
}
function fitWhenSized(fn, tries = 0) {
  const el = map.getContainer();
  if ((el.clientWidth < 50 || el.clientHeight < 50) && tries < 40) return setTimeout(() => fitWhenSized(fn, tries + 1), 60);
  map.invalidateSize(); fn();
}
function chromeOffsets() {
  const narrow = window.innerWidth <= 760, h = $('#timebar').offsetHeight;
  $('#legend').style.bottom = narrow ? '' : (h + 24) + 'px';
  const bot = map.getContainer().querySelector('.leaflet-bottom.leaflet-right');
  if (bot) bot.style.bottom = narrow ? '0px' : (h + 14) + 'px';
  const tr = map.getContainer().querySelector('.leaflet-top.leaflet-right'); if (tr) tr.style.top = '44px';
  // Keep map chrome inside the map: the legend list and the simulation card
  // scroll rather than spilling over the case summary on short windows.
  const mh = $('#mapcol').clientHeight;
  if (!narrow) {
    $('#legendList').style.maxHeight = Math.max(80, mh - (h + 24) - 60 - 70) + 'px';
    $('#simHud').style.maxHeight = Math.max(160, mh - (h + 24) - 66) + 'px';
  } else { $('#legendList').style.maxHeight = ''; $('#simHud').style.maxHeight = ''; }
  map.invalidateSize();
}
new ResizeObserver(chromeOffsets).observe($('#timebar'));
window.addEventListener('resize', chromeOffsets);
if (window.innerWidth <= 1000) { $('#legend').classList.add('collapsed'); $('#legendFold').textContent = 'Show'; }

$('#panelToggle').onclick = () => {
  const closed = document.body.classList.toggle('panel-closed');
  $('#panelToggle').setAttribute('aria-expanded', !closed);
  setTimeout(() => map.invalidateSize(), 220);
};

/* basemap popover */
$('#baseBtn').onclick = e => { e.stopPropagation(); const p = $('#basePop'); p.hidden = !p.hidden; $('#baseBtn').setAttribute('aria-expanded', !p.hidden); };
document.addEventListener('click', e => { if (!e.target.closest('#basePop') && !e.target.closest('#baseBtn')) { $('#basePop').hidden = true; $('#baseBtn').setAttribute('aria-expanded', 'false'); } });
$$('input[name=base]').forEach(r => r.onchange = () => { BASE = r.value; applyBase(); });
$('#optLabels').onchange = applyBase;
$('#optGrid').onchange = drawGrid;
$('#optIndia').onchange = drawIndia;
$('#optCoast').onchange = e => e.target.checked ? G_coast.addTo(map) : map.removeLayer(G_coast);
L.DomEvent.disableClickPropagation($('#legend'));
L.DomEvent.disableScrollPropagation($('#legend'));
[$('#timebar'), $('#simHud'), $('#satCard'), $('.mapctl.tl'), $('.mapctl.tr')].forEach(el => { L.DomEvent.disableClickPropagation(el); L.DomEvent.disableScrollPropagation(el); });

/* tooltips: one positioned element for every [data-tip] */
const tip = $('#tip');
function showTip(el) {
  const t = el.getAttribute('data-tip'); if (!t) return;
  tip.textContent = t; tip.classList.add('on');
  const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  let x = r.left + r.width / 2 - w / 2, y = r.bottom + 8;
  x = Math.max(8, Math.min(window.innerWidth - w - 8, x));
  if (y + h > window.innerHeight - 8) y = r.top - h - 8;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => tip.classList.remove('on');
document.addEventListener('mouseover', e => { const el = e.target.closest('[data-tip]'); el ? showTip(el) : hideTip(); });
document.addEventListener('focusin', e => { const el = e.target.closest('[data-tip]'); el ? showTip(el) : hideTip(); });
document.addEventListener('focusout', hideTip);
document.addEventListener('scroll', hideTip, true);

/* keyboard: space play/pause, arrows step an hour, Esc leaves a simulation */
window.addEventListener('keydown', e => {
  if (document.querySelector('dialog[open]')) return;
  const tag = e.target.tagName;
  if ((tag === 'INPUT' && e.target.type !== 'range') || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (e.key === 'Escape' && S.sim) exitSim();
  if (tag === 'BUTTON') return;
  if (e.code === 'Space') { e.preventDefault(); $('#play').click(); }
  if (e.code === 'ArrowLeft' && tag !== 'INPUT') { stopPlay(); setTime(S.T - (e.shiftKey ? 6 : 1) * HR); }
  if (e.code === 'ArrowRight' && tag !== 'INPUT') { stopPlay(); setTime(S.T + (e.shiftKey ? 6 : 1) * HR); }
});

boot();
})();
