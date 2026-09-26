/* ISRO sensor context layers — INSAT-3DS SST and EOS-06 OCM-3 chlorophyll.
 *
 * READ THIS BEFORE SHOWING IT TO ANYONE: these two rasters are MODELLED fields,
 * not MOSDAC products. They exist to show what the ingest layer would look like
 * once INSAT-3DS and EOS-06 are wired in, and every place they appear in the UI
 * is badged PROPOSED. The wind and current animation on the dashboard is a
 * different thing entirely — that reads the pipeline's own env.npz.
 *
 * Swapping in real data means replacing sstAt() and chlAt() with a sampler over
 * a MOSDAC GeoTIFF. Nothing else in this file changes.
 */
(function (global) {
  'use strict';

  const rad = d => d * Math.PI / 180;
  const deg = r => r * 180 / Math.PI;
  const R_E = 6371;

  function dest(lat, lon, brg, km) {
    const b = rad(brg), d = km / R_E, p1 = rad(lat), l1 = rad(lon);
    const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(b));
    const l2 = l1 + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(p1),
                               Math.cos(d) - Math.sin(p1) * Math.sin(p2));
    return [deg(p2), ((deg(l2) + 540) % 360) - 180];
  }

  /* ---- fields -------------------------------------------------------- */

  function sstAt(lat, lon) {
    let t = lat >= 0 ? 29.8 - 0.06 * lat - 0.0025 * lat * lat
                     : 29.8 - 0.30 * (-lat) - 0.0040 * lat * lat;
    t -= 5.2 * Math.exp(-(((lat - 9) ** 2) / 44 + ((lon - 52) ** 2) / 30));   // Somali upwelling
    t -= 3.0 * Math.exp(-(((lat - 19) ** 2) / 24 + ((lon - 59) ** 2) / 24));  // Oman upwelling
    t += 1.5 * Math.exp(-(((lat - 9) ** 2) / 64 + ((lon - 88) ** 2) / 96));   // BoB warm pool
    t += 1.1 * Math.exp(-(((lat - 1) ** 2) / 90 + ((lon - 99) ** 2) / 70));
    t += 2.4 * Math.exp(-(((lat - 26) ** 2) / 11 + ((lon - 52) ** 2) / 17));  // Persian Gulf
    t += 0.55 * Math.sin(lat * 0.9 + lon * 0.55) * Math.cos(lat * 0.6 - lon * 0.4);
    t += 0.28 * Math.sin(lon * 1.7) * Math.cos(lat * 1.3);
    return t;
  }

  function chlAt(lat, lon) {
    let c = 0.055 + 0.02 * Math.sin(lat * 1.1 + lon * 0.7);
    const g = (la, lo, sa, so, amp) =>
      amp * Math.exp(-(((lat - la) ** 2) / sa + ((lon - lo) ** 2) / so));
    c += g(10, 51, 34, 20, 7.5);      // Somali
    c += g(19, 59, 20, 17, 4.2);      // Oman
    c += g(22.2, 68.6, 7, 9, 5.6);    // Gujarat / Kutch shelf
    c += g(13, 74.0, 26, 4, 3.0);     // west India shelf
    c += g(20.5, 88.5, 12, 10, 4.4);  // Ganges plume
    c += g(-8, 106, 16, 40, 2.6);
    c += g(-3, 42, 9, 8, 2.4);
    c *= 1 + 0.30 * Math.sin(lat * 2.3 + lon * 1.4) * Math.cos(lat * 1.1 - lon * 0.9);
    return Math.max(0.03, c);
  }

  /* ---- ramps (sequential, monotone lightness, cool hues only) --------
   * Cool families deliberately: orange and red are reserved for the slick and
   * the ranked suspects, and an environmental raster must never compete with
   * the thing the investigator is looking at. */

  function ramp(stops) {
    return t => {
      t = Math.max(0, Math.min(1, t));
      for (let i = 1; i < stops.length; i++) {
        if (t <= stops[i][0]) {
          const a = stops[i - 1], b = stops[i], f = (t - a[0]) / (b[0] - a[0] || 1);
          return [0, 1, 2].map(k => Math.round(a[1][k] + (b[1][k] - a[1][k]) * f));
        }
      }
      return stops[stops.length - 1][1];
    };
  }

  const SST_RAMP = ramp([
    [0.00, [24, 42, 78]], [0.20, [27, 76, 124]], [0.42, [28, 118, 161]],
    [0.62, [40, 158, 184]], [0.78, [78, 195, 201]], [0.90, [146, 221, 216]],
    [1.00, [212, 241, 233]]]);
  const CHL_RAMP = ramp([
    [0.00, [14, 52, 34]], [0.30, [20, 90, 54]], [0.60, [36, 143, 76]],
    [0.82, [82, 192, 112]], [1.00, [156, 228, 166]]]);

  const sstT = t => (t - 16) / 16;
  const chlT = c => (Math.log10(c) - Math.log10(0.05)) / (Math.log10(10) - Math.log10(0.05));

  function cssRamp(fn, n) {
    const s = [];
    for (let i = 0; i <= n; i++) {
      const c = fn(i / n);
      s.push(`rgb(${c[0]},${c[1]},${c[2]}) ${(i / n * 100).toFixed(1)}%`);
    }
    return `linear-gradient(90deg,${s.join(',')})`;
  }

  /* ---- land mask ------------------------------------------------------
   * Natural Earth 110 m rings, punched out of each canvas tile so the raster
   * stops at the coast instead of painting over Gujarat. */

  let LAND = [], LANDBB = [];

  function setLand(polys) {
    LAND = polys || [];
    LANDBB = LAND.map(p => {
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (const pt of p[0]) {
        if (pt[0] < x0) x0 = pt[0];
        if (pt[0] > x1) x1 = pt[0];
        if (pt[1] < y0) y0 = pt[1];
        if (pt[1] > y1) y1 = pt[1];
      }
      return [x0, y0, x1, y1];
    });
  }

  function punchLand(map, ctx, coords, size) {
    if (!LAND.length) return;
    const z = coords.z, ox = coords.x * size.x, oy = coords.y * size.y;
    const nw = map.unproject([ox, oy], z), se = map.unproject([ox + size.x, oy + size.y], z);
    const tb = [nw.lng, se.lat, se.lng, nw.lat];
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = '#000';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = Math.max(1.5, size.x / 110);   // erode the coast a touch
    for (let i = 0; i < LAND.length; i++) {
      const b = LANDBB[i];
      if (b[2] < tb[0] || b[0] > tb[2] || b[3] < tb[1] || b[1] > tb[3]) continue;
      const p = new Path2D();
      for (const ring of LAND[i]) {
        for (let k = 0; k < ring.length; k++) {
          const q = map.project([ring[k][1], ring[k][0]], z);
          k ? p.lineTo(q.x - ox, q.y - oy) : p.moveTo(q.x - ox, q.y - oy);
        }
        p.closePath();
      }
      ctx.fill(p);
      ctx.stroke(p);
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  /* ---- tile layers ---------------------------------------------------- */

  function fieldLayer(map, sample) {
    return L.GridLayer.extend({
      createTile(coords, done) {
        const size = this.getTileSize(), tile = L.DomUtil.create('canvas');
        tile.width = size.x;
        tile.height = size.y;
        const N = 32, off = document.createElement('canvas');
        off.width = N;
        off.height = N;
        const octx = off.getContext('2d'), img = octx.createImageData(N, N);
        const ox = coords.x * size.x, oy = coords.y * size.y, z = coords.z;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const p = map.unproject([ox + (i + 0.5) * size.x / N, oy + (j + 0.5) * size.y / N], z);
          const r = sample(p.lat, p.lng);
          const k = (j * N + i) * 4;
          img.data[k] = r[0][0];
          img.data[k + 1] = r[0][1];
          img.data[k + 2] = r[0][2];
          img.data[k + 3] = r[1];
        }
        octx.putImageData(img, 0, 0);
        const ctx = tile.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(off, 0, 0, N, N, -1, -1, size.x + 2, size.y + 2);
        punchLand(map, ctx, coords, size);
        // A canvas tile is painted synchronously, but Leaflet holds it at
        // opacity 0 until done() fires. Fire it on the next tick.
        setTimeout(() => done(null, tile), 0);
        return tile;
      }
    });
  }

  function sstLayer(map, opts) {
    const C = fieldLayer(map, (lat, lon) => [SST_RAMP(sstT(sstAt(lat, lon))), 255]);
    return new C(opts);
  }

  function chlLayer(map, opts) {
    const C = fieldLayer(map, (lat, lon) => {
      const t = chlT(chlAt(lat, lon));
      return [CHL_RAMP(t), Math.round(255 * Math.max(0, Math.min(1, (t - 0.42) / 0.42)) ** 1.25)];
    });
    return new C(opts);
  }

  /* ---- EOS-04 SAR swath ----------------------------------------------
   * MRS mode: 115 km swath on a sun-synchronous descending pass. Drawn around
   * the scene so the footprint the detection came from is explicit. */

  function swathPolygon(lat, lon, trackDeg, lenKm, widthKm) {
    const h = widthKm / 2, n = 24, ring = [];
    const half = lenKm / 2;
    for (let i = 0; i <= n; i++) {
      const p = dest(lat, lon, trackDeg, -half + lenKm * i / n);
      const q = dest(p[0], p[1], trackDeg + 90, h);
      ring.push([q[1], q[0]]);
    }
    for (let i = n; i >= 0; i--) {
      const p = dest(lat, lon, trackDeg, -half + lenKm * i / n);
      const q = dest(p[0], p[1], trackDeg - 90, h);
      ring.push([q[1], q[0]]);
    }
    ring.push(ring[0]);
    return { type: 'Polygon', coordinates: [ring] };
  }

  global.ISRO = {
    sstAt, chlAt, sstLayer, chlLayer, setLand, cssRamp,
    SST_RAMP, CHL_RAMP, sstT, chlT, swathPolygon, dest,
    SST_DOMAIN: [16, 32], CHL_DOMAIN: [0.05, 10]
  };
})(window);
