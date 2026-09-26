/* Animated wind / current layer.
 *
 * Reads the pipeline's own environment field (artifacts/<id>/env.npz, served
 * decimated by /api/incident/<id>/env) and advects display particles through
 * it. This is the SAME field the drift model integrates, so what you see moving
 * is what moved the oil — not a decorative loop.
 *
 * It is a DISPLAY integrator, not a physics one: it uses a display time scale
 * so a 0.2 m/s current is visible in a few seconds, and it does not apply the
 * 3 % wind leeway, the 15 deg Coriolis deflection or the diffusion term that
 * attribution/drift.py applies. Never read a trajectory off this layer.
 */
(function (global) {
  'use strict';

  const FlowLayer = L.Layer.extend({

    options: {
      kind: 'curr',          // 'curr' | 'wind'
      colour: '#39c5cf',
      count: 2200,
      maxAgeFrames: 90,
      fade: 0.955,           // trail persistence per frame
      speedScale: 1800,      // display seconds of drift per real second
      width: 1.15,
      // Stroke opacity. Separated from `colour` so the field can be made
      // recessive without changing its hue, which is what identifies it in the
      // legend. The defaults are the original values; the dashboard turns them
      // well down, because at incident zoom a full-opacity field competes with
      // the slick and the vessel tracks for attention and wins.
      alphaGlow: 0.16,       // wide soft pass under the line
      alphaCore: 0.92,       // crisp 1 px line on top
      // animate:false gives arrows only — no drifting particles, no trails,
      // no requestAnimationFrame loop. A still field reads as "here is the
      // direction", where a moving one reads as "look at me", and on a map
      // whose subject is the slick the second is the wrong instruction.
      animate: true,
      glyphs: true,          // direction arrows on a coarse grid
      glyphWidth: 1,         // arrow stroke weight; drives head size too
      glyphLen: 7,           // base half-length in px, before the speed term
      glyphColour: 'rgba(255,255,255,.22)'
    },

    initialize(opts) {
      L.setOptions(this, opts);
      this._field = null;
      this._ti = 0;
      this._particles = [];
      this._raf = null;
    },

    onAdd(map) {
      this._map = map;
      const pane = map.getPane('overlayPane');
      this._canvas = L.DomUtil.create('canvas', 'leaflet-zoom-hide flow-canvas');
      this._canvas.style.position = 'absolute';
      this._canvas.style.pointerEvents = 'none';
      // First child, so the slick, envelope and tracks always draw on top of
      // the streaks. The flow is context; it must never bury the evidence.
      pane.insertBefore(this._canvas, pane.firstChild);
      this._ctx = this._canvas.getContext('2d');

      map.on('moveend zoomend resize', this._reset, this);
      map.on('movestart zoomstart', this._pause, this);
      this._reset();
      return this;
    },

    onRemove(map) {
      this._pause();
      map.off('moveend zoomend resize', this._reset, this);
      map.off('movestart zoomstart', this._pause, this);
      if (this._canvas && this._canvas.parentNode) this._canvas.parentNode.removeChild(this._canvas);
      this._canvas = null;
      return this;
    },

    /** field: {lats, lons, times(ISO), wind_u, wind_v, curr_u, curr_v} */
    setField(field) {
      this._field = field;
      this._times = (field.times || []).map(s => Date.parse(s) / 1000);
      this._reset();
      return this;
    },

    /** t: epoch seconds. Snaps to the nearest field timestep. */
    setTime(t) {
      if (!this._times || !this._times.length) return this;
      let best = 0, bd = Infinity;
      for (let i = 0; i < this._times.length; i++) {
        const d = Math.abs(this._times[i] - t);
        if (d < bd) { bd = d; best = i; }
      }
      if (best !== this._ti) { this._ti = best; this._drawGlyphs(); }
      return this;
    },

    /** Peak speed in the current frame — drives the legend. */
    maxSpeed() {
      const f = this._field;
      if (!f) return 0;
      const U = f[this.options.kind + '_u'][this._ti];
      const V = f[this.options.kind + '_v'][this._ti];
      let m = 0;
      for (let j = 0; j < U.length; j++)
        for (let i = 0; i < U[j].length; i++)
          m = Math.max(m, Math.hypot(U[j][i], V[j][i]));
      return m;
    },

    /* ---------------- sampling ---------------- */

    _sample(lat, lon) {
      const f = this._field;
      if (!f) return null;
      const la = f.lats, lo = f.lons;
      // lats may be ascending or descending; handle both.
      const asc = la[la.length - 1] > la[0];
      const fi = asc ? (lat - la[0]) / (la[la.length - 1] - la[0]) * (la.length - 1)
                     : (la[0] - lat) / (la[0] - la[la.length - 1]) * (la.length - 1);
      const fj = (lon - lo[0]) / (lo[lo.length - 1] - lo[0]) * (lo.length - 1);
      if (!(fi >= 0 && fi <= la.length - 1 && fj >= 0 && fj <= lo.length - 1)) return null;

      const i0 = Math.floor(fi), j0 = Math.floor(fj);
      const i1 = Math.min(i0 + 1, la.length - 1), j1 = Math.min(j0 + 1, lo.length - 1);
      const di = fi - i0, dj = fj - j0;
      const U = f[this.options.kind + '_u'][this._ti];
      const V = f[this.options.kind + '_v'][this._ti];
      const bl = (M) =>
        M[i0][j0] * (1 - di) * (1 - dj) + M[i1][j0] * di * (1 - dj) +
        M[i0][j1] * (1 - di) * dj + M[i1][j1] * di * dj;
      return [bl(U), bl(V)];
    },

    /* ---------------- lifecycle ---------------- */

    _reset() {
      if (!this._canvas || !this._map) return;
      const size = this._map.getSize();
      const tl = this._map.containerPointToLayerPoint([0, 0]);
      L.DomUtil.setPosition(this._canvas, tl);
      const dpr = Math.min(2, global.devicePixelRatio || 1);
      this._canvas.width = size.x * dpr;
      this._canvas.height = size.y * dpr;
      this._canvas.style.width = size.x + 'px';
      this._canvas.style.height = size.y + 'px';
      this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this._ctx.clearRect(0, 0, size.x, size.y);
      this._drawGlyphs();
      if (this.options.animate) {
        this._seed();
        this._play();
      } else {
        // Static: arrows only, painted once per view change. Nothing moves, so
        // there is no loop to run and no trail to fade.
        this._particles = [];
        this._paintGlyphs();
      }
    },

    _seed() {
      const b = this._map.getBounds();
      this._particles = [];
      for (let i = 0; i < this.options.count; i++) this._particles.push(this._spawn(b, true));
    },

    _spawn(b, randomAge) {
      return {
        lat: b.getSouth() + Math.random() * (b.getNorth() - b.getSouth()),
        lon: b.getWest() + Math.random() * (b.getEast() - b.getWest()),
        age: randomAge ? Math.floor(Math.random() * this.options.maxAgeFrames) : 0
      };
    },

    _play() {
      if (this._raf) return;
      const step = () => { this._frame(); this._raf = global.requestAnimationFrame(step); };
      this._raf = global.requestAnimationFrame(step);
    },

    _pause() {
      if (this._raf) { global.cancelAnimationFrame(this._raf); this._raf = null; }
    },

    /* ---------------- drawing ---------------- */

    _drawGlyphs() {
      // Static arrows on a coarse grid, redrawn only when the timestep changes.
      // Streaks show where things go; the arrows let you read a direction at a
      // glance without waiting for the animation.
      this._glyphs = null;
      if (!this.options.glyphs || !this._field || !this._map) return;
      const m = this._map, b = m.getBounds(), size = m.getSize();
      const n = 9;
      const out = [];
      for (let r = 0; r <= n; r++) for (let c = 0; c <= n; c++) {
        const lat = b.getSouth() + (b.getNorth() - b.getSouth()) * (r + .5) / (n + 1);
        const lon = b.getWest() + (b.getEast() - b.getWest()) * (c + .5) / (n + 1);
        const uv = this._sample(lat, lon);
        if (!uv) continue;
        const s = Math.hypot(uv[0], uv[1]);
        if (s < 1e-3) continue;
        const p = m.latLngToContainerPoint([lat, lon]);
        if (p.x < -20 || p.y < -20 || p.x > size.x + 20 || p.y > size.y + 20) continue;
        out.push({ x: p.x, y: p.y, ux: uv[0] / s, uy: uv[1] / s, s });
      }
      this._glyphs = out;
    },

    _paintGlyphs() {
      // Stroke the arrows the current timestep produced. Split out of _frame so
      // the static mode can draw them without running the animation loop at
      // all: in that mode there is no trail to fade, and a fading pass over a
      // still image just makes the arrows flicker and dim.
      const ctx = this._ctx, o = this.options;
      if (!ctx || !this._glyphs) return;
      ctx.strokeStyle = o.glyphColour;
      ctx.fillStyle = o.glyphColour;
      ctx.lineWidth = o.glyphWidth;
      for (const g of this._glyphs) {
        const len = o.glyphLen + Math.min(11, g.s * (o.kind === 'wind' ? 1.1 : 14));
        const ex = g.x + g.ux * len, ey = g.y - g.uy * len;
        ctx.beginPath();
        ctx.moveTo(g.x - g.ux * len, g.y + g.uy * len);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        const hw = o.glyphWidth * 3.0, hl = o.glyphWidth * 5.0;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - g.ux * hl + g.uy * hw, ey + g.uy * hl + g.ux * hw);
        ctx.lineTo(ex - g.ux * hl - g.uy * hw, ey + g.uy * hl - g.ux * hw);
        ctx.closePath();
        ctx.fill();
      }
    },

    _frame() {
      const ctx = this._ctx, m = this._map;
      if (!ctx || !m || !this._field) return;
      const size = m.getSize(), b = m.getBounds();
      const o = this.options;

      // Fade the previous frame instead of clearing: that residue IS the trail.
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = `rgba(0,0,0,${(1 - o.fade).toFixed(3)})`;
      ctx.fillRect(0, 0, size.x, size.y);
      ctx.globalCompositeOperation = 'source-over';

      this._paintGlyphs();

      // Advance every particle once, collecting screen-space segments.
      const dtDeg = o.speedScale / 111320;   // m/s -> deg lat per frame
      const seg = [];
      for (const p of this._particles) {
        const uv = this._sample(p.lat, p.lon);
        if (!uv || p.age++ > o.maxAgeFrames) { Object.assign(p, this._spawn(b, false)); continue; }
        const a = m.latLngToContainerPoint([p.lat, p.lon]);
        p.lat += uv[1] * dtDeg;
        p.lon += uv[0] * dtDeg / Math.max(0.2, Math.cos(p.lat * Math.PI / 180));
        if (!b.contains([p.lat, p.lon])) { Object.assign(p, this._spawn(b, false)); continue; }
        const c = m.latLngToContainerPoint([p.lat, p.lon]);
        if (Math.abs(c.x - a.x) > 60 || Math.abs(c.y - a.y) > 60) continue;  // wrap guard
        seg.push(a.x, a.y, c.x, c.y);
      }

      // Stroke twice: a wide low-alpha glow under a crisp core. A 1 px line on
      // a dark basemap is technically correct and practically invisible.
      const stroke = (w, alpha) => {
        ctx.globalAlpha = alpha;
        ctx.lineWidth = w;
        ctx.beginPath();
        for (let i = 0; i < seg.length; i += 4) {
          ctx.moveTo(seg[i], seg[i + 1]);
          ctx.lineTo(seg[i + 2], seg[i + 3]);
        }
        ctx.stroke();
      };
      ctx.lineCap = 'round';
      ctx.strokeStyle = o.colour;
      stroke(o.width * 3.4, o.alphaGlow);
      stroke(o.width, o.alphaCore);
      ctx.globalAlpha = 1;
    }
  });

  global.FlowLayer = opts => new FlowLayer(opts);
})(window);
