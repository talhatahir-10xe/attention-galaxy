/**
 * Attention Galaxy renderer.
 *
 * A single full-width canvas draws BOTH panels:
 *   - left  (~34%): the prediction panel (candidate words + probability bars,
 *                   or, in spotlight mode, the clicked word's attention list)
 *   - right (~66%): the galaxy (word-stars placed by meaning, wired by attention)
 *
 * One requestAnimationFrame loop drives a small plain state object. There is no
 * framework. The generation animation runs the per-word sequence described in
 * SPEC.md Section 3: ask -> gather -> predict -> lock -> fly.
 */

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
const PHASE = { ask: 500, gather: 1300, predict: 1000, lock: 600, fly: 800 };

const PHYS = {
  k_home: 0.07, // spring to the assigned grid slot (keeps the even spread)
  k_rep: 600, // gentle repulsion (slots already separate stars); avoids drift
  k_att: 0.00012, // only a faint attention tug, so spread isn't undone
  damping: 0.82,
  cap: 10,
};

const PANEL_FRAC = 0.34; // left panel width as a fraction of canvas width

// Ambient link thinning (keeps the galaxy readable as it grows): per word, only
// its strongest few look-back links are drawn when nothing is focused.
const AMBIENT_TOPK = 2;
const AMBIENT_MIN = 0.1; // don't draw ambient links weaker than this
// Exaggerate the strong-vs-weak contrast with a gamma curve (<1 = punchier).
const W_GAMMA = 0.7;
const emph = (w) => Math.pow(Math.max(0, w), W_GAMMA);

// Cluster palette (fixed, colour-blind-friendly-ish, reads on dark bg).
const PALETTE = [
  "#7aa2ff", // blue
  "#ffd37a", // amber
  "#7ff0c4", // teal
  "#ff9db1", // rose
  "#c4a2ff", // violet
  "#9be36b", // green
];

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
// Deterministic PRNG so the layout is identical across frames and replays.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const lerp = (a, b, t) => a + (b - a) * t;
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
const easeInOutCubic = (t) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

function displayText(text) {
  // Trim the tokenizer's leading space for display; show a visible marker for
  // raw newlines so the galaxy never shows an empty label.
  const t = text.replace(/\n/g, "⏎");
  return t.trim() === "" ? t.replace(/ /g, "·") || "·" : t.trim();
}

// Greedy word-wrap for a canvas context: splits `text` into lines no wider
// than `maxW` pixels (using the context's current font).
function wrapLines(ctx, text, maxW) {
  const words = text.split(" ");
  const lines = [];
  let cur = "";
  for (const w of words) {
    const test = cur ? cur + " " + w : w;
    if (cur && ctx.measureText(test).width > maxW) {
      lines.push(cur);
      cur = w;
    } else {
      cur = test;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${a})`;
}

// ---------------------------------------------------------------------------
// Galaxy class
// ---------------------------------------------------------------------------
export class Galaxy {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} cb callbacks: { onCaption, onStatus, onRunStateChange }
   */
  constructor(canvas, cb = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cb = cb;

    this.data = null;
    this.nodes = [];
    this.particles = [];

    this.W = 0;
    this.H = 0;
    this.dpr = 1;

    this.currentLayer = 0;
    this.beautiful = true; // Phase 2 visuals on by default

    // animation / run state
    this.running = false;
    this.step = 0; // index into data.steps
    this.phase = "idle";
    this.phaseStart = 0;
    this.landedCount = 0; // number of nodes currently landed/visible
    this.flyNode = null;
    this.flyFrom = null;

    // interaction
    this.selected = null; // spotlight node index
    this.hover = null;

    // panel layout cache (for fly origin + hit-testing)
    this._panelRows = [];

    this._bindEvents();
    this.resize();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // ------------------------------------------------------------------ setup
  _bindEvents() {
    window.addEventListener("resize", () => this.resize());
    this.canvas.addEventListener("click", (e) => this._onClick(e));
    this.canvas.addEventListener("mousemove", (e) => this._onMove(e));
    this.canvas.addEventListener("mouseleave", () => (this.hover = null));
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.W = rect.width;
    this.H = rect.height;
    this.canvas.width = Math.round(this.W * this.dpr);
    this.canvas.height = Math.round(this.H * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    if (this.data) this._recomputeHomes();
  }

  // Geometry of the two regions (in CSS pixels).
  _geom() {
    const panelW = Math.max(200, this.W * PANEL_FRAC);
    const pad = 46;
    return {
      panelW,
      panel: { x: 0, y: 0, w: panelW, h: this.H },
      galaxy: {
        x0: panelW + pad,
        y0: pad + 10,
        w: this.W - panelW - pad * 1.5,
        h: this.H - pad * 2.2,
      },
    };
  }

  /**
   * Blue-noise layout: give every token its own well-separated slot on a
   * jittered grid that fills the galaxy region, then assign tokens to slots in
   * a deterministic SHUFFLED order. This spreads every star — including
   * duplicate words — randomly but far apart, never stacked and never in a line.
   */
  _recomputeHomes() {
    const g = this._geom().galaxy;
    const N = this.nodes.length;
    if (!N) return;

    const aspect = g.w / Math.max(1, g.h);
    let cols = Math.max(1, Math.round(Math.sqrt(N * aspect)));
    let rows = Math.ceil(N / cols);
    cols = Math.ceil(N / rows); // tighten so the grid isn't lopsided
    const cw = g.w / cols;
    const ch = g.h / rows;

    // Jittered cell centres (deterministic seed → stable across frames/replays).
    const rnd = mulberry32(1337 + N);
    const slots = [];
    for (let r = 0; r < rows && slots.length < N; r++) {
      for (let c = 0; c < cols && slots.length < N; c++) {
        const jx = (rnd() - 0.5) * cw * 0.6;
        const jy = (rnd() - 0.5) * ch * 0.6;
        slots.push({
          x: g.x0 + (c + 0.5) * cw + jx,
          y: g.y0 + (r + 0.5) * ch + jy,
        });
      }
    }

    // Deterministic Fisher–Yates shuffle of the slots, so token order (and any
    // duplicate words) scatter across the whole region instead of lining up.
    for (let i = slots.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [slots[i], slots[j]] = [slots[j], slots[i]];
    }

    this.nodes.forEach((node, idx) => {
      const slot = slots[idx];
      node.homeX = slot.x;
      node.homeY = slot.y;
      if (node.landed && node.x === undefined) {
        node.x = node.homeX;
        node.y = node.homeY;
      }
    });
  }

  // ------------------------------------------------------------------ data
  setData(data) {
    this.data = data;
    this.currentLayer = data.defaultLayer ?? Math.floor((data.numLayers || 1) / 2);
    this.selected = null;

    const assign = data.clusters?.assign || [];
    this.nodes = data.tokens.map((tok) => ({
      index: tok.index,
      text: tok.text,
      label: displayText(tok.text),
      isPrompt: tok.isPrompt,
      cluster: assign[tok.index] ?? 0,
      color: PALETTE[(assign[tok.index] ?? 0) % PALETTE.length],
      x: undefined,
      y: undefined,
      vx: 0,
      vy: 0,
      landed: false,
      born: 0,
      pulse: 0,
    }));

    this.resize(); // ensure canvas dimensions are current, then place nodes
    this._recomputeHomes();
    this._resetRun();
  }

  // Number of prompt tokens.
  get _promptLen() {
    return this.nodes.filter((n) => n.isPrompt).length;
  }

  // -------------------------------------------------- attention display math
  /**
   * Look-back weights for token `i` at `layer`: the attention it pays to each
   * STRICTLY earlier token j<i, re-normalised to sum to 1 (self-attention
   * excluded so the beams of one word add up to 100%).
   * @returns {Array<{j:number, w:number}>} sorted by weight desc
   */
  lookback(i, layer = this.currentLayer) {
    const row = this.data?.attention?.byLayer?.[layer]?.[i];
    if (!row) return [];
    const out = [];
    let sum = 0;
    for (let j = 0; j < i; j++) {
      const w = row[j] || 0;
      out.push({ j, w });
      sum += w;
    }
    if (sum > 0) for (const o of out) o.w /= sum;
    out.sort((a, b) => b.w - a.w);
    return out;
  }

  // ------------------------------------------------------------------ run
  _resetRun() {
    this.running = false;
    this.step = 0;
    this.phase = "idle";
    this.flyNode = null;
    this.landedCount = this._promptLen;
    const now = performance.now();
    this.nodes.forEach((n, i) => {
      if (n.isPrompt) {
        n.landed = true;
        n.born = now - i * 60; // slight stagger on first show
        n.x = n.homeX + (Math.random() - 0.5) * 2;
        n.y = n.homeY + (Math.random() - 0.5) * 2;
        n.vx = n.vy = 0;
      } else {
        n.landed = false;
        n.x = n.y = undefined;
        n.vx = n.vy = 0;
      }
    });
    this._updateCaption();
  }

  play() {
    if (!this.data || !this.data.steps.length) return;
    this.selected = null;
    this._resetRun();
    this.running = true;
    this.step = 0;
    this._enterPhase("ask");
    this.cb.onRunStateChange?.(true);
    this.cb.onStatus?.("");
  }

  replay() {
    this.play();
  }

  /**
   * Stop an in-progress animation and jump straight to the settled view (all
   * current words placed and clickable). Safe to call when not running.
   */
  finish() {
    this.showFinal();
  }

  /**
   * Show the finished galaxy immediately: every token landed at its slot, no
   * animation, ready for exploration. Used after a Stop, or to display a
   * generation without replaying it.
   */
  showFinal() {
    if (!this.data) return;
    this.running = false;
    this.step = 0;
    this.phase = "idle";
    this.flyNode = null;
    this.selected = null;
    const now = performance.now();
    this.nodes.forEach((n) => {
      n.landed = true;
      n.born = now - 500; // already settled: skip the scale-in pop
      if (n.x === undefined) {
        n.x = n.homeX;
        n.y = n.homeY;
      }
      n.vx = n.vy = 0;
      n.pulse = 0;
    });
    this.landedCount = this.nodes.length;
    this._updateCaption();
    this.cb.onRunStateChange?.(false);
    this.cb.onStatus?.("Click any star to explore its attention.");
  }

  _enterPhase(phase) {
    this.phase = phase;
    this.phaseStart = performance.now();

    if (phase === "ask") {
      const asker = this._askerIndex();
      if (asker >= 0 && this.nodes[asker]) this.nodes[asker].pulse = 1;
    }
    if (phase === "lock") {
      // show the winning word in the caption as it locks in (not yet landed)
      this._updateCaption(this._curStep()?.token || "");
    }
    if (phase === "fly") {
      this._startFly();
    }
  }

  // The asker is the most recent landed token before the one being generated.
  _askerIndex() {
    return this.landedCount - 1;
  }

  // The asker whose look-back scene stays on screen for the whole prediction
  // step — lit during gather, then held through predict/lock/fly until the next
  // word lands — so the attention stays shown until the next token is predicted.
  _stepAsker() {
    if (
      this.running &&
      (this.phase === "gather" ||
        this.phase === "predict" ||
        this.phase === "lock" ||
        this.phase === "fly")
    ) {
      return this._askerIndex();
    }
    return -1;
  }

  _curStep() {
    return this.data.steps[this.step];
  }

  _advancePhase() {
    switch (this.phase) {
      case "ask":
        this._enterPhase("gather");
        break;
      case "gather":
        this._enterPhase("predict");
        break;
      case "predict":
        this._enterPhase("lock");
        break;
      case "lock":
        this._enterPhase("fly");
        break;
      case "fly":
        this._finishFly();
        if (this.step + 1 < this.data.steps.length) {
          this.step += 1;
          this._enterPhase("ask");
        } else {
          this.running = false;
          this.phase = "idle";
          this.cb.onRunStateChange?.(false);
          this.cb.onStatus?.("Click any star to explore its attention.");
        }
        break;
    }
  }

  _startFly() {
    const s = this._curStep();
    const idx = s.generatedIndex;
    const node = this.nodes[idx];
    if (!node) return;
    // Fly origin: the winner's row in the prediction panel.
    const row = this._panelRows[0]; // winner is row 0 (sorted desc)
    const from = row
      ? { x: row.x + row.w, y: row.y + row.h / 2 }
      : { x: this._geom().panelW, y: this.H / 2 };
    node.x = from.x;
    node.y = from.y;
    node.vx = node.vy = 0;
    node.landed = false;
    this.flyNode = node;
    this.flyFrom = from;
  }

  _finishFly() {
    const s = this._curStep();
    const node = this.nodes[s.generatedIndex];
    if (node) {
      node.landed = true;
      node.born = performance.now();
      node.pulse = 1;
      node.x = node.homeX;
      node.y = node.homeY;
      node.vx = node.vy = 0;
    }
    this.landedCount = s.generatedIndex + 1;
    this.flyNode = null;
    this._updateCaption();
  }

  _updateCaption(extraToken = "") {
    // Caption = all landed tokens, plus (optionally) the winner currently
    // locking in before it has landed. `_finishFly` passes no extra because the
    // winner is already counted in `landedCount` by then.
    let text = "";
    for (let i = 0; i < this.landedCount && i < this.nodes.length; i++) {
      text += this.data.tokens[i].text;
    }
    text += extraToken;
    this.cb.onCaption?.(text.replace(/\n/g, " ").replace(/^\s+/, ""));
  }

  setLayer(layer) {
    this.currentLayer = clamp(layer, 0, (this.data?.numLayers || 1) - 1);
  }

  setBeautiful(on) {
    this.beautiful = !!on;
  }

  // ------------------------------------------------------------------ input
  _evtPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _nodeAt(px, py) {
    let best = null;
    let bestD = 22 * 22; // hit radius^2
    for (let i = 0; i < this.landedCount; i++) {
      const n = this.nodes[i];
      if (!n.landed || n.x === undefined) continue;
      const dx = n.x - px;
      const dy = n.y - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  _onClick(e) {
    if (this.running) return; // explore only once generation has settled
    const { x, y } = this._evtPos(e);
    const n = this._nodeAt(x, y);
    if (n && n.index > 0) {
      this.selected = this.selected === n.index ? null : n.index;
    } else {
      this.selected = null;
    }
    this.cb.onSelect?.(this.selected);
  }

  /** Programmatically select a token (e.g. from clicking a caption word). */
  selectIndex(index) {
    if (this.running) return;
    this.selected = index != null && index > 0 ? index : null;
    this.cb.onSelect?.(this.selected);
  }

  _onMove(e) {
    if (this.running) {
      this.hover = null;
      this.canvas.style.cursor = "default";
      return;
    }
    const { x, y } = this._evtPos(e);
    const n = this._nodeAt(x, y);
    this.hover = n ? n.index : null;
    this.canvas.style.cursor = n && n.index > 0 ? "pointer" : "default";
  }

  // ------------------------------------------------------------------ loop
  _loop(now) {
    if (this.running) {
      const dur = PHASE[this.phase] || 0;
      if (dur && now - this.phaseStart >= dur) this._advancePhase();
    }
    this._physics();
    this._draw(now);
    requestAnimationFrame(this._loop);
  }

  // ------------------------------------------------------------------ physics
  _physics() {
    if (!this.data) return;
    const layout = this.data.defaultLayer ?? this.currentLayer; // layout = middle layer
    const landed = [];
    for (let i = 0; i < this.landedCount; i++) {
      const n = this.nodes[i];
      if (n && n.landed && n.x !== undefined) landed.push(n);
    }

    for (const n of landed) {
      // spring to home
      let fx = (n.homeX - n.x) * PHYS.k_home;
      let fy = (n.homeY - n.y) * PHYS.k_home;

      // repulsion from other nodes
      for (const m of landed) {
        if (m === n) continue;
        const dx = n.x - m.x;
        const dy = n.y - m.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) d2 = 1;
        const f = PHYS.k_rep / d2;
        const d = Math.sqrt(d2);
        fx += (dx / d) * f;
        fy += (dy / d) * f;
      }

      // attention attraction (layout layer only, to attended earlier words)
      const lb = this.lookback(n.index, layout);
      for (const { j, w } of lb) {
        const m = this.nodes[j];
        if (!m || !m.landed || m.x === undefined) continue;
        fx += (m.x - n.x) * PHYS.k_att * w * 60;
        fy += (m.y - n.y) * PHYS.k_att * w * 60;
      }

      n.vx = (n.vx + fx) * PHYS.damping;
      n.vy = (n.vy + fy) * PHYS.damping;
      const sp = Math.hypot(n.vx, n.vy);
      if (sp > PHYS.cap) {
        n.vx = (n.vx / sp) * PHYS.cap;
        n.vy = (n.vy / sp) * PHYS.cap;
      }
      n.x += n.vx;
      n.y += n.vy;

      // keep inside galaxy region
      const g = this._geom().galaxy;
      n.x = clamp(n.x, g.x0 - 20, g.x0 + g.w + 20);
      n.y = clamp(n.y, g.y0 - 20, g.y0 + g.h + 20);
    }

    // decay pulses
    for (const n of this.nodes) if (n.pulse > 0) n.pulse = Math.max(0, n.pulse - 0.02);

    this._updateParticles();
  }

  // ------------------------------------------------------------------ particles
  _updateParticles() {
    // advance existing
    for (const p of this.particles) p.t += p.speed;
    this.particles = this.particles.filter((p) => p.t < 1);

    if (!this.beautiful) return;

    // Emit along "active" links: gather beams / spotlight links (focused), or,
    // when nothing is focused, the single strongest link of each word.
    const { links, focused } = this._activeLinks();
    for (const link of links) {
      if (link.w < 0.06) continue;
      const rate = focused ? 0.18 + emph(link.w) * 0.6 : 0.03 + emph(link.w) * 0.12;
      if (Math.random() < rate) {
        const from = this.nodes[link.j]; // attended word
        const to = this.nodes[link.i]; // attending word (particles flow in)
        if (!from || !to || from.x === undefined || to.x === undefined) continue;
        this.particles.push({
          ax: from.x,
          ay: from.y,
          bx: to.x,
          by: to.y,
          t: 0,
          speed: 0.016 + Math.random() * 0.018,
          color: to.color,
          w: link.w,
        });
      }
    }
    if (this.particles.length > 500) this.particles.splice(0, this.particles.length - 500);
  }

  // Links currently "active", as {links:[{i,j,w}], focused:bool}.
  _activeLinks() {
    const stepA = this._stepAsker();
    if (stepA >= 0) {
      return {
        links: this.lookback(stepA).map(({ j, w }) => ({ i: stepA, j, w })),
        focused: true,
      };
    }
    if (!this.running && this.selected != null) {
      return {
        links: this.lookback(this.selected).map(({ j, w }) => ({ i: this.selected, j, w })),
        focused: true,
      };
    }
    // ambient: strongest link of each word, so the galaxy quietly flows
    const out = [];
    for (let i = 1; i < this.landedCount; i++) {
      const top = this.lookback(i)[0];
      if (top && top.w >= 0.18) out.push({ i, j: top.j, w: top.w });
    }
    return { links: out, focused: false };
  }

  // ------------------------------------------------------------------ draw
  _draw(now) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    this._drawBackground();

    if (!this.data) {
      this._drawEmpty();
      return;
    }

    const geom = this._geom();
    this._drawNebula(geom);
    this._drawLinks(geom, now);
    this._drawParticles();
    this._drawStars(geom, now);
    this._drawFlyNode(now);
    this._drawPanel(geom, now);
    this._drawDivider(geom);
  }

  _drawBackground() {
    const ctx = this.ctx;
    const grd = ctx.createRadialGradient(
      this.W * 0.7,
      this.H * 0.4,
      50,
      this.W * 0.7,
      this.H * 0.4,
      Math.max(this.W, this.H)
    );
    grd.addColorStop(0, "#0a0d1e");
    grd.addColorStop(1, "#05060f");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, this.W, this.H);

    // faint starfield in galaxy region
    if (!this._bg) {
      this._bg = [];
      for (let i = 0; i < 90; i++)
        this._bg.push({ x: Math.random(), y: Math.random(), r: Math.random() * 1.2 });
    }
    const g = this._geom().galaxy;
    ctx.save();
    for (const s of this._bg) {
      ctx.fillStyle = `rgba(255,255,255,${0.04 + s.r * 0.04})`;
      ctx.beginPath();
      ctx.arc(g.x0 + s.x * g.w, g.y0 - 20 + s.y * (g.h + 40), s.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  _drawEmpty() {
    const ctx = this.ctx;
    ctx.fillStyle = "#5b6488";
    ctx.font = "16px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Type a prompt and press Generate…", this.W / 2, this.H / 2);
    ctx.textAlign = "left";
  }

  _drawNebula(geom) {
    if (!this.beautiful || !this.data.clusters) return;
    const ctx = this.ctx;
    const g = geom.galaxy;

    // Centre each nebula on the actual stars of that cluster (positions come
    // from the grid layout now, not the embedding), so the glow follows them.
    const sums = {};
    for (let i = 0; i < this.landedCount; i++) {
      const n = this.nodes[i];
      if (!n || !n.landed || n.x === undefined) continue;
      const c = n.cluster;
      (sums[c] ||= { x: 0, y: 0, n: 0 });
      sums[c].x += n.x;
      sums[c].y += n.y;
      sums[c].n += 1;
    }

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    Object.keys(sums).forEach((key) => {
      const i = +key;
      const s = sums[i];
      const cx = s.x / s.n;
      const cy = s.y / s.n;
      const rad = Math.min(g.w, g.h) * 0.4;
      const col = PALETTE[i % PALETTE.length];
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
      grd.addColorStop(0, hexToRgba(col, 0.1));
      grd.addColorStop(0.5, hexToRgba(col, 0.04));
      grd.addColorStop(1, hexToRgba(col, 0));
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  _drawLinks(geom, now) {
    const ctx = this.ctx;
    const spotlight = !this.running && this.selected != null;
    const stepA = this._stepAsker();
    const gathering = this.running && this.phase === "gather";
    const gatherElapsed = gathering ? (now - this.phaseStart) / PHASE.gather : 0;
    const focus = stepA >= 0 ? stepA : spotlight ? this.selected : -1;

    ctx.save();
    if (focus >= 0) {
      // Focused: draw ONLY the focused word's beams, bright and bold. Everything
      // else is hidden so the look-back reads cleanly. The beams grow in during
      // gather, then stay at full strength through predict/lock/fly.
      const grow = gathering ? easeOutCubic(clamp(gatherElapsed * 1.3, 0, 1)) : 1;
      const ni = this.nodes[focus];
      for (const { j, w } of this.lookback(focus)) {
        const nj = this.nodes[j];
        if (!ni || !nj || ni.x === undefined || nj.x === undefined) continue;
        const s = emph(w);
        this._strokeLink(nj, ni, {
          alpha: (0.22 + s * 0.78) * grow,
          width: 1.2 + s * 6.5,
          color: this.beautiful ? ni.color : "#9fb0e8",
          glow: this.beautiful ? 10 * s : 0,
        });
      }
    } else {
      // Ambient: per word, only its strongest few links, with punchy contrast.
      for (let i = 0; i < this.landedCount; i++) {
        const ni = this.nodes[i];
        if (!ni || !ni.landed || ni.x === undefined) continue;
        const lb = this.lookback(i).slice(0, AMBIENT_TOPK);
        for (const { j, w } of lb) {
          if (w < AMBIENT_MIN) continue;
          const nj = this.nodes[j];
          if (!nj || !nj.landed || nj.x === undefined) continue;
          const s = emph(w);
          this._strokeLink(nj, ni, {
            alpha: 0.06 + s * 0.6,
            width: 0.5 + s * 5,
            color: this.beautiful ? ni.color : "#8296c8",
            glow: this.beautiful ? 5 * s : 0,
          });
        }
      }
    }
    ctx.shadowBlur = 0;
    ctx.restore();

    this._drawLinkNumbers(stepA, spotlight, gatherElapsed, gathering);
  }

  _strokeLink(from, to, { alpha, width, color, glow }) {
    const ctx = this.ctx;
    ctx.strokeStyle = hexToRgba(color, clamp(alpha, 0, 1));
    ctx.lineWidth = width;
    ctx.shadowColor = glow ? hexToRgba(color, 0.6) : "transparent";
    ctx.shadowBlur = glow || 0;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  _drawLinkNumbers(stepA, spotlight, gatherElapsed, gathering) {
    const ctx = this.ctx;
    const which = stepA >= 0 ? stepA : spotlight ? this.selected : -1;
    if (which < 0) return;
    const reveal = gathering ? clamp(gatherElapsed * 1.4, 0, 1) : 1;
    const ni = this.nodes[which];
    if (!ni || ni.x === undefined) return;

    ctx.save();
    ctx.font = "700 14px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // Place pills at varying points along each beam so they don't overlap.
    const links = this.lookback(which).filter((o) => o.w >= 0.03);
    links.forEach((o, idx) => {
      const nj = this.nodes[o.j];
      if (!nj || nj.x === undefined) return;
      const t = links.length > 1 ? 0.62 - (idx / (links.length - 1)) * 0.3 : 0.5;
      const mx = lerp(ni.x, nj.x, t);
      const my = lerp(ni.y, nj.y, t);
      const pct = `${Math.round(o.w * 100)}%`;
      ctx.globalAlpha = reveal;
      const tw = ctx.measureText(pct).width + 12;
      const hot = o.w >= 0.18;
      ctx.fillStyle = "rgba(6,9,22,0.9)";
      ctx.beginPath();
      ctx.roundRect(mx - tw / 2, my - 11, tw, 22, 7);
      ctx.fill();
      ctx.strokeStyle = hexToRgba(hot ? "#ffd37a" : ni.color, 0.8);
      ctx.lineWidth = hot ? 1.6 : 1;
      ctx.stroke();
      ctx.fillStyle = hot ? "#ffe8b0" : "#eaf0ff";
      ctx.fillText(pct, mx, my + 0.5);
    });
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  _drawParticles() {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const p of this.particles) {
      const t = easeInOutCubic(p.t);
      const x = lerp(p.ax, p.bx, t);
      const y = lerp(p.ay, p.by, t);
      const a = Math.sin(p.t * Math.PI); // fade in/out
      const rad = 1.6 + emph(p.w || 0.2) * 3.5; // bigger packets on stronger links
      ctx.fillStyle = hexToRgba(p.color, 0.9 * a);
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  _drawStars(geom, now) {
    const ctx = this.ctx;
    const spotlight = !this.running && this.selected != null;
    const stepA = this._stepAsker();
    // The word whose attention scene is on screen (spotlight, or the current
    // asker during a prediction step). Its attended words glow; the rest dim.
    const focusNode = spotlight ? this.selected : stepA;
    const attendedSet = new Set();
    if (focusNode >= 0)
      for (const { j } of this.lookback(focusNode)) attendedSet.add(j);

    for (let i = 0; i < this.landedCount; i++) {
      const n = this.nodes[i];
      if (!n || !n.landed || n.x === undefined) continue;

      const age = (now - n.born) / 420;
      const scaleIn = easeOutCubic(clamp(age, 0, 1));
      let dim = false;
      let emphasise = false;
      if (focusNode >= 0) {
        if (n.index === focusNode) emphasise = true;
        else if (attendedSet.has(n.index)) emphasise = true;
        else dim = true;
      }
      const isAsker = this.running && n.index === this._askerIndex();

      this._drawStar(n, scaleIn, { dim, emphasise, isAsker, now });
    }
  }

  _drawStar(n, scaleIn, opts) {
    const ctx = this.ctx;
    const base = 8;
    let r = base * scaleIn;
    const alpha = opts.dim ? 0.28 : 1;

    // asker pulse / general pulse
    let pulse = n.pulse;
    if (opts.isAsker) pulse = Math.max(pulse, 0.6 + 0.4 * Math.sin(opts.now / 140));

    if (this.beautiful) {
      // glow
      const glowR = r * (3.4 + pulse * 2.6);
      const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, glowR);
      grd.addColorStop(0, hexToRgba(n.color, (opts.emphasise ? 1 : 0.6) * alpha));
      grd.addColorStop(0.4, hexToRgba(n.color, 0.28 * alpha));
      grd.addColorStop(1, hexToRgba(n.color, 0));
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(n.x, n.y, glowR, 0, Math.PI * 2);
      ctx.fill();
    }

    // core
    ctx.beginPath();
    ctx.fillStyle = this.beautiful
      ? hexToRgba("#ffffff", alpha)
      : hexToRgba(n.color, alpha);
    ctx.arc(n.x, n.y, r + (opts.emphasise ? 2 : 0), 0, Math.PI * 2);
    ctx.fill();
    if (this.beautiful) {
      ctx.strokeStyle = hexToRgba(n.color, alpha);
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // label with a dark pill behind it for legibility
    const fontPx = opts.emphasise ? 16 : 14.5;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = `${opts.emphasise ? "700" : "600"} ${fontPx}px Inter, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const ly = n.y - r - 12;
    const tw = ctx.measureText(n.label).width;
    ctx.fillStyle = `rgba(5,7,16,${opts.dim ? 0.35 : 0.72})`;
    ctx.beginPath();
    ctx.roundRect(n.x - tw / 2 - 6, ly - fontPx / 2 - 3, tw + 12, fontPx + 6, 6);
    ctx.fill();
    ctx.fillStyle = opts.dim ? "#7079a3" : opts.emphasise ? "#ffffff" : "#eef2ff";
    ctx.fillText(n.label, n.x, ly + 0.5);
    ctx.restore();
  }

  _drawFlyNode(now) {
    if (!this.flyNode) return;
    const n = this.flyNode;
    const t = easeInOutCubic(clamp((now - this.phaseStart) / PHASE.fly, 0, 1));
    const x = lerp(this.flyFrom.x, n.homeX, t);
    const y = lerp(this.flyFrom.y, n.homeY, t);
    n.x = x;
    n.y = y;
    const ctx = this.ctx;
    // trail
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const grd = ctx.createRadialGradient(x, y, 0, x, y, 26);
    grd.addColorStop(0, hexToRgba(n.color, 0.8));
    grd.addColorStop(1, hexToRgba(n.color, 0));
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.beginPath();
    ctx.fillStyle = "#ffffff";
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "600 13px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#eef2ff";
    ctx.textAlign = "center";
    ctx.fillText(n.label, x, y - 12);
    ctx.textAlign = "left";
  }

  // ---------------------------------------------------------- prediction panel
  _drawDivider(geom) {
    const ctx = this.ctx;
    ctx.strokeStyle = "rgba(122,162,255,0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(geom.panelW, 0);
    ctx.lineTo(geom.panelW, this.H);
    ctx.stroke();
  }

  _drawPanel(geom, now) {
    const ctx = this.ctx;
    const p = geom.panel;
    const padX = 20;
    const innerW = p.w - padX * 2;

    // panel bg
    ctx.fillStyle = "rgba(10,12,26,0.55)";
    ctx.fillRect(0, 0, p.w, p.h);

    const spotlight = !this.running && this.selected != null;

    // title
    ctx.font = "700 13px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#9fb0e8";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    const title = spotlight
      ? `“${this.nodes[this.selected].label}” looks back at…`
      : this.running
      ? "predicting next word"
      : "explore attention";
    ctx.fillText(title.toUpperCase(), padX, 34);
    ctx.strokeStyle = "rgba(122,162,255,0.15)";
    ctx.beginPath();
    ctx.moveTo(padX, 44);
    ctx.lineTo(padX + innerW, 44);
    ctx.stroke();

    this._panelRows = [];

    if (spotlight) {
      this._drawAttentionList(padX, innerW, 64);
    } else if (this.running) {
      this._drawCandidates(padX, innerW, 64, now);
    } else {
      // Settled after a generation: invite exploration and explain the slider.
      ctx.textAlign = "left";
      ctx.fillStyle = "#ffd37a";
      ctx.font = "700 17px Inter, system-ui, sans-serif";
      ctx.fillText("★ Click any star", padX, 80);
      ctx.fillStyle = "#eef2ff";
      ctx.font = "600 15px Inter, system-ui, sans-serif";
      ctx.fillText("to explore its attention.", padX, 104);

      const modelShort = (this.data.model || "This model").split("/").pop();
      ctx.fillStyle = "#9aa6d4";
      ctx.font = "500 13px Inter, system-ui, sans-serif";
      const explainer =
        `${modelShort} has ${this.data.numLayers} attention layers. ` +
        `Use the slider below to see the attention any word pays — in any layer.`;
      let yy = 140;
      for (const line of wrapLines(ctx, explainer, innerW)) {
        ctx.fillText(line, padX, yy);
        yy += 19;
      }
    }
  }

  _rowMetrics(top, innerW) {
    return { rowH: 30, gap: 14, barH: 16, top, innerW };
  }

  _drawCandidates(padX, innerW, top, now) {
    const s = this._curStep();
    if (!s) return;
    const ctx = this.ctx;
    const cands = s.candidates;
    const m = this._rowMetrics(top, innerW);

    // bar growth progress across predict/lock/fly
    let grow = 0;
    if (this.phase === "ask" || this.phase === "gather") grow = 0;
    else if (this.phase === "predict")
      grow = easeOutCubic(clamp((now - this.phaseStart) / PHASE.predict, 0, 1));
    else grow = 1;

    // loser fade during lock/fly
    let loserAlpha = 1;
    if (this.phase === "lock")
      loserAlpha = 1 - easeInOutCubic(clamp((now - this.phaseStart) / PHASE.lock, 0, 1));
    else if (this.phase === "fly") loserAlpha = 0;

    cands.forEach((c, idx) => {
      const y = top + idx * (m.rowH + m.gap);
      const isWinner = idx === 0;
      const alpha = isWinner ? 1 : loserAlpha;
      if (alpha <= 0.01) {
        if (isWinner) this._panelRows[0] = { x: padX, y, w: innerW, h: m.rowH };
        return;
      }
      ctx.save();
      ctx.globalAlpha = alpha;

      // label + pct
      ctx.font = "500 13.5px Inter, system-ui, sans-serif";
      ctx.fillStyle = isWinner && (this.phase === "lock" || this.phase === "fly")
        ? "#ffd37a"
        : "#dfe6ff";
      ctx.textAlign = "left";
      ctx.fillText(displayText(c.text), padX, y + 13);

      const pct = `${Math.round(c.prob * 100)}%`;
      ctx.textAlign = "right";
      ctx.fillStyle = "#9fb0e8";
      ctx.fillText(pct, padX + innerW, y + 13);

      // bar track
      const by = y + 18;
      ctx.fillStyle = "rgba(122,162,255,0.08)";
      ctx.beginPath();
      ctx.roundRect(padX, by, innerW, m.barH, 5);
      ctx.fill();

      // bar fill
      const fillW = innerW * c.prob * grow;
      const col = isWinner ? "#7aa2ff" : "#49608f";
      const hot = isWinner && (this.phase === "lock" || this.phase === "fly");
      ctx.fillStyle = hot ? "#ffd37a" : col;
      if (this.beautiful && isWinner) {
        ctx.shadowColor = hexToRgba(hot ? "#ffd37a" : "#7aa2ff", 0.6);
        ctx.shadowBlur = 10;
      }
      ctx.beginPath();
      ctx.roundRect(padX, by, Math.max(0, fillW), m.barH, 5);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.restore();

      this._panelRows[idx] = { x: padX, y, w: innerW, h: m.rowH };
    });

    // phase hint
    ctx.fillStyle = "#5b6488";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    const hint = {
      ask: "the newest word is about to predict…",
      gather: "looking back — gathering from earlier words",
      predict: "scoring candidate next words",
      lock: "locked in the winner",
      fly: "sending it into the galaxy",
    }[this.phase];
    if (hint)
      ctx.fillText(hint, padX, top + cands.length * (m.rowH + m.gap) + 8);
  }

  _drawAttentionList(padX, innerW, top) {
    const ctx = this.ctx;
    const lb = this.lookback(this.selected).filter((o) => o.w > 0.001);
    const m = this._rowMetrics(top, innerW);

    if (!lb.length) {
      ctx.fillStyle = "#5b6488";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("This is the first word — nothing to look back at.", padX, top + 10);
      return;
    }

    lb.forEach((o, idx) => {
      const y = top + idx * (m.rowH + m.gap);
      const node = this.nodes[o.j];
      ctx.font = "500 13.5px Inter, system-ui, sans-serif";
      ctx.fillStyle = "#dfe6ff";
      ctx.textAlign = "left";
      ctx.fillText(node ? node.label : "?", padX, y + 13);

      ctx.textAlign = "right";
      ctx.fillStyle = "#9fb0e8";
      ctx.fillText(`${Math.round(o.w * 100)}%`, padX + innerW, y + 13);

      const by = y + 18;
      ctx.fillStyle = "rgba(122,162,255,0.08)";
      ctx.beginPath();
      ctx.roundRect(padX, by, innerW, m.barH, 5);
      ctx.fill();

      const col = node ? node.color : "#7aa2ff";
      ctx.fillStyle = col;
      if (this.beautiful) {
        ctx.shadowColor = hexToRgba(col, 0.5);
        ctx.shadowBlur = 8;
      }
      ctx.beginPath();
      ctx.roundRect(padX, by, Math.max(0, innerW * o.w), m.barH, 5);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    ctx.fillStyle = "#5b6488";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("click empty space to clear", padX, top + lb.length * (m.rowH + m.gap) + 8);
  }
}
