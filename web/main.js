import init, { Session } from "./pkg/loopclosure_wasm.js";

const LOG_URL = "data/intel.script.gz";
const FRAME_BUDGET_MS = 14;
const REBUILD_INTERVAL_MS = 400;
const COLOURS = {
  without: "#cf7d14",
  with: "#2a9fd0",
  truth: "rgba(219, 232, 242, 0.42)",
  wall: "rgba(219, 232, 242, 0.20)",
  closure: "rgba(219, 232, 242, 0.30)",
  grid: "#1d3550",
  axis: "#35536f",
  muted: "#9ab2c6",
};

const el = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const state = {
  session: null,
  running: true,
  generation: 0,
  view: null,
  lastRebuild: 0,
  images: new Map(),
  stepTimes: [],
  logText: null,
};

/** Shared with the browser test, which waits on these instead of on timers. */
window.loopclosure = { finished: false, keyframes: 0, closures: 0, stepMs: 0, error: null };

function setupCanvas(canvas) {
  const scale = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  const width = Math.round(box.width * scale);
  const height = Math.round(box.height * scale);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { ctx: canvas.getContext("2d"), width, height, scale };
}

/** Fit a world-space box into a canvas, keeping aspect ratio, north up. */
function transform(view, width, height) {
  const span = Math.max((view.maxX - view.minX) / width, (view.maxY - view.minY) / height);
  const offsetX = (width - (view.maxX - view.minX) / span) / 2;
  const offsetY = (height - (view.maxY - view.minY) / span) / 2;
  return {
    span,
    x: (x) => offsetX + (x - view.minX) / span,
    y: (y) => height - offsetY - (y - view.minY) / span,
  };
}

function worldView(session) {
  const walls = session.walls();
  if (walls.length === 0) return null;
  const box = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (let i = 0; i < walls.length; i += 2) {
    box.minX = Math.min(box.minX, walls[i]);
    box.maxX = Math.max(box.maxX, walls[i]);
    box.minY = Math.min(box.minY, walls[i + 1]);
    box.maxY = Math.max(box.maxY, walls[i + 1]);
  }
  return padded(box, 3);
}

function padded(box, margin) {
  return {
    minX: box.minX - margin,
    maxX: box.maxX + margin,
    minY: box.minY - margin,
    maxY: box.maxY + margin,
  };
}

/** A recorded log has no known extent, so the view follows both trajectories. */
function followView(session) {
  const box = { minX: -8, minY: -8, maxX: 8, maxY: 8 };
  for (const withClosure of [true, false]) {
    const t = session.trajectory(withClosure);
    for (let i = 0; i < t.length; i += 2) {
      box.minX = Math.min(box.minX, t[i]);
      box.maxX = Math.max(box.maxX, t[i]);
      box.minY = Math.min(box.minY, t[i + 1]);
      box.maxY = Math.max(box.maxY, t[i + 1]);
    }
  }
  return padded(box, 10);
}

function updateImage(session, withClosure) {
  const width = session.map_width(withClosure);
  const height = session.map_height(withClosure);
  const key = withClosure ? "closed" : "open";
  let image = state.images.get(key);
  if (!image || image.width !== width || image.height !== height) {
    image = new OffscreenCanvas(width, height);
    state.images.set(key, image);
  }
  const pixels = new ImageData(new Uint8ClampedArray(session.map_rgba(withClosure)), width, height);
  image.getContext("2d").putImageData(pixels, 0, 0);
  return image;
}

function strokePath(ctx, points, map, colour, lineWidth) {
  if (points.length < 4) return;
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(map.x(points[0]), map.y(points[1]));
  for (let i = 2; i < points.length; i += 2) ctx.lineTo(map.x(points[i]), map.y(points[i + 1]));
  ctx.stroke();
}

function drawMap(canvas, session, withClosure) {
  const { ctx, width, height, scale } = setupCanvas(canvas);
  const map = transform(state.view, width, height);
  ctx.fillStyle = "#0b1a2b";
  ctx.fillRect(0, 0, width, height);

  // Metre grid, 5 m pitch, drawn in world space so it stays put when the view
  // changes and reads as a scale bar.
  ctx.strokeStyle = COLOURS.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const pitch = 5;
  for (let x = Math.ceil(state.view.minX / pitch) * pitch; x <= state.view.maxX; x += pitch) {
    ctx.moveTo(Math.round(map.x(x)) + 0.5, 0);
    ctx.lineTo(Math.round(map.x(x)) + 0.5, height);
  }
  for (let y = Math.ceil(state.view.minY / pitch) * pitch; y <= state.view.maxY; y += pitch) {
    ctx.moveTo(0, Math.round(map.y(y)) + 0.5);
    ctx.lineTo(width, Math.round(map.y(y)) + 0.5);
  }
  ctx.stroke();

  const image = updateImage(session, withClosure);
  const resolution = session.map_resolution();
  const originX = session.map_origin_x(withClosure);
  const originY = session.map_origin_y(withClosure);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    image,
    map.x(originX),
    map.y(originY + image.height * resolution),
    (image.width * resolution) / map.span,
    (image.height * resolution) / map.span,
  );

  if (el("walls").checked) {
    const walls = session.walls();
    ctx.strokeStyle = COLOURS.wall;
    ctx.lineWidth = scale;
    ctx.beginPath();
    for (let i = 0; i < walls.length; i += 4) {
      ctx.moveTo(map.x(walls[i]), map.y(walls[i + 1]));
      ctx.lineTo(map.x(walls[i + 2]), map.y(walls[i + 3]));
    }
    ctx.stroke();
  }

  if (withClosure) {
    const segments = session.closure_segments();
    ctx.strokeStyle = COLOURS.closure;
    ctx.lineWidth = scale;
    ctx.beginPath();
    for (let i = 0; i < segments.length; i += 4) {
      ctx.moveTo(map.x(segments[i]), map.y(segments[i + 1]));
      ctx.lineTo(map.x(segments[i + 2]), map.y(segments[i + 3]));
    }
    ctx.stroke();
  }

  const colour = withClosure ? COLOURS.with : COLOURS.without;
  const trajectory = session.trajectory(withClosure);
  ctx.globalAlpha = 0.85;
  strokePath(ctx, trajectory, map, colour, (state.fixedView ? 2 : 1.25) * scale);
  ctx.globalAlpha = 1;
  if (trajectory.length >= 2) {
    const n = trajectory.length;
    ctx.fillStyle = colour;
    ctx.strokeStyle = "#0b1a2b";
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    ctx.arc(map.x(trajectory[n - 2]), map.y(trajectory[n - 1]), 5 * scale, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
}

function chartScales(errors, width, height, scale) {
  const pad = { left: 46 * scale, right: 12 * scale, top: 10 * scale, bottom: 22 * scale };
  let maxDistance = 1;
  let maxError = 0.5;
  for (let i = 0; i < errors.length; i += 3) {
    maxDistance = Math.max(maxDistance, errors[i]);
    maxError = Math.max(maxError, errors[i + 1], errors[i + 2]);
  }
  maxError = Math.ceil(maxError * 2) / 2;
  return {
    pad,
    maxDistance,
    maxError,
    x: (d) => pad.left + (d / maxDistance) * (width - pad.left - pad.right),
    y: (e) => height - pad.bottom - (e / maxError) * (height - pad.top - pad.bottom),
  };
}

function drawChart(session) {
  const canvas = el("chart");
  const { ctx, width, height, scale } = setupCanvas(canvas);
  ctx.fillStyle = "#0f2236";
  ctx.fillRect(0, 0, width, height);
  const errors = session.errors();
  const s = chartScales(errors, width, height, scale);
  state.chart = { errors, s, scale };

  ctx.font = `${11 * scale}px ui-monospace, Menlo, monospace`;
  ctx.fillStyle = COLOURS.muted;
  ctx.textAlign = "right";
  ctx.strokeStyle = COLOURS.grid;
  ctx.lineWidth = 1;
  const tick = s.maxError > 4 ? 2 : s.maxError > 2 ? 1 : 0.5;
  for (let e = 0; e <= s.maxError + 1e-6; e += tick) {
    ctx.beginPath();
    ctx.moveTo(s.pad.left, Math.round(s.y(e)) + 0.5);
    ctx.lineTo(width - s.pad.right, Math.round(s.y(e)) + 0.5);
    ctx.stroke();
    ctx.fillText(`${e.toFixed(1)} m`, s.pad.left - 6 * scale, s.y(e) + 4 * scale);
  }
  ctx.textAlign = "center";
  const step = s.maxDistance > 300 ? 100 : s.maxDistance > 120 ? 50 : 20;
  for (let d = 0; d <= s.maxDistance; d += step) {
    ctx.fillText(`${d}`, s.x(d), height - 7 * scale);
  }

  for (const [offset, colour] of [[1, COLOURS.without], [2, COLOURS.with]]) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2 * scale;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i + 2 < errors.length; i += 3) {
      const [x, y] = [s.x(errors[i]), s.y(errors[i + offset])];
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  if (state.hoverX !== undefined && errors.length > 0) {
    const row = nearestRow(errors, s, state.hoverX * scale);
    ctx.strokeStyle = COLOURS.axis;
    ctx.lineWidth = scale;
    ctx.beginPath();
    ctx.moveTo(s.x(errors[row]), s.pad.top);
    ctx.lineTo(s.x(errors[row]), height - s.pad.bottom);
    ctx.stroke();
  }
}

function nearestRow(errors, s, pixelX) {
  const target = ((pixelX - s.pad.left) / (s.x(s.maxDistance) - s.pad.left)) * s.maxDistance;
  let lo = 0;
  let hi = errors.length / 3 - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (errors[mid * 3] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo * 3;
}

function showTip() {
  const tip = el("tip");
  const chart = state.chart;
  if (state.hoverX === undefined || !chart || chart.errors.length === 0) {
    tip.hidden = true;
    return;
  }
  const row = nearestRow(chart.errors, chart.s, state.hoverX * chart.scale);
  tip.hidden = false;
  tip.textContent =
    `${chart.errors[row].toFixed(0)} m driven: ` +
    `${chart.errors[row + 1].toFixed(2)} m without, ${chart.errors[row + 2].toFixed(2)} m with`;
  const area = el("chart").getBoundingClientRect();
  const left = Math.min(state.hoverX + 12, area.width - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, left)}px`;
}

function updateReadout(session) {
  const errors = session.errors();
  const n = errors.length;
  el("stat-path").textContent = `${session.path_length().toFixed(0)} m`;
  el("stat-open").textContent = n ? `${errors[n - 2].toFixed(2)} m` : "-";
  el("stat-closed").textContent = n ? `${errors[n - 1].toFixed(2)} m` : "-";
  el("stat-keyframes").textContent = session.keyframes();
  el("stat-closures").textContent = session.closures();
  const times = state.stepTimes;
  const mean = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  el("stat-time").textContent = times.length ? `${mean.toFixed(1)} ms` : "-";
  Object.assign(window.loopclosure, {
    finished: session.finished(),
    keyframes: session.keyframes(),
    closures: session.closures(),
    stepMs: mean,
    errorOpen: n ? errors[n - 2] : null,
    errorClosed: n ? errors[n - 1] : null,
  });
}

function draw(session) {
  if (!state.fixedView) state.view = followView(session);
  drawMap(el("open"), session, false);
  drawMap(el("closed"), session, true);
  if (session.has_truth()) drawChart(session);
  updateReadout(session);
  showTip();
}

function frame(generation) {
  if (generation !== state.generation) return;
  const session = state.session;
  if (state.running && !session.finished()) {
    const start = performance.now();
    const target = Number(el("speed").value);
    let made = 0;
    while (made < target && performance.now() - start < FRAME_BUDGET_MS) {
      const before = session.keyframes();
      const t = performance.now();
      session.advance(1);
      state.stepTimes.push(performance.now() - t);
      if (state.stepTimes.length > 200) state.stepTimes.shift();
      made += session.keyframes() - before;
      if (session.finished()) break;
    }
    const now = performance.now();
    const rebuild =
      session.finished() || (session.maps_stale() && now - state.lastRebuild > REBUILD_INTERVAL_MS);
    session.refresh_maps(rebuild);
    if (rebuild) state.lastRebuild = now;
    draw(session);
  }
  requestAnimationFrame(() => frame(generation));
}

async function loadLog() {
  if (state.logText) return state.logText;
  const response = await fetch(LOG_URL);
  if (!response.ok) throw new Error(`could not fetch ${LOG_URL}: ${response.status}`);
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  state.logText = await new Response(stream).text();
  return state.logText;
}

async function restart() {
  state.generation += 1;
  const generation = state.generation;
  const input = el("input").value;
  const real = input === "intel";
  el("label-path").textContent = real ? "Odometric path" : "Distance driven";
  document.querySelectorAll(".sim-only").forEach((node) => node.classList.toggle("hidden", real));
  const note = el("note");
  try {
    if (real) {
      note.textContent = "Loading the log...";
      const text = await loadLog();
      if (generation !== state.generation) return;
      state.session = Session.recorded(text);
      note.textContent =
        "Intel Research Lab log, DSpace@MIT record 1721.1/62287, CC0 1.0. It has no ground truth, so no error is shown.";
    } else {
      state.session = Session.simulated(input, Number(el("seed").value) || 0, Number(el("noise").value));
      note.textContent = "Simulated run. Faint lines are the true walls; a map that drifts away from them has lost its place.";
    }
    state.stepTimes = [];
    state.images.clear();
    state.chart = null;
    state.lastRebuild = 0;
    state.view = worldView(state.session);
    state.fixedView = state.view !== null;
    if (!state.fixedView) state.view = followView(state.session);
    window.loopclosure.finished = false;
    draw(state.session);
    requestAnimationFrame(() => frame(generation));
  } catch (error) {
    window.loopclosure.error = String(error);
    note.textContent = `Could not start: ${error}`;
  }
}

function wireControls() {
  for (const id of ["input", "seed", "noise"]) el(id).addEventListener("change", restart);
  el("restart").addEventListener("click", restart);
  el("walls").addEventListener("change", () => state.session && draw(state.session));
  el("play").addEventListener("click", () => {
    state.running = !state.running;
    el("play").textContent = state.running ? "Pause" : "Play";
  });
  const chart = el("chart");
  chart.addEventListener("pointermove", (event) => {
    state.hoverX = event.offsetX;
    if (state.session) drawChart(state.session);
    showTip();
  });
  chart.addEventListener("pointerleave", () => {
    state.hoverX = undefined;
    if (state.session) drawChart(state.session);
    showTip();
  });
  new ResizeObserver(() => state.session && draw(state.session)).observe(document.body);
}

async function main() {
  for (const [param, id] of [["input", "input"], ["seed", "seed"], ["speed", "speed"], ["noise", "noise"]]) {
    if (params.has(param)) el(id).value = params.get(param);
  }
  await init();
  wireControls();
  await restart();
}

main().catch((error) => {
  window.loopclosure.error = String(error);
  el("note").textContent = `Could not load the WebAssembly module: ${error}`;
});
