/* Breakout Policy Visualizer — GridPolicyMLP
 * Reconstructed env: paddle on row 8, ball, bricks on rows 1-3, wall border.
 * Observation: 10x10x5 channels flattened as (row*10+col)*5+channel. */

"use strict";

const GRID = 10;
const ACTIONS = ["no-op", "up", "down", "left", "right"];
const PADDLE_ROW = 8;
const PADDLE_W = 2;
const BRICK_ROWS = [1, 2, 3];
// classic Breakout cabinet palette, top row first
const BRICK_COLORS = { 1: "#ff5252", 2: "#ff9f2e", 3: "#ffe14d" };

// ---------- model ----------
let W = null; // weights json

function forward(x) {
  const h1 = new Float64Array(32);
  for (let j = 0; j < 32; j++) {
    let s = W.b1[j];
    for (let i = 0; i < 500; i++) if (x[i] !== 0) s += x[i] * W.W1[i][j];
    h1[j] = Math.tanh(s);
  }
  const h2 = new Float64Array(32);
  for (let j = 0; j < 32; j++) {
    let s = W.b2[j];
    for (let i = 0; i < 32; i++) s += h1[i] * W.W2[i][j];
    h2[j] = Math.tanh(s);
  }
  const logits = new Float64Array(5);
  for (let a = 0; a < 5; a++) {
    let s = W.b3[a];
    for (let i = 0; i < 32; i++) s += h2[i] * W.W3[i][a];
    logits[a] = s;
  }
  return logits;
}

function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const t = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / t);
}

function sample(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ---------- environment ----------
const env = {
  paddleCol: 4, // left cell of 2-wide paddle on row 8
  ball: { r: 6, c: 4, vr: -1, vc: 1 },
  bricks: new Set(),
  prev: [], // previous-frame occupied cells (paddle + ball)
  trail: [], // recent ball positions for the motion trail
  over: false,
  score: 0,
  step: 0,
  episode: 1,
};

function resetEnv() {
  env.paddleCol = 4;
  env.ball = {
    r: 6,
    c: 2 + Math.floor(Math.random() * 5),
    vr: -1,
    vc: Math.random() < 0.5 ? -1 : 1,
  };
  env.bricks = new Set();
  for (const r of BRICK_ROWS) for (let c = 1; c <= 8; c++) env.bricks.add(r * 10 + c);
  env.prev = [];
  env.trail = [];
  env.over = false;
  env.score = 0;
  env.step = 0;
}

function observation() {
  const x = new Float64Array(500);
  const put = (r, c, ch) => { x[(r * 10 + c) * 5 + ch] = 1; };
  for (let i = 0; i < 10; i++) {
    put(0, i, 3); put(9, i, 3); put(i, 0, 3); put(i, 9, 3);
  }
  for (let k = 0; k < PADDLE_W; k++) put(PADDLE_ROW, env.paddleCol + k, 0);
  put(env.ball.r, env.ball.c, 1);
  for (const b of env.bricks) put(Math.floor(b / 10), b % 10, 2);
  for (const p of env.prev) put(p.r, p.c, 4);
  return x;
}

function stepEnv(action) {
  if (env.over) return null;
  const occ = [{ r: env.ball.r, c: env.ball.c }];
  for (let k = 0; k < PADDLE_W; k++) occ.push({ r: PADDLE_ROW, c: env.paddleCol + k });
  env.prev = occ;
  env.trail.push({ r: env.ball.r, c: env.ball.c });
  if (env.trail.length > 6) env.trail.shift();

  if (action === 3) env.paddleCol = Math.max(1, env.paddleCol - 1);
  if (action === 4) env.paddleCol = Math.min(9 - PADDLE_W, env.paddleCol + 1);

  let { r, c, vr, vc } = env.ball;
  let nr = r + vr, nc = c + vc, ev = null;

  if (nc <= 0 || nc >= 9) { vc = -vc; nc = c + vc; blip(220, 0.04); }
  if (nr <= 0) { vr = -vr; nr = r + vr; blip(220, 0.04); }

  if (env.bricks.has(nr * 10 + nc)) {
    env.bricks.delete(nr * 10 + nc);
    env.score += 1;
    vr = -vr; nr = r + vr;
    ev = `brick +1 (${env.bricks.size} left)`;
    blip(880, 0.07);
  }
  if (nr === PADDLE_ROW && nc >= env.paddleCol && nc < env.paddleCol + PADDLE_W) {
    vr = -Math.abs(vr); nr = PADDLE_ROW - 1;
    ev = ev || "paddle bounce";
    blip(440, 0.05);
  }
  if (nr >= 9) {
    env.over = true;
    env.ball = { r: 8, c: nc, vr, vc };
    env.step += 1;
    gameOverSound();
    return "game-over";
  }

  env.ball = { r: nr, c: nc, vr, vc };
  env.step += 1;

  if (env.bricks.size === 0) {
    env.over = true;
    clearSound();
    return "cleared";
  }
  return ev;
}

// ---------- sound (WebAudio, off until enabled) ----------
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function blip(freq, dur) {
  if (!document.getElementById("sound").checked) return;
  try {
    const ctx = ensureAudio();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = "square";
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.05, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
    o.connect(g).connect(ctx.destination);
    o.start();
    o.stop(ctx.currentTime + dur);
  } catch (e) { /* audio unavailable — fine */ }
}
function gameOverSound() {
  if (!document.getElementById("sound").checked) return;
  [330, 220, 147].forEach((f, i) => setTimeout(() => blip(f, 0.12), i * 130));
}
function clearSound() {
  if (!document.getElementById("sound").checked) return;
  [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => blip(f, 0.1), i * 110));
}

// ---------- rendering ----------
const canvas = document.getElementById("grid");
const ctx = canvas.getContext("2d");
const CELL = 52; // 10*52 = 520

function glowCell(r, c, fill, inset, blur) {
  ctx.save();
  ctx.shadowColor = fill;
  ctx.shadowBlur = blur;
  ctx.fillStyle = fill;
  ctx.fillRect(c * CELL + inset, r * CELL + inset, CELL - inset * 2, CELL - inset * 2);
  ctx.restore();
}

function draw(showGhost) {
  ctx.clearRect(0, 0, 520, 520);
  // grid lines
  ctx.strokeStyle = "#18212a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, 520); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(520, i * CELL); ctx.stroke();
  }
  // walls
  for (let i = 0; i < 10; i++) {
    for (const [r, c] of [[0, i], [9, i], [i, 0], [i, 9]]) {
      ctx.fillStyle = "#2b3742";
      ctx.fillRect(c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6);
    }
  }
  // bricks — classic row colors with a soft glow
  for (const b of env.bricks) {
    const r = Math.floor(b / 10), c = b % 10;
    glowCell(r, c, BRICK_COLORS[r] || "#ff6a5c", 3, 10);
  }
  // ball trail (oldest faintest)
  env.trail.forEach((p, i) => {
    const t = (i + 1) / env.trail.length;
    ctx.fillStyle = `rgba(255, 178, 36, ${0.06 + t * 0.16})`;
    ctx.fillRect(p.c * CELL + 16, p.r * CELL + 16, CELL - 32, CELL - 32);
  });
  // ghost of previous frame
  if (showGhost) {
    for (const p of env.prev) {
      ctx.strokeStyle = "rgba(255,178,36,0.35)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 2;
      ctx.strokeRect(p.c * CELL + 5, p.r * CELL + 5, CELL - 10, CELL - 10);
      ctx.setLineDash([]);
    }
  }
  // paddle — hot phosphor green
  for (let k = 0; k < PADDLE_W; k++) glowCell(PADDLE_ROW, env.paddleCol + k, "#3ef26a", 3, 14);
  // ball — bright amber orb
  glowCell(env.ball.r, env.ball.c, "#ffb224", 13, 18);

  if (env.over) {
    ctx.fillStyle = "rgba(12,16,20,0.65)";
    ctx.fillRect(0, 0, 520, 520);
    ctx.fillStyle = env.bricks.size === 0 ? "#3ef26a" : "#ff5252";
    ctx.font = "800 24px 'Press Start 2P', 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 20;
    ctx.fillText(env.bricks.size === 0 ? "BOARD CLEARED" : "GAME OVER", 260, 250);
    ctx.shadowBlur = 0;
    ctx.font = "400 14px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#d7e0e7";
    ctx.fillText("score " + env.score + " · press R to restart", 260, 282);
  }
}

// ---------- UI ----------
const $ = (id) => document.getElementById(id);
const barsEl = $("bars");
const barRows = ACTIONS.map((name) => {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `<span class="name">${name}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="p">0%</span>`;
  barsEl.appendChild(row);
  return row;
});

function logEvent(html) {
  const li = document.createElement("li");
  li.innerHTML = html;
  const log = $("log");
  log.prepend(li);
  while (log.children.length > 30) log.removeChild(log.lastChild);
}

function getHi() { return parseInt(localStorage.getItem("kgb-hiscore") || "0", 10); }
function setHi(v) { localStorage.setItem("kgb-hiscore", String(v)); }

function updateStats(lastAction, probs) {
  const hi = Math.max(getHi(), env.score);
  if (env.score >= hi && env.score > 0) setHi(env.score);
  $("s-ep").textContent = env.episode;
  $("s-step").textContent = env.step;
  $("s-score").textContent = env.score;
  $("s-hi").textContent = hi;
  $("s-bricks").textContent = env.bricks.size;
  $("s-act").textContent = lastAction == null ? "—" : ACTIONS[lastAction];
  barRows.forEach((row, i) => {
    row.querySelector(".bar-fill").style.width = ((probs?.[i] ?? 0) * 100).toFixed(1) + "%";
    row.querySelector(".p").textContent = ((probs?.[i] ?? 0) * 100).toFixed(0) + "%";
    row.classList.toggle("chosen", i === lastAction);
  });
  $("banner").classList.toggle("show", env.over && env.bricks.size > 0);
}

let playing = false;
let timer = null;

function tick() {
  if (env.over) { setPlaying(false); return; }
  const x = observation();
  const logits = forward(x);
  const probs = softmax(Array.from(logits));
  const mode = $("mode").value;
  const a = mode === "greedy"
    ? probs.indexOf(Math.max(...probs))
    : sample(probs);
  const ev = stepEnv(a);
  draw($("ghost").checked);
  updateStats(a, probs);
  if (ev === "brick +1" || (ev && ev.startsWith("brick"))) {
    logEvent(`step <b>${env.step}</b>: ${ACTIONS[a]} → ${ev}`);
  } else if (ev === "game-over") {
    logEvent(`episode <b>${env.episode}</b> over at step ${env.step}, score <b>${env.score}</b>`);
    setPlaying(false);
  } else if (ev === "cleared") {
    logEvent(`episode <b>${env.episode}</b> CLEARED at step ${env.step}`);
    setPlaying(false);
  }
}

function setPlaying(p) {
  playing = p;
  $("btn-play").textContent = p ? "Pause" : "Play";
  if (timer) { clearInterval(timer); timer = null; }
  if (p) {
    const hz = parseInt($("speed").value, 10);
    timer = setInterval(tick, 1000 / hz);
  }
}

$("btn-play").addEventListener("click", () => {
  if (env.over) doReset();
  setPlaying(!playing);
});
$("btn-step").addEventListener("click", () => {
  if (env.over) doReset();
  setPlaying(false);
  tick();
});
$("btn-reset").addEventListener("click", doReset);
$("speed").addEventListener("input", () => {
  $("speed-out").textContent = $("speed").value + " steps/s";
  if (playing) setPlaying(true);
});
$("ghost").addEventListener("change", () => draw($("ghost").checked));
$("sound").addEventListener("change", (e) => { if (e.target.checked) ensureAudio(); });

function doReset() {
  setPlaying(false);
  const clearedOrOver = env.over;
  env.episode += clearedOrOver || env.step > 0 ? 1 : 0;
  resetEnv();
  draw($("ghost").checked);
  updateStats(null, null);
  logEvent(`episode <b>${env.episode}</b> started`);
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); $("btn-play").click(); }
  else if (e.key === "s" || e.key === "S") $("btn-step").click();
  else if (e.key === "r" || e.key === "R") $("btn-reset").click();
});

// ---------- boot ----------
function loadWeights(url) {
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(url + " HTTP " + r.status);
    return r.json();
  }).then((w) => {
    W = w;
    doReset();
    logEvent(`weights loaded: <b>${url}</b> — 17,253 params, inference in-browser`);
  }).catch((err) => {
    logEvent("failed to load " + url + ": " + err.message);
    $("btn-play").disabled = true;
    $("btn-step").disabled = true;
  });
}

$("wsel").addEventListener("change", (e) => loadWeights(e.target.value));
loadWeights($("wsel").value);
