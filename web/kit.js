// Shared small things: element helpers, the quiet interface sounds, the theme switch, the loading canvas.
// The sounds and the theme switch follow the portfolio site: sounds on actions only, never on hover,
// and nothing before the first press.
export const $ = id => document.getElementById(id);
export const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
export const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export function h(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v; else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...kids.flat().filter(k => k != null && k !== false));
  return node;
}
export const plural = (n, one, many) => n + " " + (n === 1 ? one : many || one + "s");
const root = document.documentElement;

/* ---- Sound ---- */
export const sound = (function () {
  const AC = window.AudioContext || window.webkitAudioContext;
  let ctx = null, out = null, noise = null, on = true, unlocked = false;
  try { on = localStorage.getItem("sound") !== "off"; } catch (e) {}
  function init() {
    ctx = new AC({ latencyHint: "interactive" });
    out = ctx.createGain(); out.gain.value = 0.5; out.connect(ctx.destination);
    noise = ctx.createBuffer(1, ctx.sampleRate / 2, ctx.sampleRate);
    const d = noise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  function unlock() { if (!AC || unlocked) return; try { if (!ctx) init(); if (ctx.state !== "running") ctx.resume(); unlocked = true; } catch (e) {} }
  ["pointerdown", "keydown"].forEach(t => addEventListener(t, unlock, { capture: true, passive: true }));
  document.addEventListener("visibilitychange", () => { if (ctx) document.hidden ? ctx.suspend() : unlocked && ctx.resume(); });
  const rnd = (a, b) => a + Math.random() * (b - a);
  function env(g, t, a, peak, d) { g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + a); g.gain.exponentialRampToValueAtTime(0.0001, t + a + d); }
  function click(t, freq, dur, gain, q) {
    const s = ctx.createBufferSource(); s.buffer = noise; const f = ctx.createBiquadFilter(); f.type = "bandpass"; f.frequency.value = freq; f.Q.value = q || 1.5;
    const g = ctx.createGain(); env(g, t, 0.002, gain, dur); s.connect(f).connect(g).connect(out); s.start(t, rnd(0, 0.3)); s.stop(t + dur + 0.05);
  }
  function tone(t, f0, f1, dur, gain, type) {
    const o = ctx.createOscillator(); o.type = type || "sine"; o.frequency.setValueAtTime(f0, t); if (f1) o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const g = ctx.createGain(); env(g, t, 0.004, gain, dur); o.connect(g).connect(out); o.start(t); o.stop(t + dur + 0.05);
  }
  const voices = {
    tick: t => click(t, rnd(3800, 4600), 0.012, 0.05),
    open: t => { tone(t, 420, 640, 0.07, 0.06); click(t, 2600, 0.01, 0.025); },
    close: t => { tone(t, 600, 400, 0.06, 0.05); click(t, 2200, 0.01, 0.025); },
    toggle: t => tone(t, 520, 780, 0.07, 0.06),
    step: t => click(t, 3000, 0.014, 0.04),
    start: t => tone(t, 330, 495, 0.09, 0.05),
    bad: t => tone(t, 300, 220, 0.12, 0.05),
    land: t => [523.25, 783.99].forEach((f, i) => tone(t + i * 0.07, f, 0, 0.3, 0.04)),
    success: t => [659.25, 987.77].forEach((f, i) => tone(t + i * 0.085, f, 0, 0.4, 0.05)),
  };
  const api = {
    play(name) { if (!on || !unlocked || !ctx || ctx.state !== "running" || !voices[name]) return; try { voices[name](ctx.currentTime + 0.001); } catch (e) {} },
    get on() { return on; },
    set(v) { on = !!v; root.classList.toggle("sound-off", !on); try { localStorage.setItem("sound", on ? "on" : "off"); } catch (e) {} sync(); },
  };
  function sync() { document.querySelectorAll("[data-sound-toggle]").forEach(b => { b.setAttribute("aria-pressed", String(on)); b.textContent = on ? "Sound on" : "Sound off"; }); }
  root.classList.toggle("sound-off", !on); sync();
  document.addEventListener("click", e => { const b = e.target.closest("[data-sound-toggle]"); if (b) { api.set(!on); api.play("toggle"); } });
  return api;
})();

/* ---- Theme: follows the system until the switch is used. The black band does not change. ---- */
const isDark = () => getComputedStyle(root).colorScheme.indexOf("dark") >= 0;
const themeListeners = [];
export const onTheme = fn => themeListeners.push(fn);
document.querySelectorAll("[data-theme-toggle]").forEach(btn => {
  btn.setAttribute("aria-pressed", String(isDark()));
  btn.addEventListener("click", () => {
    const dark = !isDark();
    const apply = () => { root.dataset.theme = dark ? "dark" : "light"; try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {} btn.setAttribute("aria-pressed", String(dark)); themeListeners.forEach(fn => fn(dark)); };
    sound.play("toggle");
    if (!document.startViewTransition || reduced) return apply();
    const r = btn.getBoundingClientRect();
    root.style.setProperty("--vx", r.left + r.width / 2 + "px"); root.style.setProperty("--vy", r.top + r.height / 2 + "px");
    root.classList.add("vt"); document.startViewTransition(apply).finished.finally(() => root.classList.remove("vt"));
  });
});

/* ---- The loading canvas: two rows of people, drawn as square cells in the portfolio hero's greys.
   Each loading step that finishes pairs more of them up, in orange. Still under reduced motion. ---- */
export function bootCanvas(cv) {
  const c = cv.getContext("2d"); let W = 0, H = 0, cols = 0, rows = 0, raf = 0, done = 0, t0 = performance.now(), stopped = false;
  const CELL = 4, GREYS = ["#262626", "#545454", "#9a9a9a"];
  let people = [];
  function size() {
    W = cv.clientWidth; H = cv.clientHeight; if (!W || !H) return;
    cols = Math.ceil(W / CELL); rows = Math.ceil(H / CELL); cv.width = cols; cv.height = rows;
    const n = Math.max(10, Math.floor(cols / 7)); let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    people = Array.from({ length: n }, (_, i) => ({ x: Math.round((i + 0.5) * cols / n), phase: rnd() * 6.28, speed: 0.5 + rnd(), order: rnd() }));
    draw(performance.now());
  }
  function draw(now) {
    const t = reduced ? 0 : (now - t0) / 1000;
    c.clearRect(0, 0, cols, rows);
    const mid = rows / 2, gap = Math.max(5, rows * 0.2), amp = Math.max(2, rows * 0.07);
    for (const p of people) {
      const paired = p.order < done;
      const wob = paired ? 0 : Math.sin(t * p.speed + p.phase) * amp;
      const ya = Math.round(mid - gap + wob), yb = Math.round(mid + gap - Math.sin(t * p.speed * 0.8 + p.phase * 1.7) * (paired ? 0 : amp));
      if (paired) { c.fillStyle = "#ff5904"; for (let y = ya; y <= yb; y += 2) c.fillRect(p.x, y, 1, 1); c.fillRect(p.x - 1, ya - 1, 3, 3); c.fillRect(p.x - 1, yb - 1, 3, 3); }
      else { c.fillStyle = GREYS[2]; c.fillRect(p.x - 1, ya - 1, 3, 3); c.fillStyle = GREYS[1]; c.fillRect(p.x - 1, yb - 1, 3, 3); }
    }
    if (!reduced && !stopped) raf = requestAnimationFrame(draw);
  }
  const ro = new ResizeObserver(size); ro.observe(cv); size();
  return {
    progress(share) { done = share; if (reduced) draw(0); },
    stop() { stopped = true; cancelAnimationFrame(raf); ro.disconnect(); },
  };
}
