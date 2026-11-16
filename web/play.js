// The playground: build a world, write a rule, watch the weeks play out, read the score.
const $ = id => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, text) => {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
};

// Settings shown up front. Everything else lives under "Every setting".
const MAIN = ["yes_rate", "majority_share", "open_rate", "fatigue", "preference_diversity", "intent_openness", "endogamy_share", "family_approval_share", "feature_coverage", "chemistry_share"];

const state = { meta: null, preset: "today", overrides: {}, result: null, week: 0, playing: false, token: 0, timer: 0 };

/* ---------- worker ---------- */
const worker = new Worker("worker.js", { type: "module" });
const pending = new Map();
let nextId = 1;
const call = (type, args) => new Promise(resolve => { const id = nextId++; pending.set(id, resolve); worker.postMessage({ id, type, args }); });
worker.onmessage = e => {
  const m = e.data;
  if (m.type === "progress") $("boot-text").textContent = m.text + " in your browser";
  else if (m.type === "fatal") { $("boot").classList.add("failed"); $("boot-text").textContent = "The engine could not start: " + m.message; }
  else if (m.type === "ready") start(m.meta);
  else if (m.type === "result") { const done = pending.get(m.id); pending.delete(m.id); done && done(m); }
};
worker.onerror = e => { $("boot").classList.add("failed"); $("boot-text").textContent = "The engine could not start: " + e.message; };

/* ---------- world builder ---------- */
const presetOf = name => state.meta.presets.find(p => p.name === name);
const valueOf = key => key in state.overrides ? state.overrides[key] : presetOf(state.preset).settings[key];
const fmt = (v, s) => s.type === "integer" ? String(Math.round(v)) : (s.step >= 1 ? String(v) : Number(v).toFixed(2));

function knob(s) {
  const row = document.createElement("label"); row.className = "knob"; row.dataset.key = s.key;
  const name = document.createElement("span"); name.className = "knob-name"; name.textContent = s.label;
  const out = document.createElement("output");
  const input = document.createElement("input");
  Object.assign(input, { type: "range", min: s.min, max: s.max, step: s.step });
  const reset = document.createElement("button"); reset.type = "button"; reset.className = "knob-reset"; reset.textContent = "reset"; reset.setAttribute("aria-label", "reset " + s.label);
  const sync = () => {
    const v = valueOf(s.key); input.value = v; out.textContent = fmt(v, s);
    row.classList.toggle("changed", s.key in state.overrides);
    input.style.setProperty("--fill", ((v - s.min) / (s.max - s.min) * 100).toFixed(1) + "%");
  };
  input.addEventListener("input", () => {
    const v = Number(input.value); const base = presetOf(state.preset).settings[s.key];
    if (Math.abs(v - base) < s.step / 2) delete state.overrides[s.key]; else state.overrides[s.key] = v;
    syncKnobs(s.key); queueRun();
  });
  reset.addEventListener("click", e => { e.preventDefault(); delete state.overrides[s.key]; syncKnobs(s.key); queueRun(); });
  const explain = () => { $("explain").innerHTML = ""; const b = document.createElement("b"); b.textContent = s.label + ". "; $("explain").append(b, s.meaning + " ", Object.assign(document.createElement("i"), { textContent: "Default: " + s.basis })); };
  row.addEventListener("pointerenter", explain); input.addEventListener("focus", explain);
  row.append(name, out, reset, input); row._sync = sync; sync();
  return row;
}
const syncKnobs = key => document.querySelectorAll(key ? `.knob[data-key="${CSS.escape(key)}"]` : ".knob").forEach(k => k._sync());

function buildWorld() {
  const { settings, presets } = state.meta;
  presets.forEach(p => {
    const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = p.name; b.dataset.preset = p.name;
    b.addEventListener("click", () => { state.preset = p.name; state.overrides = {}; syncPreset(); syncKnobs(); queueRun(0); });
    $("presets").append(b);
  });
  const byKey = Object.fromEntries(settings.map(s => [s.key, s]));
  MAIN.filter(k => byKey[k]).forEach(k => $("main-knobs").append(knob(byKey[k])));
  const groups = new Map();
  settings.filter(s => s.key !== "pool_size").forEach(s => { if (!groups.has(s.group)) groups.set(s.group, []); groups.get(s.group).push(s); });
  for (const [name, items] of groups) {
    const d = document.createElement("details"); d.className = "group";
    const sum = document.createElement("summary"); sum.textContent = name + " (" + items.length + ")"; d.append(sum);
    items.forEach(s => d.append(knob(s))); $("groups").append(d);
  }
  ["people", "rounds", "seed"].forEach(id => $(id).addEventListener("change", () => queueRun(0)));
  syncPreset();
}
function syncPreset() {
  document.querySelectorAll("#presets .chip").forEach(c => c.setAttribute("aria-pressed", String(c.dataset.preset === state.preset)));
  $("preset-desc").textContent = presetOf(state.preset).summary;
}

/* ---------- rule editor ---------- */
function buildRules() {
  state.meta.rules.forEach(r => {
    const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = r.name; b.title = r.description;
    b.addEventListener("click", () => { $("code").value = r.source; markRule(r.name); queueRun(0, true); });
    b.dataset.rule = r.name; $("rules").append(b);
  });
  const code = $("code");
  code.addEventListener("keydown", e => {
    if (e.key === "Tab") { e.preventDefault(); const s = code.selectionStart; code.setRangeText("    ", s, code.selectionEnd, "end"); }
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(true); }
  });
  code.addEventListener("input", () => markRule(null));
}
const markRule = name => document.querySelectorAll("#rules .chip").forEach(c => c.setAttribute("aria-pressed", String(c.dataset.rule === name)));

/* ---------- running ---------- */
const settingsArg = () => ({ preset: state.preset, ...state.overrides });
const sizeArgs = () => ({ seed: +$("seed").value || 0, people: Math.max(100, Math.min(1200, +$("people").value || 400)), rounds: Math.max(2, Math.min(16, +$("rounds").value || 8)) });
let runTimer = 0;
function queueRun(delay = 280, animate = false) { clearTimeout(runTimer); runTimer = setTimeout(() => run(animate), delay); }

async function run(animate) {
  const token = ++state.token; document.body.classList.add("busy");
  const m = await call("run", { settings: settingsArg(), rule: $("code").value, ...sizeArgs() });
  if (token !== state.token) return;
  document.body.classList.remove("busy");
  const data = m.data, err = $("err");
  if (data.error) {
    err.hidden = false;
    err.textContent = (data.error.where === "rule" ? "Your rule" : "The world") + (data.error.line ? ", line " + data.error.line : "") + ": " + data.error.message;
    return;
  }
  err.hidden = true; state.result = data; $("took").textContent = "ran in " + m.ms + " ms"; saveHash();
  $("stage-rule").textContent = data.rule; $("run").textContent = "Run the " + data.rounds + " weeks";
  drawScore(data); drawStage(data);
  if (animate && !reduced) playFrom(0); else showWeek(data.rounds - 1, false);
}

/* ---------- the stage ---------- */
const W = 720, H = 380, PADX = 46, PADY = 30;
const px = p => PADX + p.x * (W - 2 * PADX), py = p => PADY + p.y * (H - 2 * PADY);
function outcome(ev) {
  if (!(ev.a_answered && ev.b_answered)) return "exp";
  if (!(ev.a_yes && ev.b_yes)) return "one";
  return ev.stage >= 4 ? "hit" : "yes";
}
function drawStage(data) {
  const svg = $("stage"); svg.replaceChildren();
  const lines = el("g"), dots = el("g"); svg.append(lines, dots);
  const byId = Object.fromEntries(data.cast.map(p => [p.id, p]));
  data.events.forEach((ev, i) => {
    const a = byId[ev.a], b = byId[ev.b]; if (!a || !b) return;
    const ln = el("line", { x1: px(a), y1: py(a), x2: px(b), y2: py(b), class: "intro " + outcome(ev) });
    ln.dataset.round = ev.round; ln.dataset.i = i; ln.style.setProperty("--len", Math.hypot(px(b) - px(a), py(b) - py(a)).toFixed(1));
    lines.append(ln);
  });
  data.cast.forEach(p => {
    const g = el("g", { class: "person " + p.side, transform: `translate(${px(p).toFixed(1)},${py(p).toFixed(1)})` }); g.dataset.id = p.id;
    g.append(el("circle", { r: 9, class: "halo" }), el("circle", { r: 5.5, class: "body" }));
    dots.append(g);
  });
  $("sample-note").textContent = data.cast.length + " of the " + data.people + " people are drawn, with every introduction among them";
}
function showWeek(w, animate) {
  const data = state.result; if (!data) return;
  state.week = w; const svg = $("stage"), round = w + 1;
  const hit = new Set();
  svg.querySelectorAll(".intro").forEach(ln => {
    const r = +ln.dataset.round;
    ln.classList.toggle("off", r > round); ln.classList.toggle("past", r < round);
    ln.classList.toggle("draw", animate && r === round);
    if (animate && r === round) ln.style.animationDelay = (Math.random() * 500).toFixed(0) + "ms";
    if (r <= round && ln.classList.contains("hit")) { const ev = data.events[+ln.dataset.i]; hit.add(ev.a); hit.add(ev.b); }
  });
  svg.querySelectorAll(".person").forEach(g => g.classList.toggle("paired", hit.has(+g.dataset.id)));
  const pr = data.per_round[w];
  $("stage-n").textContent = "WEEK " + round; $("stage-count").textContent = "week " + round + " of " + data.rounds;
  const line = `${pr.introductions} introductions across the whole world. ${pr.answered} answered by both people, ${pr.both_said_yes} said yes both ways, ${pr.hit_it_off} want to meet again.` + (pr.gave_up ? ` ${pr.gave_up} ${pr.gave_up === 1 ? "person" : "people"} gave up.` : "");
  type(line, animate);
}
let typeToken = 0;
function type(str, animate) {
  const t = ++typeToken, out = $("stage-text");
  if (!animate) { out.textContent = str; return; }
  out.textContent = ""; let i = 0;
  (function step() { if (t !== typeToken) return; out.textContent = str.slice(0, i += 2); if (i < str.length) setTimeout(step, 14); })();
}
function setPlaying(on) {
  state.playing = on;
  $("pause").innerHTML = on ? '<svg viewBox="0 0 12 12"><path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z"/></svg>' : '<svg viewBox="0 0 12 12"><path d="M3 1.5v9l7-4.5z"/></svg>';
}
function playFrom(w) {
  clearTimeout(state.timer); setPlaying(true);
  (function tick(k) {
    showWeek(k, true);
    if (k + 1 < state.result.rounds) state.timer = setTimeout(() => state.playing && tick(k + 1), 2100); else setPlaying(false);
  })(w);
}
function stepTo(w) { clearTimeout(state.timer); setPlaying(false); const n = state.result.rounds; showWeek((w + n) % n, true); }

/* ---------- the scorecard ---------- */
function drawScore(data) {
  const svg = $("score"); svg.replaceChildren();
  const X0 = 250, BW = 320, ROW = 44;
  data.stages.forEach((s, i) => {
    const y = 14 + i * ROW;
    svg.append(el("text", { x: 20, y: y + 12, class: "t" }, s.stage + "  " + s.meaning));
    svg.append(el("rect", { x: X0, y: y + 2, width: BW, height: 10, class: "bar-track" }));
    const inf = el("rect", { x: X0, y: y + 16, width: 0, height: 10, fill: "url(#st-grey)", class: "bar" });
    const got = el("rect", { x: X0, y: y + 2, width: 0, height: 10, fill: "url(#st-hot)", class: "bar" });
    svg.append(inf, got);
    requestAnimationFrame(() => { got.setAttribute("width", (BW * s.share_of_best).toFixed(1)); inf.setAttribute("width", (BW * s.informed_share_of_best).toFixed(1)); });
    svg.append(el("text", { x: X0 + BW + 12, y: y + 12, class: "t" }, `${s.couples} of ${s.best_possible}`));
    svg.append(el("text", { x: X0 + BW + 12, y: y + 26, class: "s" }, `perfect reader ${s.informed_benchmark}`));
  });
  $("score-big").textContent = (data.score * 100).toFixed(1) + "% of the best possible";
  const it = data.introductions, po = data.people_outcomes;
  const facts = [["introductions made", it.made], ["answered by both", it.answered], ["never answered", it.expired], ["one-sided", it.one_sided],
    ["against a known dealbreaker", it.known_dealbreaker_violations], ["people who hit it off", po.hit_it_off], ["people who gave up", po.gave_up], ["never introduced", po.never_introduced]];
  $("facts").replaceChildren(...facts.map(([k, v]) => { const d = document.createElement("div"); d.append(Object.assign(document.createElement("dt"), { textContent: k }), Object.assign(document.createElement("dd"), { textContent: v })); return d; }));
}

/* ---------- leaderboard ---------- */
async function compare() {
  const btn = $("compare"); btn.disabled = true; btn.textContent = "Running every rule";
  const names = state.meta.rules.map(r => r.name), mine = $("code").value;
  const isDemo = state.meta.rules.some(r => r.source === mine);
  const m = await call("compare", { settings: settingsArg(), rules: isDemo ? names : [...names, mine], ...sizeArgs() });
  btn.disabled = false; btn.textContent = "Run every demo rule";
  const rows = (m.data.results || []).filter(r => !r.error).sort((a, b) => b.score - a.score);
  const top = Math.max(0.0001, ...rows.map(r => r.score));
  $("board").replaceChildren(...rows.map(r => {
    const li = document.createElement("li"); const yours = !names.includes(r.rule);
    li.className = yours ? "yours" : "";
    li.innerHTML = `<span class="b-name"></span><span class="b-bar"><i style="width:${(r.score / top * 100).toFixed(1)}%"></i></span><span class="b-num">${(r.score * 100).toFixed(1)}%</span>`;
    li.querySelector(".b-name").textContent = r.rule + (yours ? " (yours)" : "");
    return li;
  }));
  $("board-card").hidden = false;
}

/* ---------- sharing ---------- */
const enc = o => btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(o)))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const dec = s => JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), c => c.charCodeAt(0))));
function saveHash() { history.replaceState(null, "", "#" + enc({ p: state.preset, o: state.overrides, r: $("code").value, ...sizeArgs() })); }
function loadHash() {
  try {
    const o = dec(location.hash.slice(1)); if (!presetOf(o.p)) return false;
    state.preset = o.p; state.overrides = o.o || {}; $("code").value = o.r || "";
    $("seed").value = o.seed ?? 0; $("people").value = o.people ?? 400; $("rounds").value = o.rounds ?? 8; return true;
  } catch (e) { return false; }
}

/* ---------- start ---------- */
function start(meta) {
  state.meta = meta;
  const article = meta.rules.find(r => r.name.includes("article")) || meta.rules[0];
  $("code").value = article.source;
  const restored = location.hash.length > 1 && loadHash();
  buildWorld(); buildRules(); if (!restored) markRule(article.name); else markRule((meta.rules.find(r => r.source === $("code").value) || {}).name || null);
  syncPreset(); syncKnobs();
  $("boot").hidden = true; $("app").hidden = false;
  $("run").addEventListener("click", () => run(true)); $("compare").addEventListener("click", compare);
  $("prev").addEventListener("click", () => stepTo(state.week - 1)); $("next").addEventListener("click", () => stepTo(state.week + 1));
  $("pause").addEventListener("click", () => { if (state.playing) { clearTimeout(state.timer); setPlaying(false); } else playFrom(state.week + 1 < state.result.rounds ? state.week : 0); });
  $("share").addEventListener("click", async () => { saveHash(); try { await navigator.clipboard.writeText(location.href); $("share").textContent = "Link copied"; setTimeout(() => $("share").textContent = "Copy a link to this world and rule", 1600); } catch (e) {} });
  setPlaying(false); run(true);
}
