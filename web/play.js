// The dating environment's page: choose a world, run or edit a strategy, read the result.
import { $, h, esc, plural, reduced, sound, bootCanvas } from "./kit.js";
import { mountGraph, linkKind, LINK_KINDS } from "./graph.js";

const REPO = "https://github.com/Mohit-Dhawan98/dating-env";
const DEFAULT_SIZE = { seed: 0, people: 400, rounds: 8 };
const STORE = "dating-env.strategies.v1";

/* What each demo strategy does, in plain words, written from the rule sources in rules/. */
const ABOUT = {
  article: {
    short: "the article's rule",
    what: "Scores every pair on what they wrote, their photos and their stated preferences, weighted by what each person seems to care about. Then it introduces only pairs who are in each other's top ten.",
    caveat: "Rebuilt from the article's description. It is not the production implementation: how signals were extracted from conversations and photos, the learned weights and the production tuning are not reproduced.",
  },
  best_for_me: { what: "Gives each person their own favourite on a combined score. It never asks whether the favourite would pick them back, and it does not check dealbreakers." },
  dealbreakers_then_greedy: { what: "First rules out pairs who break a dealbreaker the app knows about. Then gives each person their favourite. Still one-sided: nobody asks what the favourite thinks." },
  most_similar: { what: "Introduces each person to whoever writes most like them. It ignores photos, stated preferences and dealbreakers." },
  random_introductions: { what: "Introduces anyone to anyone they have not met. This is the floor. A rule that cannot beat it is not using what it sees." },
  spread_attention: { what: "The article's rule, approximated, with one change: anyone who said no twice lately, or let two introductions in a row go unanswered, gets a rest. Introductions go to people who are responding." },
  stable_matching: { what: "The textbook method, from Gale and Shapley. It keeps trading partners until no two people would both prefer each other, as far as the scores can tell." },
};
const ORDER = ["article", "spread_attention", "stable_matching", "dealbreakers_then_greedy", "best_for_me", "most_similar", "random_introductions"];
const TEMPLATE = `"""Describe your rule in one line."""

from wavelength.play import rule


@rule("my rule")
def my_rule(v):
    """Half words, a third voice, the rest stated preferences. Pairs in each other's top ten."""
    s = 0.5 * v.text + 0.3 * v.voice + 0.2 * v.features()
    return v.pair_mutual(s, top=10)
`;
/* The guide beside the editor. Checked against src/wavelength/view.py. */
const GUIDE = [
  ["How alike two people look", [
    ["v.text", "How alike they are in what they write."],
    ["v.photo", "How alike their photos say their lives are. Neutral when either hides their photos."],
    ["v.voice", "How alike they are in how they talk."],
    ["v.features()", "Agreement on stated preferences, by what the app has learned each person cares about."],
    ['v.features("global")', "The same, with one set of weights for everyone."],
    ["v.family", "How alike their family expectations are."],
    ["v.distance", "How far apart they live."],
    ["v.age", "Each person's age. One number per person."],
    ["v.same_community", "True where two people share a community."],
  ]],
  ["Who may be introduced", [
    ["v.ok", "Pairs you may introduce that break no dealbreaker the app knows about."],
    ["v.open", "Every pair you may introduce: opposite sides, both still here, not introduced before."],
    ["v.available", "Who is still in the pool."],
    ["v.larger_side", "Which side each person is on."],
  ]],
  ["What has happened so far", [
    ["v.round", "This week's number, from 0. v.rounds is how many weeks there are."],
    ["v.history.said_yes", "Per pair: did person i say yes to j. Also v.history.introduced and v.history.answered."],
    ["v.history.recent_declines", "Per person: how often they said no in the last two weeks. Also introductions, unanswered, yeses, declines."],
    ["v.history.idle_rounds", "Per person: introductions in a row they let go unanswered."],
    ["v.random()", "A table of random numbers, the same for every rule."],
  ]],
  ["Turning a score into pairs", [
    ["v.pair_mutual(s, top=10)", "Pairs people who are each in the other's top ten, most promising first."],
    ["v.pair_best(s)", "Gives each person, in turn, their best-scored candidate who is still free."],
    ["v.pair_stable(s)", "A stable matching on the scores."],
    ["v.rank(s)", "Each person's ranking of their candidates, to build your own pairing."],
    ["[(a, b), ...]", "Or return your own list of pairs. Each must cross the two sides, and nobody may appear twice."],
  ]],
];
/* One line per world, for the chooser. The engine's fuller description is in "Tune this world". */
const WORLD_LINE = {
  today: "Today's mix of people on an introductions app.",
  casual: "Most want something casual. Few dealbreakers.",
  serious: "Marriage-minded, firm dealbreakers, high intent.",
  swipe: "Today's people on a swipe feed. Quick judgements, few yeses.",
  dystopia: "A lopsided market, thin profiles, attention piled onto a few.",
};
const SEEDS = 5;            // each strategy is also played with this many seeds, to show how much is luck
const SEEDS_MAX_PEOPLE = 600;

const state = {
  meta: null, preset: "today", overrides: {}, size: { ...DEFAULT_SIZE },
  builtins: [], mine: [], activeId: "article",
  cache: new Map(),          // world key -> Map(strategy key -> scorecard or {error})
  seeds: new Map(),          // world key without the seed -> Map(strategy key -> Map(seed -> {score, couples, possible, bench} or {error}))
  shown: null,               // { card, strategy }
  running: null, editing: null, graph: null, week: 8, playing: 0, tab: "graph",
};

/* ---------- worker ---------- */
const worker = new Worker("worker.js", { type: "module" });
const pending = new Map();
let nextId = 1;
const send = (type, args) => new Promise(resolve => { const id = nextId++; pending.set(id, resolve); worker.postMessage({ id, type, args }); });
// What the visitor asked for goes first. Background jobs (the other seeds) run one at a time when nothing is waiting.
let foreground = 0, backgroundBusy = false; const background = [];
const call = async (type, args) => { foreground++; try { return await send(type, args); } finally { foreground--; pump(); } };
async function pump() {
  if (foreground || backgroundBusy || !background.length) return;
  backgroundBusy = true; try { await background.shift()(); } catch (e) {} backgroundBusy = false; pump();
}
const boot = bootCanvas($("boot-cv"));
function bootStep(text) {
  const steps = [...$("boot-steps").children]; const at = steps.findIndex(li => li.dataset.step === text);
  steps.forEach((li, i) => { li.classList.toggle("done", i < at); li.classList.toggle("doing", i === at); });
  boot.progress(at / steps.length);
  if (text !== "ready") $("boot-head").textContent = text + " in your browser";
}
function bootFailed(message) {
  boot.stop(); $("boot").classList.add("failed");
  $("boot-head").textContent = "The engine could not start";
  $("boot-note").textContent = message + " Reload the page to try again. The page needs a recent browser and a connection for its first visit.";
}
worker.onmessage = e => {
  const m = e.data;
  if (m.type === "progress") bootStep(m.text);
  else if (m.type === "fatal") bootFailed(m.message);
  else if (m.type === "ready") { bootStep("ready"); start(m.meta); }
  else if (m.type === "result") { const done = pending.get(m.id); pending.delete(m.id); done && done(m); }
};
worker.onerror = e => bootFailed(e.message || "The engine's files did not load.");
bootStep("Starting Python");

/* ---------- worlds ---------- */
const presetOf = name => state.meta.presets.find(p => p.name === name);
const title = s => s.charAt(0).toUpperCase() + s.slice(1);
const worldArgs = () => ({ settings: { preset: state.preset, ...state.overrides }, ...state.size });
const worldKey = () => JSON.stringify(worldArgs());
const worldCache = () => { const k = worldKey(); if (!state.cache.has(k)) { if (state.cache.size > 24) state.cache.clear(); state.cache.set(k, new Map()); } return state.cache.get(k); };
const nChanged = () => Object.keys(state.overrides).length;
const baseKey = () => JSON.stringify({ ...worldArgs(), seed: null });
const seedList = () => Array.from({ length: SEEDS }, (_, i) => state.size.seed + i);
function seedStore(s) {
  const k = baseKey(); if (!state.seeds.has(k)) { if (state.seeds.size > 24) state.seeds.clear(); state.seeds.set(k, new Map()); }
  const world = state.seeds.get(k); if (!world.has(keyOf(s))) world.set(keyOf(s), new Map());
  return world.get(keyOf(s));
}
const briefOf = r => r.error ? { error: r.error } : { score: r.score, couples: r.stages[3].couples, possible: r.stages[3].best_possible, bench: r.stages[3].informed_benchmark, benchShare: r.stages[3].informed_share_of_best };
function seedStats(s) {
  const got = [...seedStore(s).values()].filter(x => !x.error); if (!got.length) return null;
  const shares = got.map(x => x.score);
  return { n: got.length, mean: shares.reduce((a, b) => a + b, 0) / got.length, min: Math.min(...shares), max: Math.max(...shares) };
}
/* Play every strategy with the other seeds, quietly, after the visitor's own run has been shown. */
function scheduleBackground() {
  background.length = 0;
  if (state.size.people > SEEDS_MAX_PEOPLE) return;
  const key = baseKey(), args = worldArgs(), first = activeStrategy();
  const order = [first, ...allStrategies().filter(s => s !== first)];
  for (const seed of seedList()) for (const s of order) {
    if (seedStore(s).has(seed)) continue;
    background.push(async () => {
      if (baseKey() !== key || !findStrategy(s.id) || seedStore(s).has(seed)) return;
      const store = seedStore(s), m = await send("brief", { ...args, seed, rule: ruleArg(s) });
      store.set(seed, briefOf(m.data)); if (baseKey() === key) refreshScores();
    });
  }
  pump();
}
const seedsLeft = () => state.size.people > SEEDS_MAX_PEOPLE ? 0 : allStrategies().reduce((a, s) => a + seedList().filter(seed => !seedStore(s).has(seed)).length, 0);

function renderWorlds() {
  $("world-list").replaceChildren(...state.meta.presets.map(p => {
    const on = p.name === state.preset;
    return h("button", { type: "button", class: "world", role: "radio", "aria-checked": String(on), tabindex: on ? "0" : "-1", "data-world": p.name, onclick: () => chooseWorld(p.name) },
      h("b", { text: title(p.name) }), h("span", { text: WORLD_LINE[p.name] || p.summary }));
  }));
  const s = state.size, bits = [s.people + " people", s.rounds + " weeks"];
  if (s.seed) bits.push("seed " + s.seed);
  if (nChanged()) bits.push(plural(nChanged(), "setting") + " changed");
  $("world-size").textContent = bits.join(", ");
  $("world-size").classList.toggle("changed", nChanged() > 0);
  $("lead-weeks").textContent = WORDS[s.rounds] || s.rounds;
}
$("world-list").addEventListener("keydown", e => {
  const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!d) return;
  e.preventDefault(); const names = state.meta.presets.map(p => p.name);
  const next = names[(names.indexOf(state.preset) + d + names.length) % names.length];
  chooseWorld(next); $("world-list").querySelector('[aria-checked="true"]').focus();
});
function chooseWorld(name) {
  if (name === state.preset && !nChanged()) return;
  state.preset = name; state.overrides = {}; sound.play("toggle");
  renderWorlds(); renderStrategies(); runStrategy(activeStrategy());
}

/* ---------- the "adjust this world" sheet ---------- */
let draft = {};
const isShare = s => s.type === "number" && s.min >= 0 && s.max <= 1;
const fmt = (v, s) => s.type === "integer" ? String(Math.round(v)) : isShare(s) ? Math.round(v * 100) + "%" : Number(v).toFixed(2);
function knob(s) {
  const base = () => presetOf(state.preset).settings[s.key];
  const value = () => s.key in draft ? draft[s.key] : base();
  const input = h("input", { type: "range", min: s.min, max: s.max, step: s.step, "aria-label": s.label });
  const out = h("output"), reset = h("button", { type: "button", class: "knob-reset", text: "reset", "aria-label": "Reset " + s.label });
  const why = h("p", { class: "knob-why" }, s.meaning + " ", h("i", { text: "Starting value: " + s.basis }));
  const row = h("div", { class: "knob", "data-key": s.key }, h("span", { class: "knob-name", text: s.label }), out, reset, input, why);
  const touch = () => { document.querySelectorAll(".knob.touched").forEach(k => k !== row && k.classList.remove("touched")); row.classList.add("touched"); };
  row._sync = () => { const v = value(); input.value = v; out.textContent = fmt(v, s); row.classList.toggle("changed", s.key in draft); input.style.setProperty("--fill", ((v - s.min) / (s.max - s.min) * 100).toFixed(1) + "%"); };
  input.addEventListener("input", () => { const v = Number(input.value); if (Math.abs(v - base()) < s.step / 2) delete draft[s.key]; else draft[s.key] = v; row._sync(); touch(); });
  reset.addEventListener("click", () => { delete draft[s.key]; row._sync(); });
  row.addEventListener("pointerdown", touch); input.addEventListener("focus", touch);
  return row;
}
function buildSheet() {
  const groups = new Map();
  state.meta.settings.filter(s => s.key !== "pool_size").forEach(s => { if (!groups.has(s.group)) groups.set(s.group, []); groups.get(s.group).push(s); });
  let first = true;
  for (const [name, items] of groups) {
    const d = h("details", { class: "group", open: first }, h("summary", {}, name, h("small", { text: plural(items.length, "setting") })), ...items.map(knob));
    first = false; $("groups").append(d);
  }
}
function openSheet() {
  if (!$("groups").children.length) buildSheet();
  draft = { ...state.overrides };
  ["people", "rounds", "seed"].forEach(k => $(k).value = state.size[k]);
  $("sheet-sub").textContent = `Starting from ${title(state.preset)}. ${presetOf(state.preset).description}`;
  document.querySelectorAll(".knob").forEach(k => k._sync());
  $("sheet").showModal(); sound.play("open");
}
function applySheet() {
  const clamp = (id, lo, hi, d) => Math.max(lo, Math.min(hi, Math.round(+$(id).value || d)));
  state.overrides = { ...draft };
  state.size = { seed: clamp("seed", 0, 99999, 0), people: clamp("people", 100, 1200, 400), rounds: clamp("rounds", 2, 16, 8) };
  $("sheet").close(); renderWorlds(); renderStrategies(); runStrategy(activeStrategy());
}
const WORDS = ["", "", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
$("adjust").addEventListener("click", openSheet);
$("sheet-close").addEventListener("click", () => { $("sheet").close(); sound.play("close"); });
$("sheet").addEventListener("click", e => { if (e.target === $("sheet")) $("sheet").close(); });
$("sheet-apply").addEventListener("click", applySheet);
$("sheet-reset").addEventListener("click", () => { draft = {}; ["people", "rounds", "seed"].forEach(k => $(k).value = DEFAULT_SIZE[k]); document.querySelectorAll(".knob").forEach(k => k._sync()); sound.play("tick"); });

/* ---------- strategies ---------- */
const allStrategies = () => [...state.builtins, ...state.mine];
const findStrategy = id => allStrategies().find(s => s.id === id);
const activeStrategy = () => findStrategy(state.activeId) || state.builtins[0];
const keyOf = s => s.mine ? s.id + "@" + s.rev : s.id;
const ruleArg = s => s.mine ? s.source : s.name;
const docOf = source => { const m = /def\s+\w+\s*\([^)]*\)\s*:\s*\n\s*(?:"""([\s\S]*?)"""|'''([\s\S]*?)''')/.exec(source); return m ? (m[1] || m[2]).trim().replace(/\s+/g, " ") : ""; };
const NAME_RE = /@rule\(\s*(["'])((?:\\.|(?!\1).)*)\1\s*\)/;
function saveMine() { try { localStorage.setItem(STORE, JSON.stringify(state.mine.map(({ id, name, source, from }) => ({ id, name, source, from })))); } catch (e) {} }
function loadMine() {
  try { const got = JSON.parse(localStorage.getItem(STORE) || "[]"); return got.filter(s => s && typeof s.source === "string").map(s => ({ ...s, mine: true, rev: 1 })); } catch (e) { return []; }
}

function scoreLine(s) {
  const full = worldCache().get(keyOf(s)), got = full ? briefOf(full) : seedStore(s).get(state.size.seed);
  if (state.running === s.id) return h("span", { class: "strat-score running", text: "Running the " + state.size.rounds + " weeks" });
  if (!got) return h("span", { class: "strat-score none", text: backgroundBusy || background.length ? "Scoring it in this world" : "Not run in this world yet" });
  if (got.error) return h("span", { class: "strat-score bad", text: got.error.line ? "Stopped at line " + got.error.line : "Stopped with an error" });
  const st = seedStats(s);
  return h("span", { class: "strat-score" },
    h("span", { class: "meter", "aria-hidden": "true" }, h("i", { style: `width:${(got.score * 100).toFixed(1)}%` })),
    h("span", {}, h("b", { text: got.couples }), ` of ${got.possible} couples`,
      st && st.n > 1 && h("small", { text: ` ${pct(st.min)} to ${pct(st.max)} over ${st.n} seeds` })));
}
function refreshScores() {
  document.querySelectorAll(".strat").forEach(li => { const s = findStrategy(li.dataset.id), old = li.querySelector(".strat-score"); if (s && old) old.replaceWith(scoreLine(s)); });
  if (state.shown) { renderSeedNote(); if (state.tab === "all") renderAll(); }
}
function strategyRow(s) {
  const about = ABOUT[s.id];
  const what = about ? about.what : (docOf(s.source) || "Your own rule.");
  const shown = state.shown && state.shown.strategy.id === s.id && state.shown.key === worldKey();
  const busy = state.running != null;
  return h("li", { class: "strat" + (shown ? " shown" : "") + (state.running === s.id ? " running" : "") + (s.mine ? " mine" : ""), "data-id": s.id },
    h("div", { class: "strat-top" }, h("h3", { text: s.name }), s.mine && h("span", { class: "tag", text: "yours" })),
    h("p", { class: "strat-what", text: what }),
    about && about.caveat && h("p", { class: "strat-caveat", text: about.caveat }),
    s.mine && s.from && h("p", { class: "strat-caveat", text: "Started from " + s.from + "." }),
    h("div", { class: "strat-foot" }, scoreLine(s),
      h("button", { type: "button", class: "btn small solid", "data-act": "run", disabled: busy, "aria-label": `Run ${s.name}` }, h("span", { class: "btn-label", text: state.running === s.id ? "Running" : "Run" })),
      h("button", { type: "button", class: "btn small", "data-act": "edit", "aria-label": `Edit ${s.name}`, title: s.mine ? "" : "Opens a copy. The original stays as it is." , text: "Edit" })));
}
function renderStrategies() {
  const byId = Object.fromEntries(state.builtins.map(s => [s.id, s]));
  const ordered = [...ORDER.map(id => byId[id]).filter(Boolean), ...state.builtins.filter(s => !ORDER.includes(s.id))];
  $("strat-list").replaceChildren(...ordered.map(strategyRow));
  $("mine-list").replaceChildren(...state.mine.map(strategyRow));
  $("mine-empty").hidden = state.mine.length > 0;
  if (state.tab === "all" && state.shown) renderAll();
}
$("strategies").addEventListener("click", e => {
  const btn = e.target.closest("[data-act]"); if (!btn) return;
  const s = findStrategy(btn.closest("[data-id]").dataset.id); if (!s) return;
  if (btn.dataset.act === "run") runStrategy(s); else openEditor(s);
});
$("new-strategy").addEventListener("click", () => openEditor(null));
$("run-all").addEventListener("click", async () => {
  if (!state.shown) await runStrategy(activeStrategy());
  if (!state.shown) return;
  selectTab("all"); sound.play("tick");
  if (matchMedia("(max-width: 1099px)").matches) $("result").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
});

/* ---------- running ---------- */
const wait = ms => new Promise(r => setTimeout(r, ms));
function setRunning(what, text) {
  state.running = what; document.body.classList.toggle("running", what != null);
  $("result").setAttribute("aria-busy", String(what != null));
  if (text) $("live").textContent = text;
  renderStrategies();
  $("ed-run").disabled = what != null; $("ed-run").querySelector(".btn-label").textContent = what != null && state.editing ? "Running" : "Save and run";
}
async function runStrategy(s, { quiet = false } = {}) {
  if (state.running != null || !s) return null;
  state.activeId = s.id;
  const cached = worldCache().get(keyOf(s));
  setRunning(s.id, `Running ${s.name} in ${title(state.preset)}.`);
  if (!quiet) sound.play("start");
  syncJump();
  const t0 = performance.now();
  const m = cached ? { data: cached, ms: cached._ms } : await call("run", { ...worldArgs(), rule: ruleArg(s) });
  if (!reduced) await wait(Math.max(0, 420 - (performance.now() - t0)));   // long enough to see that something ran
  const data = m.data; data._ms = m.ms;
  if (!(data.error && data.error.where !== "rule")) { worldCache().set(keyOf(s), data); seedStore(s).set(state.size.seed, briefOf(data)); }
  setRunning(null);
  if (data.error) {
    sound.play("bad");
    $("live").textContent = `${s.name} stopped: ${data.error.message}`;
    if (data.error.where !== "rule") showWorldError(data.error); else if (!state.editing || state.editing.id !== s.id) showRuleError(s, data.error); else markStale(s, data.error);
    syncJump(); scheduleBackground();
    return data;
  }
  showResult(data, s);
  const last = data.stages[data.stages.length - 1];
  $("live").textContent = `Finished. ${s.name} found ${last.couples} of ${last.best_possible} couples.`;
  if (!quiet) sound.play("land");
  syncJump(); scheduleBackground();
  return data;
}
/* ---------- the result ---------- */
function showWorldError(error) {
  state.shown = null; $("result-body").hidden = true; $("result-empty").hidden = false; $("result-empty").classList.add("bad");
  $("result-empty").replaceChildren(h("p", { class: "re-head", text: "This world cannot be made." }), h("p", { text: error.message }), h("p", {}, "Open ", h("b", { text: "Tune this world" }), " and change the setting it names, or go back to the preset."));
}
function showRuleError(s, error) {
  state.shown = null; $("result-body").hidden = true; $("result-empty").hidden = false; $("result-empty").classList.add("bad");
  $("result-empty").replaceChildren(h("p", { class: "re-head", text: `${s.name} stopped${error.line ? " at line " + error.line : ""}.` }), h("p", { class: "mono", text: error.message }),
    h("p", {}, "Press ", h("b", { text: "Edit" }), " on it to see the line and fix it."));
  renderStrategies();
}
function markStale(s, error) {
  if (!state.shown) return;
  $("stale").hidden = false; $("result").classList.add("is-stale");
  $("stale").textContent = `${s.name} stopped${error.line ? " at line " + error.line : ""}. Below is still ${state.shown.strategy.id === s.id ? "its last run that worked" : "the last run that worked, " + state.shown.strategy.name}.`;
}
function showResult(card, s) {
  $("stale").hidden = true; $("result").classList.remove("is-stale");
  const first = $("result-body").hidden;
  state.shown = { card, strategy: s, key: worldKey() };
  $("result-empty").hidden = true; $("result-empty").classList.remove("bad"); $("result-body").hidden = false;
  const last = card.stages[card.stages.length - 1];
  const where = [title(state.preset) + (nChanged() ? ", adjusted" : ""), card.people + " people", card.rounds + " weeks"];
  $("verdict-what").replaceChildren(h("b", { text: s.name }), " in " + where.join(", "));
  const found = h("b", { class: "num hot", text: last.couples });
  $("result-h").replaceChildren("Of the ", h("b", { class: "num", text: last.best_possible }), ` ${last.best_possible === 1 ? "couple" : "couples"} who could have hit it off, this rule found `, found, ".");
  if (!reduced && last.couples > 0) countUp(found, last.couples);
  const share = n => last.best_possible ? (n / last.best_possible * 100).toFixed(2) + "%" : "0%";
  $("scorebar").innerHTML = last.best_possible ? `
    <div class="sb-track" role="img" aria-label="${last.couples} of ${last.best_possible} couples found. A perfect reader of profiles finds ${last.informed_benchmark}.">
      <i class="sb-hatch" style="left:${share(last.informed_benchmark)}"></i><i class="sb-fill" style="width:${share(last.couples)}"></i><i class="sb-tick" style="left:${share(last.informed_benchmark)}"></i></div>
    <p class="sb-key"><span><i class="sw-bar rule"></i><span><b class="hot">${pct(card.score)}</b> of what was possible<span id="seed-note"></span></span></span>
      <span><i class="sw-tick"></i>a perfect reader of profiles finds ${last.informed_benchmark}: a reference, not a strategy</span>
      <span><i class="sw-bar lost"></i>past that line takes chemistry and luck, which no rule can see</span></p>`
    : `<p class="sb-key">The answer key has no couple who would hit it off in this world, so there is nothing for a rule to find.</p>`;
  renderSeedNote();
  const it = card.introductions, steps = it.funnel;
  $("tiles").replaceChildren(...[[steps[0].count, "introductions made"], [steps[1].count, "answered by both people"], [steps[2].count, "a yes from both"]]
    .map(([n, label]) => h("div", {}, h("dd", { text: n.toLocaleString("en") }), h("dt", { text: label }))));
  renderStrategies();
  // the graph
  stopPlaying();
  state.week = card.rounds; $("g-week").max = card.rounds; $("g-week").value = card.rounds; $("g-find").max = card.people - 1; $("g-find").value = "";
  renderCard(null);
  state.graphFor = null;
  if (state.tab === "graph") (first ? document.fonts.ready.then(() => requestAnimationFrame(mount)) : mount()); else if (state.graph) { state.graph.destroy(); state.graph = null; }
  renderWeek(false); renderFunnel(card); renderAll();
  $("result").classList.remove("landed"); void $("result").offsetWidth; $("result").classList.add("landed");
  if (first) $("result").classList.add("first");
}
function renderSeedNote() {
  const el = $("seed-note"); if (!el || !state.shown) return;
  const st = seedStats(state.shown.strategy);
  el.textContent = st && st.n > 1 ? `. Over ${st.n} seeds it ranges from ${pct(st.min)} to ${pct(st.max)}` : "";
}
const pct = share => (share * 100).toFixed(share > 0 && share < 0.1 ? 1 : 0) + "%";
function countUp(node, to) {
  const t0 = performance.now(), dur = 520;
  (function step(now) { const k = Math.min(1, (now - t0) / dur); node.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))); if (k < 1 && node.isConnected) requestAnimationFrame(step); })(t0);
}
function mount() {
  if (!state.shown) return;
  const { card } = state.shown; if (state.graphFor === card) return;
  if (state.graph) state.graph.destroy();
  state.graph = mountGraph($("force"), card, { onFocus: id => { renderCard(id); if (id != null) sound.play("step"); } });
  state.graph.setUnanswered($("g-unanswered").getAttribute("aria-pressed") === "true");
  state.graph.setWeek(state.week, false); state.graphFor = card;
}

/* tabs */
const TABS = ["graph", "funnel", "all"];
function selectTab(k, focus) {
  state.tab = k;
  TABS.forEach(t => { const on = t === k; $("t-" + t).setAttribute("aria-selected", String(on)); $("t-" + t).tabIndex = on ? 0 : -1; $("p-" + t).hidden = !on; });
  if (k === "graph" && state.shown) requestAnimationFrame(mount);
  if (k === "all" && state.shown) renderAll();
  if (k !== "graph") stopPlaying();
  if (focus) $("t-" + k).focus();
}
$("tabs").addEventListener("click", e => { const b = e.target.closest("[role=tab]"); if (b) { selectTab(b.id.slice(2)); sound.play("tick"); } });
$("tabs").addEventListener("keydown", e => {
  const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key]; if (!d) return;
  e.preventDefault(); selectTab(TABS[(TABS.indexOf(state.tab) + d + TABS.length) % TABS.length], true);
});

/* the week control */
function renderWeek(animate) {
  const card = state.shown.card, w = state.week, r = card.per_round[w - 1];
  $("g-week").value = w; $("g-week").style.setProperty("--fill", (card.rounds > 1 ? (w - 1) / (card.rounds - 1) * 100 : 100) + "%");
  $("g-week-label").textContent = `Week ${w} of ${card.rounds}`;
  const upTo = card.per_round.slice(0, w), sum = k => upTo.reduce((a, x) => a + x[k], 0);
  $("weekline").replaceChildren(h("b", { text: `Week ${w}: ` }),
    `${plural(r.introductions, "introduction")}, ${r.answered} answered by both, ${r.both_said_yes} a yes from both, ${plural(r.hit_it_off, "new couple")}, ${sum("hit_it_off")} so far.`);
  if (state.graph) state.graph.setWeek(w, animate && !reduced);
  if (state.graph && state.graph.focused != null) renderCard(state.graph.focused);
}
function stopPlaying() {
  clearInterval(state.playing); state.playing = 0;
  $("g-play").setAttribute("aria-label", "Play the weeks"); $("g-play").classList.remove("on");
  $("g-play").firstElementChild.innerHTML = '<path d="M3 1.5v9l7-4.5z"/>';
}
$("g-play").addEventListener("click", () => {
  if (state.playing) { stopPlaying(); return sound.play("tick"); }
  const n = state.shown.card.rounds; sound.play("tick");
  state.week = state.week >= n ? 1 : state.week + 1; renderWeek(true);
  $("g-play").setAttribute("aria-label", "Pause"); $("g-play").classList.add("on");
  $("g-play").firstElementChild.innerHTML = '<path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z"/>';
  state.playing = setInterval(() => { if (state.week >= n) return stopPlaying(); state.week++; renderWeek(true); }, 1300);
});
$("g-week").addEventListener("input", () => { stopPlaying(); state.week = +$("g-week").value; renderWeek(false); });
$("g-unanswered").addEventListener("click", () => {
  const on = $("g-unanswered").getAttribute("aria-pressed") !== "true"; $("g-unanswered").setAttribute("aria-pressed", String(on));
  if (state.graph) state.graph.setUnanswered(on); sound.play("toggle");
});
$("g-reset").addEventListener("click", () => { if (state.graph) state.graph.reset(); $("g-find").value = ""; sound.play("tick"); });
function focusPerson(id) {
  if (!state.graph || !state.graph.focus(id)) { $("g-find").classList.add("miss"); setTimeout(() => $("g-find").classList.remove("miss"), 600); return false; }
  $("g-find").value = id; return true;
}
$("g-find").addEventListener("change", () => { if ($("g-find").value === "") { state.graph && state.graph.focus(null); } else focusPerson(+$("g-find").value); });
$("card").addEventListener("click", e => { const b = e.target.closest("[data-person]"); if (b) focusPerson(+b.dataset.person); if (e.target.closest("[data-clear]")) { state.graph.focus(null); $("g-find").value = ""; } });
$("fwrap").addEventListener("keydown", e => { if (e.key === "Escape" && state.graph) state.graph.focus(null); });

function legendHTML(card) {
  const sides = { larger: 0, scarce: 0 }; card.cast.forEach(p => sides[p.side]++);
  const dot = (cls, shape = "c") => `<svg class="sw ${cls}" viewBox="0 0 14 14" aria-hidden="true">${shape === "c" ? '<circle cx="7" cy="7" r="5"/>' : '<rect x="2.5" y="2.5" width="9" height="9" rx="1"/>'}</svg>`;
  const line = k => `<svg class="sw-l" viewBox="0 0 26 8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" class="gl k-${k}"/></svg>`;
  const row = (mark, text) => `<li><span class="mk">${mark}</span><span>${text}</span></li>`;
  return `<p class="how">Where people sit means something. The further a person got with anyone, the nearer the middle: couples in the centre, then people who got a yes from both, then people who were answered, and on the rim people nobody answered. So a rule that finds more couples draws a bigger orange centre. ${matchMedia("(pointer: coarse)").matches ? "Use two fingers to move or zoom the graph." : "Drag to move, scroll to zoom."} Numbers appear as you zoom in.${card.cast.length < card.people ? ` ${card.cast.length} of the ${card.people} people are drawn: the first 400 on each side.` : ""}</p>
  <ul class="key">
    ${row(dot("s-looking"), `A circle is a person on the side with more people (${sides.larger}).`)}
    ${row(dot("s-looking", "r"), `A square is a person on the side with fewer (${sides.scarce}).`)}
    ${row('<svg class="sw-size" viewBox="0 0 30 14" aria-hidden="true"><circle cx="5" cy="7" r="3"/><circle cx="20" cy="7" r="6.5"/></svg>', "Bigger means the answer key has more people they could have hit it off with.")}
    ${row(dot("s-matched"), "Orange left as a couple.")}
    ${row(dot("s-looking"), "Grey is still looking.")}
    ${row(dot("s-gaveup"), "Black gave up.")}
    ${row(dot("s-unmet"), "Hollow has not been introduced to anyone yet.")}
    ${row(line("hit"), "An introduction where both want to meet again.")}
    ${row(line("yes"), "Both said yes, and it stopped later.")}
    ${row(line("no"), "Someone said no.")}
    ${row(line("none"), "One or both never answered. Hidden until you press Unanswered.")}
  </ul>`;
}

/* one person's story */
function storyOf(card, id) {
  const person = card.cast.find(p => p.id === id);
  const events = card.events.filter(e => e.a === id || e.b === id).sort((x, y) => x.round - y.round);
  const gave = card.fates.gave_up[id], left = card.fates.left_as_couple[id];
  const lines = []; let partner = null, lostChances = 0, quiet = 0;
  const flush = upTo => { if (quiet) lines.push({ week: quiet === 1 ? `Week ${upTo - 1}` : `Weeks ${upTo - quiet} to ${upTo - 1}`, text: "No introduction.", dim: true }); quiet = 0; };
  for (let w = 1; w <= card.rounds; w++) {
    const ev = events.find(e => e.round === w);
    if (left != null && w > left) break;
    if (!ev) { quiet++; if (gave === w) { flush(w + 1); lines.push({ week: `Week ${w}`, text: "Gave up after too many letdowns.", kind: "gaveup" }); } continue; }
    flush(w);
    const me = ev.a === id ? "a" : "b", them = me === "a" ? "b" : "a", other = ev[them];
    const iAns = ev[me + "_answered"], iYes = ev[me + "_yes"], tAns = ev[them + "_answered"], tYes = ev[them + "_yes"];
    const o = `<button type="button" class="pid" data-person="${other}">#${other}</button>`;
    let text, kind = linkKind(ev);
    if (!iAns && !tAns) text = "Neither of them answered.";
    else if (!iAns) text = `#${id} never answered${gave != null && gave < w ? ", having given up" : ""}. ${o} said ${tYes ? "yes" : "no"}.`;
    else if (!tAns) text = `#${id} said ${iYes ? "yes" : "no"}. ${o} never answered.`;
    else if (iYes && tYes) text = ["", "Both said yes, then the conversation died out.", "Both said yes and kept talking, but they never met.", "Both said yes, kept talking and met. They did not both want to meet again.", "Both said yes, kept talking, met, and both want to meet again. They left as a couple."][ev.stage];
    else if (!iYes && !tYes) text = "Both said no.";
    else text = iYes ? `#${id} said yes. ${o} said no.` : `#${id} said no. ${o} said yes.`;
    if (ev.fated_stage >= 4 && ev.stage < 4) { lostChances++; text += " The answer key says these two would have hit it off."; kind = "lost"; }
    if (ev.known_ok === false) text += " The app knew of a dealbreaker between them.";
    if (ev.stage >= 4) partner = other;
    lines.push({ week: `Week ${w}`, text: `Introduced to ${o}. ${text}`, kind });
    if (gave === w) lines.push({ week: `Week ${w}`, text: "Gave up after too many letdowns, and stopped answering.", kind: "gaveup" });
  }
  if (left == null) flush(card.rounds + 1);
  let end, endKind;
  if (left != null) { end = `Left as a couple${partner != null ? ` with <button type="button" class="pid" data-person="${partner}">#${partner}</button>` : ""} in week ${left}.`; endKind = "matched"; }
  else if (gave != null) { end = `Gave up in week ${gave} and stopped answering.`; endKind = "gaveup"; }
  else if (!events.length) { end = `Was never introduced to anyone in ${card.rounds} weeks.`; endKind = "unmet"; }
  else { end = `Still looking after ${card.rounds} weeks.`; endKind = "looking"; }
  const n = person.possible;
  let key = n === 0 ? "The answer key has nobody in this world they would have hit it off with. No rule could have matched them."
    : `The answer key has ${plural(n, "person", "people")} they would have hit it off with.`;
  if (n > 0 && left != null) key += " The rule found one of them.";
  else if (n > 0 && lostChances) key += ` The rule introduced them to ${lostChances === 1 ? "one" : lostChances} of those, and the introduction went unanswered.`;
  else if (n > 0) key += " The rule never introduced them to any of those.";
  return { person, lines, end, endKind, key, count: events.length };
}
function renderCard(id) {
  const card = state.shown && state.shown.card, el = $("card");
  if (id == null || !card) {
    el.className = "card";
    el.innerHTML = `<h3>How to read it</h3><p>Click a person, or type their number in the Person box, to read what the ${card ? card.rounds : 8} weeks were like for them.</p>${card ? legendHTML(card) : ""}`;
    return;
  }
  const s = storyOf(card, id), p = s.person;
  const upTo = s.lines;   // the whole run: a story is read to its end whatever week the graph shows
  el.className = "card story";
  el.innerHTML = `<div class="card-head"><h3><svg class="sw s-${s.endKind}" viewBox="0 0 14 14" aria-hidden="true">${p.side === "larger" ? '<circle cx="7" cy="7" r="5"/>' : '<rect x="2.5" y="2.5" width="9" height="9" rx="1"/>'}</svg>Person #${p.id}</h3><button type="button" class="link-btn" data-clear>How to read it</button></div>
    <p class="k">${p.age} years old, on the side with ${p.side === "larger" ? "more" : "fewer"} people. ${plural(s.count, "introduction")} in ${card.rounds} weeks.</p>
    <ol class="weeks">${upTo.map(l => `<li class="${l.kind ? "e-" + l.kind : ""}${l.dim ? " dim" : ""}"><span class="wk">${l.week}</span><span>${l.text}</span></li>`).join("")}</ol>
    <p class="ending e-${s.endKind}"><b>How it turned out.</b> ${s.end}</p>
    <p class="keysays"><b>What was possible.</b> ${esc(s.key)}</p>`;
  el.scrollTop = 0;
}

/* The funnel. First in introductions, one unit and one scale, from made to wanting to meet again.
   Then the same last four steps in couples, against what the answer key allowed. */
const LOSS = ["", "never answered by both", "answered, but not a yes from both", "went quiet", "never met", "did not both want to meet again"];
const STEP_NAMES = ["Introductions made", "Answered by both", "Both said yes", "Conversation kept going", "They met", "Both want to meet again"];
function renderFunnel(card) {
  const steps = card.introductions.funnel.map(s => s.count), n = steps.length, st = card.stages, last = st[st.length - 1];
  const bar = (i, max, from) => {
    const c = steps[i], prev = i > from ? steps[i - 1] : null, w = v => (max ? v / max * 100 : 0).toFixed(2) + "%";
    return `<div class="ibar">${prev != null && prev > c ? `<i class="lost" style="left:${w(c)};width:calc(${w(prev)} - ${w(c)})"></i>` : ""}<i class="fill${i === n - 1 ? " rule" : ""}" style="width:${w(c)}"></i></div>`;
  };
  const rows = steps.map((c, i) => `<div class="irow"><div class="istep">${STEP_NAMES[i]}</div>
    <div class="icell">${bar(i, steps[0], 0)}<p><b${i === n - 1 ? ' class="hot"' : ""}>${c.toLocaleString("en")}</b>${i ? `<span>${(steps[i - 1] - c).toLocaleString("en")} ${LOSS[i]}</span>` : ""}</p></div>
    <div class="icell zoom">${i >= 2 ? bar(i, steps[2], 2) : i === 0 ? `<p class="restart">The last four steps again, larger. The scale restarts at ${steps[2]} introductions.</p>` : ""}</div></div>`).join("");
  // the eight weeks as small multiples, all on the scale of the busiest week
  const keys = ["introductions", "answered", "both_said_yes", "kept_talking", "met", "hit_it_off"], top = Math.max(1, ...card.per_round.map(r => r.introductions));
  const weeks = card.per_round.map(r => `<div class="wk-col"><b>Week ${r.round}</b>${keys.map((k, i) => `<div class="wk-row"><i class="${i === keys.length - 1 ? "rule" : ""}" style="width:${(r[k] / top * 100).toFixed(1)}%"></i><span>${r[k]}</span></div>`).join("")}<small>${r.gave_up} gave up</small></div>`).join("");
  // couples, against what was possible
  const max = Math.max(1, st[0].best_possible), w = v => (v / max * 100).toFixed(2) + "%";
  const level = (count, prev, cls, bench) => `<div class="fbar">${prev != null && prev > count ? `<i class="lost" style="width:${w(prev)}"></i>` : ""}${bench != null ? `<i class="bench" style="width:${w(bench)}"></i>` : ""}<i class="fill ${cls}" style="width:${w(count)}"></i></div>`;
  const couples = st.map((s, i) => {
    const pPrev = i ? st[i - 1].best_possible : null, rPrev = i ? st[i - 1].couples : null;
    return `<div class="frow"><div class="fstage"><b>${STEP_NAMES[i + 2]}</b></div>
      <div class="fcell">${level(s.best_possible, pPrev, "possible")}<div class="fnum"><b>${s.best_possible}</b>${pPrev != null ? `<small>${pPrev - s.best_possible} lost</small>` : "<small>couples</small>"}</div></div>
      <div class="fcell">${level(s.couples, rPrev, i === 3 ? "rule" : "possible", s.informed_benchmark)}<div class="fnum"><b${i === 3 ? ' class="hot"' : ""}>${s.couples}</b>${rPrev != null ? `<small>${rPrev - s.couples} lost</small>` : "<small>couples</small>"}<small class="bn">reader ${s.informed_benchmark}</small></div></div></div>`;
  }).join("");
  const it = card.introductions, po = card.people_outcomes;
  $("p-funnel").innerHTML = `
    <p class="lead-line">Of <b>${steps[0].toLocaleString("en")}</b> introductions, <b class="hot">${steps[n - 1]}</b> became couples who want to meet again. ${steps[0] ? `Most were lost at the first step: ${(steps[0] - steps[1]).toLocaleString("en")} were never answered by both people.` : ""}</p>
    <div class="ifunnel" role="img" aria-label="Introductions at each step: ${steps.map((c, i) => STEP_NAMES[i] + " " + c).join(", ")}.">${rows}</div>
    <div class="flegend"><span><i class="sw-bar possible"></i>introductions that reached this step</span><span><i class="sw-bar lost"></i>lost at this step</span><span><i class="sw-bar rule"></i>couples who want to meet again</span></div>
    <p class="note">Every bar counts introductions, and the bars in a column share one scale. ${po.gave_up} people gave up along the way and ${po.never_introduced} were never introduced to anyone${it.known_dealbreaker_violations ? `. ${it.known_dealbreaker_violations.toLocaleString("en")} introductions were made against a dealbreaker the app knew about` : ""}.</p>

    <h3 class="sub">Week by week</h3>
    <p class="note first">The same six steps for each week, all on the scale of the busiest week. Couples who want to meet again leave the pool, and people who are let down too often stop answering.</p>
    <div class="wk-wrap"><div class="wk-grid" style="--weeks:${card.per_round.length}"><div class="wk-col labels"><b>&nbsp;</b>${STEP_NAMES.map(s => `<div class="wk-row">${s}</div>`).join("")}<small>&nbsp;</small></div>${weeks}</div></div>

    <h3 class="sub">Against what was possible</h3>
    <p class="note first">Now counting couples, with nobody in two couples at once. On the left, everything the answer key allows. On the right, what this rule found, on the same scale, so a bar half as wide is half as many couples.</p>
    <div class="funnel" role="img" aria-label="Two funnels on the same scale. Possible couples by stage: ${st.map(s => s.best_possible).join(", ")}. This rule's couples by stage: ${st.map(s => s.couples).join(", ")}.">
      <div class="frow fhead"><div></div><div><b>What was possible</b><small>every couple the answer key allows</small></div><div><b>What this rule found</b><small>${esc(card.rule)}</small></div></div>
      ${couples}
    </div>
    <div class="flegend"><span><i class="sw-bar possible"></i>couples</span><span><i class="sw-bar lost"></i>lost since the stage above</span><span><i class="sw-bar rule"></i>couples who want to meet again</span>
      <span><i class="sw-bar bench"></i>how far a perfect reader of profiles gets. A reference, not a strategy.</span></div>
    <p class="note">The perfect reader knows every profile exactly and still cannot see chemistry or luck, so the dashed outline is roughly the most that reading profiles can do.</p>`;
}
/* Every strategy on one axis, from nothing to everything the answer key allows. A dot is the mean over
   the seeds, the line runs from the worst seed to the best. Never scaled to the leader. */
function renderAll() {
  if (!state.shown) return;
  const keep = document.activeElement && document.activeElement.dataset && document.activeElement.dataset.show;
  const rows = allStrategies().map(s => ({ s, st: seedStats(s), now: seedStore(s).get(state.size.seed), bad: [...seedStore(s).values()].find(x => x.error) }));
  const ran = rows.filter(r => r.st).sort((a, b) => b.st.mean - a.st.mean), rest = rows.filter(r => !r.st);
  const all = [...state.seeds.get(baseKey()).values()].flatMap(m => [...m.values()]).filter(x => !x.error);
  const bench = all.length ? all.reduce((a, x) => a + x.benchShare, 0) / all.length : null;
  const random = rows.find(r => r.s.id === "random_introductions"), floor = random && random.st ? random.st.mean : null;
  const x = v => (v * 100).toFixed(2) + "%", left = seedsLeft(), big = state.size.people > SEEDS_MAX_PEOPLE;
  const refs = `${bench != null ? `<i class="ref bench" style="left:${x(bench)}"></i>` : ""}${floor != null ? `<i class="ref floor" style="left:${x(floor)}"></i>` : ""}`;
  const tr = ({ s, st, now }) => `<tr class="${state.shown.strategy.id === s.id ? "shown" : ""}"><td><button type="button" class="row-btn" data-show="${esc(s.id)}">${esc(s.name)}</button>${s.mine ? ' <span class="tag">yours</span>' : ""}</td>
    <td class="plotcell"><div class="plot">${refs}<i class="range" style="left:${x(st.min)};width:${x(st.max - st.min)}"></i><i class="dot" style="left:${x(st.mean)}"></i></div></td>
    <td class="n strong">${pct(st.mean)}</td><td class="n">${st.n > 1 ? pct(st.min) + " to " + pct(st.max) : "one seed"}</td><td class="n">${now && !now.error ? now.couples + " of " + now.possible : ""}</td></tr>`;
  $("p-all").innerHTML = `<p class="lead-line">Every strategy in the same world. Each is played with ${big ? "one seed" : SEEDS + " seeds"}, because the same rule in the same world can move by several points on luck alone.${left ? ` <span class="pending">${left} more ${left === 1 ? "run" : "runs"} to go</span>` : ""}</p>
    <div class="tablew"><table class="alltab"><thead><tr><th>Strategy, sorted by mean share of the possible couples found</th>
      <th class="plothead"><div class="plot axis">${bench != null ? `<span class="reflab" style="left:${x(bench)}">perfect reader of profiles</span>` : ""}${floor != null ? `<span class="reflab low" style="left:${x(floor)}">random</span>` : ""}</div>
        <div class="plot ticks"><span style="left:0">0</span><span class="minor" style="left:25%">25%</span><span style="left:50%">50%</span><span class="minor" style="left:75%">75%</span><span style="left:100%">100%</span></div></th>
      <th class="n">Mean</th><th class="n">Worst to best seed</th><th class="n">Seed ${state.size.seed}</th></tr></thead><tbody>
    ${ran.map(tr).join("")}
    ${rest.map(({ s, bad }) => `<tr class="unrun"><td>${esc(s.name)}${s.mine ? ' <span class="tag">yours</span>' : ""}</td><td colspan="4">${bad ? "Stopped with an error" + (bad.error.line ? " at line " + bad.error.line : "") : "Waiting for its first run"}</td></tr>`).join("")}
    </tbody></table></div>
    <div class="flegend"><span><i class="sw-dot"></i>mean share of the possible couples found</span><span><i class="sw-range"></i>worst seed to best seed</span><span><i class="sw-tick"></i>a perfect reader of profiles, a reference and not a strategy</span><span><i class="sw-tick dotted"></i>random introductions, the floor</span><span><i class="sw-dot hot"></i>the strategy shown now</span></div>
    <p class="note">The axis runs from no couples to every couple the answer key allows, so a strategy a quarter of the way along found a quarter of them. When two lines overlap, this world cannot tell those strategies apart. ${big ? "Worlds of more than " + SEEDS_MAX_PEOPLE + " people are played with one seed only, to keep the page quick. " : ""}Press a name to see that strategy's run.</p>`;
  if (keep) { const b = $("p-all").querySelector(`[data-show="${CSS.escape(keep)}"]`); if (b) b.focus(); }
}
$("p-all").addEventListener("click", e => {
  const b = e.target.closest("[data-show]"); if (!b) return;
  const s = findStrategy(b.dataset.show); if (s) runStrategy(s).then(() => selectTab("all"));
});

/* ---------- the editor ---------- */
function uniqueName(base) { const names = new Set(allStrategies().map(s => s.name)); if (!names.has(base)) return base; let i = 2; while (names.has(base + " " + i)) i++; return base + " " + i; }
function openEditor(s) {
  let ed;
  if (s && s.mine) ed = { id: s.id, name: s.name, source: s.source, from: s.from, saved: true };
  else {
    const baseName = s ? (ABOUT[s.id] && ABOUT[s.id].short || s.name) : null;
    const name = uniqueName(s ? "my " + baseName.replace(/^the /, "") : "my rule");
    ed = { id: null, name, source: (s ? s.source : TEMPLATE).replace(NAME_RE, `@rule(${JSON.stringify(name)})`), from: s ? s.name : "a short example", saved: false };
  }
  state.editing = ed;
  $("ed-name").value = ed.name; $("code").value = ed.source;
  $("ed-origin").textContent = ed.saved ? "Yours. Changes are saved when you run." : `A copy of ${ed.from}. It becomes yours when you run it.`;
  $("ed-delete").hidden = !ed.saved; $("ed-status").textContent = ""; setEditorError(null);
  $("pane-list").hidden = true; $("pane-edit").hidden = false; $("work").classList.add("editing");
  syncGutter(); sound.play("open"); $("strategies").scrollTop = 0;
  $("code").focus({ preventScroll: true }); $("code").setSelectionRange(0, 0); $("code").scrollTop = 0;
  if (matchMedia("(max-width: 1099px)").matches) $("strategies").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
}
function closeEditor() {
  state.editing = null; $("pane-edit").hidden = true; $("pane-list").hidden = false; $("work").classList.remove("editing");
  renderStrategies(); sound.play("close"); $("strategies").scrollTop = 0;
}
function syncGutter() {
  const n = $("code").value.split("\n").length, bad = state.editing && state.editing.errorLine;
  $("gutter").innerHTML = Array.from({ length: n }, (_, i) => `<span${i + 1 === bad ? ' class="bad"' : ""}>${i + 1}</span>`).join("\n");
  $("gutter").scrollTop = $("code").scrollTop;
}
function setEditorError(error) {
  const ed = state.editing; if (ed) ed.errorLine = error && error.line || null;
  $("ed-err").hidden = !error; $("code-wrap").classList.toggle("bad", !!error);
  if (error) $("ed-err").replaceChildren(h("b", { text: error.line ? `Line ${error.line}. ` : "The rule stopped. " }), error.message);
  syncGutter();
  if (error && error.line) { const lh = parseFloat(getComputedStyle($("code")).lineHeight); $("code").scrollTop = Math.max(0, (error.line - 4) * lh); }
}
async function saveAndRun() {
  const ed = state.editing; if (!ed || state.running != null) return;
  const name = $("ed-name").value.trim() || "my rule";
  let source = $("code").value;
  if (NAME_RE.test(source)) source = source.replace(NAME_RE, `@rule(${JSON.stringify(name)})`);
  $("code").value = source;
  let s = ed.id && findStrategy(ed.id);
  if (s) { if (s.source !== source) s.rev++; s.name = name; s.source = source; }
  else { s = { id: "m" + Date.now().toString(36), name, source, from: ed.from, mine: true, rev: 1 }; state.mine.push(s); ed.id = s.id; ed.saved = true; }
  saveMine(); $("ed-delete").hidden = false; $("ed-origin").textContent = "Yours. Changes are saved when you run.";
  setEditorError(null); $("ed-status").textContent = "Saved. Running the " + state.size.rounds + " weeks.";
  const data = await runStrategy(s);
  if (!data) return;
  if (data.error) { if (data.error.where === "rule") setEditorError(data.error); $("ed-status").textContent = "Saved. The run stopped" + (data.error.line ? " at line " + data.error.line + "." : "."); }
  else { const last = data.stages[data.stages.length - 1]; $("ed-status").textContent = `Saved. Finished in ${data._ms} ms: ${last.couples} of ${last.best_possible} couples found.`; }
}
$("ed-back").addEventListener("click", closeEditor);
$("ed-run").addEventListener("click", saveAndRun);
$("ed-delete").addEventListener("click", () => {
  const ed = state.editing; if (!ed || !ed.id) return;
  if (!$("ed-delete").classList.contains("sure")) { $("ed-delete").classList.add("sure"); $("ed-delete").textContent = "Delete it for good"; setTimeout(() => { $("ed-delete").classList.remove("sure"); $("ed-delete").textContent = "Delete"; }, 3000); return; }
  $("ed-delete").classList.remove("sure"); $("ed-delete").textContent = "Delete";
  state.mine = state.mine.filter(s => s.id !== ed.id); saveMine();
  if (state.activeId === ed.id) state.activeId = "article";
  closeEditor();
});
const code = $("code");
code.addEventListener("keydown", e => {
  if (e.key === "Tab" && !e.shiftKey && !e.altKey && !code.dataset.leave) { e.preventDefault(); code.setRangeText("    ", code.selectionStart, code.selectionEnd, "end"); syncGutter(); }
  if (e.key === "Escape") { code.dataset.leave = "1"; $("ed-status").textContent = "Tab now moves on from the editor."; }
  else if (e.key !== "Tab") delete code.dataset.leave;
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); saveAndRun(); }
});
code.addEventListener("blur", () => delete code.dataset.leave);
code.addEventListener("input", syncGutter);
code.addEventListener("scroll", () => { $("gutter").scrollTop = code.scrollTop; });
$("guide-list").append(...GUIDE.flatMap(([group, items]) => [h("dt", { text: group }), ...items.map(([name, what]) =>
  h("dd", {}, h("button", { type: "button", class: "sig", text: name, "aria-label": "Insert " + name, onclick: () => { code.focus(); code.setRangeText(name, code.selectionStart, code.selectionEnd, "end"); syncGutter(); sound.play("tick"); } }), h("span", { text: what })))]));

/* ---------- the jump to the result, for narrow screens ---------- */
let resultVisible = true;
new IntersectionObserver(es => { resultVisible = es[0].isIntersecting && es[0].intersectionRect.height > 120; syncJump(); }, { threshold: [0, 0.1, 0.3] }).observe($("result"));
function syncJump() {
  const el = $("jump"), busy = state.running != null, shown = state.shown;
  if (resultVisible || !(busy || shown) || $("sheet").open) { el.hidden = true; return; }
  el.hidden = false; el.classList.toggle("busy", busy);
  const last = shown && shown.card.stages[3];
  $("jump-text").textContent = busy ? `Running ${activeStrategy().name}` : !$("stale").hidden ? "The run stopped. See the last result" : `${last.couples} of ${last.best_possible} couples, ${pct(shown.card.score)}. See the result`;
}
$("jump").addEventListener("click", () => { $("result").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" }); $("result").focus({ preventScroll: true }); sound.play("tick"); });

/* ---------- sharing. The address bar is left alone: a link is built only when asked for,
   and a link someone opens is read once and then removed from the address bar. ---------- */
const b64 = bytes => btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64 = s => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), c => c.charCodeAt(0));
async function pipe(bytes, stream) { const out = new Blob([bytes]).stream().pipeThrough(stream); return new Uint8Array(await new Response(out).arrayBuffer()); }
async function pack(text) {
  const raw = new TextEncoder().encode(text);
  try { return "z" + b64(await pipe(raw, new CompressionStream("deflate-raw"))); } catch (e) { return "u" + b64(raw); }
}
async function unpack(s) {
  const bytes = unb64(s.slice(1));
  return new TextDecoder().decode(s[0] === "z" ? await pipe(bytes, new DecompressionStream("deflate-raw")) : bytes);
}
async function shareLink() {
  const s = state.shown ? state.shown.strategy : activeStrategy(), parts = ["w=" + state.preset];
  if (s.mine) parts.push("c=" + await pack(JSON.stringify([s.name, s.source]))); else parts.push("s=" + s.id);
  const o = Object.entries(state.overrides).map(([k, v]) => k + ":" + v).join(",");
  if (o) parts.push("o=" + encodeURIComponent(o));
  if (state.size.seed !== DEFAULT_SIZE.seed) parts.push("seed=" + state.size.seed);
  if (state.size.people !== DEFAULT_SIZE.people) parts.push("n=" + state.size.people);
  if (state.size.rounds !== DEFAULT_SIZE.rounds) parts.push("wk=" + state.size.rounds);
  return location.origin + location.pathname + "#" + parts.join("&");
}
function adoptShared(name, source, from) {
  const builtin = state.builtins.find(b => b.source === source); if (builtin) return builtin.id;
  let s = state.mine.find(m => m.source === source);
  if (!s) { s = { id: "m" + Date.now().toString(36), name: uniqueName(name || "a shared rule"), source, from, mine: true, rev: 1 }; s.source = source.replace(NAME_RE, `@rule(${JSON.stringify(s.name)})`); state.mine.push(s); saveMine(); }
  return s.id;
}
async function readHash() {
  const raw = location.hash.slice(1); if (!raw) return;
  history.replaceState(null, "", location.pathname + location.search);   // read once, then clean
  try {
    if (!raw.includes("=")) {   // links made by the first version of the page
      const o = JSON.parse(new TextDecoder().decode(unb64(raw))); if (!presetOf(o.p)) return;
      state.preset = o.p; state.overrides = o.o || {};
      state.size = { seed: o.seed ?? 0, people: o.people ?? 400, rounds: o.rounds ?? 8 };
      if (o.r) state.activeId = adoptShared((NAME_RE.exec(o.r) || [])[2], o.r, "a shared link");
      return;
    }
    const q = new URLSearchParams(raw);
    if (presetOf(q.get("w"))) state.preset = q.get("w");
    const known = new Set(state.meta.settings.map(s => s.key));
    (q.get("o") || "").split(",").forEach(pair => { const [k, v] = pair.split(":"); if (known.has(k) && isFinite(+v)) state.overrides[k] = +v; });
    const num = (k, d, lo, hi) => { const v = parseInt(q.get(k), 10); return isFinite(v) ? Math.max(lo, Math.min(hi, v)) : d; };
    state.size = { seed: num("seed", 0, 0, 99999), people: num("n", 400, 100, 1200), rounds: num("wk", 8, 2, 16) };
    if (q.get("c")) { const [name, source] = JSON.parse(await unpack(q.get("c"))); state.activeId = adoptShared(name, source, "a shared link"); }
    else if (state.builtins.some(b => b.id === q.get("s"))) state.activeId = q.get("s");
  } catch (e) { /* a damaged link opens the default page */ }
}
$("share").addEventListener("click", async () => {
  const link = await shareLink(), label = $("share").querySelector("[data-label]");
  try { await navigator.clipboard.writeText(link); } catch (e) { window.prompt("Copy this link", link); }
  $("share").classList.add("copied"); label.textContent = "Link copied"; sound.play("success");
  clearTimeout($("share")._t); $("share")._t = setTimeout(() => { $("share").classList.remove("copied"); label.textContent = "Copy a link to this run"; }, 1600);
});

/* ---------- start ---------- */
async function start(meta) {
  state.meta = meta;
  state.builtins = meta.rules.map(r => ({ id: r.id, name: r.name, source: r.source, description: r.description }));
  state.mine = loadMine();
  await readHash();
  if (!findStrategy(state.activeId)) state.activeId = state.builtins[0].id;
  renderWorlds(); renderStrategies(); renderCard(null);
  boot.stop(); $("boot").hidden = true; $("app").hidden = false; document.body.classList.remove("booting"); document.body.classList.add("ready");
  runStrategy(activeStrategy(), { quiet: true });
}
// for the checks in dev_check.py
window.__env = { state, shareLink, seedsLeft };
