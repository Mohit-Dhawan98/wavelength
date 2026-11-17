// The graph of one run: people are nodes, introductions are links. D3 force layout, laid out once
// from the whole run so that people stay put while the week control reveals what happened to them.
/* global d3 */
import { reduced } from "./kit.js";

export const LINK_KINDS = {
  hit: "both want to meet again",
  yes: "both said yes, then it stopped",
  no: "someone said no",
  none: "one or both never answered",
};
export function linkKind(ev) {
  if (!(ev.a_answered && ev.b_answered)) return "none";
  if (!(ev.a_yes && ev.b_yes)) return "no";
  return ev.stage >= 4 ? "hit" : "yes";
}
export const radius = possible => 3 + 1.7 * Math.sqrt(possible);

// Pull each person towards an ellipse whose size depends on how far they got: couples in the middle,
// people nobody answered on the rim.
function ringForce(R, aspect) {
  const share = [1, 0.74, 0.46, 0.2]; let nodes;
  const force = alpha => {
    for (const n of nodes) {
      const x = n.x / Math.sqrt(aspect), y = n.y * Math.sqrt(aspect), d = Math.hypot(x, y) || 1e-6, k = (share[n.depth] * R - d) / d * alpha * (n.depth === 3 ? 0.1 : 0.22);
      n.vx += x * k * Math.sqrt(aspect); n.vy += y * k / Math.sqrt(aspect);
    }
  };
  force.initialize = ns => { nodes = ns; };
  return force;
}

export function mountGraph(host, data, { onFocus }) {
  host.replaceChildren();
  const left = data.fates.left_as_couple, gave = data.fates.gave_up;
  const nodes = data.cast.map(p => ({ id: p.id, side: p.side, possible: p.possible, r: radius(p.possible), first: Infinity, left: left[p.id] ?? Infinity, gave: gave[p.id] ?? Infinity }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  const links = [];
  // How far each person got with anyone: 3 found a partner, 2 had a yes from both, 1 was answered, 0 nothing.
  const rank = { hit: 3, yes: 2, no: 1, none: 0 };
  nodes.forEach(n => { n.depth = 0; });
  data.events.forEach((ev, i) => {
    const a = byId.get(ev.a), b = byId.get(ev.b), kind = linkKind(ev);
    // In a world too large to draw whole, one end of an introduction may be off the picture.
    for (const n of [a, b]) if (n) { n.first = Math.min(n.first, ev.round); n.depth = Math.max(n.depth, rank[kind]); }
    if (a && b) links.push({ source: a, target: b, kind, round: ev.round, i });
  });
  const pulling = links.filter(l => l.kind !== "none");
  // The layout pulls people who got further towards the middle, so the picture changes with the rule.

  let W = Math.max(320, host.clientWidth), H = Math.max(280, host.clientHeight);
  // Lay out in a box with the panel's proportions so the picture fills it.
  const aspect = W / H;
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(pulling).distance(l => l.kind === "hit" ? 16 : l.kind === "yes" ? 26 : 44).strength(l => l.kind === "hit" ? 0.9 : l.kind === "yes" ? 0.45 : 0.05))
    .force("charge", d3.forceManyBody().strength(-14).distanceMax(160))
    .force("collide", d3.forceCollide(d => d.r + 2.2))
    .force("ring", ringForce(Math.sqrt(nodes.length) * 13, aspect))
    .stop();
  for (let i = 0; i < 240; i++) sim.tick();

  const svg = d3.select(host).append("svg").attr("role", "img").attr("class", "gsvg")
    .attr("aria-label", `${nodes.length} people and the ${links.length} introductions a rule made between them. Use the Person box to read one person's story.`);
  const g = svg.append("g");
  const link = g.append("g").selectAll("line").data(links).join("line").attr("class", d => "gl k-" + d.kind)
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y)
    .style("--len", d => Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y).toFixed(1));
  const node = g.append("g").selectAll("g").data(nodes).join("g").attr("class", d => "gn side-" + d.side)
    .attr("transform", d => `translate(${d.x},${d.y})`);
  node.filter(d => d.side === "larger").append("circle").attr("r", d => d.r);
  node.filter(d => d.side !== "larger").append("rect").attr("x", d => -d.r * 0.9).attr("y", d => -d.r * 0.9).attr("width", d => d.r * 1.8).attr("height", d => d.r * 1.8).attr("rx", 1);
  node.append("title").text(d => `#${d.id}: read their story`);
  const label = g.append("g").selectAll("text").data(nodes).join("text").attr("class", "nl").attr("text-anchor", "middle")
    .attr("x", d => d.x).attr("y", d => d.y - d.r - 3).text(d => "#" + d.id);

  let fitK = 1, week = data.rounds, showNone = false, focused = null;
  const zoomer = d3.zoom().scaleExtent([0.3, 12]).on("zoom", ev => { g.attr("transform", ev.transform); svg.classed("zin", ev.transform.k > fitK * 2.7); });
  // A finger on the graph scrolls the page. Two fingers move and zoom the graph.
  zoomer.filter(ev => ev.type.startsWith("touch") ? ev.touches.length > 1 : (!ev.ctrlKey || ev.type === "wheel") && !ev.button);
  svg.call(zoomer).on("dblclick.zoom", null);
  function fit(animate) {
    W = Math.max(320, host.clientWidth); H = Math.max(280, host.clientHeight);
    svg.attr("viewBox", `${-W / 2} ${-H / 2} ${W} ${H}`).attr("width", W).attr("height", H);
    const pad = 16, xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    fitK = Math.min(W / (x1 - x0), H / (y1 - y0), 3);
    const t = d3.zoomIdentity.scale(fitK).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
    (animate && !reduced ? svg.transition().duration(450) : svg).call(zoomer.transform, t);
  }
  fit(false);
  const ro = new ResizeObserver(() => { if (Math.abs(host.clientWidth - W) > 2 || Math.abs(host.clientHeight - H) > 2) { fit(false); if (focused != null) frame(focused); } });
  ro.observe(host);

  const stateOf = d => d.left <= week ? "matched" : d.gave <= week ? "gaveup" : d.first <= week ? "looking" : "unmet";
  function paint(animate) {
    node.attr("data-state", stateOf);
    const near = focused == null ? null : new Set([focused]);
    link.each(function (d) {
      const mine = focused != null && (d.source.id === focused || d.target.id === focused);
      const shown = d.round <= week && (d.kind !== "none" || showNone || mine);
      if (mine && shown) { near.add(d.source.id); near.add(d.target.id); }
      const el = this.classList;
      el.toggle("off", !shown); el.toggle("on", mine && shown); el.toggle("old", d.round < week && d.kind !== "hit");
      el.toggle("draw", !!animate && shown && d.round === week);
    });
    svg.classed("focus", focused != null);
    node.classed("on", d => near != null && near.has(d.id)).classed("me", d => d.id === focused);
    label.classed("on", d => near != null && near.has(d.id));
    if (focused != null) node.filter(d => d.id === focused).raise();
  }
  paint(false);

  function focus(id, tell = true) {
    if (id != null && !byId.has(id)) return false;
    focused = id; paint(false); if (tell) onFocus(id);
    if (id != null) frame(id);
    return true;
  }
  // Bring a focused person and everyone they were introduced to into view, gently.
  function frame(id) {
    const near = [byId.get(id)]; links.forEach(l => { if (l.round <= week) { if (l.source.id === id) near.push(l.target); if (l.target.id === id) near.push(l.source); } });
    const pad = 46, xs = near.map(n => n.x), ys = near.map(n => n.y);
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    const k = Math.max(fitK, Math.min(W / (x1 - x0), H / (y1 - y0), fitK * 2.6));
    const t = d3.zoomIdentity.scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
    (reduced ? svg : svg.transition().duration(500)).call(zoomer.transform, t);
  }
  node.on("click", (ev, d) => { ev.stopPropagation(); focus(d.id); });
  // Drag a person to move them out of a crowd. Their links and label follow.
  node.call(d3.drag().filter(ev => !ev.ctrlKey && !ev.button && !ev.type.startsWith("touch")).on("drag", function (ev, d) {
    d.x = ev.x; d.y = ev.y; d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
    link.filter(l => l.source === d || l.target === d).attr("x1", l => l.source.x).attr("y1", l => l.source.y).attr("x2", l => l.target.x).attr("y2", l => l.target.y);
    label.filter(n => n === d).attr("x", d.x).attr("y", d.y - d.r - 3);
  }));
  svg.on("click", () => { if (focused != null) focus(null); });

  return {
    setWeek(w, animate) { week = w; paint(animate); },
    setUnanswered(on) { showNone = on; paint(false); },
    focus, get focused() { return focused; },
    reset() { focus(null); fit(true); },
    zoomBy(k) { svg.call(zoomer.scaleBy, k); },
    destroy() { ro.disconnect(); sim.stop(); host.replaceChildren(); },
  };
}
