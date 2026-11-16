// Runs the real Python engine in the visitor's browser. Nothing is sent to a server.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/pyodide.mjs";

let py = null;

const boot = (async () => {
  postMessage({ type: "progress", text: "Starting Python" });
  py = await loadPyodide();
  postMessage({ type: "progress", text: "Loading numpy" });
  await py.loadPackage(["numpy", "micropip"]);
  postMessage({ type: "progress", text: "Loading the engine" });
  const index = await (await fetch(new URL("./wheel/index.json", self.location.href))).json();
  py.globals.set("WHEEL", new URL("./wheel/" + index.file, self.location.href).href);
  await py.runPythonAsync("import micropip\nawait micropip.install(WHEEL)\nimport json\nfrom wavelength import web");
  const meta = py.runPython(
    'json.dumps({"settings": web.describe_settings(), "presets": web.list_presets(), "rules": web.list_rules()})'
  );
  postMessage({ type: "ready", meta: JSON.parse(meta) });
})().catch(err => postMessage({ type: "fatal", message: String(err) }));

const CALLS = {
  run: 'a = json.loads(ARGS)\njson.dumps(web.run(a["settings"], a["rule"], a["seed"], a["people"], a["rounds"]))',
  compare: 'a = json.loads(ARGS)\njson.dumps(web.compare(a["settings"], a["rules"], a["seed"], a["people"], a["rounds"]))',
};

onmessage = async e => {
  await boot;
  const { id, type, args } = e.data;
  try {
    py.globals.set("ARGS", JSON.stringify(args));
    const t0 = performance.now();
    const out = py.runPython(CALLS[type]);
    postMessage({ type: "result", id, ms: Math.round(performance.now() - t0), data: JSON.parse(out) });
  } catch (err) {
    postMessage({ type: "result", id, data: { error: { where: "engine", line: null, message: String(err).split("\n").slice(-2).join(" ") } } });
  }
};
