/* Exercise the dashboard's render builders against real collector output.
 *
 * The page has no build step and no test framework, but its render functions
 * are pure string builders, so a minimal shim is enough to prove they neither
 * throw nor silently render nothing -- including on the cases that only show up
 * once, like an empty runs/ directory or a run that died during setup.
 *
 *   harness-arena collect --out /tmp/results.json
 *   node tests/test_dashboard.mjs /tmp/results.json
 *
 * With no argument it synthesizes a fixture, so it runs before any benchmark has.
 */
import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import vm from "node:vm";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, "..", "dashboard", "index.html"), "utf8");

const script = html.match(/^<script>$([\s\S]*?)^<\/script>$/m);
if (!script) {
  console.error("Could not find the <script> block in dashboard/index.html");
  process.exit(1);
}

function fixture() {
  const mk = (id, harness, label, opts = {}) => ({
    run_id: id, path: `/runs/${id}`, harness, harness_label: label,
    harness_vendor: null, harness_repo: null, agent_ref: `harnesses.${harness}:X`,
    model_label: "Test Model", model_fingerprint: "fp1", model_served_id: "m",
    model_quant: "TESTQ", model_params: 1e11, model_n_ctx: 131072,
    dataset: "terminal-bench@2.0", n_concurrent: 1, n_attempts: 1,
    is_partial: true, n_tasks_requested: null, subset: "stratified-25",
    agent_timeout_multiplier: 8.0, n_timeouts: opts.timeouts || 0,
    harbor_version: "0.20.0", started_at: opts.started || "2026-08-08T20:00:00+00:00",
    finished_at: null, status: opts.status || "complete",
    n_total: 3, n_done: 3, n_resolved: opts.resolved ?? 1,
    n_errors: opts.timeouts || 0, pass_rate: (opts.resolved ?? 1) / 3,
    ci_low: 0.05, ci_high: 0.8,
    // Partial credit, matching the per-task checks below: only `beta` carries
    // any, at 5 of 6. The fixture has to carry every field collect emits or a
    // panel reading a newer one silently renders nothing here.
    n_checks_total: 6, n_checks_passed: 5, check_rate: 5 / 6,
    // Deliberately different from the all-inclusive pair above: a solved task
    // passes every check, so the two figures only coincide when nothing was
    // solved, and a fixture where they matched would not prove the panel is
    // reading the right one.
    n_checks_missed_total: 4, n_checks_missed_passed: 1,
    missed_check_rate: 1 / 4, n_missed: 2,
    wall_clock_s: 5400, llm_busy_pct: 92.0,
    mean_output_tokens_per_solve: opts.tokens ?? 42000,
    // The mean over both tasks below (40k, 50k), deliberately unequal to the
    // per-solve figure: they only coincide when every trial was solved, and a
    // fixture where they matched would not prove the panel reads the right one.
    mean_output_tokens_per_trial: opts.tokensPerTrial ?? 45000,
    n_token_samples: 2,
    median_duration_s: opts.dur ?? 1800, total_duration_s: 5400,
    error_types: opts.timeouts ? { AgentTimeoutError: opts.timeouts } : {},
    tasks: [
      { task_name: "alpha", resolved: true, reward: 1, error_type: null,
        error_message: null, duration_s: 1500, agent_s: 1400,
        started_at: null, finished_at: null,
        n_input_tokens: 90000, n_output_tokens: 40000, cost_usd: null },
      // A near miss: all-or-nothing scoring makes this a zero, but 5 of 6
      // checks passing is a very different result from 0 of 6.
      { task_name: "beta", resolved: false, reward: 0, error_type: null,
        error_message: null, duration_s: 2000, agent_s: 1900,
        started_at: null, finished_at: null,
        n_input_tokens: 120000, n_output_tokens: 50000, cost_usd: null,
        n_checks: 6, n_checks_passed: 5,
        checks: [
          { name: "output files exist", status: "passed" },
          { name: "input data integrity", status: "passed" },
          { name: "generate and schema", status: "passed" },
          { name: "shape feasibility", status: "passed" },
          { name: "coverage no duplicates", status: "passed" },
          { name: "performance thresholds", status: "failed" },
        ] },
      { task_name: "gamma", resolved: false, reward: null,
        error_type: opts.timeouts ? "AgentTimeoutError" : null,
        error_message: null, duration_s: 7200, agent_s: 7200,
        started_at: null, finished_at: null,
        n_input_tokens: null, n_output_tokens: null, cost_usd: null },
      // Harbor's verifier runs even when the agent is killed, so a trial can
      // time out *and* pass. The reward is what the pass rate counts, so the
      // matrix must agree with it rather than showing a solved task as a timeout.
      { task_name: "delta", resolved: true, reward: 1,
        error_type: "AgentTimeoutError",
        error_message: null, duration_s: 7200, agent_s: 7200,
        started_at: null, finished_at: null,
        n_input_tokens: 50000, n_output_tokens: 20000, cost_usd: null },
    ],
  });

  const a = mk("hermes__x__1", "hermes", "Hermes Agent", { resolved: 2, timeouts: 1 });
  const b = mk("omp__x__2", "omp", "oh-my-pi",
               { resolved: 1, tokens: 90000, dur: 2600, status: "running",
                 started: "2026-08-08T21:00:00+00:00" });
  return {
    generated_at: "2026-08-08T22:00:00+00:00",
    runs: [b, a],
    task_names: ["alpha", "beta", "gamma", "delta"],
    models: [{ fingerprint: "fp1", label: "Test Model", quant: "TESTQ",
               params: 1e11, n_ctx: 262144, runs: [a.run_id, b.run_id] }],
    comparisons: [{
      a: a.run_id, b: b.run_id, a_label: a.harness_label, b_label: b.harness_label,
      n_shared: 3, only_a: ["beta"], only_b: [], both: ["alpha"], neither: ["gamma"],
      agreement: 2 / 3,
    }],
    summary: { n_runs: 2, n_running: 1, n_models: 1, n_harnesses: 2 },
  };
}

const data = process.argv[2]
  ? JSON.parse(fs.readFileSync(process.argv[2], "utf8"))
  : fixture();

const stub = new Proxy({
  classList: { add() {}, remove() {}, toggle() {} },
  setAttribute() {}, getAttribute: () => null,
  // An unfilled control, which is what every form field is under this stub.
  // Without these, reading the form throws on `.value.trim()` rather than
  // returning the empty spec that an untouched form should produce.
  value: "", checked: false, options: [], selectedIndex: -1,
  style: {}, textContent: "", innerHTML: "",
  getBoundingClientRect: () => ({ width: 0, height: 0 }),
  getContext: () => null, closest: () => null,
}, { get: (t, k) => (k in t ? t[k] : undefined), set: (t, k, v) => ((t[k] = v), true) });

const sandbox = {
  console,
  document: {
    querySelector: () => stub, querySelectorAll: () => [],
    addEventListener() {}, body: stub,
  },
  // A token present means the page believes it has a live control plane, which
  // is what the control-plane render paths are gated on.
  window: { __HARNESS_ARENA_TOKEN__: "test-token" },
  fetch: async () => { throw new Error("no server"); },
  matchMedia: () => ({ matches: true }),
  setInterval: () => 0, setTimeout: () => 0, clearTimeout: () => {},
  requestAnimationFrame: () => 0, addEventListener() {},
  devicePixelRatio: 1, innerWidth: 1440, innerHeight: 900, Date,
};
sandbox.globalThis = sandbox;

const ctx = vm.createContext(sandbox);
vm.runInContext(script[1], ctx);
Object.assign(ctx, { SEED: data });
vm.runInContext(
  "state.data = SEED;" +
  "state.model = (state.data.models[0] || {}).fingerprint || (state.data.models[0] || {}).label;" +
  // render() resolves the benchmark scope before anything is drawn. These
  // checks call the render functions directly, so they resolve it the same way
  // -- otherwise every one of them would be exercising the unscoped fallback
  // rather than the path the page actually takes.
  "state.dataset = 'terminal-bench@2.0';",
  ctx
);

let failed = 0;
const run = (label, expr, minLen = 0) => {
  try {
    const out = vm.runInContext(expr, ctx) || "";
    const ok = out.length >= minLen;
    if (!ok) failed++;
    console.log(`${ok ? "PASS" : "FAIL"}  ${label.padEnd(38)} ${out.length} chars`);
    return out;
  } catch (err) {
    failed++;
    console.log(`THROW ${label.padEnd(38)} ${err.message}`);
    return "";
  }
};

const V = "visibleRuns(state.data)";
run("filters", `renderFilters(state.data)`, 200);
run("tiles", `renderTiles(state.data, ${V})`, 200);
run("leaderboard", `renderLeaderboard(${V})`, 200);
const matrix = run("task matrix", `renderMatrix(state.data, ${V})`, 200);
run("efficiency", `renderEfficiency(${V})`, 100);
const cost = run("cost of a run", `renderCost(${V})`, 100);
run("run log", `renderRunLog(state.data)`, 200);

vm.runInContext("state.diffOnly = true;", ctx);
run("matrix (disagreements only)", `renderMatrix(state.data, ${V})`, 100);
vm.runInContext("state.diffOnly = false;", ctx);

// Regressions worth pinning: a timeout must not read as a crash, and repeat runs
// of one harness must be distinguishable.
if (!process.argv[2]) {
  // It has to plot the per-trial cost, not the per-solve one its sibling
  // already shows. The two panels are the same chart against different costs,
  // and a copy that quietly plotted the same series would look right.
  const perTrialAxis = /mean output tokens per trial/.test(cost);
  console.log(`${perTrialAxis ? "PASS" : "FAIL"}  cost-of-a-run plots the per-trial cost`);
  if (!perTrialAxis) failed++;

  // The fixture is 45,000 per trial against 42,000 per solve.
  const plotted = /sc-label/.test(cost) && /45,000/.test(cost);
  console.log(`${plotted ? "PASS" : "FAIL"}  ...with a labelled point per harness`);
  if (!plotted) failed++;

  const hasTimeoutCell = /class="cell timeout"/.test(matrix);
  console.log(`${hasTimeoutCell ? "PASS" : "FAIL"}  timeouts get their own cell`);
  if (!hasTimeoutCell) failed++;

  // A trial that timed out but passed its tests must render as a pass, because
  // that is what the leaderboard counts. Showing it as a timeout would make the
  // matrix and the pass rate disagree about the same trial.
  const latePass = /class="cell pass late"/.test(matrix);
  console.log(`${latePass ? "PASS" : "FAIL"}  passed-then-timed-out renders as a pass`);
  if (!latePass) failed++;

  // A failing task must show how many checks it passed, and the tooltip must
  // name them -- otherwise "failed" hides the difference between 5/6 and 0/6.
  const ratio = /class="cell fail near"[^>]*>5\/6</.test(matrix);
  console.log(`${ratio ? "PASS" : "FAIL"}  near-miss cell shows its check ratio`);
  if (!ratio) failed++;

  // The tint marks "got somewhere", not "got most of the way". Ratios are not
  // comparable across tasks -- different check counts, unequal difficulty --
  // so a fraction threshold would rank what the data cannot rank. One check
  // passing is a fact; "most of them" would be a judgement.
  {
    const cell = (passed, total) => vm.runInContext(
      `cellFor({_byTask:{t:{resolved:false,n_checks:${total},n_checks_passed:${passed}}}}, "t")`,
      ctx);
    const cases = [
      ["one of six is tinted", cell(1, 6).near, true],
      ["one of two is tinted", cell(1, 2).near, true],
      ["five of six is tinted", cell(5, 6).near, true],
      ["zero of six is not", cell(0, 6).near, false],
    ];
    for (const [label, got, want] of cases) {
      console.log(`${got === want ? "PASS" : "FAIL"}  ${label}`);
      if (got !== want) failed++;
    }
    // Zero still shows its ratio; only the colour is withheld.
    const zeroGlyph = cell(0, 6).glyph === "0/6";
    console.log(`${zeroGlyph ? "PASS" : "FAIL"}  a zero still shows 0/6 rather than a dot`);
    if (!zeroGlyph) failed++;
  }

  const named = /performance thresholds/.test(matrix) && /t-chk/.test(matrix);
  console.log(`${named ? "PASS" : "FAIL"}  tooltip lists each check by name`);
  if (!named) failed++;

  // The legend must describe what is on screen. Two green ticks differing only
  // by an underline read as a duplicate when the second state never occurred.
  const legendHasLate = /killed at the time limit/.test(matrix);
  console.log(`${legendHasLate ? "PASS" : "FAIL"}  legend explains late passes when present`);
  if (!legendHasLate) failed++;

  const timeoutCells = (matrix.match(/class="cell timeout"/g) || []).length;
  const passCells = (matrix.match(/class="cell pass/g) || []).length;
  const lateCells = (matrix.match(/class="cell pass late"/g) || []).length;
  // Two fixture runs over the same 4 tasks. Both solve alpha and delta (delta
  // late in both); only the hermes run has gamma as a timeout. So: 4 pass cells,
  // 2 of them late, and exactly 1 timeout cell.
  const consistent = timeoutCells === 1 && passCells === 4 && lateCells === 2;
  console.log(
    `${consistent ? "PASS" : "FAIL"}  cell counts match the rewards ` +
    `(${passCells} pass / ${lateCells} late, ${timeoutCells} timeout)`
  );
  if (!consistent) failed++;

  vm.runInContext(
    "state.data.runs.push(Object.assign({}, state.data.runs[1], " +
    "{run_id:'hermes__x__3', started_at:'2026-08-08T23:00:00+00:00'}));",
    ctx
  );
  const dup = run("matrix with repeat harness", `renderMatrix(state.data, ${V})`, 200);
  const headers = (dup.match(/<th><span class="swatch"/g) || []).length;
  const distinct = new Set(dup.match(/Hermes Agent[^<]*/g) || []).size > 1;
  console.log(`${distinct ? "PASS" : "FAIL"}  repeat runs get distinct labels (${headers} cols)`);
  if (!distinct) failed++;
}

// ---- legend is data-driven -------------------------------------------------
// With no late pass and no near miss in view, neither entry may appear.
if (!process.argv[2]) {
  vm.runInContext(
    "state.data.runs.forEach(r => r.tasks.forEach(t => {" +
    "  if (t.task_name === 'delta') { t.resolved = false; t.error_type = 'AgentTimeoutError'; }" +
    "  if (t.task_name === 'beta') { t.n_checks = 0; t.n_checks_passed = 0; t.checks = []; }" +
    "}));",
    ctx
  );
  const plain = vm.runInContext("renderMatrix(state.data, " + V + ")", ctx);
  const noLate = !/killed at the time limit/.test(plain);
  const noNear = !/all-or-nothing/.test(plain);
  console.log(`${noLate ? "PASS" : "FAIL"}  no late-pass legend entry when none occurred`);
  console.log(`${noNear ? "PASS" : "FAIL"}  no all-or-nothing note without a near miss`);
  if (!noLate) failed++;
  if (!noNear) failed++;
}

// ---- stopped runs ----------------------------------------------------------
// A job killed mid-flight writes no finished_at. Without the explicit stop
// marker it reads as "running" forever, so a preserved baseline would look like
// a benchmark still in progress.
if (!process.argv[2]) {
  vm.runInContext(
    "state.data.runs[0].status = 'stopped';" +
    "state.data.runs[0].stopped_reason = 'stopped to adopt pipelining';" +
    "state.data.runs[0].n_done = 18; state.data.runs[0].n_total = 25;",
    ctx
  );
  const log = vm.runInContext("renderRunLog(state.data)", ctx);
  const shown = /status-pill stopped/.test(log) && /stopped/.test(log);
  console.log(`${shown ? "PASS" : "FAIL"}  stopped run renders as stopped`);
  if (!shown) failed++;

  const lb = vm.runInContext("renderLeaderboard(" + V + ")", ctx);
  const labelled = /stopped ·/.test(lb);
  console.log(`${labelled ? "PASS" : "FAIL"}  leaderboard marks the partial run`);
  if (!labelled) failed++;

  vm.runInContext("state.data.runs[0].status = 'complete';", ctx);
}

// ---- the elapsed column reports the model's clock ---------------------------
// A run's headline time is agent execution, not wall clock. The two differ by
// the container work -- pulls, builds, harness installs, verifiers -- which is
// the same cost for every harness on a dataset, so putting it in the headline
// narrows every gap this rig exists to measure. Wall clock stays visible
// underneath, because it is what the run actually cost you.
if (!process.argv[2]) {
  vm.runInContext(
    "state.data.runs[0].agent_total_s = 4968;" +
    "state.data.runs[0].wall_clock_s = 5400;" +
    "state.data.runs[0].llm_busy_pct = 92;",
    ctx
  );
  const log = vm.runInContext("renderRunLog(state.data)", ctx);
  // 4968s of agent inside 5400s of wall. Asserted as an order, not just as two
  // present strings: the whole change is which of the two is the headline.
  const context = /1h 30m wall/.test(log);
  const headline = /1h 23m/.test(log)
    && log.indexOf("1h 23m") < log.indexOf("1h 30m wall");
  console.log(`${headline ? "PASS" : "FAIL"}  elapsed leads with model time`);
  if (!headline) failed++;
  console.log(`${context ? "PASS" : "FAIL"}  ...and keeps wall clock as context`);
  if (!context) failed++;

  // A run recorded before agent time was tracked has no such field. It must
  // still show the time it does have rather than an em dash.
  vm.runInContext("delete state.data.runs[0].agent_total_s;", ctx);
  const legacy = vm.runInContext("renderRunLog(state.data)", ctx);
  const kept = /1h 30m/.test(legacy);
  console.log(`${kept ? "PASS" : "FAIL"}  an older run still reports its wall clock`);
  if (!kept) failed++;
  vm.runInContext("state.data.runs[0].agent_total_s = 4968;", ctx);
}

// ---- layout ----------------------------------------------------------------
// Every panel must claim a grid area, and every area the CSS defines must be
// filled -- an unassigned panel silently collapses into the wrong cell.
if (!process.argv[2]) {
  vm.runInContext("state.activity = null;", ctx);
  const all =
    vm.runInContext("renderLeaderboard(" + V + ")", ctx) +
    vm.runInContext("renderMatrix(state.data, " + V + ")", ctx) +
    vm.runInContext("renderEfficiency(" + V + ")", ctx) +
    vm.runInContext("renderCost(" + V + ")", ctx) +
    vm.runInContext("renderRunLog(state.data)", ctx);
  const areas = (all.match(/--area:([a-z0-9]+)/g) || []).map((s) => s.split(":")[1]);
  const panels = (all.match(/<section class="panel/g) || []).length;
  const want = ["rate", "matrix", "eff", "cost", "log"];
  const complete = want.every((a) => areas.includes(a)) && areas.length === panels;
  console.log(
    `${complete ? "PASS" : "FAIL"}  every panel claims a grid area ` +
    `(${panels} panels, areas: ${areas.join(",")})`
  );
  if (!complete) failed++;
}

// ---- the setup form's draft --------------------------------------------
// Every action refetches /api/state on success and replaces cp.server. The URL
// being typed is, by definition, the one not stored yet, so a successful
// connection test used to snap the box back to empty. Its placeholder renders
// the provider default, which reads as a filled-in value, and the save that
// follows then sends nothing: on a fresh install the endpoint could not be set
// through the UI at all.
if (!process.argv[2]) {
  const TYPED = "http://192.0.2.10:8080/v1";
  const fields = { "ep-base-url": TYPED, "ep-provider": "openai-compatible" };
  const el = (id) => ({
    value: fields[id] ?? "", checked: false,
    classList: { add() {}, remove() {}, toggle() {} },
    setAttribute() {}, getAttribute: () => null, style: {},
    textContent: "", innerHTML: "",
    getBoundingClientRect: () => ({ width: 0, height: 0 }),
    getContext: () => null, closest: () => null,
  });
  const box = vm.createContext(Object.assign({}, sandbox, {
    document: {
      querySelector: (s) => el(String(s).replace(/^#/, "")),
      querySelectorAll: () => [], addEventListener() {}, body: el("body"),
    },
  }));
  box.globalThis = box;
  vm.runInContext(script[1], box);
  const stored = `{config:{endpoint:{base_url:"",provider:"openai-compatible",model:""}}}`;
  vm.runInContext(`cp.tab="setup"; cp.server=${stored};`, box);
  vm.runInContext(
    `renderTab=()=>{}; notice=()=>{}; setPaneBusy=()=>{};` +
    `refreshServerState=async()=>{ cp.server=${stored}; };`, box);
  const kept = await vm.runInContext(`(async()=>{
    captureSetupDraft();
    await act("setup","test", async()=>({ok:true}), "tested");
    return cp.server.config.endpoint.base_url;
  })()`, box);
  const ok = kept === TYPED;
  console.log(`${ok ? "PASS" : "FAIL"}  a successful test keeps the URL being typed`);
  if (!ok) failed++;
}

// ---- live feed -------------------------------------------------------------
if (!process.argv[2]) {
  const feedState = (extra) => JSON.stringify(Object.assign({
    active: true, run_id: "r", task: "some-task", harness: "hermes",
    harness_label: "Hermes Agent", model_label: "Test Model", phase: "agent",
    elapsed_s: 2700, log_bytes: 198939, silent_s: 3,
    entries: [
      { kind: "assistant", text: "Let me compute the P95 latency first." },
      { kind: "tool", text: "→ bash\npython solve.py" },
    ],
  }, extra));

  vm.runInContext(`state.activity = ${feedState({})};`, ctx);
  const live = run("live feed (streaming)", "renderFeed()", 200);
  const streaming = /streaming/.test(live) && /feed-cursor/.test(live);
  console.log(`${streaming ? "PASS" : "FAIL"}  feed shows streaming + cursor`);
  if (!streaming) failed++;

  // The feed scrolls inside `.feed`, not inside `.panel-body`. mountExpanded
  // finds it with querySelector(".feed") to drive the stick-to-newest logic,
  // and the expanded feed sat wherever it opened when that lookup was pointed
  // at the body instead. A rename here would break it silently, so the marker
  // the lookup depends on is asserted rather than assumed.
  const hasTail = /class="feed"/.test(live)
    && live.indexOf('class="feed"') > live.indexOf('class="panel-body"');
  console.log(`${hasTail ? "PASS" : "FAIL"}  the feed's own scroller is inside the panel body`);
  if (!hasTail) failed++;

  // A quiet feed must explain itself rather than looking like a hang.
  vm.runInContext(`state.activity = ${feedState({ silent_s: 600 })};`, ctx);
  const quiet = run("live feed (quiet)", "renderFeed()", 200);
  const explained = /no new output/.test(quiet) && /flushes one line/.test(quiet);
  console.log(`${explained ? "PASS" : "FAIL"}  quiet feed explains the buffering`);
  if (!explained) failed++;

  // A trial that has not reached the agent yet reports no agent clock at all.
  // It must say it is setting up rather than show a zero, which would claim the
  // model had been working for no time instead of that it had not started --
  // and it must never present the setup interval as elapsed model time, which
  // is what charging image pulls and harness installs to the harness looked
  // like on screen.
  vm.runInContext(
    `state.activity = ${feedState({ elapsed_s: null, setup_s: 214, phase: "setting up" })};`,
    ctx
  );
  const settingUp = vm.runInContext("renderFeed()", ctx);
  const named = /setting up/.test(settingUp) && !/elapsed/.test(settingUp);
  console.log(`${named ? "PASS" : "FAIL"}  setup time is labelled setup, not elapsed`);
  if (!named) failed++;

  // And once the agent starts, the setup it cost is still reachable rather than
  // silently dropped -- it is the number that explains the gap between a run's
  // wall clock and its model time.
  vm.runInContext(
    `state.activity = ${feedState({ elapsed_s: 1320, setup_s: 214 })};`, ctx
  );
  const running = vm.runInContext("renderFeed()", ctx);
  const both = /elapsed/.test(running) && /Setup took/.test(running);
  console.log(`${both ? "PASS" : "FAIL"}  the agent clock names what it excludes`);
  if (!both) failed++;

  /* ---- one tab per concurrent trial ------------------------------------ *
   *
   * At --n-concurrent 4 there are four trials in flight and the feed used to
   * show whichever log had been written to most recently, discarding the other
   * three -- and swapping between them from poll to poll, since which log is
   * newest changes every few seconds. Observed on a live run: four trials
   * running, one visible, no way to reach the rest.
   */
  const trial = (task, extra) => Object.assign({
    key: `run1/${task}__aaa`, run_id: "run1", task,
    harness: "hermes", harness_label: "Hermes Agent", model_label: "Test Model",
    phase: "agent", elapsed_s: 600, setup_s: 12, log_bytes: 1234,
    silent_s: 2, entries: [],
  }, extra);

  const four = [
    trial("write-compressor"),
    trial("winning-avg-corewars", { silent_s: 600 }),
    trial("dna-assembly", { phase: "setting up", elapsed_s: null, log_bytes: 0 }),
    trial("merge-diff-arc-agi-task"),
  ];

  // A single trial must NOT grow a tab strip: one tab is a control that cannot
  // do anything, which reads as broken rather than as nothing to switch to.
  vm.runInContext(
    `state.feedTrial = null; state.activity = ${feedState({
      key: "run1/some-task__aaa", trials: [trial("some-task")],
    })};`, ctx);
  const lone = vm.runInContext("renderFeed()", ctx);
  const noStrip = !/feed-tabs/.test(lone);
  console.log(`${noStrip ? "PASS" : "FAIL"}  a lone trial grows no tab strip`);
  if (!noStrip) failed++;

  vm.runInContext(
    `state.feedTrial = null; state.activity = ${feedState({
      key: "run1/write-compressor__aaa", task: "write-compressor", trials: four,
    })};`, ctx);
  const many = run("live feed (4 concurrent)", "renderFeed()", 200);

  const everyTab = four.every((t) => many.includes(`data-feed-trial="${t.key}"`));
  console.log(`${everyTab ? "PASS" : "FAIL"}  every running trial gets a tab`);
  if (!everyTab) failed++;

  // Exactly one selected, and it is the one whose tail is being shown. Two
  // selected tabs, or none, is how a strip stops explaining which feed you are
  // reading.
  const selected = (many.match(/aria-selected="true"/g) || []).length;
  const rightOne = selected === 1
    && /data-feed-trial="run1\/write-compressor__aaa"\s+aria-selected="true"/.test(many);
  console.log(`${rightOne ? "PASS" : "FAIL"}  exactly the shown trial is selected`);
  if (!rightOne) failed++;

  // The strip is the only place a quiet or still-building trial is visible once
  // another trial is selected, so its state has to survive being unselected.
  const dots = /<i class="quiet">/.test(many) && /<i class="setup">/.test(many);
  console.log(`${dots ? "PASS" : "FAIL"}  tabs carry quiet and setting-up state`);
  if (!dots) failed++;

  const counted = /4 trials running/.test(many);
  console.log(`${counted ? "PASS" : "FAIL"}  the panel note counts the trials`);
  if (!counted) failed++;

  // Unpinned it follows the newest and says so; pinned it offers the way back.
  const followsByDefault = /following newest/.test(many)
    && !/data-feed-trial=""/.test(many);
  console.log(`${followsByDefault ? "PASS" : "FAIL"}  unpinned feed says it follows newest`);
  if (!followsByDefault) failed++;

  vm.runInContext(`state.feedTrial = "run1/dna-assembly__aaa";`, ctx);
  const pinned = vm.runInContext("renderFeed()", ctx);
  const canUnpin = /data-feed-trial=""/.test(pinned) && /follow newest/.test(pinned);
  console.log(`${canUnpin ? "PASS" : "FAIL"}  a pinned feed offers follow-newest`);
  if (!canUnpin) failed++;

  // A task name repeats among the trials in flight two legitimate ways:
  // --n-attempts 2 runs one task twice inside a run, and two benchmarks in
  // flight share a task list. Tabs labelled identically can be selected but not
  // told apart, which is the same confusion the strip was added to remove.
  const twins = [
    trial("merge-diff-arc-agi-task", { key: "run1/merge-diff-arc-agi-task__aaa" }),
    trial("merge-diff-arc-agi-task", { key: "run2/merge-diff-arc-agi-task__zzz" }),
    trial("write-compressor", { key: "run1/write-compressor__bbb" }),
  ];
  vm.runInContext(
    `state.feedTrial = null; state.activity = ${feedState({
      key: "run1/merge-diff-arc-agi-task__aaa", task: "merge-diff-arc-agi-task",
      trials: twins,
    })};`, ctx);
  const dup = vm.runInContext("renderFeed()", ctx);
  // The colliding pair is qualified by the trial's own suffix, which is unique
  // in both cases; the name that does not collide is left alone.
  const qualified = /merge-diff-arc-agi-task · aaa/.test(dup)
    && /merge-diff-arc-agi-task · zzz/.test(dup);
  console.log(`${qualified ? "PASS" : "FAIL"}  colliding tab labels are qualified`);
  if (!qualified) failed++;

  const untouched = dup.includes("</i>write-compressor")
    && !dup.includes("write-compressor ·");
  console.log(`${untouched ? "PASS" : "FAIL"}  a unique label is left unqualified`);
  if (!untouched) failed++;

  // Still three separately selectable tabs, one selected.
  const three = (dup.match(/data-feed-trial="run/g) || []).length === 3
    && (dup.match(/aria-selected="true"/g) || []).length === 1;
  console.log(`${three ? "PASS" : "FAIL"}  every twin stays separately selectable`);
  if (!three) failed++;

  // A pinned trial that finished is the case where the feed legitimately shows
  // something other than what was asked for. Silence there is indistinguishable
  // from the panel mislabelling its own contents.
  vm.runInContext(
    `state.activity = ${feedState({
      key: "run1/write-compressor__aaa", task: "write-compressor",
      trials: four, selected_finished: true,
    })};`, ctx);
  const moved = vm.runInContext("renderFeed()", ctx);
  const saidSo = /finished/.test(moved) && /write-compressor/.test(moved);
  console.log(`${saidSo ? "PASS" : "FAIL"}  a finished pin says the feed moved`);
  if (!saidSo) failed++;

  // The pin has to reach the server, or every tab shows the same trial.
  vm.runInContext(`state.feedTrial = "run1/a b__x";`, ctx);
  const pinnedUrl = vm.runInContext("activityUrl()", ctx);
  const encoded = pinnedUrl === "/api/activity?trial=run1%2Fa%20b__x";
  console.log(`${encoded ? "PASS" : "FAIL"}  the pinned trial is sent url-encoded`);
  if (!encoded) failed++;

  vm.runInContext("state.feedTrial = null;", ctx);
  const plainUrl = vm.runInContext("activityUrl()", ctx);
  const bare = plainUrl === "/api/activity";
  console.log(`${bare ? "PASS" : "FAIL"}  unpinned asks for no particular trial`);
  if (!bare) failed++;

  // Older payloads have no `trials` at all. The panel predates the strip and
  // must not start throwing on a server that has not been restarted yet.
  vm.runInContext(
    `state.feedTrial = null; state.activity = ${feedState({})};`, ctx);
  const legacy = run("live feed (no trials key)", "renderFeed()", 200);
  const survives = !/feed-tabs/.test(legacy) && /some-task/.test(legacy);
  console.log(`${survives ? "PASS" : "FAIL"}  a payload with no trials list still renders`);
  if (!survives) failed++;

  // No run in flight -> no panel at all, not an empty box.
  vm.runInContext(`state.activity = {"active": false};`, ctx);
  const off = vm.runInContext("renderFeed()", ctx);
  console.log(`${off === "" ? "PASS" : "FAIL"}  no feed panel when idle`);
  if (off !== "") failed++;

  vm.runInContext("state.activity = null;", ctx);
  const nul = vm.runInContext("renderFeed()", ctx);
  console.log(`${nul === "" ? "PASS" : "FAIL"}  feed tolerates a failed fetch`);
  if (nul !== "") failed++;
}

vm.runInContext(
  "state.data = {runs:[],task_names:[],models:[],comparisons:[]," +
  "summary:{n_runs:0,n_running:0,n_models:0,n_harnesses:0}}; state.model=null;",
  ctx
);
run("empty dataset", `renderRunLog(state.data) + renderLeaderboard([]) + renderMatrix(state.data, [])`);

/* ------------------------- control plane tabs ---------------------------- */

/* The tabs that configure and launch runs. These render from GET /api/state,
 * so the fixture below is that payload's shape. What is being pinned is that
 * every tab renders without a server, on partial state, and -- the one that
 * actually matters -- that a stored API key is never written into the page.
 */
const SERVER_STATE = {
  read_only: false,
  config: {
    endpoint: {
      provider: "openai-compatible", base_url: "http://localhost:8080/v1",
      api_key_env: "", api_key: "***redacted***", model: "", label: "",
    },
    dashboard: { host: "127.0.0.1", port: 8420, open_browser: true },
    runs_dir: "runs", record_hostname: false,
  },
  config_path: "/repo/config.yaml", config_exists: true, api_key_set: true,
  providers: [
    { id: "openai-compatible", label: "OpenAI-compatible endpoint",
      default_base_url: "http://localhost:8080/v1", default_api_key_env: "OPENAI_API_KEY",
      requires_api_key: false, model_is_discoverable: true, default_agent_concurrency: 1 },
    { id: "openrouter", label: "OpenRouter",
      default_base_url: "https://openrouter.ai/api/v1", default_api_key_env: "OPENROUTER_API_KEY",
      requires_api_key: true, model_is_discoverable: false, default_agent_concurrency: 4 },
  ],
  harnesses: {
    hermes: { label: "Hermes Agent", vendor: "Nous Research", agent: "harnesses.hermes:Hermes",
              model_ref: "local/{model_id}", agent_kwargs: { base_url: "{base_url}" } },
    omp: { label: "oh-my-pi", vendor: "can1357", agent: "harnesses.omp:Omp",
           model_ref: "local/{model_id}" },
  },
  defaults: { dataset: "terminal-bench@2.0", n_concurrent: 2, n_concurrent_agents: 1,
              n_attempts: 1, agent_timeout_multiplier: 16.0,
              environment_build_timeout_multiplier: 4.0 },
  editable_defaults: ["agent_timeout_multiplier", "n_concurrent"],
  datasets: [
    { id: "terminal-bench@2.0", label: "Terminal-Bench 2", tasks: 89,
      image_repo: "alexgshaw", image_tag: "20251031" },
    { id: "aider/aider-polyglot", label: "Aider Polyglot", tasks: 225 },
  ],
  subsets: ["stratified-25"],
  // Which benchmark each subset's task names came from. A subset is a list of
  // names, so it only means anything on the dataset it was drawn from.
  subset_datasets: { "stratified-25": "terminal-bench@2.0" },
  supervisor: { active: false, job: null },
  runs_dir: "/repo/runs",
};

Object.assign(ctx, { SRV: SERVER_STATE });
vm.runInContext("cp.server = SRV; state.data = SEED;", ctx);

run("setup tab", "renderSetup()", 800);
run("run tab", "renderRun()", 800);
{
  const html = vm.runInContext("renderRun()", ctx);
  const checks = [
    ["benchmark dropdown renders", /id="run-dataset"/.test(html)],
    ["  lists a catalog entry", /aider\/aider-polyglot/.test(html)],
    ["  shows the task count", /225 tasks/.test(html)],
    ["  preselects the catalog default",
     /value="terminal-bench@2\.0"\s+selected/.test(html)],

    // How much of the benchmark to run is one exclusive choice. It used to be
    // two independent controls joined by the word "or" in a label, while
    // runSpecFromForm sent both and the runner honoured both -- so a run could
    // go out as "subset stratified-25" and be silently capped at N tasks, then
    // be recorded under the subset's name where the results view would pair it
    // against real subset runs.
    ["task scope is a single exclusive choice",
     (html.match(/name="run-scope"/g) || []).length === 3],
    ["  defaulting to the whole benchmark",
     /value="full"[^>]*checked/.test(html)],
    ["  with the other two inert until chosen",
     /id="run-ntasks"[^>]*disabled/.test(html) && /id="run-subset"[^>]*disabled/.test(html)],

    // A subset carries the benchmark it belongs to, so the gating survives a
    // change of benchmark -- which does not re-render this form.
    ["a subset declares which benchmark it belongs to",
     /data-subset-dataset="terminal-bench@2\.0"/.test(html)],
    ["  and is offered for that benchmark",
     /value="stratified-25" data-subset-dataset="terminal-bench@2\.0">/.test(html)],

    // The most consequential fact about a run was absent from the page that
    // starts one, while the results view treats the model as its primary scope.
    ["the page names the model it will benchmark",
     /localhost:8080\/v1/.test(html)],

    // Every checkbox on this tab was being drawn by the results-filter pill
    // rule, which was never scoped to the filters.
    ["harnesses are a labelled group", /<fieldset>/.test(html)],
  ];
  for (const [label, ok] of checks) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}
{
  // A catalog with no datasets must still render a usable form rather than an
  // empty select the user cannot start a run from.
  vm.runInContext("cp.server.datasets = [];", ctx);
  const html = vm.runInContext("renderRun()", ctx);
  const ok = /id="run-dataset"/.test(html) && /catalog default/.test(html);
  console.log(`${ok ? "PASS" : "FAIL"}  empty dataset catalog falls back cleanly`);
  if (!ok) failed++;
  vm.runInContext("cp.server.datasets = SRV.datasets;", ctx);
}
{
  // The payload the control plane receives is the contract the Run tab's layout
  // is not allowed to change. Reorganising a form is presentation; quietly
  // adding or dropping a key the server acts on is not.
  //
  // With the DOM stubbed, no radio reads as checked, so this also pins the
  // default: "full" -- no subset, no task cap. The bug this replaces did the
  // opposite, sending whatever both boxes happened to hold.
  const spec = vm.runInContext("runSpecFromForm()", ctx);
  const keys = ["harnesses", "dataset", "subset", "agent_timeout_multiplier",
                "n_concurrent", "n_concurrent_agents", "allow_hosts",
                "debug_capture"];
  const shape = keys.every((k) => k in spec);
  console.log(`${shape ? "PASS" : "FAIL"}  the run payload keeps every key the server reads`);
  if (!shape) failed++;
  const exclusive = spec.subset === null && !("n_tasks" in spec);
  console.log(`${exclusive ? "PASS" : "FAIL"}  ...and never sends a subset and a task cap together`);
  if (!exclusive) failed++;
}

run("harnesses tab", "renderHarnesses()", 800);
run("maintenance tab", "renderMaintenance()", 800);

// The editor is a separate render path and is where a harness gets defined.
vm.runInContext("cp.editing = 'hermes';", ctx);
run("harness editor (existing)", "renderHarnesses()", 1200);
vm.runInContext("cp.editing = '+';", ctx);
run("harness editor (new)", "renderHarnesses()", 1200);
vm.runInContext("cp.editing = null;", ctx);

// A provider that needs a key and cannot report its model renders different
// hints and a required model field.
vm.runInContext("cp.server.config.endpoint.provider = 'openrouter';", ctx);
const or = run("setup tab (openrouter)", "renderSetup()", 800);
{
  const ok = /required/.test(or);
  console.log(`${ok ? "PASS" : "FAIL"}  openrouter marks the model required`);
  if (!ok) failed++;
}
vm.runInContext("cp.server.config.endpoint.provider = 'openai-compatible';", ctx);

// The leak that would matter most: a key rendered into the DOM.
{
  const key = "sk-should-never-appear-0123456789";
  vm.runInContext(`cp.server.config.endpoint.api_key = ${JSON.stringify(key)};`, ctx);
  const html = vm.runInContext("renderSetup()", ctx);
  const leaked = html.includes(key);
  console.log(`${leaked ? "FAIL" : "PASS"}  setup never renders a key into the page`);
  if (leaked) failed++;
  // The password field must not be pre-filled at all, redacted or otherwise.
  const prefilled = /id="ep-key"[^>]*value="(?!")/.test(html);
  console.log(`${prefilled ? "FAIL" : "PASS"}  key field is never pre-filled`);
  if (prefilled) failed++;
  vm.runInContext("cp.server.config.endpoint.api_key = '***redacted***';", ctx);
}

// A run in flight changes which controls exist; both states must render.
vm.runInContext(
  "cp.server.supervisor = { active: true, job: { id: 'j1', kind: 'bench',"
  + " status: 'running', harnesses: ['hermes'], started_at: '2026-08-09T00:00:00+00:00',"
  + " finished_at: null, returncode: null, stopped_reason: null, pid: 1, active: true } };",
  ctx);
const running = run("run tab (active)", "renderRun()", 800);
{
  const ok = /id="btn-stop"/.test(running) && /id="btn-start"[^>]*disabled/.test(running);
  console.log(`${ok ? "PASS" : "FAIL"}  active run offers stop and blocks start`);
  if (!ok) failed++;
}
run("maintenance (active run)", "renderMaintenance()", 800);
vm.runInContext("cp.server.supervisor = { active: false, job: null };", ctx);

// Console output classification, including a line that must not look like an error.
vm.runInContext(
  "cp.log = ['===== header =====', '  $ harbor run --dataset x',"
  + " '  [!] hermes exited 1', 'warning: slow', 'ordinary line'];", ctx);
const logHtml = run("console lines", "renderLogLines()", 80);
{
  const ok = /class="ln rule"/.test(logHtml) && /class="ln bad"/.test(logHtml)
    && /class="ln warn"/.test(logHtml);
  console.log(`${ok ? "PASS" : "FAIL"}  console classifies rule/error/warning lines`);
  if (!ok) failed++;
}
vm.runInContext("cp.log = [];", ctx);

// The failure box above the console. Harbor's stdout says a command failed and
// what it was classified as; it never quotes the agent's error, so a run whose
// every trial dies at its first request showed an unexplained console. The
// trial's own result.json is the only place the reason exists, and the page is
// handed it with the log.
{
  const TRIAL = JSON.stringify({
    run: "claude-code__m__tb2__stratified-25__20260817T145602Z",
    trial: "write-compressor__7xfeYgt",
    error_type: "UnknownApiError",
    message: "Command failed (exit 1): claude --print\n[...]\n"
      + "Error: Jinja Exception: System message must be at the beginning.",
    finding: {
      id: "system-message-position", ok: false, severity: "fail",
      title: "The model's chat template rejects a system message that is not first",
      detail: "Claude Code sends its agent and skill listings as a system-role message.",
      fixes: ["Patch the template: harness-arena template-fix"],
      docs: "docs/TROUBLESHOOTING.md",
    },
  });
  vm.runInContext(`cp.failure = null; cp.trialFailure = ${TRIAL};`, ctx);
  const box = vm.runInContext("renderFailure()", ctx);
  const named = /chat template rejects a system message/.test(box);
  const fix = /harness-arena template-fix/.test(box);
  const quoted = /Jinja Exception/.test(box);
  const trial = /write-compressor/.test(box);
  const ok = named && fix && quoted && trial;
  console.log(`${ok ? "PASS" : "FAIL"}  a trial failure is named, quoted and fixable`
    + ` (named=${named} fix=${fix} quoted=${quoted} trial=${trial})`);
  if (!ok) failed++;

  // The same finding must not be printed twice when the console already
  // produced it -- the two sources agree far more often than they disagree.
  vm.runInContext(
    "cp.failure = { id: 'system-message-position', ok: false, severity: 'fail',"
    + " title: 'T', detail: 'd', fixes: ['f'], docs: '' };", ctx);
  const once = vm.runInContext("renderFailure()", ctx);
  const dup = (once.match(/chat template rejects a system message/g) || []).length;
  console.log(`${dup === 0 ? "PASS" : "FAIL"}  a finding the console already named is not repeated`);
  if (dup !== 0) failed++;

  // Nothing wrong, nothing rendered.
  vm.runInContext("cp.failure = null; cp.trialFailure = null;", ctx);
  const empty = vm.runInContext("renderFailure()", ctx);
  console.log(`${empty === "" ? "PASS" : "FAIL"}  no failure box when nothing has failed`);
  if (empty !== "") failed++;
}

// The matrix cell for an errored trial has to carry the same explanation: the
// exception type alone ("UnknownApiError") names nothing anyone can act on.
if (!process.argv[2]) {
  vm.runInContext(
    "state.data.runs[0].tasks[0].error_type = 'UnknownApiError';"
    + "state.data.runs[0].tasks[0].resolved = false;"
    + "state.data.runs[0].tasks[0].error_message = 'Error: Jinja Exception: "
    + "System message must be at the beginning.';"
    + "state.data.runs[0].tasks[0].error_finding = { id: 'system-message-position',"
    + " title: 'The chat template rejects a non-first system message',"
    + " fix: 'harness-arena template-fix' };", ctx);
  // Earlier cases narrow the view; this one is about what a cell says, so
  // restore the scope and filters the page starts from rather than inheriting
  // whatever was last selected.
  vm.runInContext(
    "state.model = (state.data.models[0] || {}).fingerprint"
    + " || (state.data.models[0] || {}).label;"
    + "state.dataset = 'terminal-bench@2.0';"
    + "state.hidden = new Set(); state.hidePartial = false;"
    + "state.runPick = null; state.diffOnly = false;", ctx);
  const mx = vm.runInContext(`renderMatrix(state.data, ${V})`, ctx);
  const named = /chat template rejects a non-first system message/.test(mx);
  const fix = /harness-arena template-fix/.test(mx);
  const quoted = /Jinja Exception/.test(mx);
  const ok = named && fix && quoted;
  console.log(`${ok ? "PASS" : "FAIL"}  an errored cell explains itself`
    + ` (named=${named} fix=${fix} quoted=${quoted})`);
  if (!ok) failed++;
}

// Empty and partial state: a fresh checkout has no harnesses and no subsets, and
// that is exactly when someone is most likely to be looking at these tabs.
vm.runInContext(
  "cp.server = { ...SRV, harnesses: {}, subsets: [], defaults: {},"
  + " supervisor: { active: false, job: null } }; state.data = { runs: [] };", ctx);
{
  const html = vm.runInContext("renderRun()", ctx);
  const ok = /none defined/.test(html);
  console.log(`${ok ? "PASS" : "FAIL"}  a checkout with no subsets says so`);
  if (!ok) failed++;
}
run("run tab (no harnesses)", "renderRun()", 200);
run("harnesses tab (empty)", "renderHarnesses()", 200);
run("maintenance (no runs)", "renderMaintenance()", 200);

// No server state at all -- the first paint before /api/state returns.
vm.runInContext("cp.server = null;", ctx);
run("setup tab (no state)", "renderSetup()", 10);
run("run tab (no state)", "renderRun()", 10);
run("harnesses tab (no state)", "renderHarnesses()", 10);
run("maintenance tab (no state)", "renderMaintenance()", 10);

/* ---------------------------- model field -------------------------------- */

/* The field adapts to the catalog: a plain box before anything is fetched, a
 * real dropdown for a server with a handful of models, and a typeahead for an
 * aggregator with hundreds. Each shape is a separate branch, so each is
 * rendered here. */
{
  vm.runInContext("cp.server = JSON.parse(JSON.stringify(SRV)); cp.models = null;", ctx);
  const LOCAL = "cp.server.providers[0]";
  const HOSTED = "cp.server.providers[1]";

  const bare = run("model field (catalog not loaded)",
    `renderModelField({model:""}, ${LOCAL})`, 100);
  {
    const ok = /<input[^>]*id="ep-model"/.test(bare) && !/<select[^>]*id="ep-model"/.test(bare);
    console.log(`${ok ? "PASS" : "FAIL"}  unloaded catalog stays a text field`);
    if (!ok) failed++;
  }

  vm.runInContext(
    "cp.models = { models: [{id:'my-model', name:'my-model', context_length: 131072}],"
    + " default: 'my-model', discoverable: true };", ctx);
  const small = run("model field (small catalog)",
    `renderModelField({model:""}, ${LOCAL})`, 100);
  {
    const isSelect = /<select[^>]*id="ep-model"/.test(small);
    console.log(`${isSelect ? "PASS" : "FAIL"}  a short catalog renders a dropdown`);
    if (!isSelect) failed++;
    // The served model is what should be pre-selected; a local endpoint should
    // need no decision from the user at all.
    const preselected = /value="my-model" selected/.test(small)
      || /auto-detect \(my-model\)/.test(small);
    console.log(`${preselected ? "PASS" : "FAIL"}  the served model is offered by default`);
    if (!preselected) failed++;
  }

  const chosen = run("model field (explicit choice)",
    `renderModelField({model:"my-model"}, ${LOCAL})`, 100);
  {
    const ok = /value="my-model" selected/.test(chosen);
    console.log(`${ok ? "PASS" : "FAIL"}  an explicit choice stays selected`);
    if (!ok) failed++;
  }

  vm.runInContext(
    "cp.models = { models: Array.from({length: 300}, (_, i) =>"
    + " ({id: 'vendor/model-' + i, name: 'Model ' + i, context_length: 32768})),"
    + " default: '', discoverable: false };", ctx);
  const big = run("model field (300-model catalog)",
    `renderModelField({model:""}, ${HOSTED})`, 500);
  {
    const isTypeahead = /<datalist id="model-options">/.test(big)
      && /list="model-options"/.test(big);
    console.log(`${isTypeahead ? "PASS" : "FAIL"}  a huge catalog becomes a typeahead`);
    if (!isTypeahead) failed++;
    const noSelect = !/<select[^>]*id="ep-model"/.test(big);
    console.log(`${noSelect ? "PASS" : "FAIL"}  and not a 300-option dropdown`);
    if (!noSelect) failed++;
  }
  vm.runInContext("cp.models = null; cp.probe = null;", ctx);
}

/* The label field's placeholder must show the name you actually get by leaving
 * it empty -- which is not the derived name once one has been pinned. */
{
  vm.runInContext(
    "cp.server = JSON.parse(JSON.stringify(SRV));"
    + "cp.probe = { ok: true, description: 'x', suggested_label: 'Derived Name',"
    + "             effective_label: 'Pinned Name' };", ctx);
  const html = vm.runInContext("renderSetup()", ctx);
  const showsEffective = /id="ep-label"[^>]*placeholder="Pinned Name"/.test(html);
  console.log(`${showsEffective ? "PASS" : "FAIL"}  label placeholder shows the effective name`);
  if (!showsEffective) failed++;
  vm.runInContext("cp.probe = null;", ctx);
}

/* --------------------- action / form-read ordering ------------------------ */

/* The regression this pins cost a user their configuration silently.
 *
 * act() used to re-render the tab before invoking fn(). renderTab() replaces
 * the pane's innerHTML, so the inputs the user had just filled in were
 * destroyed and rebuilt from stored state -- and fn(), which reads the form,
 * then read the *rebuilt* fields. Every save and every connection test
 * submitted the previous values, or empty ones the server resolved to provider
 * defaults. Nothing errored. The form simply snapped back and the endpoint you
 * typed was never contacted.
 *
 * The contract: fn() observes the DOM before anything re-renders it.
 */
{
  vm.runInContext(
    "globalThis.__order = [];" +
    "globalThis.__formValue = 'typed-by-user';" +
    "renderTab = function () { __order.push('render'); __formValue = 'REBUILT'; };" +
    "globalThis.__done = act('setup', 'probe'," +
    "  async () => { __order.push('read:' + __formValue); return {}; }, 'ok');",
    ctx
  );
  await ctx.__done;
  const order = ctx.__order;

  const readFirst = order[0] === "read:typed-by-user";
  console.log(`${readFirst ? "PASS" : "FAIL"}  act() reads the form before re-rendering`);
  if (!readFirst) {
    failed++;
    console.log(`      order was: ${JSON.stringify(order)}`);
  }

  const rendersAfter = order.includes("render");
  console.log(`${rendersAfter ? "PASS" : "FAIL"}  act() still re-renders when finished`);
  if (!rendersAfter) failed++;
}

/* A failed save or test must not discard what the user typed: the tab
 * re-renders from cp.server, so the draft has to be folded back into it. */
{
  const FORM = {
    "#ep-provider": { value: "openrouter" },
    "#ep-base-url": { value: "https://example.invalid/api/v1" },
    "#ep-key-env": { value: "MY_KEY_VAR" },
    "#ep-key": { value: "sk-typed-secret-value-1234" },
    "#ep-model": { value: "vendor/model" },
    "#ep-label": { value: "My Label" },
    "#ep-context": { value: "65536" },
  };
  const prev = ctx.document.querySelector;
  ctx.document.querySelector = (sel) => FORM[sel] || stub;

  vm.runInContext("cp.server = JSON.parse(JSON.stringify(SRV));", ctx);
  const captured = vm.runInContext("captureSetupDraft()", ctx);
  const kept = vm.runInContext("cp.server.config.endpoint", ctx);
  ctx.document.querySelector = prev;

  const preserved = kept.base_url === "https://example.invalid/api/v1"
    && kept.provider === "openrouter" && kept.model === "vendor/model"
    && kept.context_window === "65536";
  console.log(`${preserved ? "PASS" : "FAIL"}  typed values survive a re-render`);
  if (!preserved) failed++;

  const sent = captured.base_url === "https://example.invalid/api/v1"
    && captured.api_key === "sk-typed-secret-value-1234"
    // The window is handed to every harness, so losing it on a re-render
    // would silently change what the next run measures.
    && captured.context_window === "65536";
  console.log(`${sent ? "PASS" : "FAIL"}  the captured payload is what gets sent`);
  if (!sent) failed++;

  // The draft is rendered back into the page; the key must never ride along.
  const keyLeaked = JSON.stringify(kept).includes("sk-typed-secret-value-1234");
  console.log(`${keyLeaked ? "FAIL" : "PASS"}  the key is not kept in render state`);
  if (keyLeaked) failed++;
}

/* The whole results view is rebuilt on every 5s poll. Without capture/restore
 * that throws the reader back to the top of an 89-row matrix every five
 * seconds, which makes the panel unusable for exactly the long runs it exists
 * to read. Both axes: the matrix scrolls sideways once harnesses pile up. */
{
  const panes = {
    mx: { scrollTop: 420, scrollLeft: 60, getAttribute: () => "mx" },
    lb: { scrollTop: 0, scrollLeft: 0, getAttribute: () => "lb" },
  };
  const prevAll = ctx.document.querySelectorAll;
  const prevOne = ctx.document.querySelector;
  ctx.document.querySelectorAll = (sel) =>
    sel === "[data-pane]" ? Object.values(panes) : [];
  ctx.document.querySelector = (sel) => {
    const hit = /^\[data-pane="(.+)"\]$/.exec(sel);
    return hit ? panes[hit[1]] || null : stub;
  };

  const saved = vm.runInContext("capturePaneScroll()", ctx);
  const onlyScrolled = saved.size === 1 && saved.has("mx");

  // Simulate the re-render: fresh nodes, scrolled back to the top.
  panes.mx.scrollTop = 0;
  panes.mx.scrollLeft = 0;
  ctx.SAVED = saved;
  vm.runInContext("restorePaneScroll(SAVED)", ctx);

  ctx.document.querySelectorAll = prevAll;
  ctx.document.querySelector = prevOne;

  const restored = panes.mx.scrollTop === 420 && panes.mx.scrollLeft === 60;
  console.log(`${restored ? "PASS" : "FAIL"}  a scrolled panel keeps its place across a re-render`);
  if (!restored) failed++;

  console.log(`${onlyScrolled ? "PASS" : "FAIL"}  an unscrolled panel is not tracked`);
  if (!onlyScrolled) failed++;

  // A panel that vanished between renders must not throw and take the whole
  // page down with it.
  ctx.document.querySelector = () => null;
  let survived = true;
  try {
    vm.runInContext("restorePaneScroll(SAVED)", ctx);
  } catch {
    survived = false;
  }
  ctx.document.querySelector = prevOne;
  console.log(`${survived ? "PASS" : "FAIL"}  a panel that disappeared is skipped, not thrown on`);
  if (!survived) failed++;
}

/* index.html is re-read from disk on every request; the server process is not.
 * A dashboard left running across an upgrade renders new controls backed by old
 * code. A control that silently does nothing is worse than one that is missing,
 * because the run starts believing it is recording evidence that it is not. */
{
  const withCaps = (caps) => {
    vm.runInContext(`cp.server = Object.assign({}, cp.server || {}, {capabilities: ${JSON.stringify(caps)}});`, ctx);
    return vm.runInContext(`can("debug_capture")`, ctx);
  };

  const newServer = withCaps(["debug_capture"]);
  console.log(`${newServer ? "PASS" : "FAIL"}  a capability the server advertises is usable`);
  if (!newServer) failed++;

  const oldServer = withCaps([]);
  console.log(`${!oldServer ? "PASS" : "FAIL"}  a capability it does not advertise is refused`);
  if (oldServer) failed++;

  // The case that actually happened: a server old enough to send no list at all.
  vm.runInContext("delete cp.server.capabilities;", ctx);
  const missing = vm.runInContext(`can("debug_capture")`, ctx);
  console.log(`${!missing ? "PASS" : "FAIL"}  a server with no capability list refuses everything`);
  if (missing) failed++;
}

/* The leftover panel. Containers and networks are reported by different server
 * versions, so the panel must never assert a count the server never sent --
 * "0 orphaned networks" from a server that cannot count them is a false claim,
 * and this is the panel someone consults before starting a six-hour run. */
{
  const render = (orphans, caps) => {
    vm.runInContext(
      `cp.orphans = ${JSON.stringify(orphans)};`
      + `cp.server = Object.assign({}, cp.server || {}, {capabilities: ${JSON.stringify(caps)}});`,
      ctx);
    // Collapse whitespace after stripping tags: "<b>2</b> orphaned" leaves a
    // double space where the tag was, which no assertion should have to know.
    return String(vm.runInContext("renderOrphans()", ctx))
      .replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  };
  const BOTH = ["orphan_counts", "network_reaping"];

  const unchecked = render(null, BOTH);
  console.log(`${unchecked === "" ? "PASS" : "FAIL"}  nothing is claimed before the check is run`);
  if (unchecked !== "") failed++;

  const clean = render(
    { orphaned: [], orphaned_networks: [], network_pool: { n_bridge_networks: 3, capacity: 32 } },
    BOTH);
  const cleanOk = /Nothing left over/.test(clean) && /3 of ~32/.test(clean);
  console.log(`${cleanOk ? "PASS" : "FAIL"}  a clean host says so, with pool headroom`);
  if (!cleanOk) failed++;

  const dirty = render(
    { orphaned: ["a", "b"], orphaned_networks: ["n1"], network_pool: { n_bridge_networks: 9, capacity: 32 } },
    BOTH);
  const dirtyOk = /2 orphaned container/.test(dirty) && /1 orphaned network/.test(dirty);
  console.log(`${dirtyOk ? "PASS" : "FAIL"}  both kinds of leftover are counted`);
  if (!dirtyOk) failed++;

  // The pool runs dry long before disk does, and the failure it causes never
  // says the word "subnet", so it has to be called out before the run starts.
  const tight = render(
    { orphaned: [], orphaned_networks: ["n1"], network_pool: { n_bridge_networks: 31, capacity: 32, tight: true } },
    BOTH);
  const tightOk = /subnet pool is nearly exhausted/.test(tight);
  console.log(`${tightOk ? "PASS" : "FAIL"}  an exhausted subnet pool is called out`);
  if (!tightOk) failed++;

  // A server that predates the split reports containers only.
  const old = render({ orphaned: ["a"] }, ["debug_capture"]);
  const oldOk = /1 orphaned container/.test(old) && !/network/.test(old);
  console.log(`${oldOk ? "PASS" : "FAIL"}  an older server is not credited with network counts`);
  if (!oldOk) failed++;

  // The dangerous case. A container's name does not say whether it is a
  // leftover: during a run these are the run's own trials, and calling them
  // orphans beside a Remove button is how someone kills their own benchmark.
  const live = render(
    { orphaned: ["a", "b"], orphaned_networks: [], network_pool: {}, run_active: true },
    BOTH);
  const liveOk = /belong to it/.test(live) && !/orphaned container/.test(live);
  console.log(`${liveOk ? "PASS" : "FAIL"}  a running benchmark's containers are not called orphans`);
  if (!liveOk) failed++;
}

/* The run picker. Selecting by name alone stops working the moment a subset is
 * re-run -- the normal case here, since every model swap re-runs everything --
 * so a run is identified by scope *and* start time, and the panels follow it. */
{
  // Earlier blocks drive the control-plane tabs and leave state pointing at
  // their own fixtures, so restore the results view before asserting on it.
  vm.runInContext(
    "state.data = SEED;" +
    "state.model = (SEED.models[0] || {}).fingerprint || (SEED.models[0] || {}).label;" +
    "state.hidden = new Set(); state.hidePartial = false; state.runPick = null;",
    ctx
  );
  const offered = vm.runInContext("runGroups(state.data).flatMap(g => g.runs)", ctx);
  const total = vm.runInContext("visibleRuns(state.data).length", ctx);
  const cases = [
    ["the picker offers the runs there are", offered.length > 0],
    ["no filter shows every run", total === offered.length],
  ];

  vm.runInContext(`state.runPick = new Set(${JSON.stringify([offered[0].run_id])});`, ctx);
  cases.push(["picking one run shows one",
    vm.runInContext("visibleRuns(state.data).length", ctx) === 1]);
  cases.push(["the run log follows the picker",
    (vm.runInContext("renderRunLog(state.data)", ctx).match(/<tr>/g) || []).length - 1 === 1]);

  // An empty set is a deliberate "show nothing" and must not collapse back to
  // "show everything", which would silently contradict the user.
  vm.runInContext("state.runPick = new Set();", ctx);
  cases.push(["selecting none shows none",
    vm.runInContext("visibleRuns(state.data).length", ctx) === 0]);

  vm.runInContext("state.runPick = null;", ctx);
  const picker = vm.runInContext("renderRunPicker(state.data)", ctx);
  cases.push(["the picker labels each run with its start time", /pick-when/.test(picker)]);
  cases.push(["...and groups them by scope name", /data-pick-group=/.test(picker)]);

  const log = vm.runInContext("renderRunLog(state.data)", ctx);
  cases.push(["the run log reports elapsed time", />elapsed</.test(log)]);

  // Partial credit, kept apart from the score. All-or-nothing scoring makes a
  // run that passed most checks on every task identical to one that wrote
  // nothing, and that is the difference this recovers.
  // A sweep is one `bench` invocation across its harnesses -- what "a run"
  // means to whoever started it. Grouping on the subset name instead merges
  // two sweeps of the same subset into one, and then "3 of 7" counts harnesses
  // while the person reading it is counting runs.
  {
    const fp = vm.runInContext("state.model", ctx);
    const mk = (id, batch, subset, started) => ({
      run_id: id, batch_id: batch, harness: "h" + id, harness_label: "H" + id,
      model_fingerprint: fp, model_label: "M", subset, is_partial: true,
      // Sweep grouping is scoped to the selected benchmark like everything else
      // on the page, so these have to sit inside it to be grouped at all.
      dataset: "terminal-bench@2.0",
      started_at: started, n_done: 1, n_total: 1, n_resolved: 0, pass_rate: 0,
      status: "complete", tasks: [],
    });
    const twoSweeps = {
      ...vm.runInContext("SEED", ctx),
      runs: [
        mk("a", "b1", "stratified-25", "2026-08-01T10:00:00+00:00"),
        mk("b", "b1", "stratified-25", "2026-08-01T11:00:00+00:00"),
        mk("c", "b2", "stratified-25", "2026-08-02T10:00:00+00:00"),
      ],
    };
    ctx.TWO = twoSweeps;
    vm.runInContext("state.data = TWO; state.runPick = null;", ctx);
    const gs = vm.runInContext("runGroups(state.data)", ctx);
    cases.push(["two sweeps of one subset stay two groups", gs.length === 2]);
    cases.push(["...split by which sweep they ran in",
      gs.every((g) => new Set(g.runs.map((r) => r.batch_id)).size === 1)]);

    // Runs recorded before sweeps had an identity still group by subset.
    ctx.LEGACY = { ...twoSweeps, runs: twoSweeps.runs.map(({ batch_id, ...r }) => r) };
    vm.runInContext("state.data = LEGACY;", ctx);
    cases.push(["runs with no sweep id fall back to one group per subset",
      vm.runInContext("runGroups(state.data).length", ctx) === 1]);

    vm.runInContext("state.data = SEED;", ctx);
  }

  /* The smoke filter switches itself on as soon as any real run has results,
   * so it can hide a run nobody chose to hide. It did: a fresh --n-tasks run
   * was invisible while three subset runs sat beside it, and the control said
   * so only by being marginally less dim than its own off state. It must now
   * read as switched on, and say what it is holding back. */
  {
    const fp = vm.runInContext("state.model", ctx);
    const smoke = {
      run_id: "s1", batch_id: "b9", harness: "claude-code",
      harness_label: "Claude Code", model_fingerprint: fp, model_label: "M",
      // An ad-hoc "first N" run: partial, and with no named subset. That
      // combination, and only that, is what isSmoke means.
      subset: null, is_partial: true, n_tasks_requested: 5,
      dataset: "terminal-bench@2.0",
      started_at: "2026-08-03T10:00:00+00:00", n_done: 2, n_total: 5,
      n_resolved: 0, pass_rate: 0, status: "running", tasks: [],
    };
    ctx.SMOKE = { ...vm.runInContext("SEED", ctx), runs: [smoke] };
    vm.runInContext(
      "state.data = SMOKE; state.runPick = null; state.hidden = new Set();"
      + "state.hidePartial = true;", ctx
    );

    const on = vm.runInContext("renderFilters(state.data, visibleRuns(state.data))", ctx);
    cases.push(["the smoke filter is a checkbox", /type="checkbox"/.test(on)]);
    cases.push(["...that renders checked when it is on", /\bchecked\b/.test(on)]);
    // The count is the part that would have answered "where did my run go".
    cases.push(["...and says how many runs it is hiding", /1 hidden/.test(on)]);
    cases.push(["...and actually hides them",
      vm.runInContext("visibleRuns(state.data).length", ctx) === 0]);

    vm.runInContext("state.hidePartial = false;", ctx);
    const off = vm.runInContext("renderFilters(state.data, visibleRuns(state.data))", ctx);
    cases.push(["unchecked when off", !/\bchecked\b/.test(off)]);
    // A count while nothing is hidden would be a second wrong signal.
    cases.push(["...and reports no count", !/hidden</.test(off)]);
    cases.push(["...and the run comes back",
      vm.runInContext("visibleRuns(state.data).length", ctx) === 1]);

    // The filter used to switch itself on as soon as any non-smoke run had
    // results, which hid a run nobody chose to hide. Nothing disappears now
    // unless it is asked to. Asserted two ways because the state literal alone
    // would not catch a re-introduced auto-enable elsewhere in render().
    cases.push(["smoke runs are shown by default", /hidePartial:\s*false/.test(html)]);
    // Scoped to render()'s own body: the checkbox handler assigns it too, and
    // that assignment is the whole point of the control.
    const renderStart = html.indexOf("function render()");
    const renderBody = html.slice(renderStart,
      html.indexOf("\nfunction ", renderStart + 1));
    cases.push(["...and nothing switches the filter on by itself",
      vm.runInContext("typeof hasRealRun === 'undefined'", ctx)
      && !/state\.hidePartial\s*=/.test(renderBody)]);

    vm.runInContext("state.data = SEED; state.hidePartial = false;", ctx);
  }

  /* The filter block moved out of a full-width strip and into the grid's first
   * column, above the feed. As a strip it took a band of height off the top of
   * every column, and the two panels that most need height -- the pass rate and
   * the 89-row matrix -- were the ones paying for it. In one column the
   * controls have to stack, and the order is what makes it readable: what you
   * are looking at, then its headline numbers, then what you filter out. */
  {
    const filters = vm.runInContext("renderFilters(state.data, visibleRuns(state.data))", ctx);
    const at = (needle) => filters.indexOf(needle);
    cases.push(["the model selector comes first", at("model-select") > -1]);
    cases.push(["...then the headline tiles",
      at('class="tiles"') > at("model-select")]);
    cases.push(["...then the harness chips",
      at("data-harness") > at('class="tiles"')]);
    cases.push(["...then the run and smoke filters",
      at("partial-toggle") > at("data-harness")]);
    cases.push(["the controls are stacked in rows",
      (filters.match(/class="filter-row"/g) || []).length >= 3]);
  }

  /* Every panel can be opened full-screen, which is the answer to six panels
   * on one page rather than making them draggable. */
  {
    const withExpand = ["renderLeaderboard(visibleRuns(state.data))",
                        "renderMatrix(state.data, visibleRuns(state.data))",
                        "renderRunLog(state.data)"];
    const all = withExpand.every((expr) =>
      /data-expand="/.test(vm.runInContext(expr, ctx)));
    cases.push(["every panel offers an expand control", all]);
    cases.push(["...and is addressable by the modal",
      /data-panel="/.test(vm.runInContext(withExpand[0], ctx))]);
    cases.push(["the expand/close plumbing exists",
      vm.runInContext("typeof mountExpanded === 'function' && typeof closeExpanded === 'function'", ctx)]);

    /* The expanded live feed follows the newest output, but only while the
     * reader is already at the bottom. Scrolled up, a poll must not drag them
     * down -- that is the difference between a live view and one that fights
     * you. The feed is a tail (old lines drop off the top as new arrive), so
     * position is held as distance from the bottom rather than an absolute
     * scrollTop, which means something different after every poll. */
    const pinned = (t, c, h) => vm.runInContext(`tailIsPinned(${t}, ${c}, ${h})`, ctx);
    const topFor = (h, p, f) => vm.runInContext(`tailScrollTop(${h}, ${p}, ${f})`, ctx);

    cases.push(["sitting at the bottom counts as pinned", pinned(900, 100, 1000) === true]);
    cases.push(["...and so does a few pixels short of it", pinned(880, 100, 1000) === true]);
    cases.push(["scrolled up is not pinned", pinned(400, 100, 1000) === false]);

    // Content grew 1000 -> 1400 between polls.
    cases.push(["pinned follows the new bottom", topFor(1400, true, 0) === 1400]);
    cases.push(["...and scrolled up keeps its distance from the tail",
      topFor(1400, false, 600) === 800]);
    // The case that actually bit: 400px from the bottom must stay 400px from
    // the bottom, not jump to it.
    cases.push(["...so the reader is never yanked to the bottom",
      1400 - topFor(1400, false, 400) === 400]);
    cases.push(["a shrunken tail cannot scroll negative",
      topFor(200, false, 600) === 0]);
    // mountExpanded appends the close button to `.panel-head > div:last-child`,
    // so the head has to keep exactly that shape: a title and an actions box.
    const head = vm.runInContext(withExpand[0], ctx)
      .match(/<div class="panel-head">[\s\S]*?<\/div>\s*<\/div>/);
    cases.push(["the panel head keeps its actions container",
      !!head && /<h2>[\s\S]*<div style="display:flex/.test(head[0])]);
  }

  /* Every named area must be a rectangle. A name that appears in two blocks
   * with a gap between is not, and the browser's response is to drop the whole
   * grid-template-areas declaration -- so the layout collapses to a stack, with
   * nothing logged. Checked against the stylesheet because it is a CSS rule
   * rather than a render path, and the header move rewrote both templates. */
  for (const [, label, block] of html.matchAll(
    /(\.grid(?:\[data-feed="off"\])?)\s*\{[^}]*?grid-template-areas:\s*([^;]+);/g
  )) {
    const rows = [...block.matchAll(/"([^"]+)"/g)].map((m) => m[1].trim().split(/\s+/));
    // The narrow-viewport rule sets `grid-template-areas: none`: no areas, so
    // nothing to validate.
    if (!rows.length) continue;
    const width = rows[0].length;
    const names = new Set(rows.flat().filter((n) => n !== "."));
    const bad = [...names].filter((name) => {
      const cells = [];
      rows.forEach((row, y) => row.forEach((cell, x) => { if (cell === name) cells.push([y, x]); }));
      const ys = cells.map((c) => c[0]), xs = cells.map((c) => c[1]);
      const h = Math.max(...ys) - Math.min(...ys) + 1;
      const w = Math.max(...xs) - Math.min(...xs) + 1;
      return h * w !== cells.length;   // a rectangle's area is its cell count
    });
    cases.push([`${label} rows are all the same width`,
      rows.every((r) => r.length === width)]);
    cases.push([`${label} areas are all rectangles`, bad.length === 0]);
    cases.push([`${label} places the filter block`, names.has("head")]);
  }

  /* A timed-out trial still gets verified, so it can carry partial credit. The
   * cell used to say "T" while the tooltip said "1 of 3 checks passed" -- the
   * fact was collected and then thrown away at the point of display. Which of
   * the two you need depends on the question, so the cell carries both. */
  {
    const cell = (task) => vm.runInContext(
      `cellFor({_byTask: {t: ${JSON.stringify(task)}}}, "t")`, ctx);

    const late = cell({ task_name: "t", resolved: false, error_type: "AgentTimeoutError",
                        n_checks: 3, n_checks_passed: 1 });
    cases.push(["a timeout shows what it had passed", late.glyph === "T 1/3"]);
    cases.push(["...and stays a timeout, not a near miss", late.cls === "timeout"]);
    cases.push(["...and says so in words", /ran out of time/.test(late.word)
      && /1 of 3 checks passed/.test(late.word)]);

    // Zero is still worth printing: it says the verifier ran and found nothing,
    // which a bare "T" leaves ambiguous.
    const none = cell({ task_name: "t", resolved: false, error_type: "AgentTimeoutError",
                        n_checks: 3, n_checks_passed: 0 });
    cases.push(["a timeout with no credit shows 0/3", none.glyph === "T 0/3"]);

    // No checks recorded at all -- nothing to report, so no fraction invented.
    const bare = cell({ task_name: "t", resolved: false, error_type: "AgentTimeoutError" });
    cases.push(["a timeout with no checks stays bare", bare.glyph === "T"]);

    // The verifier grades whatever was on disk, so a trial can time out *and*
    // pass. The reward is ground truth and the pass rate counts it, so this
    // must not regress into a timeout cell.
    const solved = cell({ task_name: "t", resolved: true, error_type: "AgentTimeoutError" });
    cases.push(["a solved-but-killed trial is still a pass",
      solved.cls === "pass" && solved.late === true]);
  }

  const lb = vm.runInContext("renderLeaderboard(visibleRuns(state.data))", ctx);
  cases.push(["checks passed are shown", /class="lb-checks"/.test(lb)]);

  /* Partial credit leads with the tasks the run did *not* solve. Including the
   * solved ones mostly restates the pass rate -- a solved task passes every
   * check by definition -- and on real data that inversion mattered: the run
   * that led on the all-inclusive figure came last on what it salvaged from its
   * failures. Both numbers are shown; the informative one is the headline. */
  /* Expectations are derived from the runs being rendered rather than written
   * as literals. The literals only ever described the synthesized fixture, so
   * pointing this file at a real results.json -- the invocation its own header
   * documents -- failed four checks that were not finding anything wrong. A
   * test that can only pass on one input is not testing the render path. */
  const shown = JSON.parse(
    vm.runInContext("JSON.stringify(visibleRuns(state.data))", ctx));
  const fmtPct = (v) => vm.runInContext(`pct(${v})`, ctx);
  const withMissed = shown.filter((r) => r.n_checks_missed_total);

  // A run that solved everything has no partial credit to report, and that is
  // a real case (see test_collect), not a gap in the fixture.
  if (withMissed.length) {
    cases.push(["partial credit leads with the unsolved tasks",
      withMissed.every((r) => lb.includes(
        `<b>${fmtPct(r.missed_check_rate)}</b> of checks on ${r.n_missed} unsolved`))]);
    cases.push(["...with its own numerator and denominator",
      withMissed.every((r) => lb.includes(
        `${r.n_checks_missed_passed}/${r.n_checks_missed_total}`))]);
    cases.push(["...and the all-inclusive total kept as context",
      withMissed.every((r) => lb.includes(
        `${r.n_checks_passed}/${r.n_checks_total} incl. solved`))]);
    // Leading with the all-inclusive rate is the bug being fixed. It may only
    // appear as context, never as the headline -- unless the two rates round to
    // the same string, which proves nothing either way.
    cases.push(["the all-inclusive rate is not the headline",
      withMissed.every((r) =>
        fmtPct(r.check_rate) === fmtPct(r.missed_check_rate) ||
        !lb.includes(`<b>${fmtPct(r.check_rate)}</b> of checks on`))]);
  }
  cases.push(["...marked as the quieter of the two",
    /class="lb-checks-all"/.test(lb)]);

  // Older result files predate the split and still have to render.
  {
    const legacy = JSON.parse(JSON.stringify(data));
    legacy.runs.forEach((r) => {
      delete r.n_checks_missed_total; delete r.n_checks_missed_passed;
      delete r.missed_check_rate; delete r.n_missed;
    });
    ctx.LEGACY_CHECKS = legacy;
    vm.runInContext("state.data = LEGACY_CHECKS;", ctx);
    const old = vm.runInContext("renderLeaderboard(visibleRuns(state.data))", ctx);
    cases.push(["a run collected before the split still renders its checks",
      /class="lb-checks"/.test(old) &&
      shown.filter((r) => r.n_checks_total).every((r) =>
        old.includes(`${r.n_checks_passed}/${r.n_checks_total}`))]);
    cases.push(["...without inventing an unsolved denominator",
      !/unsolved/.test(old)]);
    vm.runInContext("state.data = SEED;", ctx);
  }
  cases.push(["...in their own element, not merged into the rate",
    lb.indexOf('class="lb-checks"') > lb.indexOf('class="lb-val"')]);
  cases.push(["...and are labelled as checks, not as a score",
    /of checks/.test(lb) && /not comparable to the pass rate/i.test(lb)]);

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

/* ---------------------------------------------------------------------------
   Benchmark scoping.

   The results view is scoped to one model AND one benchmark. It used to be
   scoped to the model alone, which pooled a Terminal-Bench 2 run and an
   aider-polyglot run of the same model into one pass rate -- an average over 89
   tasks and 225 tasks, with nothing on screen saying so.
--------------------------------------------------------------------------- */
{
  const cases = [];
  const inCtx = (expr) => vm.runInContext(expr, ctx);
  const fp = inCtx("state.model");

  // A second benchmark, same model. Built here rather than in the shared
  // fixture: forty other checks count the runs in that one.
  const base = inCtx("SEED");
  const poly = {
    run_id: "codex__x__polyglot", batch_id: "bp", harness: "codex",
    harness_label: "Codex CLI", model_fingerprint: fp, model_label: "M",
    dataset: "aider/aider-polyglot", subset: null, is_partial: false,
    started_at: "2026-08-09T10:00:00+00:00", n_done: 2, n_total: 2,
    n_resolved: 2, pass_rate: 1, status: "complete", tasks: [],
  };
  ctx.MIXED = {
    ...base,
    runs: [poly, ...base.runs],
    datasets: [
      { id: "terminal-bench@2.0", label: "Terminal-Bench 2", slug: "tb2",
        n_tasks: 89, runs: base.runs.map((r) => r.run_id) },
      { id: "aider/aider-polyglot", label: "Aider Polyglot", slug: "polyglot",
        n_tasks: 225, runs: [poly.run_id] },
    ],
  };
  inCtx("state.data = MIXED; state.runPick = null; state.hidden = new Set();"
        + "state.hidePartial = false; state.dataset = 'terminal-bench@2.0';");

  const filters = inCtx("renderFilters(state.data, visibleRuns(state.data))");
  cases.push(["the benchmark selector renders", /id="dataset-select"/.test(filters)]);
  cases.push(["...listing every benchmark this model ran",
    /Terminal-Bench 2/.test(filters) && /Aider Polyglot/.test(filters)]);
  cases.push(["...with the selected one preselected",
    /value="terminal-bench@2\.0" selected/.test(filters)]);
  // Shown even with one benchmark: it is what tells you which one you are
  // looking at, and a scope you cannot see is a scope you forget.
  inCtx("state.data = SEED;");
  cases.push(["...and shown even when there is only one to pick",
    /id="dataset-select"/.test(inCtx("renderFilters(state.data, visibleRuns(state.data))"))]);
  inCtx("state.data = MIXED;");

  const scoped = inCtx("runsInScope(state.data).map((r) => r.run_id)");
  // Counted against the seed rather than a literal: earlier blocks append to
  // SEED, so a hard-coded total here would be asserting their run count.
  cases.push(["only the selected benchmark's runs are in scope",
    scoped.length === base.runs.length && !scoped.includes(poly.run_id)]);

  // The bug the whole scope exists to prevent: a harness that only ran the
  // other benchmark must not be ranked in this one.
  const lb = inCtx("renderLeaderboard(visibleRuns(state.data))");
  cases.push(["the leaderboard excludes the other benchmark's harness",
    !/Codex CLI/.test(lb)]);

  // Controls offering something that cannot affect the view read as broken.
  cases.push(["harness chips are scoped to the benchmark",
    !/data-harness="codex"/.test(filters)]);
  cases.push(["the run picker is scoped too",
    !inCtx("renderRunPicker(state.data)").includes(poly.run_id)]);

  inCtx("state.dataset = 'aider/aider-polyglot';");
  const other = inCtx("visibleRuns(state.data).map((r) => r.run_id)");
  cases.push(["switching benchmark switches the runs",
    other.length === 1 && other[0] === poly.run_id]);
  const otherLb = inCtx("renderLeaderboard(visibleRuns(state.data))");
  cases.push(["...and the leaderboard follows",
    /Codex CLI/.test(otherLb) && !/Hermes Agent/.test(otherLb)]);

  // A run recorded before --dataset was captured is its own bucket, not a run
  // that silently disappears.
  ctx.LEGACY_DS = { ...base, runs: [{ ...poly, dataset: null }] };
  inCtx("state.data = LEGACY_DS; state.dataset = '';");
  cases.push(["a run with no recorded dataset is still selectable",
    inCtx("visibleRuns(state.data).length") === 1]);
  cases.push(["...and is labelled rather than left blank",
    /unknown benchmark/.test(
      inCtx("renderFilters(state.data, visibleRuns(state.data))"))]);

  // null is "not resolved yet", distinct from "" -- render() resolves it before
  // drawing, and until it does nothing may silently vanish.
  inCtx("state.data = MIXED; state.dataset = null;");
  cases.push(["an unresolved scope shows everything rather than nothing",
    inCtx("runsInScope(state.data).length") === base.runs.length + 1]);

  inCtx("state.data = SEED; state.dataset = 'terminal-bench@2.0';");

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

/* ---------------------------------------------------------------------------
   A submission the verifier could not score is a failure, not an error.

   Benchmarks that compile their tests against the agent's code cannot score
   code that does not compile, so "did not implement it" arrives as
   RewardFileNotFoundError. Rendered with the error glyph, the *normal* way to
   fail on aider-polyglot wore the mark reserved for the harness falling over,
   and a matrix of red exclamation marks read as infrastructure failure when it
   meant the model could not code.
--------------------------------------------------------------------------- */
{
  const cases = [];
  const inCtx = (expr) => vm.runInContext(expr, ctx);
  const fp = inCtx("state.model");

  const trial = (name, over) => Object.assign({
    task_name: name, resolved: false, reward: 0, error_type: null,
    no_reward_reason: null, duration_s: 60, agent_s: 50,
    started_at: null, finished_at: null,
    n_input_tokens: 100, n_output_tokens: 50, cost_usd: null,
  }, over);

  ctx.SCORE = {
    ...inCtx("SEED"),
    task_names: ["built-and-passed", "scored-zero", "did-not-build", "crashed"],
    runs: [{
      run_id: "h__x__score", batch_id: "bs", harness: "hermes",
      harness_label: "Hermes Agent", model_fingerprint: fp, model_label: "M",
      dataset: "terminal-bench@2.0", subset: null, is_partial: false,
      started_at: "2026-08-10T10:00:00+00:00", status: "complete",
      n_done: 4, n_total: 4, n_resolved: 1, pass_rate: 0.25,
      tasks: [
        trial("built-and-passed", { resolved: true, reward: 1 }),
        trial("scored-zero", {}),
        trial("did-not-build", { no_reward_reason: "RewardFileNotFoundError" }),
        trial("crashed", { error_type: "NonZeroAgentExitCodeError" }),
      ],
    }],
  };
  inCtx("state.data = SCORE; state.runPick = null; state.hidden = new Set();"
        + "state.hidePartial = false; state.dataset = 'terminal-bench@2.0';");

  // renderMatrix is what indexes a run's tasks by name, and cellFor reads that
  // index -- so it has to run first or every cell reports "not run".
  const mx = inCtx("renderMatrix(state.data, visibleRuns(state.data))");
  const cellOf = (task) =>
    inCtx(`cellFor(visibleRuns(state.data)[0], ${JSON.stringify(task)})`);

  const built = cellOf("did-not-build");
  cases.push(["a submission that could not be scored is a plain failure",
    built.cls === "fail" && built.glyph === "\u00b7"]);
  cases.push(["...and says so in words rather than just failing",
    /could not score/.test(built.word)]);
  // The distinction only means something if a real crash still stands out.
  const crashed = cellOf("crashed");
  cases.push(["a real crash still wears the error glyph",
    crashed.cls === "error" && crashed.glyph === "!"]);
  cases.push(["a scored zero is unchanged", cellOf("scored-zero").cls === "fail"]);
  cases.push(["a pass is unchanged", cellOf("built-and-passed").cls === "pass"]);

  // Rendered output: the reason has to reach the reader, or a failed cell with
  // no checks explains nothing. The tooltip is escaped into a data-tip
  // attribute, so its markup arrives as entities -- match on the text.
  cases.push(["the matrix explains why there is no score",
    /no score produced/.test(mx) && /RewardFileNotFoundError/.test(mx)]);
  cases.push(["...and does not call that one an exception",
    !/exception[^!]{0,20}RewardFileNotFoundError/.test(mx)]);
  cases.push(["the crash is still labelled an exception",
    /exception.{0,20}NonZeroAgentExitCodeError/.test(mx)]);

  inCtx("state.data = SEED; state.dataset = 'terminal-bench@2.0';");

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

/* ---------------------------------------------------------------------------
   The run log is bounded.

   Every other panel is bounded by what it draws; the run log is a row per run
   and grew without limit, so after a few sweeps it owned most of the page and
   pushed the charts off the bottom. It now shows five and scrolls.

   The cut itself is measured after layout (sizeRunLog), which needs a real box
   model this harness does not have -- so what is asserted here is everything
   that does not: the scroller exists, it is the thing that scrolls rather than
   the panel, its scroll position survives a repaint, and the panel says how
   many runs are out of sight.
--------------------------------------------------------------------------- */
{
  const cases = [];
  const inCtx = (expr) => vm.runInContext(expr, ctx);
  const fp = inCtx("state.model");

  const mk = (n) => ({
    run_id: `r${n}`, batch_id: "b", harness: "hermes", harness_label: "Hermes Agent",
    model_fingerprint: fp, model_label: "M", dataset: "terminal-bench@2.0",
    subset: null, is_partial: false, status: "complete",
    started_at: `2026-08-1${n % 9}T10:00:00+00:00`,
    n_done: 1, n_total: 1, n_resolved: 0, pass_rate: 0, tasks: [],
  });

  const base = inCtx("SEED");
  const many = { ...base, runs: Array.from({ length: 9 }, (_, i) => mk(i)) };
  ctx.MANY = many;
  inCtx("state.data = MANY; state.runPick = null; state.hidden = new Set();"
        + "state.hidePartial = false; state.dataset = 'terminal-bench@2.0';");

  const log = inCtx("renderRunLog(state.data)");
  cases.push(["the run log rows get their own scroller",
    /class="log-scroll"/.test(log)]);
  cases.push(["...capped at five runs",
    /data-max-rows="5"/.test(log) && inCtx("RUN_LOG_ROWS") === 5]);
  // Without this the scroller resets to the top on every 5s poll.
  cases.push(["...and its scroll position is remembered across repaints",
    /class="log-scroll" data-pane="log-rows"/.test(log)]);
  // The table has to be inside the scroller, or the panel grows instead.
  cases.push(["the table is inside the scroller",
    log.indexOf('class="log-scroll"') < log.indexOf('<table class="data"')]);

  // A scrollbar is the only other hint that there is more, and on a trackpad
  // there is not even that.
  cases.push(["the panel says how many runs are out of sight",
    /4 more below/.test(log)]);

  const few = { ...base, runs: Array.from({ length: 3 }, (_, i) => mk(i)) };
  ctx.FEW = few;
  inCtx("state.data = FEW;");
  const shortLog = inCtx("renderRunLog(state.data)");
  cases.push(["...and says nothing when they all fit",
    !/more below/.test(shortLog)]);
  cases.push(["...while still using the same scroller",
    /class="log-scroll"/.test(shortLog)]);

  // Expanding a panel is the request to see all of it.
  cases.push(["expanding the panel lifts the cap",
    /log-scroll[\s\S]{0,200}maxHeight = ""/.test(inCtx("mountExpanded.toString()"))]);

  inCtx("state.data = SEED; state.dataset = 'terminal-bench@2.0';");

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

/* ---------------------------------------------------------------------------
   sizeRunLog cuts at the row, not at a guessed height.

   A run log row is a variable number of lines -- a run with two error types is
   half again as tall as a clean one -- so a max-height in pixels shows four
   rows on one screen and six on another. The cut is measured instead. This
   drives the function against a stub whose rows are deliberately uneven, which
   checks the arithmetic; the browser is left to do the layout.
--------------------------------------------------------------------------- */
{
  const cases = [];

  // Rows at 10, 50 (tall), 70, 90, 130 (tall), 150, 170 ... header is 20.
  const tops = [10, 50, 70, 90, 130, 150, 170, 190];
  const makeBox = (nRows, insideModal) => {
    const rows = tops.slice(0, nRows).map((t) => ({
      getBoundingClientRect: () => ({ top: t, height: 0 }),
    }));
    return {
      style: {},
      _rows: rows,
      getAttribute: (k) => (k === "data-max-rows" ? "5" : null),
      closest: (sel) => (insideModal && sel === ".modal-backdrop" ? {} : null),
      querySelector: (sel) =>
        sel === "thead" ? { getBoundingClientRect: () => ({ height: 20 }) } : null,
      querySelectorAll: () => rows,
    };
  };

  const withBoxes = (boxes, fn) => {
    const real = ctx.document.querySelectorAll;
    ctx.document.querySelectorAll = (sel) =>
      sel === ".log-scroll" ? boxes : [];
    try { fn(); } finally { ctx.document.querySelectorAll = real; }
  };

  // Eight rows, cap of five: cut below the fifth, i.e. at row index 5's top.
  const box = makeBox(8, false);
  withBoxes([box], () => vm.runInContext("sizeRunLog()", ctx));
  // header 20 + (tops[5] - tops[0]) = 20 + (150 - 10) = 160
  cases.push(["the cut lands under the fifth row, however tall the rows are",
    box.style.maxHeight === "160px"]);

  // Fewer rows than the cap: no cap at all, so the panel shrinks to fit.
  const small = makeBox(3, false);
  withBoxes([small], () => vm.runInContext("sizeRunLog()", ctx));
  cases.push(["a log that already fits is not capped",
    small.style.maxHeight === ""]);

  // Exactly the cap is still not capped -- capping there would scroll a list
  // that fits, which is the thing that reads as broken.
  const exact = makeBox(5, false);
  withBoxes([exact], () => vm.runInContext("sizeRunLog()", ctx));
  cases.push(["exactly five is not capped", exact.style.maxHeight === ""]);

  // The expanded copy shows everything.
  const modal = makeBox(8, true);
  withBoxes([modal], () => vm.runInContext("sizeRunLog()", ctx));
  cases.push(["the expanded copy is never capped", modal.style.maxHeight === ""]);

  // Runs on every poll, so it must not compound its own previous answer.
  const again = makeBox(8, false);
  again.style.maxHeight = "999px";
  withBoxes([again], () => vm.runInContext("sizeRunLog()", ctx));
  cases.push(["re-measuring does not build on the last answer",
    again.style.maxHeight === "160px"]);

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

/* Harness lists are alphabetical wherever they appear.
 *
 * registry.yaml is an ordered mapping and these lists used to render it
 * verbatim, so the order was whatever order the rows were written in -- and it
 * drifted with every edit, because a harness added from the dashboard appends
 * to the end. Sorting the YAML would not hold it: save() rewrites from parsed
 * YAML in insertion order.
 *
 * The catalog below is deliberately in registry.yaml's real order, which is not
 * alphabetical, so a list that forgot to sort fails here rather than looking
 * plausible.
 */
{
  const cases = [];
  const catalog = {
    "dmfa-minion": { label: "dmfa-minion" },
    hermes: { label: "Hermes Agent" },
    minion: { label: "minion" },
    omp: { label: "oh-my-pi" },
    "claude-code": { label: "Claude Code" },
    codex: { label: "Codex CLI" },
    dsh: { label: "DeepSeek Harness" },
    opencode: { label: "opencode" },
  };
  vm.runInContext(`__catalog = ${JSON.stringify(catalog)};`, ctx);

  const ids = vm.runInContext("harnessIdsSorted(__catalog)", ctx);
  const labels = ids.map((id) => catalog[id].label);
  const wanted = [...labels].sort((a, b) => a.localeCompare(b));
  cases.push(["ids sort by the label the reader sees",
    JSON.stringify(labels) === JSON.stringify(wanted)]);
  // The catalog handed in was not alphabetical, so an unsorted implementation
  // would return it unchanged and pass a weaker assertion than this one.
  cases.push(["  and not in catalog order",
    JSON.stringify(ids) !== JSON.stringify(Object.keys(catalog))]);
  cases.push(["  every harness survives the sort", ids.length === 8]);

  // A row with no label falls back to its id rather than sorting as blank.
  const partial = vm.runInContext(
    "harnessIdsSorted({ zulu: {}, alpha: { label: 'Alpha' } })", ctx);
  cases.push(["a label-less row sorts on its id",
    JSON.stringify(partial) === JSON.stringify(["alpha", "zulu"])]);

  // The filter chips build their list from run order, not the catalog.
  const chips = vm.runInContext(
    "[{id:'omp',label:'oh-my-pi'},{id:'dsh',label:'DeepSeek Harness'}," +
    "{id:'codex',label:'Codex CLI'}].sort(byHarnessName).map(h=>h.label)", ctx);
  cases.push(["the results filter chips sort too",
    JSON.stringify(chips) ===
      JSON.stringify(["Codex CLI", "DeepSeek Harness", "oh-my-pi"])]);

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

/* --- the refetch dim must never strobe ------------------------------------
 *
 * Holding the previous render beats flashing a skeleton, and a genuinely slow
 * scan should say so -- but a bare threshold pulses. A tree whose scan sits
 * near the limit crosses it on some ticks and not others, and the page goes
 * dark and light every five seconds. That happened: adding a per-trial log read
 * to the collector took a 209 ms scan to 408 ms against a 400 ms trigger.
 *
 * So a tick may only dim when the *previous* scan was already slow.
 */
{
  const cases = [];
  const dim = (ms) => vm.runInContext(`shouldDimOnRefetch(${ms})`, ctx);
  const limit = vm.runInContext("SLOW_SCAN_MS", ctx);

  cases.push(["a fast previous scan never dims", dim(12) === false]);
  cases.push(["...nor one exactly at the threshold", dim(limit) === false]);
  cases.push(["...nor the very first tick", dim(0) === false]);
  cases.push(["a persistently slow scan dims", dim(limit + 1) === true]);
  cases.push(["...and stays dimmed while it stays slow", dim(5000) === true]);
  // The strobe itself: alternating either side of the limit must not alternate
  // the dim, because the decision is made from the previous scan only.
  cases.push(["a scan hovering at the limit does not strobe",
    dim(limit - 1) === false && dim(limit) === false]);

  for (const [label, ok] of cases) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    if (!ok) failed++;
  }
}

console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
process.exit(failed ? 1 : 0);
