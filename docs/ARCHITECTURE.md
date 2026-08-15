# Architecture

How harness-arena works internally: the data flow, each module's job, and the contract an
adapter has to satisfy.

## Contents

1. [The pipeline](#the-pipeline)
2. [Module reference](#module-reference)
3. [Writing an adapter](#writing-an-adapter)
4. [On-disk formats](#on-disk-formats)
5. [Design decisions](#design-decisions)

---

## The pipeline

```
   config.yaml / env / flags ─ which endpoint, which provider, which credential
             │
             ▼
   bench/config.py ────────── resolve the layered configuration
             │
             ▼
   <endpoint> /v1/models + /props
             │
             ▼
   bench/probe.py ─────────── fingerprint the weights, resolve a display label
             │                (cached in bench/models.json)
             ▼
   harnesses/registry.yaml ── which harnesses, which flags, which defaults
             │
             ▼
   bench/runner.py ────────── build one `harbor run` per harness, run them
             │                SEQUENTIALLY, write a manifest per job
             ▼
   harbor run (subprocess)
             │
             ├── pulls the task image, starts a Linux container
             ├── installs the harness inside it (harnesses/*.py install())
             ├── runs the agent against the endpoint  (run())
             └── copies tests/ in, runs the verifier, writes reward
             │
             ▼
   runs/<job>/<trial>/result.json          ← written as each trial finishes
   runs/<job>/harness-bench.json           ← our manifest
             │
             ▼
   bench/collect.py ───────── normalize into one comparable index
   bench/activity.py ──────── tail the in-flight trial
   bench/throughput.py ────── wall clock and LLM utilization
             │
             ▼
   dashboard/server.py ────── results API + authenticated control plane
   dashboard/index.html ───── the page: results, run, setup, harnesses, upkeep
```

**Harness runs are strictly sequential.** One endpoint backs every harness; overlapping runs
would each measure the other's queueing delay as if it were their own latency.

---

## Module reference

### `bench/config.py`

Everything that differs between two people running this repo, resolved in one place:
**defaults → `config.yaml` → `HARNESS_ARENA_*` env vars → CLI flags**.

Also holds the provider catalog. A `Provider` records what an endpoint can tell you about
itself, which is the only thing that actually varies between them:

| | `openai-compatible` | `openrouter` |
|---|---|---|
| `supports_props` | yes, weights are inspectable | no |
| `model_is_discoverable` | yes. One model is loaded | no, hundreds are served |
| `requires_api_key` | no | yes |
| `default_agent_concurrency` | 1 (one local slot) | 4 (rate limit, not a queue) |

**Credentials never reach disk in this repo.** The supported path is indirection,
`api_key_env` names an environment variable. A literal key in `config.yaml` is supported
because the setup flow has to write one somewhere, and that file is gitignored and written
`0600`. `scrub()` is the chokepoint every logging, serializing and rendering path calls;
`tests/test_config.py` proves a key cannot reach a run manifest.

### `bench/probe.py`

Queries the endpoint and builds a `ModelIdentity`. How depends on what the endpoint can say
about itself:

**Self-hosted.** The fingerprint is a SHA-256 prefix over `served_id + n_params + size +
ftype + n_ctx_train + n_embd + n_vocab + model_path`. Everything in that list changes when you
load a different file; nothing changes when you restart the same one. The alias is
deliberately *not* the key, reloading the same `--alias` with a different quant must register
as a new model.

`/props` also yields `total_slots`, which is what `n_concurrent_agents` should match, and
`model_path`, which produces a far better default label than the alias
(`my-model` → `Qwen3 Coder 30B (Q4_K_M)`).

**Hosted.** There is nothing to discover, so the configured model id *is* the identity and the
fingerprint derives from it. The probe's job becomes validation: confirm the provider actually
serves that id, and suggest near-misses if not, rather than letting the first trial die on a
404 that reads like a network fault. Note that the weights behind a hosted id can change
without notice, a hosted run measures an endpoint, not a file you hold.

`measure_speed()` times one uncontended completion and converts it into a recommended
`agent_timeout_multiplier`. That setting is the one most likely to invalidate a run and is
pure arithmetic given generation speed, so it is measured rather than guessed.

Labels are cached by fingerprint in `bench/models.json`, so the interactive prompt appears
once per distinct set of weights. That file is gitignored. It records which weights are on
*your* disk, including their paths.

### `bench/cli.py`

Dispatch only. Each command forwards its remaining arguments to the module that owns them, so
there is exactly one definition of every flag. The two commands defined here, `init` and
`doctor`, exist to get someone from `git clone` to a working run, which is the step where a
benchmarking repo usually loses people.

### `bench/supervisor.py`

Owns the benchmark subprocess on behalf of the UI. A terminal gave three things for free that
a web button has to implement:

- **One at a time.** Runs are sequential because one endpoint backs every harness; two at once
  would each measure the other's queueing delay as their own latency. A terminal enforced this
  socially. `start()` enforces it for real, and `prepull` shares the same slot rather than
  getting its own, pulling tens of GB while a benchmark runs competes for the same disk.
- **Stopping the whole tree.** `harbor run` spawns containers and children, so the child is
  started in its own process group and the *group* is signalled. Harbor tears containers down
  on interrupt, so a grace period comes before any hard kill; skipping it is what leaves
  multi-GB containers running.
- **An honest stop.** A killed job never writes `finished_at`, so on stop the supervisor writes
  `stopped_at`/`stopped_reason` into every manifest that run created. `collect.py` already
  reads them.

It also pins the child to the endpoint the *server* holds, passing it by environment rather
than reading config.yaml again. The two could otherwise disagree after a UI edit, and a run
labelled with one model but generated by another looks entirely normal in the output. The API
key travels by environment variable specifically because an argv is world-readable.

### `bench/registry.py`

Read/write access to the harness catalog. Edits go through here rather than straight to YAML
because the UI can write this file and **it is committed**: `upsert_harness` refuses anything
key-shaped in a literal value, rejects unknown `{placeholders}` that would otherwise fail at
run time with a bare `KeyError`, and bounds every editable default. `save()` keeps one
`.bak` generation, since the adapter notes in that file are the accumulated result of
debugging each upstream's quirks.

`validate_dataset_slugs` runs on every write. A dataset `slug` becomes a segment of every run
directory name, so two entries sharing one makes the directories stop distinguishing the
benchmarks — the mislabelling the slug exists to prevent, and nothing downstream would report
it.

**`catalog_drift()` and why nothing is merged.** In a checkout there is one catalog and this
is inert. Installed from a wheel, reads fall back to the packaged catalog only until the
first edit; `save()` then writes a full copy under `.harness-arena/` and `registry_path()`
prefers it permanently. A later `pip install -U` updates the code and the packaged catalog
and touches nothing of yours.

That is silent and it is not cosmetic: the catalog carries the harness `version:` pins, so an
upgrade that re-pins a harness leaves the old build installing under the new release's name —
the same "two runs measuring different harnesses under one name" failure the pins were added
to prevent. New `datasets:` entries disappear the same way.

Merging automatically is not available, because a pin you changed deliberately and one you
simply never received are identical in the file — so a merge either discards your edit or
keeps a stale harness, and both are silent. Instead `save()` stamps `snapshot_of` with the
release that wrote the copy (installed only; in a checkout it would be committed churn), and
`catalog_drift()` diffs yours against the packaged one. `harness-arena doctor` prints the
result. Recording the fact and refusing to guess is the same rule the pairing logic follows.

### `bench/runner.py`

Reads `registry.yaml`, substitutes probe values into the harness block, and builds the
`harbor run` argv. Writes `harness-bench.json` **before** starting, so a job killed early is
still identifiable.

**Run directory names.**

```
{harness}__{model_slug}__{dataset_slug}__{scope}__{stamp}
omp__qwen3-coder-30b-q4-k-m-a1b2c3d4__tb2__full__20260814T173206Z
hermes__qwen3-coder-30b-q4-k-m-a1b2c3d4__polyglot__stratified-25__20260814T181500Z
```

The name carries the variables that decide whether two runs are *the same experiment*.
Model was always there; dataset and scope were not, so a Terminal-Bench 2 run and an
aider-polyglot run of one harness were indistinguishable in a directory listing. Everything
else that varies — context window, reasoning effort, timeout multiplier, attempts — stays in
the manifest: the name is a human index, not the record.

Three constraints shape it, and each rules out an otherwise nicer scheme:

- **Harness stays at index 0.** `collect.load_run` identifies a job directory with *no*
  manifest — a bare `harbor run`, or one killed before the manifest landed — by splitting on
  `__` and taking the first field. Putting the dataset first would relabel every such run.
- **Length.** The model slug alone reaches 57 characters, and Harbor writes trial
  directories underneath this one against Windows' 260-character path limit. Hence
  `slug` in the `datasets:` catalog, bounded at 12 characters and checked for collisions
  on save — a shared slug would make two benchmarks' directories stop telling them apart.
- **Scope is three states, not two.** A *named* subset is a deliberate experiment every
  harness ran identically; an ad-hoc `--n-tasks` cap is a smoke test; neither is the full
  dataset. `scope_name()` and the dashboard's `scopeName()` agree on all three.

A dataset with no catalogued slug derives one from its id rather than refusing: `--dataset`
accepts anything Harbor resolves, and an unnameable directory is worse than a longer name.

Sets `PYTHONPATH` to the project root so Harbor can import `harnesses.*` in its own process.

Runs harnesses one after another and returns non-zero if any failed.

The manifest is written through `scrub()` and omits `host_env` entirely. A manifest is read by
the dashboard and inlined verbatim into exported snapshots, so it has to be safe to publish,
which also means the machine hostname is recorded only when `record_hostname` is set.

### `bench/collect.py`

Walks `runs/`, reads each job's manifest plus every `<trial>/result.json`, and normalizes.
Key behaviors:

- **Resolution**: a task counts as resolved only if *every* reward value is ≥ 1.0.
- **Checks**: parses `verifier/ctrf.json` for per-test outcomes, so a 5/6 near miss is
  distinguishable from 0/6 despite scoring identically.
- **Retries**: deduplicates by task name, keeping the last attempt.
- **Setup failures**: a trial that dies before writing its own `result.json` is recovered
  from the job-level `stats.evals[*].exception_stats`, so a failed install shows as a broken
  run rather than an empty one.
- **Stopped runs**: a job killed mid-flight never writes `finished_at`; the manifest's
  `stopped_at` marker turns it into `status: stopped` rather than "running" forever.
- **Wilson intervals**: every pass rate carries a 95 % score interval.
- **Pairing**: head-to-head only pairs runs sharing model, **dataset**, subset,
  partial-status and timeout multiplier, differing in harness. The dataset clause was
  missing until multi-benchmark support was audited: two *full* runs on different
  benchmarks both carry `subset: null` and `is_partial: false`, so every other clause
  passed and they paired. Task names rarely collide across datasets, which made it worse
  rather than harmless — the comparison rendered with an empty shared set, reading as two
  harnesses that agreed on nothing.
- **Benchmarks present**: `datasets` lists the benchmarks the runs actually cover, built
  from the runs rather than the catalog, so a catalogued benchmark nobody has run is not
  an empty option and a run whose dataset has since left the catalog is still selectable.

### `bench/activity.py`

Finds the most recently touched trial without a `result.json` and tails its agent log.

Format-agnostic on purpose: it tries to parse each line as JSON and treats it as prose
otherwise, so hermes (plain text) and omp (NDJSON) both render, and so will a new harness's
log without an adapter here.

**Filtering is a view concern, never a storage one.** The log on disk keeps every byte; the
feed drops per-token bookkeeping, Claude Code's `thinking_tokens` (91 % of one log) and Rust
`tracing` below `WARN` (97 % of another, courtesy of `--debug-capture`). `WARN`/`ERROR` and
all tool output survive, including the hex an agent produces by `od`-ing a binary.

That makes **bytes scanned and bytes rendered different budgets**, which is the non-obvious
part. A ceiling sized for a log that is mostly content cannot reach content in a log that is
97 % exhaust: the first attempt left the panel blank on a 10 MB log whose last 400 KB held
1,115 lines and no output. The scan ceiling is now far larger, the bytes are discarded rather
than rendered, and a normal log still costs exactly one small read because the loop stops as
soon as it has something to show. When even the full scan finds nothing, the panel says so:
silence reads as a hung agent.

**Expanding a panel clones it rather than re-rendering it.** The whole results view is rebuilt
every five seconds, so the dialog is rebuilt with it, from the freshly rendered panel, which
means there is no second rendering path to keep in step with the first, and an expanded panel
keeps updating while it is open. It is mounted *before* the event wiring pass, so its controls
are wired by the same code as the original. Ids and `data-pane` are stripped from the copy: the
originals own them, and `capturePaneScroll` keys on `data-pane`, so a duplicate would overwrite
the real panel's remembered scroll with the dialog's.

Note that every name in `grid-template-areas` must be a **rectangle**; an invalid one is dropped
whole and silently, collapsing the layout into a stack with nothing logged. `tests/test_dashboard.mjs`
validates the stylesheet's own templates against that rule.

Liveness is measured by **log size deltas across calls**, not mtime. On Windows, a file being
appended through a Docker bind mount can report a stale mtime for many minutes, observed
claiming 24 minutes of silence while the file grew 5 KB.

### `bench/throughput.py`

Two metrics per run:

- **min/trial**: wall clock per *completed* trial. Understates a run with work in flight, so
  it is only final once the run ends.
- **llm busy %**: share of elapsed time the model spent generating. This is what pipelining
  targets and it stays honest mid-run.

Anchors on the job's recorded `started_at`, and for a live run measures to *now*. Timing from
an observer's start instead reports a fictional speedup.

### `bench/prepull.py` / `bench/make_subset.py`

`prepull` pulls `<repo>/<task>:<tag>` for every task in a subset, skipping cached ones,
with a small worker pool (multi-GB layers; saturating the link makes every pull slower).

`make_subset` builds a difficulty-stratified task list by stride-sampling sorted names within
each band, with largest-remainder apportionment so the strata sum exactly. Deterministic, no
RNG seed to remember, and not biased toward whatever sorts first.

### `dashboard/`

`server.py` is stdlib-only, no build step, nothing to install. `/api/results` is cached ~2 s;
`/api/activity` is uncached because the point of a live feed is to be current.

`index.html` is one file with inline CSS and JS. Colors come from a palette validated for
contrast and colorblind separation against the near-black surface; pass/fail is encoded by
glyph as well as hue, never hue alone.

**The control plane.** Four of the five tabs write, which changed the threat model: the server
can now start processes and holds an API key, on a port every page in the browser can reach.
Four gates gate every mutating request:

| Gate | Stops |
|---|---|
| Per-process token in a custom header | Cross-origin script, which cannot set one without a preflight this server never answers |
| `Host` header validation | DNS rebinding, the attacker controls resolution, not the Host string |
| `Origin` rejection | Browser-issued cross-origin writes |
| JSON bodies only | Form encoding, the one content type sendable cross-origin without a preflight |

**A rejected request still has to answer.** Every gate above refuses *before* reading the
request body, and this speaks HTTP/1.1. Closing a socket with unread request data in it sends
an RST, so the client sees `WSAECONNABORTED` instead of the 403 or 415 saying what it did
wrong, and on a kept-alive connection the leftover bytes are parsed as the next request line.
`_send` therefore drains the body first, and `_read_json` claims `_body_consumed` only once it
is actually about to read — claiming it up front marked every early rejection as drained and
reintroduced exactly the failure the drain exists to prevent. An *oversized* body is
deliberately not drained; reading megabytes only to reject them is the denial of service this
would invite, so the close is the honest signal there.

That bug is why `tests/test_api.py` asserts a rejected request leaves the connection reusable
rather than merely returning the right status. Whether a skipped drain aborts depends on how
much the OS had already buffered, so the status-only form caught it about 1 run in 40 and read
like an environment fault; reusing the connection fails every time instead.

Redaction is separate from authentication and is not optional: `/api/state` reports *that* a
key is set, never its value. Exported snapshots are seeded with a read-only flag and no token,
so a shared file has no control plane in it.

Forms are uncontrolled and re-render only on tab switch and after an action. The results poll
runs every 5 s, and repainting a form on that tick would discard whatever was being typed, so
the Run tab's console and status bar update in place instead, touching no input.

---

## Writing an adapter

An adapter is a `BaseInstalledAgent` subclass that Harbor loads via
`--agent harnesses.<module>:<Class>`.

```python
class MyCli(BaseInstalledAgent):
    @staticmethod
    def name() -> str: ...              # free-form; not validated against Harbor's enum

    def get_version_command(self) -> str | None: ...
    async def install(self, environment) -> None: ...       # install into the container
    async def run(self, instruction, environment, context) -> None: ...
    def populate_context_post_run(self, context) -> None: ...   # token accounting
```

Useful base-class helpers: `exec_as_root`, `exec_as_agent`, `build_cli_flags`, `_get_env`,
`self.logs_dir`, `self.model_name`, `self.skills_dir`, `self.mcp_servers`.

### Rules learned the hard way

**Pin the API to the *installed* Harbor, not its main branch.** `ensure_system_dependencies`
exists on `main` but not in 0.20.0; using it produced `'Omp' object has no attribute
'ensure_system_dependencies'` at install time. Read the version you actually have.

**Don't assume a package manager.** Task images are mostly Debian but some are Alpine or
RPM-based. A hard-coded `apt-get` turns those into install failures that read like agent bugs.

**Line-buffer every stage of the log pipeline.** `grep` without `--line-buffered` holds ~4 KB,
so the log lags the run and a trial killed at the timeout loses its tail, precisely the part
worth reading. This produced a 0-byte trajectory log until fixed.

**Pass the instruction via an environment variable**, not on the command line. Task
instructions routinely contain quotes, backticks and newlines that no amount of shell quoting
survives through `exec → sh -c → pipeline`. Where a positional argument is unavoidable, use
`--` to end flag parsing so an instruction starting with `-` or `@` is not misread.

**Disable persistent memory.** A harness that learns across sessions would have task N solved
by an agent shaped by tasks 1..N-1, measuring accumulated memory rather than the harness.

**Verify the token path against real output.** Every adapter's token accounting has been
wrong at least once, in ways synthetic tests could not catch: two reported zero, and two more
reported input net of cache while the rest reported it inclusive -- understating their prompt
totals by 18x without erroring. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## On-disk formats

### `runs/<job>/harness-bench.json` (ours)

```jsonc
{
  "schema": 1,
  "harness": "omp", "harness_label": "oh-my-pi",
  "agent_ref": "harnesses.omp:Omp",
  "model": { "label": "...", "fingerprint": "...", "n_ctx": 262144, ... },
  "dataset": "terminal-bench@2.0",
  "n_concurrent": 2, "n_concurrent_agents": 1, "n_attempts": 1,
  "agent_timeout_multiplier": 16.0,
  // The window every harness was told, and where the number came from:
  // "configured" | "detected" | "fallback". Two runs at different windows are
  // not the same experiment, and a fallback window is a weaker claim than a
  // detected one, so both travel with the result.
  "context_window": 65536, "context_window_source": "configured",
  "agent_max_tokens": 8192,
  // Same idea for the reasoning effort harnesses that send one were given:
  // "configured" | "probed" | "fallback". Probed because a server can refuse
  // an effort outright, and a run that reasoned is not comparable to one that
  // did not.
  "reasoning_effort": "none", "reasoning_effort_source": "probed",
  "max_retries": 1, "debug_capture": true,
  "subset": "stratified-25", "is_partial": true,
  "harbor_version": "0.20.0",
  "command": ["harbor", "run", ...],       // secrets excluded
  "started_at": "...",
  "stopped_at": null, "stopped_reason": null
}
```

### `runs/<job>/<trial>/` (Harbor's)

| Path | Contents |
|---|---|
| `result.json` | `TrialResult`, rewards, exception, per-phase timings, token counts |
| `agent/<name>.txt` | The agent's stdout, tee'd live |
| `agent/trajectory.json` | ATIF trajectory where the adapter produces one |
| `verifier/ctrf.json` | Per-test outcomes |
| `verifier/reward.txt` | `1` or `0` |
| `config.json`, `lock.json` | Reproducibility |

Trial `result.json` appears the moment that trial ends; the job-level one is written at the
end. That is what makes the dashboard live.

---

## Design decisions

**Why subclass Harbor's agents instead of writing from scratch.** The install paths, error
classification and trajectory conversion are substantial and already correct. Each adapter
overrides only what is actually wrong, for hermes, config generation and the CLI invocation.

**Why the registry is YAML and not Python.** Adding a harness should not require reading the
runner. Everything harness-specific, agent reference, model reference format, env vars,
kwargs, is declarative, and nothing else in the codebase knows harness names.

**Why the dashboard is stdlib.** It has to work months from now with no build step and no
dependency drift. One HTML file and one stdlib server survive that; a bundler does not.

**Why liveness is measured by size delta, not mtime.** See `activity.py` above, mtime lies
on Windows bind mounts.

**Why runs are compared only within matching configuration.** A pairing that looks
authoritative but measures the time budget is worse than no pairing at all, because nothing on
screen signals the difference.
