# Troubleshooting

Every failure mode hit while building and running this, what it looked like, and what fixed
it. Several were silent. They produced plausible-looking results rather than errors, which is
the dangerous kind.

## Contents

- [Harness routing](#harness-routing)
- [Single-trial failures](#single-trial-failures)
- [Installation failures](#installation-failures)
- [Benchmarks and the catalog](#benchmarks-and-the-catalog)
- [Token accounting](#token-accounting)
- [Timeouts](#timeouts)
- [Logs and liveness](#logs-and-liveness)
- [Containers and images](#containers-and-images)
- [Measurement mistakes](#measurement-mistakes)
- [Diagnostics](#diagnostics)

---

## Harness routing

### hermes: every trial exits non-zero and nothing explains why

**Symptom.** `NonZeroAgentExitCodeError` on every task, from the first one. Zero tokens,
trials lasting a couple of minutes, and, because that exception is on the retry list,
each task failing twice. Nothing in the run's own error says more than "exit code 1".

**Diagnosis.** The reason is in the trial's agent log, not the run error:

```bash
cat runs/<job>/<task>__*/agent/hermes.txt
```

```
Failed to initialize agent: Model X has a context window of 32,768 tokens, which is
below the minimum 64,000 required by Hermes Agent.
```

**Cause.** hermes-agent refuses to start below a 64K window. This only becomes reachable
once the rig tells it a real number: a harness given nothing auto-detects and guesses high.

**Fix.** Serve at least 65536 and set `endpoint.context_window` to match. Raising the
setting alone is worse than the failure, the server then truncates in silence, which
scores as a wrong answer. On a card too small for 64K at f16, quantize the KV cache
(`OLLAMA_KV_CACHE_TYPE=q8_0`) rather than lowering the window.

Runs below a declared floor are now refused before launching, so this should present as
one message rather than 178 identical trials. The floor lives in `registry.yaml` as
`min_context_window`.

### codex: every trial exits non-zero against Ollama

**Symptom.** The same shape as the hermes failure above, `NonZeroAgentExitCodeError` on
every task from the first one, no tokens, each task failing twice because the exception is
on the retry list. The run error only quotes the whole `codex exec` command line.

**Diagnosis.** Again, the reason is in the agent log rather than the run error:

```bash
cat runs/<job>/<task>__*/agent/codex.txt
```

```
{"type":"error","message":"{\"error\":{\"message\":\"\\\"llama3.2:latest\\\" does not
support thinking\",\"type\":\"invalid_request_error\"}}"}
{"type":"turn.failed",...}
```

**Cause.** Codex puts a `reasoning` object on every request and offers no way to leave the
effort out, omitting the setting still sends `{"summary":"auto"}`. Ollama refuses a real
effort for a model that cannot think and answers 400; llama.cpp accepts one from any model,
which is why this only appears after moving endpoints. Measured against ollama 0.32.5 and
codex-cli 0.147.0.

**Fix.** Nothing to do: the effort is probed per endpoint and set to `none` when the server
refuses one. Check what a run chose in its manifest:

```bash
python -c "import json;m=json.load(open('runs/<job>/harness-bench.json'));print(m['reasoning_effort'],m['reasoning_effort_source'])"
```

`probed` means the endpoint was asked, `configured` means `endpoint.reasoning_effort` in
`config.yaml` overrode it, and `fallback` means the endpoint could not be asked and the
harness default was left alone. To pin every run to one effort, the only way to compare a
thinking run against a non-thinking one honestly, set `endpoint.reasoning_effort`.

Full history, the wire-level measurements behind it, and the validation still outstanding
on other server families: [CODEX-REASONING-EFFORT.md](CODEX-REASONING-EFFORT.md).

### Windows: a trial dies with `UnicodeDecodeError` after the agent finished

**Symptom.** `'charmap' codec can't decode byte 0x81 in position 11149`. The agent did its
work; the trial is thrown away during token accounting. Some trials in the same run are
fine. Never reproduces on Linux, and may never reproduce with a different model.

**Cause.** `Path.read_text()` with no `encoding=` uses the *locale* encoding, cp1252 on a
stock Windows install. Agent transcripts contain whatever the model emitted. Only five byte
values (`0x81 0x8d 0x8f 0x90 0x9d`) are undefined in cp1252, so almost all non-ASCII output
decodes to mojibake in silence and a narrow slice raises. In one 20 KB transcript, 37 of 39
non-ASCII characters decoded quietly and 2 raised, both CJK characters the model drifted
into.

**Fix.** `bench.runner` sets `PYTHONUTF8=1` on the harbor process, which makes every
default-encoding read in Harbor and the adapters UTF-8. If you drive `harbor run` yourself,
set it in your own environment:

```bash
PYTHONUTF8=1 harbor run -d terminal-bench@2.0 -a harnesses.hermes:Hermes ...
```

Check whether a transcript would have tripped it:

```bash
python -c "raw=open('hermes-session.jsonl','rb').read(); \
print([i for i,b in enumerate(raw) if b in (0x81,0x8d,0x8f,0x90,0x9d)])"
```

### hermes: every trial dies with `HTTP 401: Missing Authentication header`

**Symptom.** The harness installs, boots, prints a session id, then exits non-zero.
Harbor reports `NonZeroAgentExitCodeError`. Looks like a credentials problem with your
endpoint.

**Cause.** Harbor's built-in `hermes` agent forwards `OPENAI_BASE_URL`, and hermes-agent never
reads it. Its resolution order is:

```python
self.base_url = (base_url                                   # CLI/config
                 or CLI_CONFIG["model"].get("base_url", "")
                 or os.getenv("OPENROUTER_BASE_URL", "")) or None
```

With none of those set the URL stays empty and hermes falls through to OpenRouter, where the
placeholder key is rejected.

**Fix.** `harnesses/hermes.py` writes `model.base_url` into hermes's own `config.yaml` and sets
`provider: llamacpp`. The provider matters independently: hermes **discards a non-loopback
`base_url`** unless the configured provider is `custom` or a local-server alias, specifically so
a stale cloud URL cannot hijack a local session. Any LAN address is non-loopback, so if your
model server is on another machine the base URL is thrown away without the provider set, and
you are back to OpenRouter.

### omp: model not found, or requests go somewhere unexpected

**Cause.** llama-server advertises no context window, and omp needs one declared to size
compaction. `OPENAI_BASE_URL` alone leaves it guessing.

**Fix.** `harnesses/omp.py` writes an explicit `models.yml` custom provider with `auth: none`,
`contextWindow` and `maxTokens`, and pins **every** model role to the local model. Roles left
unset resolve against omp's built-in catalog and try to reach a cloud provider you have no
credentials for, a subagent or plan-mode step would silently fail mid-run.

### claude-code: every trial dies at its first request with `API Error: 500`

`UnknownApiError` on every task, each after ~3 minutes of retries and with no tokens spent.
The trial's `agent/claude-code.txt` shows only `api_retry ... error_status 500`; the sentence
that explains it is one directory further down, in
`agent/sessions/projects/-app/<session>.jsonl`:

```
API Error: 500
While executing CallExpression at line 106, column 32 in source:
...first %}  {{- raise_exception('System message must be at the beginnin...
Error: Jinja Exception: System message must be at the beginning.
```

**Cause.** The model's chat template, not the server and not the harness. Claude Code sends
`messages: [user, system]` — a second, non-first `system`-role message carrying its agent and
skill listings. llama.cpp's `/v1/messages` bridge passes that role straight to the template,
and some templates abort rather than render it:

```jinja
{%- if message.role == "system" %}
    {%- if not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
```

This is a property of the **weights**, so it appears the moment you load a different model and
nothing else changed. Measured: the same llama.cpp build (`b10269`), the same Claude Code
(`2.1.223`) and the same Harbor (`0.20.0`) ran a full 25-task sweep two days earlier against
different weights. Only the GGUF changed.

The top-level Anthropic `system` array is **not** what fails — llama.cpp merges its blocks
into one leading system message, and three blocks are accepted where one trailing system
*message* is not.

Only Anthropic-shaped harnesses are affected. Everything speaking `/v1/chat/completions` sends
one leading system message and never reaches that branch; on the run that produced this, six
of seven harnesses were fine against the very endpoint that could not serve the seventh.

**Fix.**

```
harness-arena template-fix
```

It reads the template off `/props`, replaces the `raise_exception` with the system turn the
template already emits for a leading system message, and writes the patched copy. Restart
your server with it and confirm:

```
llama-server ... --chat-template-file qwen3.8-27b.patched.jinja
harness-arena template-fix --verify
```

The edit touches only the branch that currently aborts, so every conversation that renders
today renders byte-identically — verified across seven conversation shapes (with and without
tools, with tool-call round trips, with content blocks, with and without a leading system
message).

**Why not rewrite the request instead.** A harness whose traffic the rig rewrites is no longer
the harness being measured, and a run altered that way is not comparable to one that was not.

**Prevention.** `harness-arena bench` now asks the endpoint whether it accepts the shapes the
selected harnesses send, before the first container is built, and drops only the harnesses
whose refusal it recognises — the rest of the sweep runs as asked. `harness-arena doctor` and
the dashboard's Diagnostics panel report the same check. `--skip-wire-check` turns it off.

---

## Single-trial failures

The dashboard marks these `!`, the trial raised rather than scoring. One cell,
not a whole column, which is what separates them from the harness-wide failures
above. All four below were seen in one 25-task sweep and have distinct causes.

### opencode: one task exits 1 with nothing but the usage message

`NonZeroAgentExitCodeError`, and `agent/opencode.txt` contains opencode's help
text and nothing else. No model was ever called.

The instruction is passed as a positional argument. Shell quoting makes it a
single argv element, but the CLI still reads an element *beginning* with `-` as
a flag, so a task whose text opens with a markdown bullet is parsed as unknown
options. Exactly one of Terminal-Bench 2's 89 instructions does this
(`pytorch-model-recovery`, which opens `- You are given a PyTorch state
dictionary`), which is the worst possible frequency: rare enough to look like a
bad task, common enough to cost a cell in every sweep.

**Fixed**: the command now ends flag parsing with `--` before the instruction,
as omp already did. Optional flags are emitted *before* the `--`, since anything
after it is a positional and would be pasted onto the prompt instead of parsed.

If you add a harness, pass the instruction after `--`, via stdin, or in a file.
Never as a bare positional.

### Re-running one task into a finished run

After fixing a harness bug you usually want the one affected cell corrected
without paying to re-run the other 24 tasks. Run the single task into a scratch
jobs-dir, then copy the trial directory beside the original inside the finished
run:

```bash
python -m bench.runner --harness opencode --task <task> --jobs-dir /tmp/rerun \
  --agent-timeout-multiplier <same as the original run> --n-attempts 1
cp -r /tmp/rerun/<job>/<task>__* runs/<original-job>/<task>__zz-rerun
```

Two things make this work. `load_run` keeps the **last** attempt per task name
sorted by directory name, so a suffix that sorts after the original wins and the
failed attempt stays on disk as evidence. And the copied directory must contain
a `rerun.json` marker (any JSON; record what it supersedes and why), without it
the graft ends the run's clock, because wall clock runs from the manifest's
start to the last trial to finish. A next-day re-run reported one 3.7-hour run
as 27.1 hours and dropped its LLM-busy share from 93% to 13%. With the marker,
the graft counts toward the score, the checks and the tokens, and is left out of
the two timing figures only.

Match the original run's `agent_timeout_multiplier` and attempt settings, or the
grafted trial is not comparable to the ones beside it. Check the original's
`harness-bench.json`.

If the run's summary still shows an error afterwards, it is Harbor's job-level
`result.json`, which is written once at the end and names the superseded trial
id; `collect.load_run` falls back to it only when no trial-level errors remain.

### Any vision task: `image input is not supported`

`claude-code` reports API error 500 with `image input is not supported - hint:
if this is unexpected, you may need to provide the mmproj`. On the same task
`codex` reports HTTP 400 `Output of tool call should be 'Input text'` after its
first `view_image` call.

One cause, two surfaces: **the endpoint has no multimodal projector loaded**, so
it cannot accept image content, directly, or fed back as tool output. Not a
harness bug and not a rig bug.

Restart `llama-server` with `--mmproj <projector.gguf>` if the model has one,
or accept that vision tasks are unavailable on that endpoint and read those
cells as "not attempted" rather than "failed". Note that harnesses which solve
such a task *without* looking at the image will still score, so the column is
not uniformly blank, which makes this easy to misread as a harness difference.

### claude-code: `OutputTokenExceededError`

`API Error: Claude's response exceeded the 16384 output token maximum.`

That ceiling is the rig's, not Claude Code's: `agent_max_tokens_for()` sets it
to `context_window // 8` and hands the same number to every harness, because an
output ceiling that differs per harness is a variable under test. A 131,072
window gives 16,384.

**Deliberately not changed.** Raising it would break comparability with existing
runs, and the ceiling is doing its job. What differs is how each harness reacts
to hitting it, Claude Code raises, others truncate or continue, and that
difference is a genuine measurement, not a fault. Read the cell as "the harness
ran out of output budget on this task".

Override with `endpoint.context_window` only if you intend to change the ceiling
for *every* harness, and record that you did.

---

## Installation failures

### `'Omp' object has no attribute 'ensure_system_dependencies'`

**Cause.** That helper exists on Harbor's `main` branch but not in the released 0.20.0. Writing
an adapter against the GitHub source rather than the installed package.

**Fix.** Read the installed version, not the GitHub source:

```bash
python -c "import harbor, pathlib; print(pathlib.Path(harbor.__file__).parent)"
```

### Install fails on non-Debian task images

**Cause.** Hard-coded `apt-get`. Task images are mostly Debian-derived but some are Alpine or
RPM-based.

**Fix.** Detect the package manager (`apt-get` / `apk` / `dnf` / `yum`), treat the whole step as
best-effort, and gate on `command -v curl`, the only hard requirement.

### omp installed but will not start

**Cause.** The published `@oh-my-pi/pi-coding-agent` declares `engines: {bun: ">=1.3.14"}`.
`npm install -g` succeeds and produces a binary that cannot run under Node.

**Fix.** Install from the official `install.sh --binary` release asset, a standalone binary
with no runtime prerequisites.

---

## Benchmarks and the catalog

### The benchmark dropdown is empty, or missing benchmarks the release notes mention

**Cause.** A PyPI install reads the packaged catalog only until your first edit from
the dashboard. That edit writes a full copy to `.harness-arena/registry.yaml`, and every
read prefers that copy from then on. `pip install -U` updates the code and the packaged
catalog; nothing updates yours.

This is not only cosmetic. The catalog carries the harness `version:` pins, so an upgrade
that re-pins a harness leaves you installing the **old** build under the new release's
name — two runs a week apart measuring different harnesses under one label, which is the
exact failure the pins exist to prevent.

**Diagnosis.** `harness-arena doctor` reports it explicitly, naming which pins and
benchmarks differ and which release your copy was forked from:

```
  [ok  ] harness catalog  --  .harness-arena/registry.yaml
        your own copy, forked from 0.1.14; package is 0.1.16
  [warn] Your catalog is missing what the installed package ships.
         harness omp: you pin 'v17.3.1', the package ships 'v18.0.0'
         benchmarks you do not have: brand-new/bench-9
```

**Fix.** Nothing is merged for you, because a pin you changed deliberately and one you
never received are indistinguishable in the file. Either copy the entries you want out of
the packaged catalog (doctor prints both paths), or delete
`.harness-arena/registry.yaml` to start over from the packaged one. A clone has a single
catalog and cannot drift.

### `No dataset to run`

**Cause.** No `--dataset` was passed and the catalog being read has no `defaults.dataset`
— usually a hand-edited or partial `.harness-arena/registry.yaml`.

**Fix.** Pass `--dataset`, or restore `defaults.dataset` in the catalog doctor names.

Before 0.1.15 this did not report itself: the missing value was spliced into the command
line as a `None` and died several frames later inside credential scrubbing with
`TypeError: expected string or bytes-like object`, which reads like a bug in redaction.

### `RewardFileNotFoundError`, often on most C++/Go/Rust/Java tasks

**Cause.** Not a fault. The verifier ran, the agent's code did not compile, and a benchmark
that compiles its tests *against* that code cannot produce a score for it. `Verifier.verify`
runs the test script and only then looks for `reward.txt` / `reward.json`, so this exception
is reachable only after the work was evaluated.

On aider-polyglot, SWE-bench and anything else built this way, that is the **normal** way to
fail: "did not implement it" is a build error, not a failing test. Terminal-Bench 2 scores a
missing implementation as a plain 0 and never raises it.

**What the rig does.** Since 0.1.17 these are recorded as ordinary failures, not errors:
still unresolved, still counted in the denominator, shown as `·` with a tooltip reading
*no score produced*, and kept out of the error tally. Before that they took the `!` glyph,
so a run against a weak model produced a screen of red that read as broken infrastructure.

**When to worry.** If the verifier output shows the *task's own* files failing to compile
rather than the agent's, the environment is wrong rather than the answer. Check:

```
runs/<job>/<trial>/verifier/test-stdout.txt
```

A trial that genuinely crashed has no `test-stdout.txt` at all — that is the difference
between "could not be scored" and "never got that far".

### Every harness dies in seconds with `Missing Environment Variables`

```
  Variable         │  Phase
  OPENAI_API_KEY   │  [environment.env]
  OPENAI_BASE_URL  │  [environment.env]
  OPENAI_API_KEY   │  [verifier.env]
  OPENAI_BASE_URL  │  [verifier.env]
```

**Cause.** The benchmark's own machinery calls a model, not just the agent under test.
tau3-bench simulates the user inside its environment and judges assertions in natural
language inside its verifier, so both phases need an endpoint. Harbor reads a task's
`[environment].env` and `[verifier].env` from *its own* process environment and exits
before the first trial when a required one is unset — so all seven harnesses fail in
seconds, each writing a manifest and nothing else.

**Fix.** The catalog carries them. `datasets:` entries take a `host_env` block using the
same `{placeholders}` as a harness, and the runner puts it in the child's environment:

```yaml
- id: sierra-research/tau3-bench
  host_env:
    OPENAI_API_KEY: '{api_key}'
    OPENAI_BASE_URL: '{base_url}'
    TAU2_USER_MODEL: openai/{model_id}
    TAU2_NL_ASSERTIONS_MODEL: openai/{model_id}
    TAU2_USER_REASONING_EFFORT: ''
```

The two model names matter as much as the URL: tau3's `task.toml` defaults both to
`gpt-5.2`, so setting only the endpoint asks your local server for a model it does not
serve. The `openai/` prefix is litellm's provider routing, which is what tau2 calls. The
empty reasoning effort is deliberate — tau2 sends one only when that variable is
non-empty, and a server that refuses it fails every request.

**Read the result carefully.** A tau3 run against a local endpoint has that model playing
the user *and* grading the outcome. It is not comparable to a published tau3-bench score,
where both are frontier models. The manifest records them under `dataset_env` for exactly
that reason.

### tau3-bench: `FileNotFoundError` on a `task.toml`, on Windows

**Cause.** Path length, not a missing download. Harbor caches a task under
`<cache>/tasks/packages/<org>/<dataset>__<task>/<64-char hash>/…`, so the dataset's own
task names decide whether its files fit. tau3-bench has task names up to 118 characters —
one reaches `…telecom-mobile-data-issue-bad-network-preference-bad-vpn-user-abroad-roaming-disabled-on-persona-none` — which puts its deepest file at **287 characters** against Windows'
260-character limit. With long paths disabled the download creates the directories and
writes nothing, leaving an empty package that fails much later when Harbor reads
`task.toml`.

**Diagnosis.** `harness-arena doctor` reports it. By hand:

```powershell
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled
```

**Fix.** In an admin PowerShell, then reboot:

```powershell
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1
```

Then clear the partial download — an empty package directory is not re-fetched on its own:

```
harbor cache clean
```

Linux and macOS are unaffected.

### A benchmark id will not resolve

**Cause.** `id:` in the `datasets:` catalog is passed to `harbor run --dataset` verbatim,
so it has to be a dataset Harbor can resolve. Two forms work and they take different
resolution paths: a bare name with a version (`terminal-bench@2.0`) resolves against the
Harbor registry, and `org/name` (`aider/aider-polyglot`) resolves as a package reference.

**Diagnosis.** Resolve every id in the catalog without downloading anything:

```python
import asyncio
from bench import registry as R
from harbor.registry.client.package import PackageDatasetClient
from harbor.registry.client.factory import RegistryClientFactory

async def main():
    cat, pkg, reg = R.load(), PackageDatasetClient(), RegistryClientFactory.create()
    for e in R.datasets(cat):
        name = e["id"].split("@", 1)[0]
        client = pkg if "/" in name else reg
        md = await client.get_dataset_metadata(e["id"] if "/" in name else name)
        print(f'{e["id"]:<40} {len(md.task_ids)} tasks (catalog says {e.get("tasks")})')

asyncio.run(main())
```

A mismatch between the resolved count and the catalog's `tasks:` means the dataset moved
on; `tasks:` is display-only, so it misleads the dropdown rather than breaking a run.

### Two benchmarks share a run-directory name

**Cause.** Two `datasets:` entries resolving to the same slug. The slug is a segment of
every run directory name, so a collision makes the directories stop telling the benchmarks
apart.

It need not be written down to collide. An entry with no `slug:` gets one derived from its
id by truncating to 12 characters, so two ids sharing a prefix land on the same string —
`terminal-bench@2.0` and `terminal-bench-pro/terminal-bench-pro` both derive
`terminal-ben`. That is the form to watch for, because nothing in the file looks wrong.

**Fix.** `save()` refuses both forms, so a collision can only arrive by hand-editing the
file. Give each entry an explicit `slug` of at most 12 lowercase characters, and re-run
`harness-arena doctor`.

---

## Token accounting

hermes and omp each reported **zero tokens** at first, in different ways. Neither raised an
error; the efficiency panel was simply empty. A third failure mode -- input counted net of
cache -- came later and is the one that looked right, below.

### hermes: session export writes a 0-byte file

**Diagnosis**: run both forms inside a live container:

```bash
hermes sessions export /tmp/a.jsonl --source cli   # "Exported 0 sessions"  → 0 bytes
hermes sessions export /tmp/b.jsonl                # "Exported 1 sessions"  → 30 KB
```

`--source cli` (inherited from Harbor's built-in agent) matches nothing, even though the
session's own record reads `source: 'cli'`.

### hermes: messages carry no usage at all

Harbor's parser walks `messages[].usage.prompt_tokens`. In a real export **not one message has
a `usage` key**. The real numbers are session-level aggregates on the record itself:
`input_tokens`, `output_tokens`, `cache_read_tokens`, `reasoning_tokens`.

Both are upstream bugs inherited by subclassing. `harnesses/hermes.py` drops the filter and
reads the session totals, still calling `super()` for the ATIF trajectory.

### hermes and opencode understated their prompt totals by 18x

**Symptom.** Nothing looked broken. The efficiency panel rendered, every number was
plausible, and hermes and opencode simply appeared to be the cheapest harnesses in the
catalog by an enormous margin — 1.69M and 1.87M prompt tokens against codex's 90M for the
same 25 tasks.

**Cause.** `AgentContext.n_input_tokens` is the prompt total *including* cache reads. omp,
minion, codex and claude-code all report it that way. hermes and opencode report input
*net* of cache, and their adapters passed that through unchanged, so two rows of the
comparison were counting a different thing from the other four.

The hermes adapter said so in a comment: against a local llama-server both cache counters
read 0, so whether hermes counted cache inside its input was unobservable, and reporting
them separately was the conservative choice. That stopped being true — llama.cpp reports
cached tokens now, and the gap is most of the number.

**Diagnosis.** Cache reads are a *subset* of a request's prompt, so any trial with
`n_cache_tokens > n_input_tokens` is recording input net of cache. It is arithmetic, not a
judgement call:

```bash
# hermes, crack-7z-hash: 34,919 input against 635,146 cache read over 30 calls.
# A 34,919-token prompt total would mean each call read 21K of cache it never had.
python -c "import json;r=json.load(open('runs/<job>/<trial>/result.json'));a=r['agent_result'];print(a['n_input_tokens'], a['n_cache_tokens'])"
```

opencode settles it outright — its own `step_finish` totals only balance one way:

```json
{"total": 7687, "input": 30, "output": 91, "cache": {"write": 0, "read": 7566}}
```

**Fix.** Both adapters now sum input and cache read. Runs already on disk keep whatever
they were written with, so `bench/collect.py` repairs the old form on read, using the same
invariant: `n_input_tokens < n_cache_tokens` cannot occur in a correctly recorded trial.
Output tokens were never affected, so any comparison drawn on output alone still stands.

### omp: log is 0 bytes so nothing can be parsed

See [Logs and liveness](#logs-and-liveness), `grep` buffering. Once fixed, omp's usage parses
correctly: `{input, output, cacheRead, cacheWrite, totalTokens, cost}` on assistant
`message_end` events.

**Verify against a live run** rather than a fixture:

```bash
# copy the in-flight log and run the real parser over it
python tests/test_tokens.py
```

---

## Timeouts

### `AgentTimeoutError` on most tasks

**Cause.** Terminal-Bench budgets each task 900-1800 s, sized for frontier APIs. At ~25 tok/s
a trajectory needs on the order of 20-100 minutes of pure generation, so at `1.0×` almost
everything is killed mid-task, scoring identically to a wrong answer.

**Fix.** Measure your throughput with `harness-arena probe --speed` and take the multiplier it
recommends. Default is `16.0`.

**But check first.** Once the budget is roughly right, very few failures are timeouts. Most
end with *no exception*, meaning the agent stopped voluntarily without hitting its turn limit
either. More time cannot help a task the agent already considered finished. Look at the `T`
count before spending wall clock.

### `EnvironmentStartTimeoutError` after 600 s

**Cause.** Harbor pulls the task image inside the environment-start budget.
`mteb-retrieve` is 21.6 GB; `pytorch-model-recovery` is 14.7 GB and took 428 s alone.

**Fix.** `harness-arena prepull` before the run, plus
`environment_build_timeout_multiplier: 4.0` as a backstop.

---

## Logs and liveness

### Agent log is 0 bytes, or stops growing for minutes

**Two distinct causes:**

1. **`grep` block-buffering.** Without `--line-buffered`, `grep` holds ~4 KB before flushing.
   The log lags the run, and a trial killed at the timeout loses its tail. Fixed in
   `harnesses/omp.py`.
2. **Normal.** Output is piped through `tee`, which flushes per line. A long unbroken reasoning
   block emits no newline until it ends, silences of 6+ minutes are expected at ~30 tok/s. For
   omp specifically, `message_update` deltas are filtered out, so nothing is written until a
   whole message completes.

**Do not use log growth as a liveness check.** Ask the server instead:

```bash
curl -s http://localhost:8080/slots
# is_processing=True and a climbing n_prompt_tokens means it is generating
```

### The live feed shows machine exhaust, or shows nothing at all

Both symptoms are one cause: a log that is mostly not agent output.

| Harness | What floods it | Measured |
|---|---|---|
| Claude Code | `system`/`thinking_tokens`, one event per couple of tokens of reasoning | 32,505 of 32,666 lines; 91 % of 6.6 MB |
| Codex | Rust `tracing` at `INFO`, one line per streamed event, switched on by `--debug-capture` | 65,965 lines; 97 % of 24.9 MB |

Both are filtered from the **view** only; the log on disk keeps every line,
which is the whole point of `--debug-capture`. `WARN` and `ERROR` still render,
because a reset stream announces itself there.

**If the panel is empty**, check whether there is anything to show before
suspecting the filter:

```bash
LOG=runs/<job>/<task>__*/agent/codex.txt
# every line that is NOT Rust tracing
grep -cvE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z?[[:space:]]+(TRACE|DEBUG|INFO)' "$LOG"
```

A count in the single digits on a multi-MB log means the agent genuinely has
not emitted anything yet. It is mid-turn, and the feed says so rather than
going blank. The feed budgets *scanning* separately from *rendering* precisely
so it can reach content buried behind megabytes of exhaust.

**A screen of hex is not a bug.** An agent inspecting a binary (`od -A x -t x1z
-v model.ckpt`) produces exactly that, and it is tool output, not tracing.
Filtering it would hide the work.

### mtime says the log is stale but it is growing

Windows can report a stale mtime for many minutes on a file being appended through a Docker
bind mount, observed claiming 24 minutes of silence while the file grew 5 KB.
`bench/activity.py` tracks **size deltas across calls** instead.

---

## Containers and images

### Containers still running after a run is killed

A hard kill leaves them. Clean up:

```bash
docker ps -a --format "{{.Names}}" | grep "__env-main" | xargs -r docker rm -f
```
```powershell
# PowerShell
docker ps -a --format "{{.Names}}" | Select-String "__env-main" | ForEach-Object { docker rm -f $_ }
```

One `mteb-retrieve` container was observed still up **6 hours** after its trial failed.

### Disk filling

Task images total ~60 GB. `docker system df` shows reclaimable space; `docker image prune`
frees non-task images. Do **not** prune task images unless you want to re-pull ~60 GB.

### A run shows "running" forever

A job killed mid-flight never writes `finished_at`. Mark it stopped so its partial results read
as a baseline rather than an in-flight benchmark, write `stopped_at` and `stopped_reason` into
its `harness-bench.json`. `bench/collect.py` then reports `status: stopped`.

---

## Measurement mistakes

Recorded because they produced confident wrong numbers, not errors.

### "5.0 min/trial vs a 36 min baseline"

A watch timed from **its own start** rather than the run's, over a trial that had already
finished before it began. Implied a 7× speedup that did not exist. `bench/throughput.py` now
anchors on the job's recorded `started_at`.

### Trial-completion throughput understates a pipelined run

With trials in flight, their work is done but uncounted, so min/trial systematically punishes
pipelining. Use **llm busy %** for a live run; min/trial is only final once a run ends.

### Comparing runs at different time budgets

The pairing guard originally checked harness, subset and partial-status but **not** the timeout
multiplier, so an 8× run would pair against a 16× one and render a confident disagreement set
that was really measuring the budget. Fixed and pinned by `tests/test_collect.py`.

### KB/min as a proxy for tokens/s

Trajectory-size growth conflates tool output with generation and varies with how verbose a
model's reasoning is. Use the server's own `/slots` counters or the recorded token totals.

---

## Diagnostics

```bash
# does everything needed to run actually work?
harness-arena doctor

# what is the endpoint serving, how many slots, and how fast?
harness-arena probe --speed

# is it generating right now? (llama-server)
curl -s http://localhost:8080/slots

# what is in flight?
docker ps --format "{{.Names}} | {{.Status}}"

# progress and results
harness-arena collect
harness-arena throughput

# per-phase breakdown of a finished trial
python -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps({k:d.get(k) for k in ('task_name','verifier_result','exception_info','agent_execution')},indent=2,default=str))" \
  runs/<job>/<trial>/result.json

# why did a task fail?
grep -E "assert|FAILED" runs/<job>/<trial>/verifier/test-stdout.txt

# what did the agent actually do?
tail -60 runs/<job>/<trial>/agent/<harness>.txt
```

Harbor's own viewer gives trial-level drill-down the dashboard does not:

```bash
harbor view runs
```
