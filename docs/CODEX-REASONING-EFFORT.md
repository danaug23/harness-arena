# Codex reasoning effort: the bug, the fix, and what still needs validating

**Status:** implemented, and now measured against **Ollama 0.32.5** *and*
**llama.cpp (`llama-server` b10269)**. The second machine closed Q1. The fix is
a no-op on llama.cpp, as required, and in doing so exposed five further defects
and one wrong claim in this document, all corrected here.
See [§14](#14-second-pass-what-the-llamacpp-run-changed).
**Needs validation against:** vLLM, a hosted provider, and a *non-reasoning*
model on llama.cpp.
**Date:** 2026-08-12 · **Versions:** codex-cli 0.147.0, harbor 0.20.0,
ollama 0.32.5, llama.cpp b10269

This document exists so someone on a different machine, with different model
servers, can decide whether the fix is correct, or prove it wrong. It is
deliberately long: it records what was measured, what was only inferred, and
exactly which claims are still untested.

Sections 1-13 are the original write-up, amended where the second machine
contradicted them. §14 is what that machine changed.

---

## Contents

- [1. What this rig is trying to do](#1-what-this-rig-is-trying-to-do)
- [2. How the bug arrived](#2-how-the-bug-arrived)
- [3. What the failure looked like](#3-what-the-failure-looked-like)
- [4. Root cause](#4-root-cause)
- [5. What Codex actually puts on the wire](#5-what-codex-actually-puts-on-the-wire)
- [6. What the server actually accepts](#6-what-the-server-actually-accepts)
- [7. Why the obvious fixes are wrong](#7-why-the-obvious-fixes-are-wrong)
- [8. The fix](#8-the-fix)
- [9. Evidence the fix works](#9-evidence-the-fix-works)
- [10. What needs validating elsewhere](#10-what-needs-validating-elsewhere)
- [11. Validation procedure](#11-validation-procedure)
- [12. How to prove the fix wrong](#12-how-to-prove-the-fix-wrong)
- [13. Known remaining gap: wire_api](#13-known-remaining-gap-wire_api)
- [14. Second pass: what the llama.cpp run changed](#14-second-pass-what-the-llamacpp-run-changed)

---

## 1. What this rig is trying to do

harness-bench runs several agent harnesses (hermes, omp, opencode, minion,
Claude Code, Codex) against **one model on one endpoint**, on Terminal-Bench 2,
and compares them. The entire value of the comparison rests on one idea:

> Everything except the harness is held constant, and anything that could not
> be held constant is recorded in the run manifest.

That is why the rig tells every harness the same context window, the same
output-token ceiling, the same base URL, and records where each of those
numbers came from. A run that cannot say what it held constant is not a
measurement.

Reasoning effort is one of those held-constant settings, and until this fix it
was being held constant at a value that some servers reject outright.

---

## 2. How the bug arrived

Codex was added as a harness in commit `5b2270e` *"Add Claude Code and Codex as
harnesses against a local model"*, together with `harnesses/codex.py` and a
registry block in `harnesses/registry.yaml`.

`harnesses/codex.py` is a thin subclass of Harbor's built-in Codex agent. Its
module docstring is explicit that its three design decisions, keeping `/v1` on
the base URL, using a named provider block, and `wire_api = "responses"`, were
**measured against codex-cli 0.147.0 pointed at llama.cpp**, not read from
documentation. That is good practice, and it is also exactly where the bug came
from: everything was verified against **one server implementation**.

The adapter was then run on a second machine whose endpoint is **Ollama 0.32.5**
serving `llama3.2:latest`. Every trial failed.

Nothing in the repo was wrong *for llama.cpp*. The adapter simply inherited a
setting whose acceptability is a property of the server, and never checked it.

---

## 3. What the failure looked like

Run `runs/codex__llama3-2-latest-3735d814__20260811T194632Z`:

- 89 trials, **100 % `NonZeroAgentExitCodeError`**
- zero input tokens, zero output tokens, no trajectory, no score
- every task attempted **twice**, because the run used
  `max_retries: 1` with `retry_include: ["NonZeroAgentExitCodeError"]`, so the
  failure mode was on the retry list and doubled the wall clock
- ran for roughly 90 minutes and produced nothing

Compare the sibling harness in the same batch, `claude-code`, against the same
endpoint, which ran normally (tokens counted, verifier scored):

```
codex        n=89   all NonZeroAgentExitCodeError, 0 tokens
claude-code  n=89   agent ran, tokens counted, verifier reward 0.0
hermes / omp / opencode / minion   ran normally
```

The run-level error is unhelpful. It quotes the whole `codex exec` command line
and says `Command failed (exit 1)`. Nothing at that level mentions reasoning.

This is the dangerous shape of failure for a benchmark: **a configuration
mismatch that is indistinguishable from "this harness is bad at the tasks."**
If the exception type had been swallowed slightly differently, it would have
been published as a 0 % score for Codex.

The actual cause is one level down, in the agent log
(`runs/<job>/<task>__*/agent/codex.txt`):

```json
{"type":"item.completed","item":{"id":"item_0","type":"error",
 "message":"Model metadata for `llama3.2:latest` not found. Defaulting to
 fallback metadata; this can degrade performance and cause issues."}}
{"type":"turn.started"}
{"type":"error","message":"{\"error\":{\"message\":\"\\\"llama3.2:latest\\\"
 does not support thinking\",\"type\":\"invalid_request_error\"}}"}
{"type":"turn.failed","error":{...same...}}
```

The endpoint returned **HTTP 400 `"llama3.2:latest" does not support
thinking`** on the very first request. Codex failed the turn and exited 1.

---

## 4. Root cause

Harbor's built-in Codex agent declares a CLI flag with a hard default
(`harbor/agents/installed/codex.py`):

```python
CLI_FLAGS = [
    CliFlag(
        "reasoning_effort",
        cli="-c",
        type="str",
        default="high",                       # <-- always applied
        format="-c model_reasoning_effort={value}",
    ),
    ...
]
```

`harnesses/codex.py` extends that list rather than replacing it:

```python
CLI_FLAGS = [
    *_HarborCodex.CLI_FLAGS,     # <-- inherits reasoning_effort="high"
    ...
]
```

So every run issued `-c model_reasoning_effort=high`, Codex turned that into a
`reasoning` object on each `/v1/responses` request, and Ollama refused it.

**The key insight:** whether a `reasoning.effort` is acceptable is a fact about
the *server and the loaded model*, not about Codex and not about this rig. It
therefore cannot be a constant in the registry. It has to be discovered, the
same way the context window already is.

---

## 5. What Codex actually puts on the wire

Measured directly, not inferred. A local HTTP server logged the exact JSON body
Codex posted, then returned 400 so Codex would exit immediately. The script is
reproduced in [§11](#11-validation-procedure) so this can be repeated.

**codex-cli 0.147.0, `wire_api = "responses"`:**

| Codex configuration | `reasoning` field in the request body |
|---|---|
| `-c model_reasoning_effort=high` (Harbor's default) | `{"effort":"high","summary":"auto"}` |
| `-c model_reasoning_effort=none` | `{"effort":"none","summary":"auto"}` |
| *flag omitted entirely* | `{"summary":"auto"}` |
| `-c model_reasoning_summary=none` | `{}` |
| `-c model_reasoning_effort=none -c model_reasoning_summary=none` | `{"effort":"none"}` |
| `-c model_supports_reasoning_summaries=false` | `{"summary":"auto"}` |

**Conclusion: Codex 0.147.0 never omits the `reasoning` key on the Responses
wire.** There is no configuration that makes it stop asking. The only lever is
*what effort it asks for*.

---

## 6. What the server actually accepts

**Ollama 0.32.5, `llama3.2:latest`, POST `/v1/responses`.** Requests were made
against a warmed model, a cold server times out and returns a misleading `000`,
which is worth knowing before repeating this.

| `reasoning` payload | HTTP | Note |
|---|---|---|
| *(field absent)* | 200 | baseline works |
| `{}` | 200 | |
| `{"summary":"auto"}` | 200 | |
| `{"summary":"none"}` | 200 | |
| `{"effort":"none"}` | 200 | |
| `{"effort":"none","summary":"auto"}` | **200** | what `effort=none` produces |
| `{"effort":"minimal"}` | 400 | `does not support thinking` |
| `{"effort":"low"}` | 400 | `does not support thinking` |
| `{"effort":"medium"}` | 400 | `does not support thinking` |
| `{"effort":"high","summary":"auto"}` | **400** | what shipped |

**The boundary is precise:** Ollama rejects any *real* effort for a model that
cannot think, and accepts `"none"` or no effort key at all. The `summary`
sub-field is irrelevant to the rejection.

This endpoint also serves `/v1/messages` (which is why Claude Code works against
it) and `/v1/chat/completions`.

---

## 7. Why the obvious fixes are wrong

Three fixes suggest themselves. Two are wrong and one is unacceptable. Recording
them because a reviewer will think of them too.

**"Just omit the flag."** Wrong, measured in §5. Omitting it still sends
`{"summary":"auto"}`. That happens to be accepted by Ollama, but only by
accident: it leaves Codex's own fallback-metadata behaviour in charge, which is
unstated, unrecorded, and free to change between Codex versions. A benchmark
must not depend on an undocumented default.

**"Set `reasoning_effort: none` in `registry.yaml`."** Works on Ollama, breaks
the comparison everywhere else. It silently disables thinking on a server that
supports it, so a Codex run against llama.cpp with a reasoning model would
quietly measure a non-reasoning configuration. That is precisely the class of
error this rig exists to prevent.

**"Tell the operator to configure around it."** Unacceptable for a public repo.
The failure gives no usable signal at the run level, costs ~90 minutes, and
requires the operator to already know that Codex sends a reasoning object they
cannot see.

---

## 8. The fix

The repo already had the right pattern. `bench/runner.py::effective_context()`
resolves the context window from three sources in precedence order,
**configured → detected → fallback**: and records *which*, because "4096,
detected" and "4096, because nothing knew" are different claims about a run.

Reasoning effort is the same kind of fact, so it now works the same way.

### 8.1 Probe, `bench/probe.py::supports_reasoning_effort()`

Asks the endpoint once whether it accepts a real effort:

1. POST `/responses` with `{"reasoning":{"effort":"high"}}`,
   `max_output_tokens: 16`, `high` because that is exactly what a run sends
   when the answer comes back `True`. Probing one effort and shipping another
   leaves the only case that matters untested.
2. **404**, and `base_url` carries no `/v1` suffix → retry at
   `{base}/v1/responses`. The rig accepts a base URL written either way and the
   model lookup already tolerates both, so this has to as well.
3. **200** → `True` (endpoint accepts an effort).
4. **Nothing landed**, unreachable, or too slow to answer → `None`.
5. **Any other status** → repeat the identical request **without** the
   `reasoning` object. If that succeeds → `False`. If it also fails → `None`.

Step 5 is the subtle part and the most important thing for a reviewer to check.
Without the control request, a wrong model id or a missing credential would
produce a 400 that reads as "this server hates reasoning", and every future run
against that endpoint would be silently downgraded to non-thinking.

It is also why the status list is *not* narrowed to 400/422. The control
request, not the status code, is what makes a refusal safe to act on, so every
rejection is put through it. Servers do not agree on how to spell "I will not do
that", and one that answers 500 to an unsupported parameter would otherwise fall
through to the fallback and take a whole run with it.

Nothing in the probe raises, including the model-id lookup it needs when
`endpoint.model` is empty. It runs on the run path, where an exception would
abort exactly the runs the `None` fallback exists to leave alone.

Result is cached in `bench/models.json` under the model fingerprint **and the
base URL it was measured against**. Keying on the weights alone would be wrong
on this fix's own terms: acceptance is a property of the *server*, and the same
GGUF answers differently behind llama.cpp and behind Ollama, so repointing
`base_url` would otherwise silently reuse the first server's answer. A sweep
still costs one extra request, not six. Only definite booleans are trusted from
cache. An endpoint that was merely unreachable is asked again next time rather
than pinned to "unknown" forever, and a bare boolean left by the first version
of this fix is discarded rather than guessed at, because it cannot be attributed
to any endpoint.

Called from `probe.resolve()`, which is the run/setup path. The dashboard uses
`probe.probe()` and is unaffected, so no page load pays for this.

### 8.2 Resolution, `bench/runner.py::effective_reasoning_effort()`

Pure function, no I/O, mirrors `effective_context()`:

| source | condition | value |
|---|---|---|
| `configured` | `endpoint.reasoning_effort` set in `config.yaml` | that value |
| `probed` | endpoint accepts an effort | `high` (Harbor's default) |
| `probed` | endpoint refuses an effort | `none` |
| `fallback` | probe could not answer | `high` (unchanged behaviour) |

**The fallback is deliberately the old behaviour.** Any setup that works today
and cannot be probed keeps doing exactly what it did before this change.

### 8.3 Wiring

- `bench/registry.py`, `reasoning_effort` added to `KNOWN_PLACEHOLDERS`.
- `harnesses/registry.yaml`, codex block gains `reasoning_effort: "{reasoning_effort}"`.
- `bench/config.py`, `EndpointConfig.reasoning_effort: str = ""` (empty = ask the endpoint).
- `bench/runner.py`, manifest records `reasoning_effort` and
  `reasoning_effort_source`, derived from the same function that built the
  command so the two cannot drift.
- `bench/collect.py` / `dashboard/index.html`, carried through and displayed
  when it is not the default, because a run that reasoned and one that did not
  are not two measurements of the same thing.
- `bench/runner.py`, `main()` prints the resolved effort before a run that can
  use it, and warns when the source is `fallback`. See §14.2 for why that
  warning is not cosmetic.
- `bench/cli.py`, `doctor` reports whether the endpoint accepts an effort, so
  the answer costs one request up front instead of 90 minutes of dead trials.
- `harnesses/codex.py`, **unchanged**. The adapter did not need to know.

### 8.4 Files changed

```
bench/probe.py             capability probe + ModelIdentity.supports_reasoning
bench/runner.py            effective_reasoning_effort, substitution, manifest
bench/registry.py          {reasoning_effort} placeholder
bench/config.py            endpoint.reasoning_effort
bench/collect.py           carry both fields into the run record
harnesses/registry.yaml    codex block uses the placeholder
dashboard/index.html       show a non-default effort
config.example.yaml        document the knob
docs/ARCHITECTURE.md       manifest shape
docs/TROUBLESHOOTING.md    the symptom and how to read it
README.md                  why the effort is probed
tests/test_local_agents.py 24 new checks (53 total, from 29)
```

---

## 9. Evidence the fix works

**On Ollama 0.32.5 + llama3.2:latest only.** Everything below is reproducible.

Real Codex CLI, real endpoint, before and after:

```
=== effort=high (what the repo shipped) ===
{"type":"turn.failed","error":{"message":"... does not support thinking ..."}}

=== effort=none (what the probe now selects) ===
{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}
{"type":"turn.completed","usage":{"input_tokens":8494,"output_tokens":2,
 "reasoning_output_tokens":0}}
```

The rig end to end:

```
model: llama3.2:latest | supports_reasoning: False
effective_reasoning_effort: ('none', 'probed')
codex --ak flags: ['base_url=http://<host>:11434/v1', 'context_window=65536',
                   'wire_api=responses', 'reasoning_effort=none']
```

All seven test suites pass (`test_api`, `test_collect`, `test_config`,
`test_local_agents`, `test_supervisor`, `test_tokens`, `test_dashboard.mjs`),
`ruff` clean, all 6 registry blocks validate.

---

## 10. What needs validating elsewhere

Everything above was measured against **one server**. The fix's whole premise is
that servers disagree, so validating it against one server proves very little.
These are the open questions, most important first.

### Q1, llama.cpp (`llama-server`): does the probe return `True`? **ANSWERED: yes**

Measured 2026-08-12 against `llama-server` b10269 serving
`qwen3.6-35b-a3b` (Qwen3.6 35B A3B, UD-Q4_K_M), POST `/v1/responses`:

| `reasoning` payload | HTTP |
|---|---|
| *(absent)* | 200 |
| `{}` | 200 |
| `{"summary":"auto"}` | 200 |
| `{"effort":"none"}` | 200 |
| `{"effort":"none","summary":"auto"}` | 200 |
| `{"effort":"minimal"}` | 200 |
| `{"effort":"low"}` | 200 |
| `{"effort":"medium"}` | 200 |
| `{"effort":"high","summary":"auto"}` | 200 |

llama.cpp accepts every effort, so the probe returns `True` on its first request
and never issues the control request. Through the real code path:

```
endpoint: http://<host>:8002
supports_reasoning_effort -> True   (0.2s)
effective_reasoning_effort: ('high', 'probed')
--ak: ['base_url=http://<host>:8002', 'context_window=131072',
       'wire_api=responses', 'reasoning_effort=high']
```

`high` is `DEFAULT_REASONING_EFFORT`, i.e. byte-identical to what Harbor sent
before this change. **The fix is a no-op on llama.cpp. No regression.**

Confirmed independently by a full 25-task sweep run on that machine with the
*pre-fix* code: Codex produced tokens, trajectories and verifier scores on every
task, which is the behaviour the fix must preserve.

Two things this turned up that the original write-up did not expect:

- llama.cpp does not merely tolerate the effort, it **honours** it. Response
  bodies carry a real `"type":"reasoning"` block, and on an idle server
  `effort=none` returned in 0.9 s against 29.7 s for `effort=high`. So resolving
  to `none` on a refusing server genuinely changes the experiment. It is not
  just a wire-format difference.
- Probe latency is dominated by **slot contention**, not by cold start. The same
  request took 0.2 s on an idle server and 25-30 s while a benchmark held the
  single slot. §10 Q5 framed this as a cold-server concern; the busy case is the
  common one and the slower one.

Still open: a **non-reasoning** GGUF on llama.cpp. The belief that llama.cpp
accepts an effort from any model is now measured for one reasoning model only,
and one model is not the claim.

### Q2, vLLM: does it implement `/v1/responses` at all?

Unknown and important. Three outcomes:

- **Serves `/v1/responses` and accepts an effort** → probe `True`, effort
  `high`, Codex works. Best case.
- **Serves `/v1/responses` and refuses an effort** → probe `False`, effort
  `none`, Codex works. The fix earns its keep on a second server family.
- **Does not serve `/v1/responses` (404)** → probe returns `None`, effort falls
  back to `high`, and **Codex still fails**, for an entirely different reason
  covered in [§13](#13-known-remaining-gap-wire_api). This would not be a fault
  in this fix, but it must be reported, because it means Codex on vLLM needs
  `wire_api: chat` and that is still hard-coded.

Please record which of the three you observe, with the raw HTTP status.

### Q3, a thinking-capable model on Ollama: does the probe return `True`?

The probe must *discriminate*, not just always say `False` on Ollama. Load a
reasoning model (`qwen3`, `deepseek-r1`, `gpt-oss`) and confirm the probe flips
to `True` and the effort resolves to `high`. If it returns `False` for a model
that genuinely can think, the fix is downgrading runs it should not.

### Q4, a hosted provider (OpenAI, OpenRouter): is behaviour unchanged?

The probe should return `True` (these accept `effort: "low"`), giving `high`,
the status quo. The failure mode to watch for is a probe that returns `False` and
sends `effort: "none"` to a provider that does not accept that spelling.

### Q5, cost and latency of the probe

It runs once per fingerprint-and-endpoint and asks for 16 output tokens.
Measured at 0.2 s on an idle llama.cpp server; 25-30 s for the same request
while a benchmark held the server's only slot. Default timeout is 90 s, and a
run that needs the control request can pay it twice.

Confirm it does not add meaningful time to run startup, and that a server which
cannot answer degrades to `None` rather than hanging.

**But `None` is not harmless, and the original draft of this document was wrong
to call it that.** The fallback is safe in exactly the place it is not needed
and unsafe in exactly the place it is:

- Server *accepts* an effort (llama.cpp): probe fails → `high` → run works. This
  is the case that never needed the fix.
- Server *refuses* an effort (Ollama): probe fails → `high` → **every trial dies
  at its first request**. That is the original bug, restored in full.

So the fix turns "always broken on Ollama" into "broken on Ollama whenever the
probe cannot get an answer", a large improvement, but not a guarantee, and
every inconclusive path is a landmine on precisely the server family this exists
for. Since a busy single-slot server is the rig's normal operating state, that
path is reachable in ordinary use.

The runner therefore prints the effort and its source before any run that can
use one, and says plainly what a `fallback` means. It does not refuse to start:
the fallback is still the old behaviour, and a server without `/responses` at
all is the §13 problem, not this one.

### Q6, cache behaviour

Confirm `bench/models.json` gains `supports_reasoning` for the fingerprint, that
a second run does not re-probe, and that swapping the loaded weights (new
fingerprint) causes a fresh probe.

---

## 11. Validation procedure

Run these on the target machine. `$BASE` is the endpoint base URL **including
`/v1`**; `$MODEL` is the served model id.

```bash
BASE=http://127.0.0.1:8080/v1      # llama-server default; vLLM often :8000
MODEL=your-model-id
```

### Step 1, Does the endpoint serve the Responses API at all?

```bash
curl -s -o /dev/null -w 'responses: %{http_code}\n' -X POST "$BASE/responses" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":\"hi\",\"max_output_tokens\":16}"
```

`200` → continue. `404` → **stop and record**: this is the §13 gap, not this fix.

> Warm the model first. A cold server can time out and report `000`, which looks
> like a rejection and is not one.

### Step 2, The acceptance matrix

```bash
for R in '{}' '{"summary":"auto"}' '{"effort":"none"}' '{"effort":"low"}' \
         '{"effort":"medium"}' '{"effort":"high","summary":"auto"}'; do
  printf '%-38s -> ' "$R"
  curl -s -o /tmp/r -w '%{http_code}' -X POST "$BASE/responses" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":\"hi\",\"max_output_tokens\":16,\"reasoning\":$R}"
  echo "  $(head -c 120 /tmp/r)"
done
```

Record the full table. This is the single most valuable artifact to bring back.

### Step 3, What the rig's probe concludes

```bash
conda activate harness-bench     # or your venv
python -c "
from bench.config import load
from bench.probe import supports_reasoning_effort
cfg = load()
print('endpoint:', cfg.endpoint.resolved_base_url())
print('supports_reasoning_effort ->', supports_reasoning_effort(cfg.endpoint))"
```

Must agree with Step 2: `True` if a real effort returned 200, `False` if a real
effort returned 400 **and** the no-reasoning request returned 200, else `None`.

### Step 4, What the rig would actually run

```bash
python -c "
import tempfile; from pathlib import Path
from bench import REGISTRY_PATH
from bench.config import load
from bench.registry import load as load_registry
from bench.probe import resolve
from bench.runner import build_command, effective_reasoning_effort
cfg = load(); reg = load_registry(REGISTRY_PATH)
m = resolve(cfg.endpoint, interactive=False)
print('supports_reasoning:', m.supports_reasoning)
print('effort:', effective_reasoning_effort(m, cfg))
with tempfile.TemporaryDirectory() as t:
    argv,_ = build_command('codex', reg['harnesses']['codex'], m, reg, cfg,
        dataset='terminal-bench@2.0', jobs_dir=Path(t), name='j', n_concurrent=1,
        n_attempts=1, n_tasks=None, include_tasks=None, extra_args=None,
        allow_hosts=False, agent_timeout_multiplier=1.0, n_concurrent_agents=1,
        env_build_timeout_multiplier=None, max_retries=0, retry_include=None)
print('--ak:', [argv[i+1] for i,a in enumerate(argv) if a=='--ak'])"
```

### Step 5, Real Codex against the real endpoint

Installs Codex locally, not globally:

```bash
mkdir -p /tmp/codexcheck && cd /tmp/codexcheck
npm install @openai/codex@0.147.0 --no-audit --no-fund
export OPENAI_API_KEY=local CODEX_HOME=/tmp/codexcheck/home && mkdir -p home

CFG="-c model_provider=local \
 -c model_providers.local.base_url=\"$BASE\" \
 -c model_providers.local.wire_api=\"responses\" \
 -c model_providers.local.env_key=\"OPENAI_API_KEY\" \
 -c model_providers.local.name=\"hb\" \
 -c model_context_window=32768"

for EFFORT in high none; do
  echo "=== effort=$EFFORT ==="
  eval ./node_modules/.bin/codex exec \
    --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
    --model "$MODEL" --json $CFG -c model_reasoning_effort=$EFFORT \
    -- "'reply with the word OK'" 2>&1 \
    | grep -E '"type":"(turn\.failed|turn\.completed|agent_message)"' | head -3
done
```

On a server that accepts effort, **both** should complete. On one that refuses,
`high` fails and `none` completes.

### Step 6, Capture what Codex sends (optional, for wire-level disputes)

```python
# logsrv.py, logs the JSON body Codex posts, then 400s so Codex exits
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
OUT = sys.argv[1] if len(sys.argv) > 1 else "capture.json"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8931

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        json.dump({"path": self.path, "body": json.loads(raw)},
                  open(OUT, "w"), indent=1)
        p = b'{"error":{"message":"captured","type":"invalid_request_error"}}'
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.end_headers(); self.wfile.write(p)
    def log_message(self, *a): pass

HTTPServer(("127.0.0.1", PORT), H).serve_forever()
```

Run it, point Codex at `http://127.0.0.1:8931/v1`, and read `capture.json` to
see the exact `reasoning` object for any configuration.

### Step 7, Test suites

```bash
python tests/test_local_agents.py     # 53 checks, includes the probe logic
python -m ruff check bench harnesses tests
```

`test_local_agents.py` is hermetic (its probe tests spin a local HTTP server on
a random port) and needs a working Harbor import but no live endpoint.

### Step 8, A short real run

```bash
harness-bench run --harness codex --n-tasks 2
python -c "
import json;m=json.load(open('runs/<job>/harness-bench.json'))
print(m['reasoning_effort'], m['reasoning_effort_source'])"
```

Trials should produce tokens and reach the verifier, any score, including 0, is
fine. What must **not** happen is `NonZeroAgentExitCodeError` on every trial with
zero tokens.

---

## 12. How to prove the fix wrong

Any of these means it needs rework, not a tweak:

1. **llama.cpp probes as anything but `True`.** Regression on the reference
   setup. Blocker.
2. **A reasoning-capable model probes as `False`.** The fix would be disabling
   thinking on servers that support it, worse than the original bug, because it
   is silent and produces plausible results.
3. **The probe returns `False` when the real cause is a bad model id or missing
   credential.** The control request in step 4 of §8.1 is supposed to prevent
   exactly this; if it does not, every run against a misconfigured endpoint gets
   silently downgraded.
4. **The probe hangs or materially slows run startup**, especially on a cold or
   contended server. It must degrade to `None`, never block, and because
   `None` hands back the effort that breaks a refusing server, a run that takes
   that path has to *say so*, not take it quietly.
5. **A hosted provider rejects `effort: "none"`** after being probed as `False`.
   Would mean `none` is not a universal spelling and the "off" value must become
   server-dependent too.
6. **The manifest disagrees with the command.** `reasoning_effort` in
   `harness-bench.json` must equal the `--ak reasoning_effort=` value in the
   recorded `command`. They are derived from one function specifically so they
   cannot diverge.

---

## 13. Known remaining gap: `wire_api`

Not fixed, and it is the same class of bug.

`harnesses/registry.yaml` hard-codes `wire_api: "responses"` for Codex. A server
that does not implement `/v1/responses`, some vLLM deployments, will fail on
**every trial**, with the same uninformative `NonZeroAgentExitCodeError` and the
same doubled cost from retries. The registry comment already tells you to set
`wire_api: chat` by hand, which is the same "make the operator work around it"
answer this fix rejected for reasoning effort.

The same three-line pattern would close it: probe which of `/v1/responses` and
`/v1/chat/completions` the endpoint answers, expose `{wire_api}`, record the
choice in the manifest. It was **deliberately not implemented here**, because
there was no vLLM deployment available to measure against, and this repo's
standard is that server behaviour is measured rather than assumed.

If your validation covers vLLM, the data from §11 Step 1 is exactly what is
needed to close this properly. Please bring back the raw status codes for both
routes:

```bash
curl -s -o /dev/null -w 'responses:        %{http_code}\n' -X POST "$BASE/responses" \
  -H 'content-type: application/json' -d "{\"model\":\"$MODEL\",\"input\":\"hi\"}"
curl -s -o /dev/null -w 'chat/completions: %{http_code}\n' -X POST "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"
```

---

## 14. Second pass: what the llama.cpp run changed

Validating on a second server was the point of writing §10, and it did its job.
Q1 came back clean. The fix is a no-op on llama.cpp, but reviewing the code
against a *live* second endpoint exposed five defects and one wrong claim. All
are fixed; each has a regression test.

The pattern worth noticing: every one of them is the same mistake the original
bug was, one level up. The first version measured Codex against one server and
froze the answer. The fix measured the *probe* against one server and froze
several answers about how servers fail.

### 14.1 The probe could abort a run it was meant to protect

`supports_reasoning_effort()` opened with
`endpoint.model or _served_id_for_speed(endpoint)`, and `_served_id_for_speed`
calls `probe()`, which **raises** `RuntimeError` when `/models` does not answer.
That exception propagated out of `resolve()` and killed the whole run.

Every other failure in the probe degrades to `None` by design. This one did not,
and it was reachable in the default configuration: `endpoint.model` is empty for
a self-hosted server, so any transient blip on the second `/models` call took
the run down. A function whose entire contract is "never make things worse" must
not have an exception path.

Fixed: the resolved `served_id` is passed in from `resolve()`, which already
has it, removing two redundant round trips as well, and the remaining lookup
is wrapped so it returns `None` instead of raising.

### 14.2 The fallback was documented as harmless; it is not

Covered in [§10 Q5](#q5--cost-and-latency-of-the-probe). `None` → `high` is
safe on servers that accept an effort and fatal on servers that refuse one,
which is backwards from where the safety is needed.

Fixed: `bench/runner.py::main()` now prints the effort and its source before any
run whose harnesses actually use one, and states what a `fallback` implies. It
mirrors the existing `context_window` fallback warning, the same shape of
problem deserved the same treatment, and did not have it.

### 14.3 The cache key ignored the server

`supports_reasoning` was cached under the **weights fingerprint**, which
contains no host and no base URL. The fix's own thesis is that acceptance is a
property of the server, so this filed a server fact under a weights key.

Point `base_url` at a different server family serving the same model id and the
first server's answer is reused, and because it is cached, stickily. Ollama
has no `/props`, so almost all of its fingerprint material is `None`, which
makes a collision easier rather than harder.

Fixed: the answer is stored per base URL. A bare boolean from the previous
version is discarded rather than attributed to whichever endpoint asks next.

### 14.4 A 500 was treated as "no answer"

The original narrowed refusals to 400/422 and sent everything else to the
fallback, and a test pinned that behaviour in place. But the safety comes from
the **control request**, not from the status code: if the same request without
`reasoning` succeeds, the server refused the reasoning, whatever number it
chose to say so with. A server answering 500 to an unsupported parameter would
have fallen through to `high` and taken the run with it.

Fixed: every non-200 that landed goes through the control request. Only "nothing
landed at all" short-circuits to `None`.

### 14.5 A `/v1`-less `base_url` sent the probe to a route that need not exist

`_post_responses` built `{base_url}/responses`. But the rig accepts a base URL
written with or without `/v1`, `_probe_self_hosted` explicitly tries both for
`/models`, and this machine's own `config.yaml` has no suffix.

llama.cpp answers both, so it was invisible here. Ollama serves the OpenAI
routes only under `/v1`, so on the very machine that motivated this fix, a
config written the other way would have 404'd the probe into the fallback and
resurrected the original bug. Measured, not assumed: both routes returned 200 on
llama.cpp b10269.

Fixed: a 404 with no `/v1` in the base URL retries at `{base}/v1/responses`, and
the control request is sent to whichever URL answered, otherwise the two
requests would not be about the same route.

### 14.6 The probe asked about an effort it would never send

It asked `{"effort":"low"}` and, on success, sent `high`. A server accepting one
and refusing the other would be misread. No such server has been seen, llama.cpp
takes both, but the asymmetry bought nothing.

Fixed: the probe asks for `high`, and a test asserts `probe.PROBE_EFFORT ==
runner.DEFAULT_REASONING_EFFORT` so the two cannot drift apart.

### 14.7 `doctor` never mentioned it

`cmd_doctor` uses `probe()`, which leaves `supports_reasoning` at `None`, so the
one command whose job is "find out before you spend hours" was silent about the
setting that costs 90 minutes when it is wrong.

Fixed: `doctor` reports it, in the same block as the model identity:

```
        fingerprint  802849a6f83a8fac
        reasoning    accepts a reasoning effort (Codex will think)
```

### 14.8 Corrections to this document

- §8.4 claimed "20 new checks" and §11 Step 7 claimed "36 checks". Both were
  wrong: the branch added 10 checks, taking the suite from 29 to 39. After this
  pass it is **53**, of which **24** are new relative to `main`.
- §10 Q5 and §12 item 4 described the `None` fallback as harmless. Corrected.
- §8.1's step list described the old 400/422 logic and the fingerprint-only
  cache. Rewritten.

### 14.9 Still not validated

Unchanged from §10: vLLM (Q2), a thinking-capable model on Ollama (Q3), and a
hosted provider (Q4). New: a **non-reasoning GGUF on llama.cpp**, which is the
cheap one, load a Llama 3 GGUF and rerun §11 Step 2. It would tell you whether
"llama.cpp accepts an effort from any model" is a fact about the server or about
the model that happened to be loaded.

The `wire_api` gap in §13 is also untouched, and still wants a vLLM deployment.
