"""Guard the two CLI harnesses that talk to the endpoint in their own dialect.

Claude Code and Codex are the only harnesses here that do not speak
OpenAI-on-/v1. Claude Code uses the Anthropic Messages API and its client
appends "/v1/messages" itself; Codex uses the Responses API and appends only
"/responses". So one wants the endpoint URL with its "/v1" removed and the
other wants it kept -- from the same `base_url` in config.yaml.

That is a silent failure in both directions: the wrong spelling produces
/v1/v1/messages or a bare /responses, every trial fails identically at the
first request, and the run looks like the harness cannot solve anything rather
than like a URL bug. Both spellings are asserted here, together, because the
bug is the pair disagreeing.

    python tests/test_local_agents.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import (
    REGISTRY_PATH,  # noqa: E402
    runner,  # noqa: E402
)
from bench.config import Config, ConfigError, EndpointConfig  # noqa: E402
from bench.probe import ModelIdentity  # noqa: E402
from bench.registry import (  # noqa: E402
    KNOWN_PLACEHOLDERS,
    RegistryError,
    load,
    validate_harness,
)
from bench.runner import base_url_root, build_command  # noqa: E402

# Keep the suite hermetic. build_command only needs argv[0]; requiring the real
# console script would make this fail on PATH rather than on the thing under
# test, and none of the other suites need an installed Harbor either.
runner.harbor_executable = lambda: "harbor"

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<56} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def check_true(label: str, got: object) -> None:
    check(label, bool(got), True)


# ---------------------------------------------------------------------------
# base_url_root
# ---------------------------------------------------------------------------

print("\n-- base_url_root --")
check("strips /v1", base_url_root("http://example.invalid:8002/v1"),
      "http://example.invalid:8002")
check("strips /v1 with trailing slash", base_url_root("http://example.invalid:8002/v1/"),
      "http://example.invalid:8002")
check("strips any version segment", base_url_root("http://example.invalid/v2"),
      "http://example.invalid")
check("no-op without a version segment", base_url_root("http://example.invalid:8002"),
      "http://example.invalid:8002")
check("idempotent", base_url_root(base_url_root("http://example.invalid/v1")),
      "http://example.invalid")
# A path that merely contains "v1" is not a version suffix; stripping it would
# silently retarget a gateway that serves the endpoint under a sub-path.
check("leaves a mid-path segment alone", base_url_root("http://example.invalid/v1/proxy"),
      "http://example.invalid/v1/proxy")
check("empty stays empty", base_url_root(""), "")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

print("\n-- registry --")
check_true("base_url_root is a known placeholder", "base_url_root" in KNOWN_PLACEHOLDERS)

registry = load(REGISTRY_PATH)
harnesses = registry.get("harnesses") or {}
check_true("claude-code is registered", "claude-code" in harnesses)
check_true("codex is registered", "codex" in harnesses)

for harness_id in ("claude-code", "codex"):
    try:
        validate_harness(harness_id, harnesses[harness_id])
        ok = True
    except RegistryError as exc:  # pragma: no cover - only on a broken block
        ok = False
        print(f"      {harness_id}: {exc}")
    check(f"{harness_id} block validates", ok, True)


# ---------------------------------------------------------------------------
# The command the runner actually builds
# ---------------------------------------------------------------------------

print("\n-- build_command --")

MODEL = ModelIdentity(
    served_id="a-model",
    fingerprint="deadbeefdeadbeef",
    label="A Model",
    base_url="http://example.invalid:8002/v1",
    host="example.invalid",
    n_ctx=131072,
)
CONFIG = Config(endpoint=EndpointConfig(base_url="http://example.invalid:8002/v1"))


def command_for(
    harness_id: str, model: ModelIdentity | None = None, config: Config | None = None
) -> tuple[list[str], dict[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        return build_command(
            harness_id,
            harnesses[harness_id],
            model or MODEL,
            registry,
            config or CONFIG,
            dataset="terminal-bench@2.0",
            jobs_dir=Path(tmp),
            name="job",
            n_concurrent=1,
            n_attempts=1,
            n_tasks=None,
            include_tasks=None,
            extra_args=None,
            allow_hosts=False,
            agent_timeout_multiplier=1.0,
            n_concurrent_agents=1,
            env_build_timeout_multiplier=None,
            max_retries=0,
            retry_include=None,
        )


claude_argv, claude_host_env = command_for("claude-code")
codex_argv, codex_host_env = command_for("codex")


def flag_values(argv: list[str], flag: str) -> dict[str, str]:
    """Collect `--ae`/`--ak` KEY=VALUE pairs out of a built argv."""
    found: dict[str, str] = {}
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            key, _, value = argv[index + 1].partition("=")
            found[key] = value
    return found


claude_ae = flag_values(claude_argv, "--ae")
codex_ak = flag_values(codex_argv, "--ak")

# The pair that matters. Same endpoint, two spellings, and each is only correct
# for its own client.
check("claude-code base URL drops /v1",
      claude_host_env.get("ANTHROPIC_BASE_URL"), "http://example.invalid:8002")
check("codex base URL keeps /v1",
      codex_ak.get("base_url"), "http://example.invalid:8002/v1")
check_true(
    "the two harnesses are handed different URLs",
    claude_host_env.get("ANTHROPIC_BASE_URL") != codex_ak.get("base_url"),
)

# Both must be told the same window as every other harness, by different means.
check("claude-code is told the window", claude_ae.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"),
      "131072")
check("codex is told the window", codex_ak.get("context_window"), "131072")

# ANTHROPIC_BASE_URL has to be on the harbor process: Harbor's built-in agent
# reads that one from os.environ, so passing it with --ae would be ignored and
# Claude Code would quietly talk to api.anthropic.com instead.
check_true("claude-code sets the base URL as host_env",
           "ANTHROPIC_BASE_URL" in claude_host_env)
check_true("claude-code does not rely on --ae for the base URL",
           "ANTHROPIC_BASE_URL" not in claude_ae)

# The model id reaches the server bare. With a custom base URL the built-in
# agent forwards model_name unchanged, so a "local/" prefix would be sent as
# part of the model name.
model_index = claude_argv.index("--model")
check("claude-code passes a bare model id", claude_argv[model_index + 1], "a-model")

check("claude-code uses the Harbor built-in",
      claude_argv[claude_argv.index("--agent") + 1], "claude-code")
check("codex uses the local adapter",
      codex_argv[codex_argv.index("--agent") + 1], "harnesses.codex:Codex")


# ---------------------------------------------------------------------------
# The Codex adapter's generated `-c` overrides
# ---------------------------------------------------------------------------

print("\n-- codex adapter --")

from harnesses.codex import Codex  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    agent = Codex(
        Path(tmp),
        base_url="http://example.invalid:8002/v1",
        context_window=131072,
    )
    flags = agent.build_cli_flags()

check_true("selects the local provider", "-c model_provider=local" in flags)
check_true(
    "sets the provider base URL",
    "-c 'model_providers.local.base_url=\"http://example.invalid:8002/v1\"'" in flags,
)
check_true("sets the wire protocol",
           "-c 'model_providers.local.wire_api=\"responses\"'" in flags)
# Not decoration: codex 0.147 refuses to load a provider without it --
# "model_providers.local: provider name must not be empty".
check_true("sets the provider name",
           "-c 'model_providers.local.name=" in flags)
check_true("names the credential variable",
           "-c 'model_providers.local.env_key=\"OPENAI_API_KEY\"'" in flags)
# The reason the adapter exists at all.
check_true("sets the context window", "-c model_context_window=131072" in flags)

# A missing endpoint must fail when the run is built, not once per task inside
# a container where it reads as the harness crashing.
try:
    with tempfile.TemporaryDirectory() as tmp:
        Codex(Path(tmp), context_window=131072)
    refused = False
except ValueError:
    refused = True
check("refuses to build without an endpoint", refused, True)


# ---------------------------------------------------------------------------
# Reasoning effort
#
# Codex sends a reasoning object on every request and gives no way to leave the
# effort out, so the effort has to be one the server will accept. A server that
# refuses one answers 400 and takes every trial with it, which is why this is
# resolved per endpoint instead of being a constant in the registry.
# ---------------------------------------------------------------------------

print("\n-- reasoning effort --")

from dataclasses import replace  # noqa: E402

from bench.probe import supports_reasoning_effort  # noqa: E402
from bench.runner import effective_reasoning_effort  # noqa: E402

THINKS = replace(MODEL, supports_reasoning=True)
CANNOT_THINK = replace(MODEL, supports_reasoning=False)
UNKNOWN = replace(MODEL, supports_reasoning=None)

check("a thinking endpoint gets the harness default",
      effective_reasoning_effort(THINKS, CONFIG), ("high", "probed"))
check("an endpoint that refuses thinking gets none",
      effective_reasoning_effort(CANNOT_THINK, CONFIG), ("none", "probed"))
# An unanswered probe must not change what a working setup was already doing.
check("an unprobed endpoint keeps the harness default",
      effective_reasoning_effort(UNKNOWN, CONFIG), ("high", "fallback"))
check(
    "an explicit setting beats the probe",
    effective_reasoning_effort(
        CANNOT_THINK,
        Config(endpoint=EndpointConfig(base_url=MODEL.base_url, reasoning_effort="low")),
    ),
    ("low", "configured"),
)

# The value has to reach the command, or none of the above matters.
check("codex is told the probed effort",
      flag_values(command_for("codex", CANNOT_THINK)[0], "--ak").get("reasoning_effort"),
      "none")
check("codex is told the default effort when the endpoint can think",
      flag_values(command_for("codex", THINKS)[0], "--ak").get("reasoning_effort"),
      "high")


# ---------------------------------------------------------------------------
# The probe itself
#
# A rejection only means "this server refuses reasoning" if the same request
# without it succeeds. Otherwise a wrong model id or a missing credential would
# read as a reasoning refusal and silently downgrade every later run.
# ---------------------------------------------------------------------------

print("\n-- reasoning probe --")

import json as _json  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from bench.probe import PROBE_EFFORT, resolve  # noqa: E402
from bench.runner import DEFAULT_REASONING_EFFORT, uses_placeholder  # noqa: E402


@contextmanager
def fake_endpoint(status_with: int = 200, status_without: int = 200, prefix: str = ""):
    """A server answering /models, /props and ``{prefix}/responses``.

    Every POST body is recorded, which is what lets a test tell a fresh probe
    from a cached answer and see which effort actually went on the wire.
    """
    posts: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("/props"):
                self._send(200, {
                    "model_alias": "a-model",
                    "model_path": "/w/a-model.gguf",
                    "default_generation_settings": {"n_ctx": 4096},
                })
            else:
                self._send(200, {"data": [{"id": "a-model", "meta": {
                    "n_ctx": 4096, "n_ctx_train": 4096, "n_params": 1,
                    "size": 2, "ftype": "F16", "n_embd": 8, "n_vocab": 9,
                }}]})

        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if prefix and not self.path.startswith(prefix):
                self._send(404, {})
                return
            posts.append(body)
            self._send(status_with if "reasoning" in body else status_without, {})

        def _send(self, code: int, payload: dict) -> None:
            raw = _json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", posts
    finally:
        server.shutdown()
        server.server_close()


def probe_against(status_with: int, status_without: int) -> bool | None:
    with fake_endpoint(status_with, status_without) as (base, _posts):
        return supports_reasoning_effort(
            EndpointConfig(base_url=f"{base}/v1", model="a-model"), timeout=10.0
        )


def closed_port() -> int:
    """A port nothing is listening on, for the endpoint-is-down cases."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


check("an accepted effort reads as supported", probe_against(200, 200), True)
check("a refused effort reads as unsupported", probe_against(400, 200), False)
# Both failing means something else is wrong -- an unknown model, no
# credential. Reporting False there would blame reasoning for someone else's
# bug and quietly change what every later run measures.
check("a server that refuses both is not an answer", probe_against(400, 400), None)
# Servers do not agree on how to spell a refusal, and the control request --
# not the status code -- is what makes a refusal safe to act on. A 500 that a
# plain request survives is a refusal like any other; treating it as "no
# answer" would hand back the default that breaks the run.
check("a 500 the plain request survives is a refusal", probe_against(500, 200), False)
check("a 404 on both is not an answer", probe_against(404, 404), None)

# The probe has to ask for the same effort a run will send. Asking about "low"
# and then shipping "high" would leave the only case that matters untested.
check("the probe asks for the effort a run sends", PROBE_EFFORT,
      DEFAULT_REASONING_EFFORT)
with fake_endpoint() as (base, posts):
    supports_reasoning_effort(
        EndpointConfig(base_url=f"{base}/v1", model="a-model"), timeout=10.0
    )
check("the effort reaches the wire", posts[0]["reasoning"]["effort"], PROBE_EFFORT)

# base_url is accepted with or without the /v1 suffix -- the model lookup takes
# both -- so a server that serves the OpenAI routes only under /v1 must still
# be answerable from a base_url written without it. Ollama is that server, and
# a 404 here would fall back to the default that takes its runs down.
with fake_endpoint(prefix="/v1") as (base, _posts):
    check(
        "a /v1-only server is found from a base_url without /v1",
        supports_reasoning_effort(
            EndpointConfig(base_url=base, model="a-model"), timeout=10.0
        ),
        True,
    )

# Nothing in the probe may raise: resolve() calls it on the run path, where an
# exception would abort exactly the runs the None fallback exists to protect.
dead = f"http://127.0.0.1:{closed_port()}"
check("an unreachable endpoint is not an answer",
      supports_reasoning_effort(EndpointConfig(base_url=dead, model="m"), timeout=2.0),
      None)
check("an unreachable endpoint with no model id does not raise",
      supports_reasoning_effort(EndpointConfig(base_url=dead), timeout=2.0),
      None)


# ---------------------------------------------------------------------------
# Caching the answer
#
# Whether an effort is accepted belongs to the server as much as to the
# weights, so it cannot be cached against the weights alone: the same GGUF
# answers differently behind llama.cpp and behind Ollama.
# ---------------------------------------------------------------------------

print("\n-- reasoning cache --")

with tempfile.TemporaryDirectory() as tmp:
    cache = Path(tmp) / "models.json"
    with fake_endpoint() as (base_a, posts_a), fake_endpoint() as (base_b, posts_b):
        first = resolve(EndpointConfig(base_url=base_a), interactive=False,
                        cache_path=cache)
        check("the endpoint is asked once", len(posts_a), 1)
        check("and its answer is recorded", first.supports_reasoning, True)

        resolve(EndpointConfig(base_url=base_a), interactive=False, cache_path=cache)
        check("a second run against the same endpoint reuses it", len(posts_a), 1)

        # Same weights, same fingerprint, different server. Reusing the first
        # answer here is the failure this cache key exists to prevent.
        resolve(EndpointConfig(base_url=base_b), interactive=False, cache_path=cache)
        check("a different endpoint is asked for itself", len(posts_b), 1)


# ---------------------------------------------------------------------------
# Reporting the effort only to runs it can affect
# ---------------------------------------------------------------------------

check_true("codex is seen to use the effort",
           uses_placeholder(harnesses["codex"], "reasoning_effort"))
check_true("a harness without the knob is not",
           not uses_placeholder(harnesses["claude-code"], "reasoning_effort"))


# ---------------------------------------------------------------------------
# opencode's command line
#
# The instruction is a positional, and a CLI reads an argument beginning with
# `-` as a flag no matter how well the shell quoted it. One Terminal-Bench 2
# task opens with a bullet, which cost a whole cell to a usage message.
# ---------------------------------------------------------------------------

print("\n-- opencode command --")

from harnesses.opencode import OpenCode  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    opencode_cmd = OpenCode(
        Path(tmp),
        model_name="local/a-model",
        base_url="http://example.invalid:8002/v1",
        context_window=131072,
    )._run_command()

check_true("flag parsing is ended before the instruction",
           '-- "$HARBOR_INSTRUCTION"' in opencode_cmd)
# Everything after `--` is a positional, so a flag left behind it would be
# pasted onto the end of the prompt instead of being parsed.
check("no flag follows the instruction",
      opencode_cmd.split('-- "$HARBOR_INSTRUCTION"')[1].strip().startswith("2>&1"),
      True)


# ---------------------------------------------------------------------------
# dsh command
#
# Same bug, one layer deeper. dsh puts two commander parsers between the shell
# and the task -- the launcher's and the one-shot app's -- and the first `--`
# is consumed by the launcher, so a single one leaves the app reading a task
# that opens with a bullet as an unknown option and exiting on its usage
# message. Measured against commander 15 with both parser configurations: one
# `--` reports "unknown option '- You are given a PyTorch state dictionary'",
# two deliver the task intact.
# ---------------------------------------------------------------------------

print("\n-- dsh command --")

from harnesses.deepseek import DeepSeekHarness  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    dsh_cmd = DeepSeekHarness(
        Path(tmp),
        model_name="local/a-model",
        base_url="http://example.invalid:8002/v1",
        context_window=131072,
    )._run_command()

check_true("the launcher's dashdash is doubled for the app",
           '-- -- "$HARBOR_INSTRUCTION"' in dsh_cmd)
check_true("the overlay is applied", "--patch /tmp/dsh/arena.cordis.yml" in dsh_cmd)
# dsh prints only its final answer, at the end, and keeps stderr empty on a
# successful run. Without the mirror the live feed has nothing to tail for the
# whole trial, and a blank panel reads as a hung agent.
check_true("the session log is mirrored to the tailable file",
           "tail -n +1 -F" in dsh_cmd and "/logs/agent/dsh.txt" in dsh_cmd)
# The mirror must not decide the trial's outcome.
check_true("the run's own exit status is what propagates",
           "status=$?" in dsh_cmd and dsh_cmd.rstrip().endswith("exit $status"))

# dsh is the only harness here installed from npm rather than as a standalone
# release binary, and its tree reaches node-pty -- a native addon with no
# prebuilt, so `npm install -g` runs node-gyp. Terminal-Bench 2's images carry
# gcc and neither python3, make nor g++, so the first run of this harness lost
# every trial in setup to `gyp ERR! not ok` before the agent started. The
# toolchain is a hard requirement, not the best-effort sharpening that omp and
# opencode install.
_packages = DeepSeekHarness._packages_command()
for _tool in ("python3", "make", "g++"):
    check_true(f"the install provides {_tool}", f" {_tool}" in _packages)
# Checked one at a time at the end: a toolchain that could not be installed
# should say which piece is missing, not surface as npm's gyp output later.
check_true("a missing build tool is named, not left to npm",
           "for tool in curl python3 make g++" in _packages)

# dsh exits 1 whenever the final turn/end reason is not `completed`, which
# folds "the model stopped early" into the same signal as "the harness broke".
# Harbor reads non-zero as a trial error and skips the verifier, so a run that
# did the work and had its last message clipped at the output cap is discarded
# ungraded. Measured on llm-inference-batching-scheduler: `max-tokens` after 13
# tool calls, workspace never scored.
#
# The asymmetry is the point. `error` and `aborted` must keep their non-zero
# exit: an `error` turn is usually the endpoint failing, and collect.py needs
# the exception to classify it as a transport fault and drop it from the
# denominator. Swallowing that would score an outage as a fair attempt.
for _reason in ("max-tokens", "blocked"):
    check_true(f"a turn ending {_reason} is graded, not errored",
               f'"kind":"{_reason}"' in dsh_cmd)
for _reason in ("error", "aborted"):
    check(f"a turn ending {_reason} still fails the trial",
          f'"kind":"{_reason}"' in dsh_cmd, False)
# Only ever downgrades a failure; a success is never reinterpreted.
check_true("the salvage is reached only on a non-zero exit",
           "if [ $status -ne 0 ]; then" in dsh_cmd)
# The last turn decides: an earlier max-tokens must not mask a later error.
check_true("the last turn/end is the one consulted",
           "tail -n 1" in dsh_cmd.split("if [ $status -ne 0 ]")[1])


# ---------------------------------------------------------------------------
# The live feed can only tail a log it knows the name of
# ---------------------------------------------------------------------------

print(chr(10) + "-- live feed coverage --")

from bench.activity import LOG_NAMES  # noqa: E402

for _mod, _cls in (("harnesses.hermes", "Hermes"), ("harnesses.omp", "Omp"),
                   ("harnesses.opencode", "OpenCode"), ("harnesses.minion", "Minion"),
                   ("harnesses.codex", "Codex"),
                   ("harnesses.deepseek", "DeepSeekHarness")):
    _agent = getattr(__import__(_mod, fromlist=[_cls]), _cls)
    _name = getattr(_agent, "_OUTPUT_FILENAME", None)
    if _name:
        # An adapter whose filename is missing here has no live feed for its
        # whole run, and a blank panel reads as a hung agent rather than as a
        # gap in this tuple.
        check(f"{_cls} log is tailable", _name in LOG_NAMES, True)


print("\n-- harness pinning --")

# Every harness installs from an upstream that keeps moving. Unpinned, the
# build a run gets is decided by when the trial happened to start, so a run
# cannot be reproduced and -- as on 2026-08-13, when hermes-agent made a failed
# npm install fatal mid-run -- one run can span two different harnesses. This
# asserts the catalog names a build for each, not that the build is any good.
_catalog = load(REGISTRY_PATH)
for _hid, _spec in _catalog["harnesses"].items():
    check_true(f"{_hid} pins a version", bool(_spec.get("version")))

# save() rewrites the catalog from parsed YAML and re-emits HEADER, so a header
# in the file that has drifted from the constant turns every harness edit made
# from the dashboard into a diff nobody asked for.
from bench.registry import HEADER, save  # noqa: E402


def _lf(text: str) -> str:
    """Content without the line-ending question.

    write_text() emits CRLF on Windows and LF elsewhere, and git hands the
    checkout either one depending on core.autocrlf. Comparing raw bytes would
    make this suite pass or fail on that setting rather than on drift, which is
    the thing it exists to catch.
    """
    return text.replace("\r\n", "\n")


check_true(
    "registry.yaml starts with the canonical header",
    _lf(REGISTRY_PATH.read_text(encoding="utf-8")).startswith(_lf(HEADER)),
)

with tempfile.TemporaryDirectory() as _tmp:
    _copy = Path(_tmp) / "registry.yaml"
    _copy.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    _before = _lf(_copy.read_text(encoding="utf-8"))
    save(load(_copy), _copy)
    # A save that is not a no-op means the committed file and the writer
    # disagree, and the next UI edit silently reformats the whole catalog.
    # Compared as a diff rather than as two strings: the catalog is ~4KB, and
    # a check that prints both copies on every run buries the suite's output.
    import difflib  # noqa: E402

    _drift = "".join(
        difflib.unified_diff(
            _before.splitlines(keepends=True),
            _lf(_copy.read_text(encoding="utf-8")).splitlines(keepends=True),
            "committed",
            "after save()",
        )
    )
    check("save(load(catalog)) rewrites nothing", _drift, "")



# ---------------------------------------------------------------------------
# Run naming, and the benchmark segment it gained
#
# A run directory is the only place the dataset and the scope are visible
# without opening a manifest. Until multi-benchmark support they were not in it
# at all, so two runs of the same harness and model on different benchmarks were
# indistinguishable in a directory listing.
# ---------------------------------------------------------------------------

print("\n-- run naming --")

from bench import registry as _reg  # noqa: E402
from bench.runner import job_name, scope_name  # noqa: E402

_catalog = _reg.load()
_model = ModelIdentity(
    served_id="a-model",
    fingerprint="a1b2c3d4e5f6",
    label="Qwen3 Coder 30B (Q4_K_M)",
    base_url="http://example.invalid:8002/v1",
    host="example.invalid",
    n_ctx=131072,
)

check("catalogued slugs are used verbatim",
      _reg.dataset_slug("aider/aider-polyglot", _catalog), "polyglot")
check("  including the versioned id",
      _reg.dataset_slug("terminal-bench@2.0", _catalog), "tb2")
# An uncatalogued dataset still has to name itself: --dataset accepts anything
# Harbor resolves, and refusing to name the directory would be worse than a
# longer name.
check("an uncatalogued dataset derives one",
      _reg.dataset_slug("some-org/brand-new-bench@3.1", _catalog), "brand-new-be")
check("  and never exceeds the bound",
      len(_reg.dataset_slug("some-org/brand-new-bench@3.1", _catalog))
      <= _reg.DATASET_SLUG_MAX, True)
check("  nor ends on a separator",
      _reg.derive_dataset_slug("org/aaaaaaaaaaa-bbb").endswith("-"), False)

check("a named subset names itself", scope_name("stratified-25", None, ["a"]),
      "stratified-25")
check("an ad-hoc task cap is a smoke run", scope_name(None, 3, None), "smoke")
check("everything else is the full dataset", scope_name(None, None, None), "full")

_name = job_name(_model_harness := "omp", _model, "20260814T173206Z",
                 dataset_slug="tb2", scope="full")
check("the name carries the benchmark", "__tb2__" in _name, True)
check("  and the scope", "__full__" in _name, True)
# collect.load_run identifies a job directory with no manifest by splitting on
# `__` and taking the first field. Inserting the new segments anywhere but after
# the model would relabel every such run as whatever now sits at index 0.
check("harness stays at index 0 for the manifest-less fallback",
      _name.split("__")[0], "omp")
check("  and the timestamp stays last",
      _name.split("__")[-1], "20260814T173206Z")

# Windows still enforces MAX_PATH on the trial directories Harbor writes under
# this one, and the model slug alone can reach 57 characters.
check("the longest catalogued name stays inside a sane bound",
      max(
          len(job_name("dmfa-minion", _model, "20260814T173206Z",
                       dataset_slug=_reg.dataset_slug(e["id"], _catalog),
                       scope="stratified-25"))
          for e in _reg.datasets(_catalog)
      ) < 120,
      True)

# Two datasets sharing a slug is the failure the slug exists to prevent: the
# run directories stop telling them apart, and nothing downstream reports it.
try:
    _reg.validate_dataset_slugs(
        {"datasets": [{"id": "a/one", "slug": "dup"}, {"id": "b/two", "slug": "dup"}]}
    )
    check("colliding slugs are refused", "accepted", "refused")
except RegistryError as exc:
    check("colliding slugs are refused", "slug" in str(exc), True)

try:
    _reg.validate_dataset_slugs({"datasets": [{"id": "a/one", "slug": "Way-Too-Long-Here"}]})
    check("an out-of-bounds slug is refused", "accepted", "refused")
except RegistryError as exc:
    check("an out-of-bounds slug is refused", "at most" in str(exc), True)

# A slug nobody wrote down is still the slug that lands in the directory name,
# so the collision check has to see it. Derived slugs are truncations, which
# collide more readily than chosen names rather than less -- these two ids are
# from the shipped catalog, and both derive 'terminal-ben'. Validating only the
# declared ones left the mislabelling reachable through the one entry no one
# thought to check.
check("two ids that derive the same slug collide",
      _reg.derive_dataset_slug("terminal-bench@2.0"),
      _reg.derive_dataset_slug("terminal-bench-pro/terminal-bench-pro"))
try:
    _reg.validate_dataset_slugs(
        {"datasets": [
            {"id": "terminal-bench@2.0"},
            {"id": "terminal-bench-pro/terminal-bench-pro"},
        ]}
    )
    check("...and a catalog relying on them is refused", "accepted", "refused")
except RegistryError as exc:
    check("...and a catalog relying on them is refused", "slug" in str(exc), True)
    # The remedy differs from a declared collision: there is nothing to correct,
    # something has to be added.
    check("   naming the fix that applies", "explicit `slug:`" in str(exc), True)

# A declared slug and a derived one landing on the same string is the same
# failure from the other direction, and the one an added dataset walks into.
try:
    _reg.validate_dataset_slugs(
        {"datasets": [{"id": "someone/polyglot"}, {"id": "y/other", "slug": "polyglot"}]}
    )
    check("a declared slug colliding with a derived one is refused",
          "accepted", "refused")
except RegistryError as exc:
    check("a declared slug colliding with a derived one is refused",
          "slug" in str(exc), True)

# Distinct ids that happen to share no prefix are still fine: the check must not
# have become a blanket refusal of catalogs without slugs.
_reg.validate_dataset_slugs({"datasets": [{"id": "a/alpha"}, {"id": "b/beta"}]})
check("distinct derived slugs are accepted", True, True)

# The committed catalog has to satisfy its own rule.
_reg.validate_dataset_slugs(_catalog)
check("the committed catalog's slugs are valid and unique", True, True)


# ---------------------------------------------------------------------------
# A missing dataset is reported as a missing dataset
#
# It used to splice None into the argv and die several frames later inside
# scrub() with "expected string or bytes-like object", which reads as a bug in
# credential redaction rather than as a catalog with no defaults.dataset.
# ---------------------------------------------------------------------------

print("\n-- no dataset --")

_cfg = Config(endpoint=EndpointConfig(base_url="http://example.invalid:8002/v1"))
for _empty in (None, "", "   "):
    try:
        build_command(
            "omp", _catalog["harnesses"]["omp"], _model, _catalog, _cfg,
            jobs_dir=Path("runs"), name="j", dataset=_empty,
            n_concurrent=1, n_attempts=1, n_tasks=None, include_tasks=None,
            extra_args=None, allow_hosts=False, agent_timeout_multiplier=8.0,
            n_concurrent_agents=1, env_build_timeout_multiplier=4.0,
            max_retries=0, retry_include=None,
        )
        check(f"{_empty!r} is refused", "accepted", "refused")
    except ConfigError as exc:
        check(f"{_empty!r} is refused with a dataset error",
              "dataset" in str(exc).lower(), True)
    except TypeError:
        check(f"{_empty!r} is refused with a dataset error",
              "TypeError from scrub", "a ConfigError")


# ---------------------------------------------------------------------------
# A subset belongs to the benchmark it was drawn from
#
# A subset is a list of task names, and a task name only means something inside
# its own dataset: stratified-25 is 25 of Terminal-Bench 2's 89 tasks, and
# against aider-polyglot it selects 25 tasks that do not exist. Nothing rejected
# that while the rig ran one benchmark -- there was only one dataset a name
# could have come from -- so multi-benchmark support made the combination
# reachable from a dropdown without making it look wrong. A run that selects
# nothing finishes, and reads like a run that measured something.
# ---------------------------------------------------------------------------

print("\n-- subsets belong to a benchmark --")

from bench import subset_dataset, subset_datasets  # noqa: E402
from bench.runner import check_subset_dataset  # noqa: E402

check("the packaged subset declares its benchmark",
      subset_dataset("stratified-25"), "terminal-bench@2.0")
check("  and every subset is reported with one",
      subset_datasets().get("stratified-25"), "terminal-bench@2.0")

# The matching case must stay silent, or the guard is just an obstruction.
check_subset_dataset("stratified-25", "terminal-bench@2.0")
check("its own benchmark is accepted", True, True)

try:
    check_subset_dataset("stratified-25", "aider/aider-polyglot")
    check("another benchmark is refused", "accepted", "refused")
except ConfigError as exc:
    check("another benchmark is refused", "would select nothing" in str(exc), True)
    # The message has to name the way out, since the subset is not the thing
    # that is wrong -- the pairing is.
    check("  naming the dataset it does belong to",
          "terminal-bench@2.0" in str(exc), True)

# The compatibility case, and the one that would break quietly: a list written
# before subsets declared anything belongs to every benchmark, exactly as it did
# before this existed.
with tempfile.TemporaryDirectory() as _sub_tmp:
    _dir = Path(_sub_tmp)
    (_dir / "undeclared.txt").write_text("task-one\ntask-two\n", encoding="utf-8")
    import bench as _bench

    _orig = (_bench.PACKAGED_SUBSET_DIR, _bench.USER_SUBSET_DIR)
    _bench.PACKAGED_SUBSET_DIR = _bench.USER_SUBSET_DIR = _dir
    try:
        check("a subset that declares nothing belongs to any benchmark",
              subset_dataset("undeclared"), None)
        check_subset_dataset("undeclared", "gaia/gaia")
        check("  and is refused on none of them", True, True)
    finally:
        (_bench.PACKAGED_SUBSET_DIR, _bench.USER_SUBSET_DIR) = _orig



# ---------------------------------------------------------------------------
# A benchmark whose own machinery calls a model
#
# Most datasets only need the agent to reach an endpoint. tau3-bench simulates
# the user in its environment and judges assertions in its verifier, so both
# need one too -- and Harbor reads a task's [environment].env / [verifier].env
# from its *own* environment and exits before the first trial when a required
# variable is unset. A seven-harness tau3 sweep failed seven for seven that way,
# each run dying in seconds with "Missing Environment Variables".
# ---------------------------------------------------------------------------

print("\n-- dataset host_env --")

from bench.runner import dataset_host_env  # noqa: E402

_cfg2 = Config(endpoint=EndpointConfig(base_url="http://example.invalid:8002/v1"))
_argv2, _host_env = build_command(
    "omp", _catalog["harnesses"]["omp"], _model, _catalog, _cfg2,
    jobs_dir=Path("runs"), name="j", dataset="sierra-research/tau3-bench",
    n_concurrent=1, n_attempts=1, n_tasks=1, include_tasks=None,
    extra_args=None, allow_hosts=False, agent_timeout_multiplier=8.0,
    n_concurrent_agents=1, env_build_timeout_multiplier=4.0,
    max_retries=0, retry_include=None,
)
# The two Harbor refuses to start without.
check("the benchmark's endpoint reaches the child",
      _host_env.get("OPENAI_BASE_URL"), "http://example.invalid:8002/v1")
check("  with a credential", bool(_host_env.get("OPENAI_API_KEY")), True)
# tau3's task.toml defaults both of these to gpt-5.2, so pointing only the URL
# at a local server would ask that server for gpt-5.2.
check("the simulated user is told which model to use",
      _host_env.get("TAU2_USER_MODEL"), "openai/a-model")
check("  and so is the judge",
      _host_env.get("TAU2_NL_ASSERTIONS_MODEL"), "openai/a-model")
# tau2 only sends a reasoning effort when the variable is non-empty, and a local
# server that refuses one fails every request of the run.
check("  and no reasoning effort is sent",
      _host_env.get("TAU2_USER_REASONING_EFFORT"), "")

# A dataset that needs nothing must not gain anything.
_, _plain_env = build_command(
    "omp", _catalog["harnesses"]["omp"], _model, _catalog, _cfg2,
    jobs_dir=Path("runs"), name="j", dataset="terminal-bench@2.0",
    n_concurrent=1, n_attempts=1, n_tasks=1, include_tasks=None,
    extra_args=None, allow_hosts=False, agent_timeout_multiplier=8.0,
    n_concurrent_agents=1, env_build_timeout_multiplier=4.0,
    max_retries=0, retry_include=None,
)
check("a dataset that needs none gets none", _plain_env, {})
check("an uncatalogued dataset is not guessed at",
      dataset_host_env("nobody/nothing", _catalog), {})

# The harness block wins: it describes the thing being measured, while the
# dataset block only has to make the benchmark run.
_clash = {
    **_catalog,
    "datasets": [{"id": "x/y", "slug": "xy", "host_env": {"SHARED": "dataset"}}],
}
_spec = {**_catalog["harnesses"]["omp"], "host_env": {"SHARED": "harness"}}
_, _merged = build_command(
    "omp", _spec, _model, _clash, _cfg2, jobs_dir=Path("runs"), name="j",
    dataset="x/y", n_concurrent=1, n_attempts=1, n_tasks=1, include_tasks=None,
    extra_args=None, allow_hosts=False, agent_timeout_multiplier=8.0,
    n_concurrent_agents=1, env_build_timeout_multiplier=4.0,
    max_retries=0, retry_include=None,
)
check("the harness wins a collision", _merged.get("SHARED"), "harness")



# ---------------------------------------------------------------------------
# The ceiling a harness is actually given, versus the one the rig computed
# ---------------------------------------------------------------------------

print("\n-- output caps --")

import os  # noqa: E402

from bench.runner import (  # noqa: E402
    CAP_APPLIED,
    NO_OUTPUT_CAP,
    WINDOWS_MAX_PATH,
    check_path_budget,
    output_cap_for,
    path_budget,
    report_output_caps,
    warn_reasoning_under_cap,
)

_cat = _catalog["harnesses"]

# Every harness gets handed the same number; not every harness has a knob for
# it. Recording the number regardless made the manifest a claim rather than a
# record, and that is how a 16K-capped run and an uncapped one came to be
# compared as though they were the same experiment.
check("a harness that takes the cap records it",
      output_cap_for(_cat["claude-code"], 131072), (16384, CAP_APPLIED))
# codex-cli 0.147.0's ConfigToml has 96 keys and none of them caps a
# completion; model_max_output_tokens does not appear in the binary at all.
# Measured against the shipped executable, not read from the docs.
check("codex is recorded as uncapped, not as capped at the rig's number",
      output_cap_for(_cat["codex"], 131072), (None, NO_OUTPUT_CAP))
check("...and the ceiling itself is unchanged for everyone else",
      output_cap_for(_cat["dsh"], 131072)[0], 16384)
check("an empty block cannot silently claim a cap",
      output_cap_for({}, 131072), (None, NO_OUTPUT_CAP))

check("a sweep says which of its harnesses it cannot cap",
      report_output_caps(_catalog, ["claude-code", "codex", "dsh"], 131072),
      ["codex"])
check("...and says nothing when they are all capped",
      report_output_caps(_catalog, ["claude-code", "dsh"], 131072), [])


# ---------------------------------------------------------------------------
# Reasoning and the cap spend the same budget
# ---------------------------------------------------------------------------

print("\n-- reasoning under a cap --")

# The DeepSeek Harness ran at high effort under a 16,384-token ceiling on
# 2026-08-18 and lost 8 of 25 trials to it -- every one scoring zero, three
# without ever emitting a tool call. Reasoning tokens count against max_tokens,
# so the two settings are spending the same budget.
_reasons_and_capped = {"agent_kwargs": {"max_tokens": "{max_tokens}",
                                        "reasoning_effort": "{reasoning_effort}"}}
check_true("a harness that reasons under a cap is warned about",
           warn_reasoning_under_cap("x", _reasons_and_capped, 131072, "high"))
# Codex is here: uncapped, so the effort has nothing to collide with.
check("an uncapped harness is not warned about",
      warn_reasoning_under_cap("codex", _cat["codex"], 131072, "high"), False)
check("a capped harness that does not reason is not warned about",
      warn_reasoning_under_cap("cc", _cat["claude-code"], 131072, "high"), False)
check("no effort, no warning",
      warn_reasoning_under_cap("x", _reasons_and_capped, 131072, "none"), False)

# Every harness with a knob is told the *same* effort, and the warning fires
# for the ones that are also capped. Removing the placeholder from a block
# would not make that harness reason less -- it would make it fall back to its
# own default, unrecorded, which is strictly worse than a value the manifest
# can state. The lever for the collision is the effort value, which is one
# number for the whole sweep and so keeps the harnesses comparable.
check("dsh is told the same effort as codex, not left to its own default",
      uses_placeholder(_cat["dsh"], "reasoning_effort"), True)
check_true("...and is warned about, because it is capped as well",
           warn_reasoning_under_cap("dsh", _cat["dsh"], 131072, "high"))


# ---------------------------------------------------------------------------
# Windows stops at 260 characters, on files it has just written itself
# ---------------------------------------------------------------------------

print("\n-- path budget --")

# Measured off the runs on disk: the deepest artifact each harness writes below
# a trial directory. Codex is the one that has already cost something -- two
# rollout paths at 260 and 264 characters, both unopenable, both trials
# recording no tokens at all.
_jobs = Path("C:/x")
_total, _which = path_budget(_jobs, "j", ["short", "a-much-longer-task-name"], "codex")
check("the longest task name is the one that decides it",
      _which, "a-much-longer-task-name")
check_true("...and a deep-session harness is budgeted deeper than a shallow one",
           path_budget(_jobs, "j", ["t"], "claude-code")[0]
           > path_budget(_jobs, "j", ["t"], "hermes")[0])
check_true("a harness nobody has profiled is not assumed to be cheap",
           path_budget(_jobs, "j", ["t"], "never-seen-before")[0]
           > path_budget(_jobs, "j", ["t"], "x")[0] - 1)

# The run that actually lost its tokens, reconstructed at the length it had.
#
# Built from the platform's own filesystem root rather than written out, for two
# reasons: path_budget resolves what it is given, so a Windows-shaped literal
# would pick up the working directory on Linux and measure something else
# entirely; and the real path is somebody's home directory, which is not a fact
# this repository needs to carry. Only its length matters, and that is stated.
_ROOT = Path(os.path.abspath(os.sep))
# 40 characters, the runs directory on the machine where this was found.
_real_jobs = _ROOT / ("d" * (40 - len(str(_ROOT))))
check("the reconstruction is the length being claimed",
      len(str(_real_jobs.resolve())), 40)

_real_name = ("codex__qwen3-8-27b-q4-k-m-q4-k-medium-96a0f273__tb2"
              "__stratified-25__20260818T010700Z")
_measured, _ = path_budget(_real_jobs, _real_name,
                           ["llm-inference-batching-scheduler"], "codex")
# Harbor could not open that rollout, and reported FileNotFoundError on a file
# that was sitting right there. The path was 264 characters; so is this.
check("the budget reproduces the path that actually broke", _measured, 264)
check_true("...which is over the limit", _measured >= WINDOWS_MAX_PATH)

# The warning itself only exists where the limit is enforced and long paths are
# off, so its content is checked only when the platform produced one.
_warning = check_path_budget(_real_jobs, _real_name,
                             ["llm-inference-batching-scheduler"], "codex")
if _warning is not None:
    check_true("...and the warning names the task that reaches it",
               "llm-inference-batching-scheduler" in _warning)
    check_true("...and says how to fix it", "LongPathsEnabled" in _warning)
check("a short path says nothing",
      check_path_budget(_ROOT / "r", "j", ["t"], "hermes"), None)



# ---------------------------------------------------------------------------
# One reasoning effort, handed to every harness that has a knob for one
# ---------------------------------------------------------------------------

print("\n-- reasoning effort reaches the harnesses --")

from bench.config import with_overrides  # noqa: E402
from bench.runner import KNOWN_REASONING_EFFORTS  # noqa: E402
from harnesses.hermes import Hermes  # noqa: E402
from harnesses.omp import Omp  # noqa: E402

# Every level the Run tab offers is one all four wired harnesses accept. The
# rig deliberately stops at `high`: omp and hermes go on to xhigh/max/ultra and
# codex and dsh do not, and a sweep where two harnesses reasoned at different
# levels because only some understood the word is not one experiment.
check("the offered vocabulary is the one every wired harness shares",
      list(KNOWN_REASONING_EFFORTS), ["none", "minimal", "low", "medium", "high"])

# hermes spells the levels exactly as this rig does -- read off `hermes chat
# --help` for the pinned build: none, minimal, low, medium, high, xhigh, max,
# ultra. So no translation, and "none" is a value it accepts rather than an
# absence it has to infer.
with tempfile.TemporaryDirectory() as _tmp:
    _hermes = Hermes(Path(_tmp), model_name="local/a-model",
                     base_url="http://example.invalid:8002/v1",
                     context_window=131072, reasoning_effort="low")
    check("hermes is told the effort on its own flag",
          _hermes._reasoning_effort, "low")
    _plain = Hermes(Path(_tmp), model_name="local/a-model",
                    base_url="http://example.invalid:8002/v1",
                    context_window=131072)
    # Absent is not "none": hermes then uses its own agent.reasoning_effort,
    # and the model reasons either way. An unset effort is one nobody recorded.
    check("...and left alone when the sweep resolved nothing",
          _plain._reasoning_effort, None)

# omp is the one that needs translating: its "do not think" level is spelled
# `off`, and handing its enum "none" would be refused at construction with a
# message about a choice list rather than about the setting somebody picked.
with tempfile.TemporaryDirectory() as _tmp:
    check("omp's refusal to think is translated into its own spelling",
          Omp(Path(_tmp), model_name="local/a-model",
              base_url="http://example.invalid:8002/v1",
              reasoning_effort="none")._resolved_flags.get("thinking"), "off")
    check("...while every other level passes through untouched",
          Omp(Path(_tmp), model_name="local/a-model",
              base_url="http://example.invalid:8002/v1",
              reasoning_effort="medium")._resolved_flags.get("thinking"), "medium")
    # The flag stays usable on omp's own terms for anyone who wants xhigh.
    check("an explicit thinking level wins over the sweep's",
          Omp(Path(_tmp), model_name="local/a-model",
              base_url="http://example.invalid:8002/v1",
              reasoning_effort="low", thinking="xhigh")
          ._resolved_flags.get("thinking"), "xhigh")
    check("no effort, no flag",
          Omp(Path(_tmp), model_name="local/a-model",
              base_url="http://example.invalid:8002/v1")
          ._resolved_flags.get("thinking"), None)

# End to end: what the Run tab picks is what each harness is handed. Checked on
# the built command rather than on the adapters, because the catalog is the part
# that decides whether a harness is offered the value at all.
_low = with_overrides(_cfg2, reasoning_effort="low")
check("a chosen effort reads as configured, not probed",
      effective_reasoning_effort(_model, _low), ("low", "configured"))
for _h in ("codex", "dsh", "hermes", "omp"):
    _argv, _ = build_command(
        _h, _catalog["harnesses"][_h], _model, _catalog, _low,
        jobs_dir=Path("runs"), name="j", dataset="terminal-bench@2.0",
        n_concurrent=1, n_attempts=1, n_tasks=1, include_tasks=None,
        extra_args=None, allow_hosts=False, agent_timeout_multiplier=8.0,
        n_concurrent_agents=1, env_build_timeout_multiplier=4.0,
        max_retries=0, retry_include=None,
    )
    check_true(f"{_h} is handed the sweep's effort",
               "reasoning_effort=low" in " ".join(_argv))

# The harnesses with no knob are not silently assumed to be at the same effort.
# The model reasons either way, so these are runs whose effort nobody recorded,
# and the manifest and the Run tab both say so rather than implying otherwise.
for _h in ("claude-code", "minion", "dmfa-minion", "opencode"):
    check(f"{_h} is recorded as not having been told",
          uses_placeholder(_catalog["harnesses"][_h], "reasoning_effort"), False)


print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
