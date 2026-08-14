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
from bench.config import Config, EndpointConfig  # noqa: E402
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
# The live feed can only tail a log it knows the name of
# ---------------------------------------------------------------------------

print(chr(10) + "-- live feed coverage --")

from bench.activity import LOG_NAMES  # noqa: E402

for _mod, _cls in (("harnesses.hermes", "Hermes"), ("harnesses.omp", "Omp"),
                   ("harnesses.opencode", "OpenCode"), ("harnesses.minion", "Minion"),
                   ("harnesses.codex", "Codex")):
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


print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
