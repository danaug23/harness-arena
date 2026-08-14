"""Verify every adapter's token accounting against synthetic agent logs.

Token counts are the one thing a real run cannot cheaply validate: a trial takes
hours, and a silent zero would make the efficiency panel quietly wrong rather
than visibly broken. So drive the parsers directly with logs in each harness's
real on-disk format, including the ugly cases -- a log truncated mid-write by a
killed trial, and no log at all.

    python tests/test_tokens.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor.models.agent.context import AgentContext  # noqa: E402

from harnesses.hermes import Hermes  # noqa: E402
from harnesses.omp import Omp  # noqa: E402

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<44} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def test_omp() -> None:
    lines = [
        {"type": "session_start", "sessionId": "abc"},
        {"type": "message_end", "message": {
            "role": "assistant",
            "usage": {"input": 1200, "output": 300, "cacheRead": 800,
                      "cost": {"total": 0.0}},
        }},
        # Tool results carry usage-shaped fields but are not assistant turns.
        {"type": "message_end", "message": {
            "role": "toolResult", "usage": {"input": 999, "output": 999}}},
        {"type": "message_end", "message": {
            "role": "assistant",
            "usage": {"input": 2400, "output": 700, "cacheRead": 1500,
                      "cost": {"total": 0.0}},
        }},
        # Streaming deltas carry no usage.
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta"}},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "omp.txt").write_text(
            "\n".join(json.dumps(line) for line in lines)
            + '\n{"type":"message_end","message":{"role":"assis',  # killed mid-write
            encoding="utf-8",
        )
        agent = Omp(logs, model_name="local/m", base_url="http://x/v1")
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)

        # AgentContext.n_input_tokens is total input *including* cache.
        check("omp n_input_tokens (input+cacheRead)", ctx.n_input_tokens, 5900)
        check("omp n_output_tokens", ctx.n_output_tokens, 1000)
        check("omp n_cache_tokens", ctx.n_cache_tokens, 2300)
        check("omp cost_usd (zero -> None)", ctx.cost_usd, None)

    with tempfile.TemporaryDirectory() as tmp:
        agent = Omp(Path(tmp), model_name="local/m", base_url="http://x/v1")
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("omp no log -> untouched", ctx.n_output_tokens, None)


def test_hermes_real_export() -> None:
    """Against a session shaped like a real `hermes sessions export`.

    The *shape* was taken from a live trial container; every value here is
    synthetic, because a fixture is committed and must not carry anything from
    the machine that produced it. The decisive detail is structural anyway: not
    one message carries a ``usage`` key. Harbor's inherited parser reads
    per-message usage and therefore always reports zero -- the true counts are
    the session-level aggregates asserted below.
    """
    session = {
        "id": "20200101_000000_aaaaaa",
        "source": "cli",
        "model": "test-model",
        "input_tokens": 16352,
        "output_tokens": 292,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "api_call_count": 1,
        "message_count": 4,
        "messages": [
            {"role": "user", "content": "solve the task"},
            {"role": "assistant", "content": "Working on it."},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "hermes-session.jsonl").write_text(json.dumps(session), encoding="utf-8")
        agent = Hermes(logs, model_name="local/m", base_url="http://x/v1")
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("hermes real export: input", ctx.n_input_tokens, 16352)
        check("hermes real export: output", ctx.n_output_tokens, 292)
        check("hermes real export: cache", ctx.n_cache_tokens, 0)


def test_hermes_session_totals_include_cache() -> None:
    """hermes reports input NET of cache; n_input_tokens must be the total.

    The fixture above carries cache_read_tokens=0, which is what let hermes
    ship reporting input alone: with no cache there is nothing to add and both
    conventions agree. Against a real llama.cpp server they diverge hard -- a
    25-task run logged 1.69M input against 27.97M cache read -- so the
    non-zero case gets its own test.
    """
    session = {
        "id": "20200101_000000_bbbbbb",
        "model": "test-model",
        "input_tokens": 34919,
        "output_tokens": 4540,
        "reasoning_tokens": 60,
        "cache_read_tokens": 635146,
        "cache_write_tokens": 0,
        "api_call_count": 30,
        "messages": [{"role": "user", "content": "solve the task"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "hermes-session.jsonl").write_text(json.dumps(session), encoding="utf-8")
        agent = Hermes(logs, model_name="local/m", base_url="http://x/v1")
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("hermes input includes cache", ctx.n_input_tokens, 34919 + 635146)
        check("hermes cache reported too", ctx.n_cache_tokens, 635146)
        # reasoning is generated like output and is billed as such.
        check("hermes output includes reasoning", ctx.n_output_tokens, 4540 + 60)
        # The invariant that makes the sum correct: cache reads are a subset of
        # the prompt, so the prompt total can never be smaller than the cache.
        check("hermes prompt total >= cache",
              ctx.n_input_tokens >= ctx.n_cache_tokens, True)


def test_hermes() -> None:
    session = {"messages": [
        {"role": "user", "content": "solve the task"},
        {"role": "assistant", "content": "I'll inspect the files.",
         "tool_calls": [{"id": "t1", "function": {"name": "bash",
                                                  "arguments": '{"cmd":"ls"}'}}],
         "usage": {"prompt_tokens": 1500, "completion_tokens": 220}},
        {"role": "tool", "tool_call_id": "t1", "content": "a.py b.py"},
        {"role": "assistant", "content": "Done.",
         "usage": {"prompt_tokens": 2600, "completion_tokens": 180}},
    ]}

    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "hermes-session.jsonl").write_text(json.dumps(session), encoding="utf-8")
        agent = Hermes(logs, model_name="local/m", base_url="http://x/v1")
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        # No session-level totals here, so the inherited per-message parse stands.
        check("hermes per-message fallback: input", ctx.n_input_tokens, 4100)
        check("hermes per-message fallback: output", ctx.n_output_tokens, 400)
        check("hermes trajectory.json written", (logs / "trajectory.json").exists(), True)

    with tempfile.TemporaryDirectory() as tmp:
        agent = Hermes(Path(tmp), model_name="local/m", base_url="http://x/v1")
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("hermes no log -> untouched", ctx.n_output_tokens, None)


# RFC 5737 TEST-NET-1, reserved for documentation. Deliberately *not* loopback:
# hermes discards a non-loopback base_url unless the provider is a local alias,
# which is the exact trap harnesses/hermes.py exists to work around. A
# 127.0.0.1 fixture here would pass while the real path stayed broken.
ENDPOINT = "http://192.0.2.10:8000/v1"


def test_configs() -> None:
    """The generated configs are what actually point each harness at the endpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        omp = Omp(Path(tmp), model_name="local/test-model",
                  base_url=ENDPOINT,
                  context_window="262144", max_tokens="32768")
        models_yaml = omp._build_models_yaml()
        check("omp models.yml has baseUrl", ENDPOINT in models_yaml, True)
        check("omp models.yml is keyless", "auth: none" in models_yaml, True)
        check("omp models.yml declares context",
              "contextWindow: 262144" in models_yaml, True)
        # Every role must resolve locally or a subagent silently calls the cloud.
        config_yaml = omp._build_config_yaml()
        check("omp pins every model role",
              config_yaml.count("'@default'"), 9)

        # Same class of bug: hermes treats an empty key as "no credentials"
        # and abandons the request before building it.
        blank_hermes = Hermes(Path(tmp), model_name="local/test-model",
                              base_url=ENDPOINT, api_key="")
        check("hermes never sends an empty key", blank_hermes._api_key, "local")

        hermes = Hermes(Path(tmp), model_name="local/test-model",
                        base_url=ENDPOINT)
        hermes_yaml = hermes._local_config_yaml()
        check("hermes config sets model.base_url",
              f"base_url: {ENDPOINT}" in hermes_yaml, True)
        check("hermes config sets a local provider",
              "provider: llamacpp" in hermes_yaml, True)
        check("hermes disables cross-task memory",
              "memory_enabled: false" in hermes_yaml, True)


def test_opencode() -> None:
    """Tokens come from step_finish events, or the export when they are missing.

    The fallback is not hypothetical: `opencode run --format json` is known to
    exit before emitting its final step_finish, and without the fallback the
    efficiency panel would show a silent zero rather than an error.
    """
    from harnesses.opencode import OpenCode

    def step(inp, out, reasoning=0, cache=0):
        return {"type": "step_finish", "part": {"tokens": {
            "input": inp, "output": out, "reasoning": reasoning,
            "cache": {"read": cache, "write": 0}}}}

    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "opencode.txt").write_text("\n".join(json.dumps(e) for e in [
            {"type": "step_start"},
            {"type": "text", "part": {"text": "working"}},
            step(9000, 400, reasoning=100, cache=8000),
            step(11000, 250),
        ]), encoding="utf-8")
        agent = OpenCode(logs, model_name="local/test-model", base_url=ENDPOINT)
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        # opencode's `input` is net of cache reads -- its own step_finish
        # arithmetic says so, since {"total": 7687, "input": 30, "output": 91,
        # "cache": {"read": 7566}} only balances that way -- and Harbor's
        # n_input_tokens is the total including cache. So 20000 + 8000.
        check("opencode input includes cache", ctx.n_input_tokens, 28000)
        # Reasoning tokens are generated and billed like output.
        check("opencode folds reasoning into output", ctx.n_output_tokens, 750)
        check("opencode reports cache separately", ctx.n_cache_tokens, 8000)
        check("opencode prompt total >= cache",
              ctx.n_input_tokens >= ctx.n_cache_tokens, True)

    # Stream truncated before the final event: the export must cover for it.
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "opencode.txt").write_text(
            json.dumps({"type": "step_start"}) + "\n", encoding="utf-8")
        (logs / "opencode-session.json").write_text(json.dumps(
            # Two shapes occur in exports: the stream event verbatim (tokens
            # under "part") and a stored session part (tokens inline).
            {"session": {"parts": [
                step(5000, 300),
                {"type": "step-finish", "tokens": {"input": 1000, "output": 50,
                                                   "reasoning": 0,
                                                   "cache": {"read": 0}}},
            ]}}), encoding="utf-8")
        agent = OpenCode(logs, model_name="local/test-model", base_url=ENDPOINT)
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("opencode falls back to the export", ctx.n_input_tokens, 6000)
        check("  and handles both export shapes", ctx.n_output_tokens, 350)


def test_minion() -> None:
    """Usage comes from minion's traffic log, since one-shot saves no session.

    Raw provider fields are summed rather than minion's normalized ones:
    minion subtracts cached tokens out of its input count, while Harbor's
    n_input_tokens is the total *including* cache. Mixing the two would quietly
    understate input on any endpoint with prompt caching.
    """
    from harnesses.minion import Minion

    def resp(prompt, completion, cached=0):
        return {"ts": 0, "dir": "resp", "data": {"usage": {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached}}}}

    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "minion-traffic.jsonl").write_text("\n".join([
            json.dumps({"ts": 0, "dir": "req", "data": {"messages": []}}),
            json.dumps(resp(12000, 400, cached=9000)),
            # Streaming chunks without usage must not be counted.
            json.dumps({"ts": 0, "dir": "resp", "data": {"choices": [{"delta": {}}]}}),
            json.dumps(resp(15000, 250)),
            '{"ts":0,"dir":"resp","data":{"usa',   # killed mid-write
        ]), encoding="utf-8")
        agent = Minion(logs, model_name="local/test-model", base_url=ENDPOINT)
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("minion sums input across calls", ctx.n_input_tokens, 27000)
        check("minion sums output across calls", ctx.n_output_tokens, 650)
        check("minion reports cache separately", ctx.n_cache_tokens, 9000)

    with tempfile.TemporaryDirectory() as tmp:
        agent = Minion(Path(tmp), model_name="local/test-model", base_url=ENDPOINT)
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        check("minion no traffic log -> untouched", ctx.n_output_tokens, None)


def test_minion_config() -> None:
    from harnesses.minion import Minion

    with tempfile.TemporaryDirectory() as tmp:
        agent = Minion(Path(tmp), model_name="local/test-model", base_url=ENDPOINT)
        env = agent._agent_env()
        # A bare MINION_BASE_URL is not read: one-shot exits with "no source
        # configured -- set MINION_SOURCE_* env vars or use --source".
        check("minion registers a named source", env["MINION_SOURCES"], "local")
        check("minion binds the endpoint to that source",
              env["MINION_SOURCE_LOCAL_BASE_URL"], ENDPOINT)
        check("minion names the model rather than discovering it",
              env["MINION_SOURCE_LOCAL_MODEL"], "test-model")
        check("minion never prompts for approval", env["MINION_APPROVAL"], "yolo")

        cmd = agent._run_command()
        # Without --yolo the agent waits at the first approval for someone who
        # is not there, and the trial burns its whole budget before failing.
        check("minion runs unattended", "--yolo" in cmd, True)
        # The instruction goes via a file, which is what -f is for and which
        # keeps a task description out of shell quoting entirely.
        check("minion takes the task from a file", "--prompt-file" in cmd, True)


def test_opencode_config() -> None:
    from harnesses.opencode import OpenCode

    with tempfile.TemporaryDirectory() as tmp:
        agent = OpenCode(Path(tmp), model_name="local/test-model",
                         base_url=ENDPOINT, context_window="262144",
                         max_tokens="32768")
        config = json.loads(agent._build_config_json())
        provider = config["provider"]["local"]
        check("opencode uses the openai-compatible sdk",
              provider["npm"], "@ai-sdk/openai-compatible")
        check("opencode points at the endpoint",
              provider["options"]["baseURL"], ENDPOINT)
        check("opencode declares the context limit",
              provider["models"]["test-model"]["limit"]["context"], 262144)
        # Both roles must resolve locally, or a subagent reaches for a cloud
        # default we have no credentials for and the run dies mid-task.
        check("opencode pins the main model", config["model"], "local/test-model")
        check("opencode pins the small model", config["small_model"], "local/test-model")
        check("opencode runs unattended", config["permission"]["*"], "allow")
        check("opencode does not self-update mid-experiment",
              config["autoupdate"], False)
        check("opencode does not share sessions", config["share"], "disabled")



def test_context_window_reaches_every_harness() -> None:
    """Every harness must be told the same window, in the key it actually reads.

    The window decides when a harness compresses or truncates, so a harness left
    to guess is not running the same experiment as one that was told. Each of
    these key names was taken from the harness's own documentation -- inventing
    one would leave the setting looking configured while doing nothing, which is
    worse than leaving it unset.
    """
    import importlib
    import json as _json

    from bench.runner import load_registry

    registry = load_registry()
    kwargs = {"base_url": ENDPOINT, "api_key": "local",
              "context_window": 131072, "max_tokens": 16384}
    #: harness -> the text that must appear in its generated configuration.
    expected = {
        "hermes": ["context_length: 131072", "max_tokens: 16384"],
        # omp's config is YAML, opencode's is JSON -- the needles differ in
        # punctuation for that reason, not by accident.
        "omp": ["contextWindow: 131072", "maxTokens: 16384"],
        "opencode": ['"context": 131072', '"output": 16384'],
        # minion reads /props from the server itself, so it is handed no window
        # -- only the output cap, which is not something it can probe.
        "minion": ['"MINION_MAX_TOKENS": "16384"'],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for harness, needles in expected.items():
            module, cls = registry["harnesses"][harness]["agent"].split(":")
            klass = getattr(importlib.import_module(module), cls)
            accepted = klass.__init__.__code__.co_varnames
            agent = klass(Path(tmp), model_name="local/test-model",
                          **{k: v for k, v in kwargs.items() if k in accepted})
            for attr in ("_local_config_yaml", "_build_models_yaml", "_build_config_json"):
                if hasattr(agent, attr):
                    text = getattr(agent, attr)()
                    break
            else:
                text = _json.dumps(agent._agent_env())
            for needle in needles:
                check(f"{harness}: {needle}", needle in text, True)

    # Absent, the setting is omitted rather than defaulted: a constant here
    # would be silently wrong for any other model.
    with tempfile.TemporaryDirectory() as tmp:
        bare = Omp(Path(tmp), model_name="local/m", base_url=ENDPOINT)
        check("no window given -> none written",
              "contextWindow" in bare._build_models_yaml(), False)

if __name__ == "__main__":
    test_omp()
    test_hermes_real_export()
    test_hermes_session_totals_include_cache()
    test_hermes()
    test_configs()
    test_opencode()
    test_opencode_config()
    test_context_window_reaches_every_harness()
    test_minion()
    test_minion_config()
    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "all checks passed"))
    raise SystemExit(1 if failures else 0)
