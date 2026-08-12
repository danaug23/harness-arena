"""Guard configuration precedence and, above all, that keys do not leak.

An API key in this rig travels from your environment into a `harbor run` command
line, and from there it is one careless line away from a run manifest, a printed
command, or an exported snapshot someone posts in an issue. Those paths are
tested here rather than reasoned about, because a leak is silent: everything
keeps working, and the credential is simply also in a file you published.

    python tests/test_config.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.config import (  # noqa: E402
    REDACTED,
    Config,
    ConfigError,
    EndpointConfig,
    load,
    scrub,
)
from bench.probe import ModelIdentity  # noqa: E402
from bench.runner import MANIFEST_NAME, write_manifest  # noqa: E402

failures: list[str] = []

KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<54} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def clear_env() -> None:
    for name in list(os.environ):
        if name.startswith("HARNESS_BENCH_") or name in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
        ):
            del os.environ[name]


def write_config(tmp: str, body: str) -> Path:
    path = Path(tmp) / "config.yaml"
    path.write_text(body, encoding="utf-8")
    os.environ["HARNESS_BENCH_CONFIG"] = str(path)
    return path


# ---------------------------------------------------------------------------


def test_precedence() -> None:
    clear_env()
    with tempfile.TemporaryDirectory() as tmp:
        write_config(tmp, "endpoint:\n  base_url: http://from-file:1/v1\n")
        check("file beats default", load().endpoint.resolved_base_url(),
              "http://from-file:1/v1")

        os.environ["HARNESS_BENCH_BASE_URL"] = "http://from-env:2/v1"
        check("env beats file", load().endpoint.resolved_base_url(),
              "http://from-env:2/v1")
    clear_env()

    # No file at all must still be a usable configuration, or `git clone &&
    # harness-bench probe` fails for someone running a stock local server.
    os.environ["HARNESS_BENCH_CONFIG"] = str(Path(tempfile.gettempdir()) / "nope.yaml")
    check("missing file is not an error", load().endpoint.provider,
          "openai-compatible")
    clear_env()


def test_api_key_resolution() -> None:
    clear_env()
    endpoint = EndpointConfig(provider="openrouter")
    check("no key found", endpoint.resolve_api_key(), "")

    os.environ["OPENROUTER_API_KEY"] = KEY
    check("provider default env var is used", endpoint.resolve_api_key(), KEY)

    os.environ["MY_KEY"] = "from-named-var"
    endpoint.api_key_env = "MY_KEY"
    check("named env var wins over default", endpoint.resolve_api_key(),
          "from-named-var")

    endpoint.api_key = "literal"
    check("literal wins over env", endpoint.resolve_api_key(), "literal")
    clear_env()
    del os.environ["MY_KEY"]


def test_validation() -> None:
    clear_env()

    def fails(endpoint: EndpointConfig) -> bool:
        try:
            endpoint.validate()
            return False
        except ConfigError:
            return True

    check("openrouter without a key is rejected",
          fails(EndpointConfig(provider="openrouter")), True)

    os.environ["OPENROUTER_API_KEY"] = KEY
    check("openrouter without a model is rejected",
          fails(EndpointConfig(provider="openrouter")), True)
    check("openrouter fully configured is accepted",
          fails(EndpointConfig(provider="openrouter", model="a/b")), False)
    check("local endpoint needs neither key nor model",
          fails(EndpointConfig()), False)
    check("unknown provider is rejected",
          fails(EndpointConfig(provider="nope")), True)
    clear_env()


def test_redaction() -> None:
    config = Config(endpoint=EndpointConfig(api_key=KEY))
    check("redacted() hides the key",
          config.redacted()["endpoint"]["api_key"], REDACTED)
    check("redacted() is not merely truncated",
          KEY in json.dumps(config.redacted()), False)
    check("scrub removes a configured key",
          scrub(f"--ak api_key={KEY}", config), f"--ak api_key={REDACTED}")
    # Also catches a key this process never held -- e.g. one a harness echoed
    # out of its own config into the captured live feed.
    check("scrub catches key-shaped strings with no config",
          scrub("Bearer sk-abcdefghijklmnop"), f"Bearer {REDACTED}")
    check("scrub leaves ordinary text alone",
          scrub("no secrets here"), "no secrets here")


def test_strip_ansi() -> None:
    """Agent output is captured verbatim, and terminals are not the destination.

    minion emits color unconditionally with no way to turn it off, so the
    reader has to cope. Rendered into the live feed unhandled, its escapes
    showed up as literal "[0m[2m" between every word, burying the text.
    """
    from bench.config import strip_ansi

    # The exact shape minion produces: DIM/RESET around every fragment.
    check(
        "color codes removed",
        strip_ansi("\x1b[0m\x1b[2musing\x1b[0m\x1b[2m each\x1b[0m"),
        "using each",
    )
    # A spinner rewrites one line; without collapsing it the feed shows every
    # frame instead of the line's final state.
    check(
        "carriage returns collapse to the final state",
        strip_ansi("thinking |\rthinking /\rthinking done\n"),
        "thinking done\n",
    )
    check("cursor toggles removed", strip_ansi("\x1b[?25lx\x1b[?25h"), "x")
    check("window titles removed", strip_ansi("\x1b]0;title\x07hi"), "hi")
    check("plain text untouched", strip_ansi("just text"), "just text")
    check("empty input is safe", strip_ansi(""), "")


def test_manifest_never_records_a_key() -> None:
    """The leak path that matters: manifests are read by the dashboard and
    inlined verbatim into exported snapshots."""
    config = Config(endpoint=EndpointConfig(provider="openrouter", api_key=KEY))
    model = ModelIdentity(
        served_id="a/b", fingerprint="deadbeefdeadbeef", label="Test Model",
        base_url="https://example.invalid/v1", host="example.invalid", n_ctx=1024,
    )
    argv = ["harbor", "run", "--ak", f"api_key={KEY}", "--model", "a/b"]

    with tempfile.TemporaryDirectory() as tmp:
        job_dir = Path(tmp) / "job"
        write_manifest(
            job_dir,
            harness_id="hermes", spec={"label": "Hermes"}, model=model,
            config=config, argv=argv, dataset="terminal-bench@2.0",
            n_concurrent=2, n_concurrent_agents=1, n_attempts=1, n_tasks=None,
            include_tasks=None, agent_timeout_multiplier=16.0, subset=None,
            started_at="2026-08-09T00:00:00+00:00",
        )
        raw = (job_dir / MANIFEST_NAME).read_text(encoding="utf-8")

    check("key absent from the manifest", KEY in raw, False)
    check("redaction marker present instead", REDACTED in raw, True)
    manifest = json.loads(raw)
    check("rest of the command survives", "--model" in manifest["command"], True)
    # A hostname identifies a machine, and manifests get shared.
    check("hostname absent unless opted in",
          "hostname" in manifest["orchestrator"], False)


def test_hostname_opt_in() -> None:
    config = Config(endpoint=EndpointConfig(), record_hostname=True)
    model = ModelIdentity(
        served_id="m", fingerprint="f" * 16, label="M",
        base_url="http://example.invalid/v1", host="example.invalid", n_ctx=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        job_dir = Path(tmp) / "job"
        write_manifest(
            job_dir, harness_id="h", spec={}, model=model, config=config,
            argv=["harbor"], dataset="d", n_concurrent=1, n_concurrent_agents=1,
            n_attempts=1, n_tasks=None, include_tasks=None,
            agent_timeout_multiplier=1.0, subset=None, started_at="t",
        )
        manifest = json.loads((job_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    check("hostname present when opted in",
          "hostname" in manifest["orchestrator"], True)


def test_saved_config_round_trips() -> None:
    """The setup UI writes this file; a key that cannot round-trip means the
    endpoint silently stops authenticating after a reload."""
    clear_env()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        Config(
            endpoint=EndpointConfig(
                provider="openrouter", api_key=KEY, model="a/b",
            ),
            runs_dir="elsewhere",
        ).save(path)
        os.environ["HARNESS_BENCH_CONFIG"] = str(path)
        reloaded = load()
        check("key round-trips", reloaded.endpoint.resolve_api_key(), KEY)
        check("model round-trips", reloaded.endpoint.model, "a/b")
        check("runs_dir round-trips", reloaded.runs_dir, "elsewhere")
    clear_env()


def test_slug_is_filesystem_safe() -> None:
    identity = ModelIdentity(
        served_id="fallback-id", fingerprint="abcdef0123456789", label="",
        base_url="http://example.invalid/v1", host="example.invalid", n_ctx=1,
    )
    check("empty label falls back to served id",
          identity.slug, "fallback-id-abcdef01")
    identity.label = "!!!"
    check("punctuation-only label does not lead with a dash",
          identity.slug.startswith("-"), False)



def test_context_window_resolution() -> None:
    """One number, three possible origins, and the origin is recorded.

    Not every server will say what window it is serving: llama.cpp reports the
    loaded value, Ollama exposes only the model's architectural maximum. Left
    unresolved each harness guesses separately, which is a difference between
    harnesses that has nothing to do with the harnesses.
    """
    from bench.config import DEFAULT_CONTEXT_WINDOW
    from bench.probe import ModelIdentity
    from bench.runner import agent_max_tokens_for, effective_context

    def served(n_ctx: int) -> ModelIdentity:
        return ModelIdentity(label="M", served_id="m", fingerprint="f" * 16,
                             base_url="http://x.invalid/v1", host="x.invalid",
                             n_ctx=n_ctx)

    def resolve(configured: int, probed: int):
        return effective_context(
            served(probed), Config(endpoint=EndpointConfig(context_window=configured))
        )

    check("the server is believed when it answers", resolve(0, 131072), (131072, "detected"))
    check("an explicit setting outranks the probe", resolve(32768, 131072), (32768, "configured"))
    check("a silent server falls back", resolve(0, 0), (DEFAULT_CONTEXT_WINDOW, "fallback"))
    check("...and the setting still wins over the fallback", resolve(8192, 0), (8192, "configured"))

    # Overshooting truncates in silence and scores as a wrong answer, so the
    # fallback must never be larger than a default install actually serves.
    check("the fallback is conservative", DEFAULT_CONTEXT_WINDOW <= 4096, True)
    check("output ceiling follows the window", agent_max_tokens_for(131072), 16384)
    check("...with a floor for small windows", agent_max_tokens_for(4096), 4096)

    # It has to reach the browser to be editable, and survive a save.
    config = Config(endpoint=EndpointConfig(base_url="http://x.invalid/v1",
                                            context_window=65536))
    check("the window reaches the UI",
          config.redacted()["endpoint"].get("context_window"), 65536)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        config.save(path)
        check("...and survives a reload", load(path).endpoint.context_window, 65536)


def test_context_floor_is_refused_up_front() -> None:
    """A harness that cannot start on this window must not be launched.

    hermes-agent exits during initialisation below 64K. Harbor can only report
    that as a non-zero exit, once per task and again for the retry, so an 89
    task run spends hours reproducing one refusal that was knowable before the
    first container started.
    """
    from bench.probe import ModelIdentity
    from bench.runner import check_context_floor

    model = ModelIdentity(label="M", served_id="m", fingerprint="f" * 16,
                          base_url="http://x.invalid/v1", host="x.invalid", n_ctx=0)
    WITH_FLOOR = {"agent": "harnesses.hermes:Hermes", "min_context_window": 64000}
    NO_FLOOR = {"agent": "harnesses.minion:Minion"}

    def refused(spec: dict, window: int) -> bool:
        config = Config(endpoint=EndpointConfig(context_window=window))
        try:
            check_context_floor("h", spec, model, config)
            return False
        except SystemExit:
            return True

    check("below the floor is refused", refused(WITH_FLOOR, 32768), True)
    check("at the floor is allowed", refused(WITH_FLOOR, 64000), False)
    check("above the floor is allowed", refused(WITH_FLOOR, 65536), False)

    # Most harnesses declare no floor, and must not be gated by this at all.
    check("no floor declared -> never refused", refused(NO_FLOOR, 4096), False)


if __name__ == "__main__":
    original = dict(os.environ)
    try:
        test_precedence()
        test_api_key_resolution()
        test_validation()
        test_redaction()
        test_strip_ansi()
        test_manifest_never_records_a_key()
        test_hostname_opt_in()
        test_saved_config_round_trips()
        test_slug_is_filesystem_safe()
    finally:
        os.environ.clear()
        os.environ.update(original)
    test_context_window_resolution()
    test_context_floor_is_refused_up_front()
    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "all checks passed"))
    raise SystemExit(1 if failures else 0)
