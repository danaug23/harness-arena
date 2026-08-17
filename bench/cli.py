"""The single entrypoint: ``harness-arena <command>``.

Dispatch is deliberately thin. Each command forwards its remaining arguments to
the module that owns them, so ``harness-arena bench --help`` is the runner's own
help and there is exactly one definition of every flag. The two commands defined
here -- ``init`` and ``doctor`` -- are the ones with no module of their own,
because they exist to get someone from ``git clone`` to a working run.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from collections.abc import Callable

from bench.config import (
    CONFIG_NAME,
    PROVIDERS,
    Config,
    ConfigError,
    EndpointConfig,
    config_path,
    describe,
    load,
)

#: command -> (module providing main(argv), one-line help)
COMMANDS: dict[str, tuple[str, str]] = {
    "init": ("", "Create config.yaml interactively"),
    "doctor": ("", "Check that everything needed to run is present and working"),
    "probe": ("bench.probe", "Identify the model at the endpoint (--speed to time it)"),
    "template-fix": (
        "bench.template",
        "Patch a chat template that refuses a harness's request shape",
    ),
    "bench": ("bench.runner", "Run the benchmark, one harness after another"),
    "dash": ("dashboard.server", "Serve the live dashboard"),
    "export": ("dashboard.server", "Write a standalone snapshot HTML"),
    "collect": ("bench.collect", "Print a text summary of all runs"),
    "throughput": ("bench.throughput", "Wall clock and LLM utilization per run"),
    "clipping": ("bench.clipping", "How often each harness hit the output ceiling"),
    "prepull": ("bench.prepull", "Cache task images ahead of a run"),
    "subset": ("bench.make_subset", "Regenerate a stratified task subset"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "harness-arena -- benchmark agent harnesses against a model.",
        "",
        "usage: harness-arena <command> [options]",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += [
        "",
        "`harness-arena <command> --help` shows that command's own options.",
        "",
        "first time here:",
        "  harness-arena init      point it at your model server",
        "  harness-arena doctor    confirm Docker, Harbor and the endpoint work",
        "  harness-arena dash      open the dashboard",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def cmd_init(argv: list[str]) -> int:
    """Write config.yaml by asking the few questions that cannot be guessed."""
    force = "--force" in argv
    target = config_path()
    if target.exists() and not force:
        print(f"{target} already exists. Re-run with --force to overwrite it.")
        print(f"\nCurrent configuration:\n{describe(load())}")
        return 1

    if not sys.stdin.isatty():
        print(
            f"harness-arena init needs a terminal. Copy config.example.yaml to "
            f"{CONFIG_NAME} and edit it instead.",
            file=sys.stderr,
        )
        return 1

    print(textwrap.dedent(f"""
        harness-arena setup
        -------------------
        This writes {target.name}, which is gitignored. Nothing here is committed.
    """).strip())

    print("\nProviders:")
    for key, provider in PROVIDERS.items():
        print(f"  {key:<20} {provider.label}")
    provider_id = _ask("\nProvider", "openai-compatible")
    while provider_id not in PROVIDERS:
        print(f"  Unknown provider {provider_id!r}.")
        provider_id = _ask("Provider", "openai-compatible")
    provider = PROVIDERS[provider_id]

    endpoint = EndpointConfig(provider=provider_id)
    endpoint.base_url = _ask("Base URL", provider.default_base_url)

    if provider.requires_api_key:
        print(
            textwrap.dedent(f"""
            {provider.label} needs an API key.

            The recommended way is to keep the key in your environment and name
            the variable here -- the key never touches disk in this repo.
              export {provider.default_api_key_env}=...
            """).strip()
        )
        endpoint.api_key_env = _ask(
            "Environment variable holding the key", provider.default_api_key_env
        )
        if not os.environ.get(endpoint.api_key_env):
            print(
                f"  [!] {endpoint.api_key_env} is not set in this shell. Set it "
                f"before running a benchmark."
            )

    if not provider.model_is_discoverable:
        print(
            f"\n{provider.label} serves many models, so name the one to benchmark "
            f"(e.g. qwen/qwen3-coder)."
        )
        endpoint.model = _ask("Model id")
        while not endpoint.model:
            endpoint.model = _ask("Model id")

    config = Config(endpoint=endpoint)
    config.runs_dir = _ask("Where to write run output", config.runs_dir)

    written = config.save()
    print(f"\nWrote {written}")
    print(f"\n{describe(config)}")
    print("\nNext:  harness-arena doctor")
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  --  {detail}" if detail else ""))
    return ok


#: Not a pass/fail: all three answers are legitimate, and what each one means
#: for a run is the part worth printing. See bench.probe.supports_reasoning_effort.
_REASONING_NOTE = {
    True: "accepts a reasoning effort (Codex will think)",
    False: "refuses a reasoning effort -- Codex will be told none",
    None: "could not be determined; harnesses keep their own default",
}


def cmd_doctor(argv: list[str]) -> int:
    """Verify the whole chain before someone spends hours discovering it broken.

    Every check and every fix lives in `bench.diagnose`, not here, because the
    dashboard shows the same ones. A fix that only exists in the terminal is a
    fix nobody running the dashboard will ever see.
    """
    from bench import diagnose

    print("harness-arena doctor\n")

    try:
        config = load()
        path = config_path()
        where = (
            str(path)
            if path.exists()
            else f"no {path.name} -- using defaults (run `harness-arena init`)"
        )
        _check("config loads", True, where)
        print(textwrap.indent(describe(config), "        "))
    except ConfigError as exc:
        _check("config loads", False, str(exc))
        return 1

    print()
    findings = diagnose.run_checks(config)
    for finding in findings:
        mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}[finding.severity]
        print(f"  [{mark}] {finding.title}"
              + (f"  --  {finding.detail}" if finding.ok and finding.detail else ""))
        if finding.ok:
            continue
        # Wrapped rather than printed raw: these are paragraphs, and a wall of
        # unwrapped text is skipped rather than read.
        if finding.detail:
            print(textwrap.indent(textwrap.fill(finding.detail, 72), "         "))
        for step in finding.fixes:
            print(textwrap.indent(textwrap.fill(step, 68), "           ")
                  .replace("           ", "         - ", 1))
        if finding.docs:
            print(f"         see {finding.docs}")

    # Not a pass/fail, so not a finding: all three answers are legitimate and
    # what each means for a run is the part worth printing.
    if any(f.id == "endpoint" and f.ok for f in findings):
        try:
            from bench.probe import supports_reasoning_effort

            accepts = supports_reasoning_effort(config.endpoint)
            print(f"\n  reasoning    {_REASONING_NOTE[accepts]}")
        except Exception:  # noqa: BLE001 - an extra note must not fail doctor
            pass

    problems = [f for f in findings if not f.ok]
    print()
    if problems:
        print(f"{len(problems)} problem(s): "
              + ", ".join(f.id for f in problems))
        return 1 if diagnose.worst(findings) == "fail" else 0
    print("All checks passed.  Next:  harness-arena bench --subset stratified-25")
    return 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

LOCAL: dict[str, Callable[[list[str]], int]] = {
    "init": cmd_init,
    "doctor": cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    # Windows defaults stdout to a legacy code page, and model labels, harness
    # labels and captured error text are all arbitrary: a served model named in
    # Chinese, or an agent that renders an arrow, would end the command in a
    # UnicodeEncodeError rather than printing its results. Replace what cannot
    # be represented instead -- a mangled character is a far better outcome than
    # a traceback in place of the leaderboard.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if args[0] in ("-V", "--version"):
        from bench import __version__

        print(f"harness-arena {__version__}")
        return 0

    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"Unknown command {command!r}.\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2

    if command in LOCAL:
        return LOCAL[command](rest)

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)

    # `export` is `dash` with a destination, which is worth a command of its own
    # because writing a shareable file is a different intent from serving one.
    if command == "export" and not any(a.startswith("--export") for a in rest):
        rest = ["--export", "harness-arena-snapshot.html", *rest]

    try:
        return int(module.main(rest) or 0)
    except ConfigError as exc:
        # A configuration problem is the user's to fix, not a traceback to read.
        print(f"\nconfiguration error: {exc}", file=sys.stderr)
        print("\nRun `harness-arena init` to set this up.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
