"""The single entrypoint: ``harness-bench <command>``.

Dispatch is deliberately thin. Each command forwards its remaining arguments to
the module that owns them, so ``harness-bench bench --help`` is the runner's own
help and there is exactly one definition of every flag. The two commands defined
here -- ``init`` and ``doctor`` -- are the ones with no module of their own,
because they exist to get someone from ``git clone`` to a working run.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

from bench import ROOT
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
    "bench": ("bench.runner", "Run the benchmark, one harness after another"),
    "dash": ("dashboard.server", "Serve the live dashboard"),
    "export": ("dashboard.server", "Write a standalone snapshot HTML"),
    "collect": ("bench.collect", "Print a text summary of all runs"),
    "throughput": ("bench.throughput", "Wall clock and LLM utilization per run"),
    "prepull": ("bench.prepull", "Cache task images ahead of a run"),
    "subset": ("bench.make_subset", "Regenerate a stratified task subset"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "harness-bench -- benchmark agent harnesses against one model.",
        "",
        "usage: harness-bench <command> [options]",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += [
        "",
        "`harness-bench <command> --help` shows that command's own options.",
        "",
        "first time here:",
        "  harness-bench init      point it at your model server",
        "  harness-bench doctor    confirm Docker, Harbor and the endpoint work",
        "  harness-bench dash      open the dashboard",
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
            f"harness-bench init needs a terminal. Copy config.example.yaml to "
            f"{CONFIG_NAME} and edit it instead.",
            file=sys.stderr,
        )
        return 1

    print(textwrap.dedent(f"""
        harness-bench setup
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
    print("\nNext:  harness-bench doctor")
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

    Ordered so the cheapest and most commonly wrong things are reported first.
    Every failure prints the specific next action rather than just a verdict.
    """
    print("harness-bench doctor\n")
    failures: list[str] = []

    # --- which code is actually running ---
    #
    # First, because everything else is relative to it and because an editable
    # install pins an absolute path at install time. Run the console script from
    # a second clone and it still imports the first one -- silently, since `cd`
    # does not affect import resolution. Every path below then belongs to a
    # checkout you are not looking at.
    cwd = Path.cwd().resolve()
    _check("code root", True, str(ROOT))
    if cwd != ROOT and (cwd / "bench" / "__init__.py").exists():
        print(
            f"  [warn] You are in {cwd}\n"
            f"         but the running code is {ROOT}.\n"
            f"         An editable install points at a fixed path, so `cd` does "
            f"not change it.\n"
            f"         Fix: reinstall here  ->  python -m pip install -e .\n"
            f"         Or run this checkout ->  python -m bench <command>"
        )
        failures.append("wrong checkout")

    # --- configuration ---
    try:
        config = load()
        path = config_path()
        where = (
            str(path)
            if path.exists()
            else f"no {path.name} -- using defaults (run `harness-bench init`)"
        )
        _check("config loads", True, where)
        print(textwrap.indent(describe(config), "        "))
    except ConfigError as exc:
        _check("config loads", False, str(exc))
        return 1

    print()

    # --- Harbor ---
    harbor = shutil.which("harbor")
    if not _check(
        "harbor on PATH",
        bool(harbor),
        harbor or "pip install harbor==0.20.0, or activate the project env",
    ):
        failures.append("harbor")
    else:
        try:
            result = subprocess.run(
                [harbor, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            _check("harbor runs", result.returncode == 0, result.stdout.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            _check("harbor runs", False, str(exc))
            failures.append("harbor")

    # --- Docker ---
    docker = shutil.which("docker")
    if not _check("docker on PATH", bool(docker), docker or "install Docker"):
        failures.append("docker")
    else:
        try:
            result = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            running = result.returncode == 0
            if not _check(
                "docker daemon running",
                running,
                result.stdout.strip() or "start Docker Desktop / the docker service",
            ):
                failures.append("docker daemon")
        except (OSError, subprocess.SubprocessError) as exc:
            _check("docker daemon running", False, str(exc))
            failures.append("docker daemon")

    # --- disk ---
    try:
        runs_dir = config.resolved_runs_dir()
        anchor = runs_dir if runs_dir.exists() else runs_dir.parent
        free_gb = shutil.disk_usage(anchor).free / 1e9
        # Task images alone are ~60GB for the full dataset; run output adds more.
        _check(
            "disk space for task images",
            free_gb >= 100,
            f"{free_gb:.0f} GB free at {anchor} (~100 GB recommended)",
        )
    except OSError as exc:
        _check("disk space", False, str(exc))

    # --- endpoint ---
    print()
    try:
        from bench.probe import describe as describe_model
        from bench.probe import probe, supports_reasoning_effort

        identity = probe(config.endpoint)
        _check("endpoint answers", True, identity.served_id)
        print(textwrap.indent(describe_model(identity), "      "))
        # Worth one extra request here. A server that refuses a reasoning
        # effort takes down every trial of a Codex run at its first request,
        # and this is the cheap place to find that out -- not 90 minutes in.
        accepts = supports_reasoning_effort(
            config.endpoint, served_id=identity.served_id
        )
        print(f"        reasoning    {_REASONING_NOTE[accepts]}")
    except (ConfigError, RuntimeError) as exc:
        _check("endpoint answers", False, str(exc))
        failures.append("endpoint")

    print()
    if failures:
        print(f"{len(failures)} problem(s): {', '.join(failures)}")
        return 1
    print("All checks passed.  Next:  harness-bench bench --subset stratified-25")
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

        print(f"harness-bench {__version__}")
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
        rest = ["--export", "harness-bench-snapshot.html", *rest]

    try:
        return int(module.main(rest) or 0)
    except ConfigError as exc:
        # A configuration problem is the user's to fix, not a traceback to read.
        print(f"\nconfiguration error: {exc}", file=sys.stderr)
        print("\nRun `harness-bench init` to set this up.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
