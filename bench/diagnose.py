"""Every way this rig is known to break, in one place, with the fix.

Two kinds of knowledge live here, and they are deliberately together:

**Checks** ask "is this machine able to run a benchmark right now" -- Docker,
Harbor, the catalog, the endpoint, Windows path length. They run before a run
does, which is the only time their answers are cheap.

**Signatures** ask "what does this failure mean" -- given the text of a failure
that already happened, name it and say what to do. A benchmark failure surfaces
as a stack trace from a subprocess inside a container, three layers from
anything the reader controls, and the distance between that text and the actual
one-line fix is where people give up.

Both produce the same `Finding`, and there is exactly one description of each
problem, because the CLI and the dashboard both render from here. `bench/cli.py`
prints them; `/api/diagnostics` serves them. A fix that only exists in the
terminal is a fix nobody running the dashboard will ever see, which is how this
started: every failure in this file cost someone a debugging session first.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench import ROOT, registry_path
from bench.config import Config, ConfigError

#: How much a problem matters. "fail" stops a run from working at all; "warn"
#: lets it run but makes the result harder to trust or the experience worse.
#: Nothing here is cosmetic -- a finding that does not change what you should do
#: is noise, and noise is why warnings stop being read.
SEVERITIES = ("ok", "warn", "fail")


@dataclass
class Finding:
    """One problem, in the terms of the person who has to fix it.

    `detail` says what was observed. `fixes` are literal next actions, in order,
    each one a thing you can paste or click -- not advice. A finding without a
    fix is a complaint.
    """

    id: str
    title: str
    severity: str = "ok"
    detail: str = ""
    fixes: list[str] = field(default_factory=list)
    #: Where to read more, as a path inside this repo.
    docs: str = ""

    @property
    def ok(self) -> bool:
        return self.severity == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "ok": self.ok,
            "detail": self.detail,
            "fixes": list(self.fixes),
            "docs": self.docs,
        }


# ---------------------------------------------------------------------------
# Signatures: naming a failure that already happened
# ---------------------------------------------------------------------------
#
# Ordered, and the order matters: the first match wins, so the specific
# signatures come before the general ones. Every entry here is a failure that
# actually happened on a real run, and the fix is the one that actually worked.

@dataclass
class _Signature:
    id: str
    title: str
    #: Matched case-insensitively against the failure text. All must be present,
    #: which is what keeps "error" from matching everything.
    needles: tuple[str, ...]
    detail: str
    fixes: tuple[str, ...]
    severity: str = "fail"
    docs: str = "docs/TROUBLESHOOTING.md"


SIGNATURES: tuple[_Signature, ...] = (
    _Signature(
        id="dataset-env",
        title="This benchmark needs its own model endpoint",
        needles=("missing environment variables",),
        detail=(
            "The benchmark's own machinery calls a model, not just the agent "
            "under test. tau3-bench simulates the user in its environment and "
            "judges assertions in its verifier, so both phases need an "
            "endpoint. Harbor reads those from its own environment and exits "
            "before the first trial when one is unset."
        ),
        fixes=(
            "Add a `host_env` block to this benchmark's entry under `datasets:` "
            "in the harness catalog, using {api_key}, {base_url} and {model_id} "
            "placeholders.",
            "tau3-bench ships with one already -- if you are seeing this for it, "
            "your catalog is older than the package. Run `harness-arena doctor` "
            "to check for catalog drift.",
        ),
    ),
    _Signature(
        id="long-paths",
        title="A task's files were never downloaded (Windows path limit)",
        needles=("filenotfounderror", "task.toml"),
        detail=(
            "Harbor caches a task under a directory named for the task plus a "
            "64-character hash. A dataset with long task names pushes its "
            "deepest file past Windows' 260-character limit, so the download "
            "creates the directories and writes nothing. tau3-bench reaches "
            "287 characters. The failure surfaces later as a read of a file "
            "that was never written."
        ),
        fixes=(
            "In an admin PowerShell: Set-ItemProperty "
            "'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' "
            "LongPathsEnabled 1",
            "Reboot.",
            "Clear the partial download: harbor cache clean  "
            "(an empty package directory is not re-fetched on its own)",
        ),
    ),
    _Signature(
        id="subset-dataset-mismatch",
        title="This task subset does not belong to this benchmark",
        needles=("no tasks matched the filter",),
        detail=(
            "Subsets are lists of task names, and task names belong to one "
            "dataset. `stratified-25` is 25 of Terminal-Bench 2's 89 tasks, so "
            "running it against another benchmark matches nothing."
        ),
        fixes=(
            "Set the task subset back to 'full dataset', or pick a subset built "
            "for this benchmark.",
            "To sample this benchmark instead, use a task count rather than a "
            "subset.",
        ),
    ),
    _Signature(
        id="no-dataset",
        title="No benchmark was selected",
        needles=("no dataset to run",),
        detail=(
            "Nothing told the runner which benchmark to use, and the catalog "
            "being read has no `defaults.dataset` either."
        ),
        fixes=(
            "Pick a benchmark on the Run tab.",
            "Or restore `defaults.dataset` in the harness catalog.",
        ),
    ),
    _Signature(
        id="unscorable",
        title="The work could not be scored (it did not build)",
        needles=("no reward file found",),
        detail=(
            "The verifier ran and the agent's code did not compile, so there "
            "was nothing to score. On benchmarks that compile their tests "
            "against the agent's code this is the ordinary way to fail, not a "
            "fault -- it is counted as a failed task."
        ),
        fixes=(
            "Nothing to fix if the model simply got it wrong.",
            "If the verifier output shows the *task's own* files failing to "
            "compile rather than the agent's, the environment is wrong: check "
            "runs/<job>/<trial>/verifier/test-stdout.txt",
        ),
        severity="warn",
    ),
    _Signature(
        id="cold-pull",
        title="A task image could not be pulled in time",
        needles=("environmentstarttimeout",),
        detail=(
            "Harbor pulls a task's image inside that trial's environment-start "
            "budget. Task images reach 21 GB, and a cold pull can exceed it, "
            "which loses the task rather than scoring it."
        ),
        fixes=(
            "Pre-pull this benchmark's images from the Upkeep tab before "
            "running it.",
            "Raise the environment build timeout multiplier if the machine is "
            "slow rather than the network.",
        ),
    ),
    _Signature(
        id="agent-timeout",
        title="The agent ran out of time",
        needles=("agenttimeout",),
        detail=(
            "The agent was still working when its budget expired. That is a "
            "statement about generation speed, not about the harness being "
            "wrong -- and a run at a different time budget is not comparable "
            "to one at this budget."
        ),
        fixes=(
            "Raise the agent timeout multiplier on the Run tab, and re-run "
            "every harness you intend to compare.",
            "Or use a faster model or endpoint.",
        ),
        severity="warn",
    ),
    _Signature(
        id="endpoint-refused-effort",
        title="The endpoint refused a reasoning effort",
        needles=("reasoning", "unsupported"),
        detail=(
            "Some servers reject a reasoning effort outright rather than "
            "ignoring it, which fails every request of the run at its first "
            "call."
        ),
        fixes=(
            "Set the reasoning effort to 'none' on the Setup tab.",
        ),
    ),
    _Signature(
        id="endpoint-down",
        title="The model endpoint could not be reached",
        needles=("connection refused",),
        detail=(
            "The request never reached the model. Trials that fail this way are "
            "excluded from the pass rate rather than charged to the harness."
        ),
        fixes=(
            "Confirm the model server is running and reachable from this "
            "machine.",
            "Check the endpoint on the Setup tab.",
        ),
    ),
)


def explain(text: str | None) -> Finding | None:
    """Name a failure from its text, or None when it is not one we know.

    Returning None is the honest answer for an unrecognised failure: inventing
    an explanation for it would be worse than showing the raw text, which the
    caller still has.
    """
    if not text:
        return None
    haystack = str(text).lower()
    for sig in SIGNATURES:
        if all(needle in haystack for needle in sig.needles):
            return Finding(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                detail=sig.detail,
                fixes=list(sig.fixes),
                docs=sig.docs,
            )
    return None


# ---------------------------------------------------------------------------
# Checks: is this machine able to run a benchmark
# ---------------------------------------------------------------------------


def _run(argv: list[str], timeout: int = 60) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def check_harbor() -> Finding:
    """Harbor is the measuring instrument; without it nothing runs."""
    found = shutil.which("harbor")
    if not found:
        return Finding(
            id="harbor", title="Harbor is not installed", severity="fail",
            detail="`harbor` is not on PATH, so no benchmark can start.",
            fixes=[
                "pip install harbor==0.20.0",
                "Or activate the project environment you installed it into.",
            ],
        )
    result = _run([found, "--version"])
    if not result or result.returncode != 0:
        return Finding(
            id="harbor", title="Harbor is installed but will not run",
            severity="fail", detail=f"`{found} --version` failed.",
            fixes=["Reinstall it: pip install --force-reinstall harbor==0.20.0"],
        )
    return Finding(id="harbor", title="Harbor", detail=result.stdout.strip())


def check_docker() -> Finding:
    """Every task runs in a container, so the daemon has to be up first."""
    from bench.dockerenv import daemon_hint, install_hint

    found = shutil.which("docker")
    if not found:
        return Finding(
            id="docker", title="Docker is not installed", severity="fail",
            detail="Every benchmark task runs in its own Linux container.",
            fixes=[install_hint()],
        )
    result = _run([found, "info", "--format", "{{.ServerVersion}}"])
    if not result or result.returncode != 0:
        return Finding(
            id="docker", title="Docker is installed but not running",
            severity="fail",
            detail="The daemon did not answer, so no task can start.",
            fixes=[daemon_hint()],
        )
    return Finding(id="docker", title="Docker",
                   detail=f"daemon {result.stdout.strip()}")


def check_long_paths() -> Finding:
    """Windows caps paths at 260 characters unless this is switched on.

    Not a nicety: a benchmark whose task names are long enough downloads an
    empty package and fails much later reading a file that was never written,
    with nothing in the error mentioning path length.
    """
    if sys.platform != "win32":
        return Finding(id="long-paths", title="Long paths",
                       detail="not applicable on this platform")
    enabled = 0
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            enabled = winreg.QueryValueEx(key, "LongPathsEnabled")[0]
    except (OSError, ImportError):
        enabled = 0
    if enabled:
        return Finding(id="long-paths", title="Long paths", detail="enabled")
    return Finding(
        id="long-paths",
        title="Windows long paths are disabled",
        severity="warn",
        detail=(
            "Benchmarks with long task names cannot fully download. Harbor "
            "writes a task's files under a directory named for the task plus a "
            "64-character hash; tau3-bench reaches 287 characters against the "
            "260-character limit, so its download silently writes nothing and "
            "the run fails later reading task.toml. Terminal-Bench 2, "
            "SWE-bench and aider-polyglot are unaffected."
        ),
        fixes=[
            "In an admin PowerShell: Set-ItemProperty "
            "'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' "
            "LongPathsEnabled 1",
            "Reboot.",
            "Then clear any partial download: harbor cache clean",
        ],
        docs="docs/TROUBLESHOOTING.md",
    )


def check_catalog() -> Finding:
    """The catalog an installed copy reads stops receiving upgrades."""
    from bench import registry as registry_mod

    try:
        drift = registry_mod.catalog_drift()
    except Exception as exc:  # noqa: BLE001 - a report must never be the failure
        return Finding(id="catalog", title="Harness catalog", severity="warn",
                       detail=f"could not be read: {exc}")
    if not drift.get("stale"):
        return Finding(id="catalog", title="Harness catalog",
                       detail=str(registry_path()))

    bits = []
    for change in drift["version_changes"]:
        bits.append(
            f"harness {change['harness']}: you pin {change['yours']!r}, "
            f"the package ships {change['packaged']!r}"
        )
    if drift["new_datasets"]:
        bits.append("benchmarks you do not have: "
                    + ", ".join(drift["new_datasets"]))
    if drift["new_harnesses"]:
        bits.append("harnesses you do not have: "
                    + ", ".join(drift["new_harnesses"]))
    origin = drift["snapshot_of"] or "a release before this was recorded"
    return Finding(
        id="catalog",
        title="Your harness catalog is older than the installed package",
        severity="warn",
        detail=(
            f"Your copy was forked from {origin}; the package is "
            f"{drift['package_version']}. It was copied on your first edit and "
            f"nothing updates it, which also holds back the harness version "
            f"pins -- so a run can install an old harness build under the new "
            f"release's name. " + " ".join(bits)
        ),
        fixes=[
            f"Edit {drift['user_path']} to bring across what you want.",
            "Or delete that file to start over from the packaged catalog.",
            "Nothing is merged automatically: a pin you changed on purpose and "
            "one you never received look identical in the file.",
        ],
        docs="docs/TROUBLESHOOTING.md",
    )


def check_default_dataset() -> Finding:
    """A catalog with no default benchmark cannot start a run from the CLI."""
    from bench import registry as registry_mod

    try:
        catalog = registry_mod.load()
    except Exception as exc:  # noqa: BLE001
        return Finding(id="default-dataset", title="Benchmark catalog",
                       severity="fail", detail=str(exc),
                       fixes=["Fix or delete the harness catalog."])
    datasets = registry_mod.datasets(catalog)
    default = (catalog.get("defaults") or {}).get("dataset")
    if not datasets:
        return Finding(
            id="default-dataset", title="No benchmarks are catalogued",
            severity="fail",
            detail="The `datasets:` block is empty, so nothing can be run.",
            fixes=["Delete your catalog copy to restore the packaged one."],
        )
    if not default:
        return Finding(
            id="default-dataset", title="No default benchmark is set",
            severity="warn",
            detail=("`defaults.dataset` is unset, so a run started without an "
                    "explicit benchmark has nothing to use."),
            fixes=["Set `defaults.dataset` in the harness catalog."],
        )
    return Finding(id="default-dataset", title="Benchmarks",
                   detail=f"{len(datasets)} catalogued, default {default}")


def check_disk(config: Config) -> Finding:
    """Task images are tens of GB and a full run adds more."""
    try:
        runs_dir = config.resolved_runs_dir()
        anchor = runs_dir if runs_dir.exists() else runs_dir.parent
        free_gb = shutil.disk_usage(anchor).free / 1e9
    except OSError as exc:
        return Finding(id="disk", title="Disk space", severity="warn",
                       detail=str(exc))
    if free_gb < 100:
        return Finding(
            id="disk", title="Low disk space", severity="warn",
            detail=(f"{free_gb:.0f} GB free at {anchor}. Terminal-Bench 2's "
                    f"task images alone are around 60 GB."),
            fixes=["Free space, or point `runs_dir` at a larger volume.",
                   "Reclaim old images from the Upkeep tab."],
        )
    return Finding(id="disk", title="Disk space",
                   detail=f"{free_gb:.0f} GB free at {anchor}")


def check_endpoint(config: Config) -> Finding:
    """Everything else can be perfect and a run still measures nothing."""
    from bench.probe import probe

    try:
        identity = probe(config.endpoint)
    except Exception as exc:  # noqa: BLE001 - probe raises many shapes
        known = explain(str(exc))
        return Finding(
            id="endpoint", title="The model endpoint did not answer",
            severity="fail",
            detail=str(exc),
            fixes=(known.fixes if known else [
                "Confirm the model server is running.",
                "Check the endpoint on the Setup tab, or run "
                "`harness-arena init`.",
            ]),
        )
    return Finding(id="endpoint", title="Model endpoint",
                   detail=f"{identity.served_id} at {identity.base_url}")


def check_workspace() -> Finding:
    """Which code is running, when that is not the code you are looking at."""
    cwd = Path.cwd().resolve()
    if cwd != ROOT and (cwd / "bench" / "__init__.py").exists():
        return Finding(
            id="workspace", title="A different checkout is running",
            severity="warn",
            detail=(f"You are in {cwd} but the running code is {ROOT}. An "
                    f"editable install pins an absolute path, so changing "
                    f"directory does not change which code runs."),
            fixes=["Reinstall from here: python -m pip install -e .",
                   "Or run this checkout directly: python -m bench <command>"],
        )
    return Finding(id="workspace", title="Code root", detail=str(ROOT))


#: Ordered cheapest and most-commonly-wrong first, so the first failure a reader
#: sees is usually the one to fix. `endpoint` is last because it is the only one
#: that makes a network request.
def run_checks(config: Config | None = None, *, include_endpoint: bool = True
               ) -> list[Finding]:
    """Every check, as findings. Never raises: a broken check is a finding."""
    if config is None:
        try:
            from bench.config import load

            config = load()
        except ConfigError as exc:
            return [Finding(id="config", title="Configuration is invalid",
                            severity="fail", detail=str(exc),
                            fixes=["Fix config.yaml, or run `harness-arena init`."])]

    findings = [
        check_workspace(),
        check_harbor(),
        check_docker(),
        check_long_paths(),
        check_catalog(),
        check_default_dataset(),
        check_disk(config),
    ]
    if include_endpoint:
        findings.append(check_endpoint(config))
    return findings


def worst(findings: list[Finding]) -> str:
    """The highest severity present, for a one-word summary."""
    for level in ("fail", "warn"):
        if any(f.severity == level for f in findings):
            return level
    return "ok"


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def explain_log(text: str | None, tail_chars: int = 20_000) -> Finding | None:
    """`explain` over the tail of a log, which is where the fatal error is."""
    if not text:
        return None
    return explain(_ANSI.sub("", str(text))[-tail_chars:])
