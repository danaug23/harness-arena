"""Drive Harbor once per harness against the currently-served model.

Everything harness-specific lives in harnesses/registry.yaml -- this module only
knows how to turn a registry block plus a probed model into a `harbor run`
invocation, and how to record what it did.

Runs are strictly sequential. One llama-server backs every harness, so two
overlapping runs would contend for the same slots and each would measure the
other's queueing delay as if it were its own latency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from bench import (
    REGISTRY_PATH,
    ROOT,
    WORKSPACE,
    subset_dataset,
    subset_names,
    subset_path,
    wireshape,
)
from bench import registry as registry_mod
from bench.config import DEFAULT_CONTEXT_WINDOW, Config, ConfigError, scrub
from bench.probe import ModelIdentity, add_endpoint_args, config_from_args, describe, resolve
from bench.supervisor import clear_run_marker, write_run_marker
from bench.watchdog import HEALTH_FILENAME, EndpointWatchdog, health_url

# Re-exported, not defined here. {base_url_root} is a statement about how an
# Anthropic client addresses an endpoint, so it lives with the rest of that
# knowledge in bench.wireshape -- which bench.diagnose can then import without
# pulling in the runner. Named here so bench.runner.base_url_root, which the
# substitutions below and tests/test_local_agents.py both use, still resolves.
from bench.wireshape import base_url_root  # noqa: F401  (re-export)

MANIFEST_NAME = "harness-bench.json"

#: Exception types worth a second attempt. A dropped connection to the model
#: endpoint reaches Harbor as a non-zero exit from the agent process, because
#: that is how every harness reacts to losing its endpoint mid-turn.
#:
#: Nothing at retry time can tell that apart from a harness crashing on its own
#: bug -- the distinction is only visible afterwards, in the log, which is where
#: bench.collect draws it. So the list is kept narrow, the budget defaults to a
#: single retry, and both are written into the manifest: a run that retried is
#: not the same experiment as one that did not, and the dashboard has to be able
#: to say so.
RETRY_INCLUDE_DEFAULT = ["NonZeroAgentExitCodeError"]

#: Environment asked for when diagnostics are on. Scoped rather than blanket
#: `debug`: the connection pool is the layer that matters for a request that
#: was never sent, and full trace output would bury the transcript it shares a
#: log with.
DEBUG_AGENT_ENV = {
    "RUST_LOG": "info,reqwest=debug,hyper=debug,hyper_util=debug",
    "RUST_BACKTRACE": "1",
}


# Subsets resolve through bench.subset_path: a list you made yourself wins
# over the packaged one of the same name.


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_subset_dataset(name: str, dataset: str | None) -> None:
    """Refuse a subset whose task names belong to a different benchmark.

    A subset is a list of task names, and a task name only means anything inside
    the dataset it was drawn from: `stratified-25` is 25 of Terminal-Bench 2's
    89 tasks, and against aider-polyglot it selects 25 tasks that do not exist.
    Nothing rejected this while the rig ran one benchmark, because there was
    only one dataset a name could come from -- multi-benchmark support made the
    combination reachable from a dropdown without making it wrong-looking.

    Refused here rather than left to Harbor, on the same grounds as the context
    floor: the failure otherwise arrives after the images are pulled, and a run
    that selects nothing is as easy to read as a run that finished.
    """
    declared = subset_dataset(name)
    if declared and dataset and declared != dataset:
        raise ConfigError(
            f"Subset '{name}' lists tasks from {declared}, but this run is on "
            f"{dataset}. Task names do not carry across benchmarks, so the run "
            f"would select nothing.\n"
            f"Use --dataset {declared}, pick a different subset, or scope this "
            f"run with --n-tasks instead."
        )


def load_subset(name: str) -> list[str]:
    """Read a named task list by name, yours if you have one."""
    path = subset_path(name)
    if not path.exists():
        available = subset_names()
        raise SystemExit(
            f"No subset '{name}' at {path}."
            + (f" Available: {', '.join(available)}" if available else "")
        )
    tasks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not tasks:
        raise SystemExit(f"Subset '{name}' lists no tasks.")
    return tasks


def harbor_executable() -> str:
    found = shutil.which("harbor")
    if not found:
        raise RuntimeError(
            "`harbor` is not on PATH. Run through the project env, e.g.\n"
            "  conda run --no-capture-output -n harness-arena python -m bench.runner ..."
        )
    return found


def harbor_version() -> str:
    try:
        result = subprocess.run(
            [harbor_executable(), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def effective_context(model: ModelIdentity, config: Config) -> tuple[int, str]:
    """The window every harness is told, and where the number came from.

    Three sources, most authoritative first: what you configured, what the
    server reported, and a conservative fallback for a server that will not
    say. The source travels with the number because "128K, detected" and
    "4K, because nothing knew" are not the same claim about a run, and only
    one of them is worth comparing against another run.
    """
    configured = int(getattr(config.endpoint, "context_window", 0) or 0)
    if configured > 0:
        return configured, "configured"
    if model.n_ctx:
        return int(model.n_ctx), "detected"
    return DEFAULT_CONTEXT_WINDOW, "fallback"


#: What a harness with a reasoning knob is told when the endpoint can think.
#: Harbor's own default for Codex, kept so that turning this into a probed
#: value changes nothing for a server that was already working.
DEFAULT_REASONING_EFFORT = "high"

#: The spelling that means "do not think". Not the same as sending no effort at
#: all: Codex emits a ``reasoning`` object either way, and only an explicit
#: "none" is both accepted by a server that refuses thinking and recorded in
#: the manifest as a deliberate choice. Measured against codex-cli 0.147.0.
NO_REASONING_EFFORT = "none"

#: Efforts some server has actually been measured to accept. Not a whitelist --
#: an unrecognised value is still passed through, because a new server or a new
#: harness release is free to add one and this rig's job is to measure that,
#: not to veto it. But a typo fails every trial with the same bare non-zero
#: exit code a refused effort produces, so it is worth saying out loud first.
KNOWN_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")


def effective_reasoning_effort(
    model: ModelIdentity, config: Config
) -> tuple[str, str]:
    """The reasoning effort every reasoning-capable harness is told, and why.

    Same three sources as effective_context, most authoritative first: what you
    configured, what the endpoint was measured to accept, and the harness
    default when nothing could be learned.

    The middle source is the point. An effort is not a preference here, it is
    something a server can refuse outright -- Ollama answers 400 "does not
    support thinking" for a model that cannot -- and Codex sends one on every
    request whether or not the model can use it. Hard-coding "high" therefore
    breaks every trial against such a server, with a bare non-zero exit code
    that reads as a broken harness rather than a mismatched setting.

    The source travels with the value for the same reason it does for the
    context window: a thinking run and a non-thinking run are not the same
    experiment, so a run has to say which one it was.
    """
    configured = (getattr(config.endpoint, "reasoning_effort", "") or "").strip()
    if configured:
        return configured, "configured"
    if model.supports_reasoning is True:
        return DEFAULT_REASONING_EFFORT, "probed"
    if model.supports_reasoning is False:
        return NO_REASONING_EFFORT, "probed"
    return DEFAULT_REASONING_EFFORT, "fallback"


def uses_placeholder(spec: Any, name: str) -> bool:
    """Whether a harness spec asks for ``{name}`` anywhere in its values.

    Lets the runner report a resolved setting only to the runs it can actually
    affect, rather than announcing a reasoning effort ahead of a harness that
    has no reasoning knob.
    """
    token = "{" + name + "}"
    if isinstance(spec, str):
        return token in spec
    if isinstance(spec, dict):
        return any(uses_placeholder(value, name) for value in spec.values())
    if isinstance(spec, list):
        return any(uses_placeholder(value, name) for value in spec)
    return False


def agent_max_tokens_for(window: int) -> int:
    """Output ceiling for a given window. One definition, used everywhere."""
    return max(4096, window // 8)


#: What the manifest records when a harness has no knob to take the ceiling.
NO_OUTPUT_CAP = "harness has no output cap"
CAP_APPLIED = "applied"


def output_cap_for(spec: dict[str, Any], window: int) -> tuple[int | None, str]:
    """The output ceiling this harness will actually run under, and how we know.

    Every harness in the catalog is handed one number, and until this existed
    the manifest recorded that number for all of them. For most that is true.
    For Codex it is not, and the gap is not small.

    codex-cli 0.147.0 takes no per-response output cap. Checked against the
    shipped binary rather than the docs, the same way everything else about
    Codex in this rig was: its `ConfigToml` carries 96 keys, including
    `model_context_window`, `model_auto_compact_token_limit` and
    `tool_output_token_limit`, and none of them caps a completion.
    `model_max_output_tokens` does not appear at all. So the catalog cannot
    hand Codex `{max_tokens}` and does not try to.

    What made this worth a function is that the manifest claimed otherwise.
    A 25-task sweep recorded `agent_max_tokens: 16384` for both Claude Code and
    Codex; Claude Code's largest completion was exactly 16,384 and Codex's was
    92,436. The two tied at 0.68, and the tie was between a harness under a 16K
    output budget and one under none. A run that cannot say which is which is
    not a comparison.

    Returns (ceiling, source). A ceiling of None means no cap reached the
    harness, which is a fact about the run and is recorded as one.
    """
    if uses_placeholder(spec, "max_tokens"):
        return agent_max_tokens_for(window), CAP_APPLIED
    return None, NO_OUTPUT_CAP


def report_output_caps(
    registry: dict[str, Any], harnesses: list[str], window: int
) -> list[str]:
    """Say, before a sweep starts, which harnesses it cannot cap.

    Printed rather than blocked. An uncapped harness is still worth measuring
    -- it is how that harness runs -- and refusing it would leave the rig
    unable to benchmark Codex at all. What is not acceptable is finding out
    afterwards, from a manifest that said otherwise.
    """
    catalog = registry.get("harnesses") or {}
    uncapped = [
        h
        for h in harnesses
        if output_cap_for(catalog.get(h) or {}, window)[1] == NO_OUTPUT_CAP
    ]
    if not uncapped:
        return []
    print(
        f"\n  [!] no output cap reaches {', '.join(sorted(uncapped))}: "
        f"{'it takes' if len(uncapped) == 1 else 'they take'} no max-tokens "
        f"setting."
    )
    print(
        f"      Every other harness in this sweep is clamped at "
        f"{agent_max_tokens_for(window):,} tokens per response. "
        f"A measured Codex completion ran to 92,436."
    )
    print(
        "      The runs are still worth having; they are not a like-for-like "
        "comparison of output budget, and the manifest records which is which."
    )
    return uncapped


def warn_reasoning_under_cap(
    harness_id: str, spec: dict[str, Any], window: int, effort: str
) -> bool:
    """Warn when a harness will reason and be capped on the same budget.

    Measured on dsh__qwen3-8-27b...__20260818T171816Z, which ran the DeepSeek
    Harness at `reasoning_effort: high` under a 16,384-token ceiling: 8 of 25
    trials ended on that ceiling and every one scored 0, three of them without
    ever producing a tool call. Reasoning tokens count against max_tokens, so
    the effort setting and the cap are spending the same budget.

    Returns whether the warning fired, so a caller can test it.
    """
    if not effort or effort == "none":
        return False
    if not uses_placeholder(spec, "reasoning_effort"):
        return False
    if not uses_placeholder(spec, "max_tokens"):
        # Uncapped: the effort has nothing to collide with. Codex is here.
        return False
    print(
        f"\n  [!] {harness_id} is being told to reason at '{effort}' *and* "
        f"capped at {agent_max_tokens_for(window):,} output tokens."
    )
    print(
        "      Reasoning tokens count against that cap, so a long think can "
        "consume the whole budget before the model emits a tool call, and the "
        "trial ends having done nothing. Measured: 8 of 25 trials, all scoring "
        "zero, on the run that produced this warning."
    )
    return True


def check_context_floor(
    harness_id: str, spec: dict[str, Any], model: ModelIdentity, config: Config
) -> None:
    """Refuse a run a harness cannot start, before it starts.

    Some harnesses will not run below a fixed window. hermes-agent is one: it
    exits during initialisation under 64K, which Harbor records as a bare
    NonZeroAgentExitCodeError -- indistinguishable from a crash, and repeated
    once per task plus its retry. A 89-task run spends hours reproducing the
    same refusal instead of reporting it once.

    The floor is a property of the harness, so it lives beside it in the
    catalog rather than being special-cased here.
    """
    floor = int(spec.get("min_context_window") or 0)
    if floor <= 0:
        return
    window, source = effective_context(model, config)
    if window >= floor:
        return
    raise SystemExit(
        f"{harness_id} needs a context window of at least {floor:,} tokens, "
        f"but this run would give it {window:,} ({source}).\n"
        f"  The harness refuses to initialise below its floor, so every trial "
        f"would fail identically.\n"
        f"  Fix: serve a larger window and set endpoint.context_window to "
        f"match it -- raising the number alone makes the server truncate "
        f"silently.\n"
        f"  Or run the other harnesses and leave {harness_id} out."
    )


#: Windows refuses to open a path at or beyond this length unless long paths
#: are enabled machine-wide, and the refusal is a plain FileNotFoundError on a
#: file that is sitting right there.
WINDOWS_MAX_PATH = 260

#: How deep, in characters, each harness writes below a trial directory.
#: Measured off the runs on disk rather than assumed -- the deepest artifact
#: each harness produced across every run in this rig:
#:
#:   claude-code  109  agent/sessions/projects/-app/<uuid>/subagents/<id>.meta.json
#:   omp          102  agent/omp/sessions/<stamp>_<uuid>/<name>.jsonl
#:   codex         96  agent/sessions/<yyyy>/<mm>/<dd>/rollout-<stamp>-<uuid>.jsonl
#:   dsh           85  agent/dsh/sessions/--app--/session-<uuid>/session.jsonl
#:   everything else 32  verifier/original-repo-ctrf.json
#:
#: Codex is the one that has already cost something. Two trials in a 25-task
#: sweep produced rollout paths of 260 and 264 characters; Harbor could not open
#: either, so its trajectory conversion raised FileNotFoundError and both trials
#: recorded no tokens at all -- one of them a solve worth 2.26M input and 140k
#: output. Two more trials in the same run sat at 258. The run directory name is
#: what pushes them over, and it is chosen before any of this exists.
_TRIAL_SUBPATH: dict[str, int] = {
    "claude-code": 109,
    "omp": 102,
    "codex": 96,
    "dsh": 85,
}

#: For a harness with no measurement of its own. Above every non-session
#: harness measured, below all four that write session trees: a harness nobody
#: has profiled should not be assumed to be the cheap kind.
_TRIAL_SUBPATH_DEFAULT = 64

#: Harbor's own trial-directory suffix: "__" plus seven random characters.
_TRIAL_SUFFIX = 9

#: Assumed longest task name when the run does not name its tasks -- a full
#: dataset sweep passes no --include-task-name, so there is no list to measure.
#: 32 is the longest in Terminal-Bench 2 ("llm-inference-batching-scheduler",
#: measured across all 89), and it is the dataset this rig runs by default. A
#: dataset with longer names would make this warn late rather than early, which
#: is the failure direction that costs a run rather than an annoyance.
_ASSUMED_TASK_NAME = 32


def path_budget(
    jobs_dir: Path, job_name: str, task_names: list[str], harness_id: str
) -> tuple[int, str]:
    """Longest path this run will try to write, and the task that produces it."""
    longest = max(task_names, key=len) if task_names else "?" * _ASSUMED_TASK_NAME
    below = _TRIAL_SUBPATH.get(harness_id, _TRIAL_SUBPATH_DEFAULT)
    # jobs_dir/job_name/<task>__<7>/<harness artifact>
    total = (
        len(str(jobs_dir.resolve()))
        + 1 + len(job_name)
        + 1 + len(longest) + _TRIAL_SUFFIX
        + 1 + below
    )
    return total, longest


def long_paths_enabled() -> bool | None:
    """Whether Windows will open a path past MAX_PATH. None if unknowable."""
    if os.name != "nt":
        return True
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except Exception:  # noqa: BLE001 - a preflight must not be the failure
        return None


def check_path_budget(
    jobs_dir: Path, job_name: str, task_names: list[str], harness_id: str
) -> str | None:
    """Warn before a run writes files this machine cannot then open.

    Not fatal. The run still produces correct results; what it loses is the
    artifacts written past the limit, and which of those matter depends on the
    harness -- for Codex it is the session rollout Harbor reads token counts
    from, so the trial scores normally and reports no tokens at all. Blocking a
    sweep over that would be worse than the loss.

    Returns the warning text, or None when there is nothing to say. Silent on
    any platform that does not enforce MAX_PATH, and on a Windows machine that
    has long paths turned on.
    """
    if long_paths_enabled():
        return None
    total, longest = path_budget(jobs_dir, job_name, task_names, harness_id)
    if total < WINDOWS_MAX_PATH:
        return None
    over = total - WINDOWS_MAX_PATH + 1
    which = (
        f"The task that reaches it is '{longest}'."
        if task_names
        else f"Estimated against a {_ASSUMED_TASK_NAME}-character task name, "
        f"the longest in Terminal-Bench 2."
    )
    return (
        f"\n  [!] {harness_id}: this run's deepest artifact path is about "
        f"{total} characters, and Windows stops at {WINDOWS_MAX_PATH}.\n"
        f"      {which} Files past the limit "
        f"are written and then cannot be reopened, which Harbor reports as "
        f"FileNotFoundError on a file that exists -- measured cost: two trials "
        f"that scored normally and recorded no tokens at all.\n"
        f"      Fix, cheapest first:\n"
        f"      - Enable long paths (needs admin, then a reboot):\n"
        f"          Set-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control"
        f"\\FileSystem' LongPathsEnabled 1\n"
        f"      - Or move runs_dir at least {over} characters shallower than "
        f"{jobs_dir}."
    )


def preflight_wire_shapes(
    config: Config,
    registry: dict[str, Any],
    harnesses: list[str],
    model: ModelIdentity,
    *,
    enabled: bool = True,
) -> list[str]:
    """Drop harnesses this endpoint cannot talk to, before any container starts.

    Same intent as `check_context_floor` and the opposite arithmetic: that one
    refuses a harness the *configuration* cannot satisfy, this one refuses a
    harness the *endpoint* will not accept a request from. Both exist because
    the alternative is a full sweep of identical first-request failures -- 25
    trials at three minutes of retries each, with nothing measured.

    Only the incompatible harnesses are dropped. A sweep is a list of
    independent runs, and a shape one harness sends says nothing about the
    others: on the run that produced this, six of seven harnesses were fine
    against the very endpoint that could not serve the seventh.

    Returns the harnesses that should still run. Never raises: a probe that
    cannot answer leaves the selection exactly as it found it.
    """
    if not enabled or not harnesses:
        return harnesses
    try:
        verdicts = wireshape.check_selection(
            config.endpoint, registry, harnesses, served_id=model.served_id
        )
    except Exception as exc:  # noqa: BLE001 - a preflight must not be the failure
        print(f"\n  [!] wire-shape preflight could not run ({exc}); continuing.")
        return harnesses
    if not verdicts:
        return harnesses

    for verdict in verdicts:
        if verdict.result == wireshape.ACCEPTED:
            continue
        if verdict.result == wireshape.UNKNOWN:
            # Reported and not acted on. An endpoint that works today must not
            # become unrunnable because a probe read an unfamiliar answer as
            # fatal -- see bench/wireshape.py.
            print(f"\n  [ ] {verdict.shape.title}: not established -- {verdict.why}")
            if verdict.message:
                print(f"      endpoint said: {verdict.message}")
            continue
        print(f"\n  [!] This endpoint refuses a request shape "
              f"{', '.join(verdict.harnesses)} sends.")
        print(textwrap.indent(textwrap.fill(verdict.shape.detail, 72), "      "))
        if verdict.message:
            print(f"\n      endpoint said: {verdict.message}")
        for step in verdict.shape.fixes:
            print(textwrap.indent(textwrap.fill(step, 68), "        ")
                  .replace("        ", "      - ", 1))
        if not verdict.shape.fatal:
            # Reported, and the run goes ahead. This shape costs the harness
            # the tasks that send it, not the run -- see WireShape.fatal.
            print(f"\n      Running {', '.join(verdict.harnesses)} anyway: this "
                  f"costs the tasks that send this shape, not the run.")

    blocked = wireshape.blocked_harnesses(verdicts)
    if not blocked:
        return harnesses

    remaining = [h for h in harnesses if h not in blocked]
    print(f"\n  not starting: {', '.join(sorted(blocked))}")
    if remaining:
        print(f"  unaffected, running as asked: {', '.join(remaining)}")
    else:
        print("  every selected harness sends a shape this endpoint refuses.")
    print("  (--skip-wire-check runs them anyway, which is what just failed.)")
    return remaining


def _substitutions(model: ModelIdentity, config: Config) -> dict[str, str]:
    window, _ = effective_context(model, config)
    max_tokens = agent_max_tokens_for(window)
    return {
        "model_id": model.served_id,
        "base_url": model.base_url,
        # For harnesses whose SDK appends "/v1" itself -- see base_url_root.
        "base_url_root": base_url_root(model.base_url),
        "host": model.host,
        "n_ctx": str(window),
        "max_tokens": str(max_tokens),
        # Only harnesses that actually send a reasoning effort reference this;
        # see effective_reasoning_effort for why it is not a constant.
        "reasoning_effort": effective_reasoning_effort(model, config)[0],
        "label": model.label,
        # Harnesses talking to a hosted provider need the credential. It is
        # substituted into the command here and scrubbed back out of anything
        # recorded or printed -- see write_manifest and run_one.
        #
        # "local" when there is no key: a self-hosted server ignores the value,
        # but several harnesses treat an *empty* key as "no credentials" and
        # abandon the request before building it, which surfaces as a confusing
        # auth error against a server that never wanted auth.
        "api_key": config.endpoint.resolve_api_key() or "local",
    }


def _fill(value: Any, subs: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**subs)
    if isinstance(value, dict):
        return {key: _fill(item, subs) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, subs) for item in value]
    return value


def new_batch_id(stamp: str) -> str:
    """Identity for one sweep -- one `bench` invocation across its harnesses.

    Each harness writes its own job directory, so without this the only thing
    tying them together is that they happen to share a subset name. That is not
    the same fact: re-running a subset produces a second sweep the first cannot
    be told apart from, and "three runs" then means three harnesses or three
    sweeps depending on who is counting.
    """
    return f"{stamp}-{secrets.token_hex(3)}"


def dataset_host_env(dataset: str | None, registry: dict[str, Any]) -> dict[str, str]:
    """Host variables the *benchmark itself* needs, before substitution.

    Distinct from a harness's `host_env`: this is what the dataset's own
    environment and verifier require, and it is the same for every harness in a
    sweep. Returns a copy, since the caller substitutes into it.
    """
    entry = registry_mod.dataset_entry(dataset, registry)
    declared = entry.get("host_env")
    return dict(declared) if isinstance(declared, dict) else {}


def scope_name(
    subset: str | None, n_tasks: int | None, include_tasks: list[str] | None
) -> str:
    """What a run covered, as one directory-name segment.

    Three states, and they are not interchangeable: a *named* subset is a
    deliberate experiment every harness ran identically, an ad-hoc task cap is a
    smoke test that answers "does the adapter work", and neither is the full
    dataset. `collect.py` already refuses to rank them against each other, and
    this puts the same distinction in the name so it survives a directory
    listing.

    The dashboard's `scopeName` draws the same three lines and no others, but
    the strings differ on purpose: it renders a label ("full dataset", the
    subset name as you typed it) and this is a path segment, so the subset is
    slugified and "full" is left short. Nothing reads the scope back out of a
    directory name -- the manifest is the record -- so the two never have to
    agree character for character, only on where the boundaries fall.
    """
    if subset:
        return re.sub(r"[^a-z0-9]+", "-", subset.lower()).strip("-") or "subset"
    if n_tasks is not None or include_tasks:
        return "smoke"
    return "full"


def job_name(
    harness: str,
    model: ModelIdentity,
    stamp: str,
    *,
    dataset_slug: str,
    scope: str,
) -> str:
    """The run directory name: what was measured, on what, over what, when.

    Harness stays first because it always has. A job directory with no manifest
    -- a bare `harbor run`, or one killed before the manifest landed -- is
    identified by splitting this on `__` and taking the first field, so
    inserting the new segments anywhere else would silently relabel every such
    run as whatever now sits at index 0. See `collect.load_run`.

    The dataset and the scope are here because they are the two facts that
    decide whether two runs are the same experiment, and until now both existed
    only inside the manifest -- so a directory listing showed a Terminal-Bench 2
    run and an aider-polyglot run of the same harness as indistinguishable.
    """
    return f"{harness}__{model.slug}__{dataset_slug}__{scope}__{stamp}"


def build_command(
    harness_id: str,
    spec: dict[str, Any],
    model: ModelIdentity,
    registry: dict[str, Any],
    config: Config,
    *,
    jobs_dir: Path,
    name: str,
    dataset: str,
    n_concurrent: int,
    n_attempts: int,
    n_tasks: int | None,
    include_tasks: list[str] | None,
    extra_args: list[str] | None,
    allow_hosts: bool,
    agent_timeout_multiplier: float,
    n_concurrent_agents: int | None,
    env_build_timeout_multiplier: float | None,
    max_retries: int,
    retry_include: list[str] | None,
    debug_capture: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, host_env_overrides) for one harness run."""
    # Caught here rather than left to argv construction. A None splices into the
    # command line as a None and dies several frames later inside scrub() with
    # "expected string or bytes-like object", which reads as a bug in credential
    # redaction rather than as a catalog missing its `defaults.dataset` -- the
    # state an installed copy lands in when its own registry.yaml predates the
    # `datasets:` block.
    if not dataset or not str(dataset).strip():
        raise ConfigError(
            "No dataset to run. Pass --dataset, or set `defaults.dataset` in "
            f"{REGISTRY_PATH}.\n"
            "An installed copy keeps its own catalog, which a package upgrade "
            "does not rewrite; delete it to pick the packaged one back up."
        )
    check_context_floor(harness_id, spec, model, config)
    subs = _substitutions(model, config)
    spec = _fill(spec, subs)

    argv = [
        harbor_executable(),
        "run",
        "--dataset",
        dataset,
        "--agent",
        spec["agent"],
        "--model",
        spec["model_ref"],
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        name,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
        "--yes",
    ]

    # Terminal-Bench tasks carry their own agent budget (900-1800s in this
    # dataset). Those were set for frontier APIs; a large local model on one slot
    # can spend the whole budget and time out with the task half-finished, which
    # scores identically to being wrong. Scaling the budget measures capability
    # instead of throughput -- at the cost of leaderboard comparability, so the
    # multiplier is recorded per run and surfaced in the dashboard.
    if agent_timeout_multiplier != 1.0:
        argv += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]

    # Overlap setup/verify/teardown across trials while keeping the LLM phase
    # strictly serialized. Harbor rejects a value above --n-concurrent.
    if n_concurrent_agents:
        argv += [
            "--n-concurrent-agents",
            str(min(n_concurrent_agents, n_concurrent)),
        ]
    if env_build_timeout_multiplier:
        argv += [
            "--environment-build-timeout-multiplier",
            str(env_build_timeout_multiplier),
        ]

    # A dropped connection to the endpoint should cost wall clock, not a data
    # point. Retries are restricted to the exception types an infrastructure
    # failure actually surfaces as -- retrying everything would quietly hand a
    # harness a second attempt at its own bugs, which is a different experiment.
    # The budget is recorded per run so two runs are never silently compared
    # across different numbers of attempts.
    if max_retries:
        argv += ["--max-retries", str(max_retries)]
        for exception_type in retry_include or []:
            argv += ["--retry-include", exception_type]

    if n_tasks is not None:
        argv += ["--n-tasks", str(n_tasks)]
    for task in include_tasks or []:
        argv += ["--include-task-name", task]

    # Terminal-Bench 2 runs its agent phase with a public network policy, so an
    # allowlist is not merely unnecessary -- Harbor warns that it is ignored.
    # Kept behind a flag for datasets that do restrict egress, where both
    # harnesses would otherwise fail to install themselves.
    if allow_hosts:
        hosts = [model.host, *(_fill(registry.get("allow_agent_hosts") or [], subs))]
        for host in dict.fromkeys(hosts):
            argv += ["--allow-agent-host", host]

    # Best-effort verbosity. A harness that failed to *send* a request logs
    # only that sending failed -- Rust's reqwest, which several of these use,
    # prints its top-level message and discards the source, so refused, reset
    # and timed-out are indistinguishable afterwards. These ask its logging
    # layer for the connection detail instead. Harnesses that are not Rust, or
    # do not read these, simply ignore them; nothing here is a guarantee, which
    # is why the watchdog is the part that does not depend on cooperation.
    if debug_capture:
        for key, value in DEBUG_AGENT_ENV.items():
            argv += ["--ae", f"{key}={value}"]

    for key, value in (spec.get("agent_env") or {}).items():
        argv += ["--ae", f"{key}={value}"]

    # The catalog's `version:` is the harness build this run installs. It
    # travels as the agent's `version` kwarg, which is where Harbor's
    # BaseInstalledAgent already keeps it (`self._version`) and what every
    # install step here reads.
    #
    # Unpinned, each install resolves whatever upstream's default branch holds
    # at the moment the trial starts. That is not merely irreproducible across
    # days: a push mid-run changes the harness *between trials of one run*, so
    # its trials stop being measurements of the same thing and the run's pass
    # rate averages two different harnesses. That is not hypothetical -- it is
    # what NousResearch/hermes-agent@6a198f8a1 did on 2026-08-13, turning a
    # warning into a fatal install error 2h42m into a 89-task run: the 31
    # trials before it installed cleanly and 28 of the 33 after it died.
    #
    # An explicit agent_kwargs entry still wins, so a catalog can override the
    # pin per harness without this reaching around it.
    pinned = spec.get("version")
    if pinned and "version" not in (spec.get("agent_kwargs") or {}):
        argv += ["--ak", f"version={pinned}"]

    for key, value in (spec.get("agent_kwargs") or {}).items():
        argv += ["--ak", f"{key}={value}"]
    argv += list(extra_args or [])

    # Some benchmarks call a model of their own, outside the agent under test:
    # tau3-bench simulates the user in its environment and judges assertions in
    # its verifier. Harbor reads a task's [environment].env / [verifier].env
    # from its *own* environment and exits before the first trial when a
    # required one is unset -- which is how a tau3 sweep failed seven for seven
    # with "Missing Environment Variables: OPENAI_API_KEY, OPENAI_BASE_URL".
    #
    # The harness block wins on a collision: it describes the thing being
    # measured, while the dataset block only has to make the benchmark run.
    dataset_env = _fill(dataset_host_env(dataset, registry), subs)
    return argv, {**dataset_env, **dict(spec.get("host_env") or {})}


def write_manifest(
    job_dir: Path,
    *,
    harness_id: str,
    spec: dict[str, Any],
    model: ModelIdentity,
    config: Config,
    argv: list[str],
    dataset: str,
    n_concurrent: int,
    n_concurrent_agents: int | None,
    n_attempts: int,
    n_tasks: int | None,
    include_tasks: list[str] | None,
    agent_timeout_multiplier: float,
    subset: str | None,
    started_at: str,
    batch_id: str = "",
    context_window: int = 0,
    context_source: str = "detected",
    max_retries: int = 0,
    retry_include: list[str] | None = None,
    debug_capture: bool = False,
    # Only needed to resolve the dataset's own host_env, which most datasets do
    # not have. Optional so a caller recording a manifest for a run it already
    # built does not have to carry the catalog along with it.
    registry: dict[str, Any] | None = None,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    # Derived from the same inputs as the command itself rather than passed in,
    # so the manifest cannot drift from what the harness was actually told.
    reasoning_effort, reasoning_source = effective_reasoning_effort(model, config)
    manifest = {
        "schema": 1,
        "harness": harness_id,
        "harness_label": spec.get("label", harness_id),
        # Which sweep this harness belonged to. Absent on runs recorded before
        # sweeps were identified; the dashboard falls back to grouping those by
        # subset, which is what it did for all of them.
        "batch_id": batch_id or None,
        "harness_vendor": spec.get("vendor"),
        "harness_repo": spec.get("repo"),
        # The pinned harness build, or null when the catalog left it floating.
        # Without this a finished run cannot say which harness produced it,
        # which is the difference between "hermes scored 20%" and a number
        # nobody can reproduce or even attribute.
        "harness_version": spec.get("version"),
        "agent_ref": spec.get("agent"),
        "model": model.to_dict(),
        # What the harnesses were told about the model, resolved from the probe.
        # Recorded because a run is only comparable to another that used the
        # same window: it decides when a harness compresses or truncates, and
        # nothing else in the run data would reveal a mismatch.
        "context_window": context_window,
        "context_window_source": context_source,
        # The ceiling this harness actually runs under, not the one the rig
        # computed. Null when the harness takes no such setting -- recording the
        # rig's number there made the manifest claim a cap that was never
        # applied, which is how a capped harness and an uncapped one came to be
        # compared as though they were the same experiment. See output_cap_for.
        "agent_max_tokens": output_cap_for(spec, context_window)[0],
        "agent_max_tokens_source": output_cap_for(spec, context_window)[1],
        # Recorded for the same reason as the window, and it is the stronger
        # case: a harness that reasoned and one that did not are not two
        # measurements of the same thing. Only some harnesses read it, but the
        # run either offered thinking or it did not.
        "reasoning_effort": reasoning_effort,
        "reasoning_effort_source": reasoning_source,
        # Whether that effort actually reached *this* harness. Only some have a
        # knob for it -- on the current catalog codex and dsh do, omp has one
        # this catalog does not fill, and the rest have none -- and the model
        # reasons either way, because it is a reasoning model. So a harness
        # that was told nothing is not one that did not think; it is one whose
        # effort nobody recorded. Without this the manifest described "told
        # high" and "used its own default" with the same two fields.
        "reasoning_effort_applied": uses_placeholder(spec, "reasoning_effort"),
        "dataset": dataset,
        # What the *benchmark's own* machinery was pointed at, when it has any.
        # tau3-bench simulates the user and judges in natural language, so these
        # name the model doing both -- and a run whose simulator and judge are a
        # small local model is not comparable to a published score where both
        # are frontier models. Nothing else on screen would reveal the
        # difference, so it travels with the result. Scrubbed like the command:
        # the endpoint's credential is substituted into these.
        "dataset_env": {
            key: scrub(str(value), config)
            for key, value in sorted(
                _fill(
                    dataset_host_env(dataset, registry or {}),
                    _substitutions(model, config),
                ).items()
            )
        }
        or None,
        "n_concurrent": n_concurrent,
        "n_concurrent_agents": n_concurrent_agents,
        "n_attempts": n_attempts,
        # How many extra attempts a trial got, and for which failures. A run
        # that retried is not the same experiment as one that did not, so this
        # is recorded rather than left implicit in the command line.
        "max_retries": max_retries,
        "retry_include": retry_include or None,
        # Whether endpoint sampling was running. Absent or false means nobody
        # may claim the endpoint was up *or* down for this run -- there is no
        # evidence either way, and that is worth recording explicitly.
        "debug_capture": debug_capture,
        # A subset run is not comparable to a full-dataset run, and the two look
        # identical once they are rows in the same table. Record the restriction
        # so the dashboard can say so instead of quietly ranking them together.
        "n_tasks_requested": n_tasks,
        "include_tasks": include_tasks or None,
        "agent_timeout_multiplier": agent_timeout_multiplier,
        "subset": subset,
        "is_partial": bool(n_tasks is not None or include_tasks),
        "harbor_version": harbor_version(),
        "orchestrator": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        # The command, with credentials removed. host_env is deliberately absent
        # entirely, and any API key substituted into an --ak/--ae value is
        # scrubbed here: a manifest is what the dashboard reads and what `export`
        # inlines into a shareable snapshot, so it must be safe to publish.
        "command": [scrub(arg, config) for arg in argv],
        "started_at": started_at,
    }
    # Opt-in, because manifests get shared and a hostname identifies a machine.
    if config.record_hostname:
        manifest["orchestrator"]["hostname"] = platform.node()
    (job_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run_one(
    harness_id: str,
    registry: dict[str, Any],
    model: ModelIdentity,
    config: Config,
    *,
    jobs_dir: Path,
    dataset: str,
    n_concurrent: int,
    n_concurrent_agents: int | None,
    env_build_timeout_multiplier: float | None,
    n_attempts: int,
    n_tasks: int | None,
    include_tasks: list[str] | None,
    extra_args: list[str] | None,
    allow_hosts: bool,
    agent_timeout_multiplier: float,
    subset: str | None,
    dry_run: bool,
    batch_id: str = "",
    max_retries: int = 0,
    retry_include: list[str] | None = None,
    debug_capture: bool = False,
) -> int:
    spec = registry["harnesses"][harness_id]
    run_window, run_window_source = effective_context(model, config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = job_name(
        harness_id,
        model,
        stamp,
        # Resolved from the catalog already in hand rather than by re-reading
        # it, so the name cannot describe a different dataset than the argv.
        dataset_slug=registry_mod.dataset_slug(dataset, registry),
        scope=scope_name(subset, n_tasks, include_tasks),
    )
    job_dir = jobs_dir / name

    argv, host_env = build_command(
        harness_id,
        spec,
        model,
        registry,
        config,
        jobs_dir=jobs_dir,
        name=name,
        dataset=dataset,
        n_concurrent=n_concurrent,
        n_attempts=n_attempts,
        n_tasks=n_tasks,
        include_tasks=include_tasks,
        extra_args=extra_args,
        allow_hosts=allow_hosts,
        agent_timeout_multiplier=agent_timeout_multiplier,
        n_concurrent_agents=n_concurrent_agents,
        env_build_timeout_multiplier=env_build_timeout_multiplier,
        max_retries=max_retries,
        retry_include=retry_include,
        debug_capture=debug_capture,
    )

    env = os.environ.copy()
    env.update(host_env)
    # Custom agents live in harnesses/, importable only if the project root is
    # on the path of the harbor process.
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)

    # Read every file as UTF-8 regardless of the machine's locale.
    #
    # Agent transcripts contain whatever the model emitted, and Path.read_text()
    # with no encoding= uses the *locale* encoding -- cp1252 on a stock Windows
    # install. Only five byte values (0x81 0x8d 0x8f 0x90 0x9d) are undefined
    # there, so the vast majority of non-ASCII output decodes to mojibake in
    # silence and the rest raises UnicodeDecodeError. Harbor reads the hermes
    # session that way, after the agent has already done its work, so a model
    # that wandered into CJK once cost the trial during token accounting -- and
    # scored as an error rather than as whatever the agent actually achieved.
    #
    # Set on the harbor process rather than patched in one adapter: the same
    # unqualified read appears throughout an installed dependency, and which
    # ones fire depends on what the model typed. UTF-8 mode makes the whole
    # subprocess locale-independent, which is the property we actually want.
    # No effect where the locale is already UTF-8, i.e. Linux and macOS.
    env["PYTHONUTF8"] = "1"

    print(f"\n{'=' * 78}")
    print(f"  {spec.get('label', harness_id)}  x  {model.label}")
    print(f"  job: {name}")
    print(f"{'=' * 78}")
    # Scrubbed: this line gets pasted into issues and screen shares.
    printable = " ".join(
        arg if " " not in arg else f'"{arg}"' for arg in (scrub(a, config) for a in argv)
    )
    print(f"  $ {printable}\n")

    # Both before the dry-run exit: a dry run exists to show what a real one
    # would do, and "this run will write files it cannot reopen" is exactly the
    # kind of thing worth learning without spending four hours first.
    warning = check_path_budget(jobs_dir, name, include_tasks or [], harness_id)
    if warning:
        print(warning)
    warn_reasoning_under_cap(
        harness_id, spec, run_window, effective_reasoning_effort(model, config)[0]
    )

    if dry_run:
        return 0

    write_manifest(
        job_dir,
        harness_id=harness_id,
        spec=spec,
        model=model,
        config=config,
        argv=argv,
        dataset=dataset,
        registry=registry,
        n_concurrent=n_concurrent,
        n_concurrent_agents=n_concurrent_agents,
        n_attempts=n_attempts,
        n_tasks=n_tasks,
        include_tasks=include_tasks,
        agent_timeout_multiplier=agent_timeout_multiplier,
        subset=subset,
        started_at=datetime.now(UTC).isoformat(),
        batch_id=batch_id,
        context_window=run_window,
        context_source=run_window_source,
        max_retries=max_retries,
        retry_include=retry_include,
        debug_capture=debug_capture,
    )

    # Ground truth for "was the endpoint actually up?". Without it, a transport
    # failure inside a harness is unattributable -- and the tempting answer,
    # blaming the endpoint, is the one that hides a bug in our own code.
    watchdog = None
    if debug_capture:
        watchdog = EndpointWatchdog(
            health_url(model.base_url),
            job_dir / HEALTH_FILENAME,
            api_key=config.endpoint.resolve_api_key(),
        )
        watchdog.start()
        print(f"  diagnostics on: sampling the endpoint into {HEALTH_FILENAME}")

    try:
        result = subprocess.run(argv, env=env, cwd=str(WORKSPACE), check=False)
    finally:
        if watchdog is not None:
            watchdog.stop()
    if result.returncode != 0:
        print(f"\n  [!] {harness_id} exited {result.returncode}", file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    registry = load_registry()
    defaults = registry.get("defaults") or {}
    known = list(registry["harnesses"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        action="append",
        choices=known,
        help=f"Harness to run; repeatable. Default: all ({', '.join(known)})",
    )
    add_endpoint_args(parser)
    parser.add_argument("--dataset", default=defaults.get("dataset"))
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=defaults.get("n_concurrent", 1),
        help="Trials in flight at once. Their setup/verify phases overlap; the "
        "agent phase is capped separately by --n-concurrent-agents.",
    )
    parser.add_argument(
        "--n-concurrent-agents",
        type=int,
        default=defaults.get("n_concurrent_agents"),
        help="Cap on concurrent agent (LLM) phases. Keep at the endpoint's slot "
        "count so trials never queue on the model server.",
    )
    parser.add_argument(
        "--env-build-timeout-multiplier",
        type=float,
        default=defaults.get("environment_build_timeout_multiplier"),
        help="Scale the environment start/build budget. Large task images can "
        "exceed the 600s default and lose the task outright.",
    )
    parser.add_argument("--n-attempts", type=int, default=defaults.get("n_attempts", 1))
    parser.add_argument(
        "--max-retries",
        type=int,
        default=defaults.get("max_retries", 1),
        help="Extra attempts for a trial that died on an infrastructure failure "
        "(see --retry-include). 0 disables retrying.",
    )
    parser.add_argument(
        "--retry-include",
        action="append",
        default=None,
        help="Exception type to retry on; repeatable. Defaults to the types a "
        "dropped endpoint connection surfaces as.",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=None, help="Smoke-test with N tasks"
    )
    parser.add_argument(
        "--task", action="append", default=None, help="Run only this task; repeatable"
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Named task list from bench/subsets/<name>.txt. Every harness runs "
        "the identical set, which is what keeps the comparison valid.",
    )
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=defaults.get("agent_timeout_multiplier", 1.0),
        help="Scale each task's agent time budget. Terminal-Bench budgets assume "
        "frontier-API speed; a large local model often needs 2-4x. 1.0 keeps "
        "results leaderboard-comparable.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help="Where Harbor job directories are written. Default: runs_dir from config.",
    )
    parser.add_argument(
        "--allow-hosts",
        action="store_true",
        help="Send the registry's egress allowlist. Only needed for datasets "
        "whose agent phase restricts network access.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--debug-capture",
        action="store_true",
        help="Sample the endpoint for the life of the run, and ask harnesses "
        "for connection-level logs where they support it, so a transport "
        "failure can be attributed afterwards instead of guessed at.",
    )
    parser.add_argument("--no-input", action="store_true")
    parser.add_argument(
        "--skip-wire-check",
        action="store_true",
        help="Do not ask the endpoint whether it accepts the request shapes the "
        "selected harnesses send. The check costs two one-token requests and "
        "only ever drops a harness whose refusal it recognises, so this exists "
        "for the case where the probe itself is the thing misbehaving.",
    )
    parser.add_argument(
        "harbor_args",
        nargs="*",
        help="Extra args forwarded to `harbor run` verbatim (put after --)",
    )
    args = parser.parse_args(argv)

    # An unset or unreachable endpoint is the first thing anyone hits, and it is
    # a question about configuration rather than a defect. `doctor` has always
    # answered it in a sentence; a run answered it with a traceback, which buries
    # the one line that says what to do in twenty that do not.
    try:
        config = config_from_args(args)
        jobs_dir = args.jobs_dir or config.resolved_runs_dir()
        model = resolve(config.endpoint, interactive=not args.no_input)
    except (ConfigError, RuntimeError) as exc:
        raise SystemExit(
            f"\n{exc}\n\n"
            f"Set the endpoint on the dashboard's Setup tab, or run "
            f"`harness-arena init`.\n"
            f"`harness-arena doctor` checks the whole chain."
        ) from None
    print(f"\nModel under test: {model.label}")
    print(describe(model))

    n_concurrent = args.n_concurrent
    # The agent cap, not the trial cap, is what must match the server. Default it
    # to the endpoint's slot count so the LLM never sees queued requests, however
    # many trials are open. A hosted provider reports no slots, so fall back to
    # the provider's own default rather than needlessly serializing.
    n_concurrent_agents = (
        args.n_concurrent_agents
        or model.total_slots
        or config.endpoint.resolved_provider().default_agent_concurrency
    )
    n_concurrent_agents = min(n_concurrent_agents, n_concurrent)
    window, window_source = effective_context(model, config)
    print(
        f"\n  context window: {window:,} ({window_source}), "
        f"max {agent_max_tokens_for(window):,} output tokens per response"
    )
    if window_source == "fallback":
        print(
            "  [!] The server did not report a context window and none is "
            "configured, so a conservative default is in use. Set it on the "
            "Setup tab (or endpoint.context_window) to match your server -- "
            "overshooting truncates silently and scores as a wrong answer."
        )

    slots = (
        f"endpoint reports {model.total_slots} slot(s)"
        if model.total_slots is not None
        else "hosted endpoint, no local slots"
    )
    print(
        f"\n  {n_concurrent} trials in flight, {n_concurrent_agents} generating at once "
        f"({slots})."
    )
    # Only meaningful for a server with a fixed slot count. Overshooting it means
    # requests queue at the server, which inflates per-request latency and falls
    # hardest on whichever harness makes more calls -- the variable under test.
    if model.total_slots is not None and n_concurrent_agents > model.total_slots:
        print(
            "  [!] More concurrent agents than server slots: requests will queue "
            "and per-request latency will rise. Restart the server with matching "
            "parallel slots first (llama-server: -np)."
        )

    include_tasks = list(args.task or [])
    if args.subset:
        # Before the images are pulled, not after: a subset from another
        # benchmark selects nothing, and a run that selected nothing is
        # indistinguishable from one that finished.
        check_subset_dataset(args.subset, args.dataset)
        subset_tasks = load_subset(args.subset)
        include_tasks = subset_tasks + [t for t in include_tasks if t not in subset_tasks]
        print(f"\n  subset '{args.subset}': {len(subset_tasks)} tasks")

    harnesses = args.harness or known

    # Said once for the sweep, beside the ceiling that was just announced --
    # otherwise that line reads as a property of every run about to start, and
    # for at least one of them it is not true.
    report_output_caps(registry, harnesses, window)

    # Reported once the selection is known, because only some harnesses send an
    # effort at all. The fallback deserves a warning even though it is the old
    # behaviour: it is safe exactly where it is not needed. On a server that
    # accepts an effort, falling back to the default is what would have
    # happened anyway; on one that refuses, the same fallback answers 400 on
    # the first request of every trial and burns the whole run.
    if any(
        uses_placeholder(registry["harnesses"].get(harness_id) or {}, "reasoning_effort")
        for harness_id in harnesses
    ):
        effort, effort_source = effective_reasoning_effort(model, config)
        print(f"\n  reasoning effort: {effort} ({effort_source})")
        if effort_source == "fallback":
            print(
                "  [!] The endpoint could not be asked whether it accepts a "
                "reasoning effort, so the harness default is in use -- "
                "unchanged from before this was probed. If the server refuses "
                "one, every trial dies at its first request with a bare "
                "non-zero exit code and no tokens. Set endpoint.reasoning_"
                "effort=none if that happens."
            )
        elif effort not in KNOWN_REASONING_EFFORTS:
            print(
                f"  [!] {effort!r} is not an effort any server here has been "
                "measured to accept. Passing it through as configured -- if it "
                "is rejected, every trial fails at its first request."
            )

    # Before the batch id, the run marker and the first container: a harness
    # this endpoint will not serve produces a job directory full of identical
    # first-request failures, and the cheapest moment to find that out is now.
    # A dry run is asking what *would* happen, and dropping a harness would
    # answer a different question, so it is left alone.
    requested = list(harnesses)
    if not args.dry_run:
        harnesses = preflight_wire_shapes(
            config, registry, harnesses, model, enabled=not args.skip_wire_check
        )
    skipped = [h for h in requested if h not in harnesses]
    if not harnesses:
        return 1

    # One id for this whole invocation, shared by every harness it runs, so a
    # sweep can be recognised as one thing afterwards rather than inferred from
    # the subset name -- which stops being a sweep the second time you use it.
    batch_id = new_batch_id(datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    failures = 0

    # Claim this machine's Harbor containers for the duration, so anything
    # looking at them from another process -- a dashboard open beside this
    # terminal -- can tell a live trial from a leftover. A dry run starts no
    # containers and must not claim any. Cleared in `finally` so an ordinary
    # Ctrl-C leaves nothing behind; a hard kill is covered by the liveness
    # check on the marker rather than by trusting this to run.
    marker_dir = jobs_dir if not args.dry_run else None
    if marker_dir:
        write_run_marker(marker_dir, harnesses)
    try:
        for harness_id in harnesses:
            failures += bool(
                run_one(
                    harness_id,
                    registry,
                    model,
                    config,
                    jobs_dir=jobs_dir,
                    dataset=args.dataset,
                    n_concurrent=n_concurrent,
                    n_concurrent_agents=n_concurrent_agents,
                    env_build_timeout_multiplier=args.env_build_timeout_multiplier,
                    n_attempts=args.n_attempts,
                    debug_capture=args.debug_capture,
                    max_retries=args.max_retries,
                    retry_include=args.retry_include or RETRY_INCLUDE_DEFAULT,
                    n_tasks=args.n_tasks,
                    include_tasks=include_tasks or None,
                    extra_args=args.harbor_args,
                    allow_hosts=args.allow_hosts,
                    agent_timeout_multiplier=args.agent_timeout_multiplier,
                    subset=args.subset,
                    dry_run=args.dry_run,
                    batch_id=batch_id,
                )
            )
    finally:
        if marker_dir:
            clear_run_marker(marker_dir)

    print(f"\nDone. {len(harnesses) - failures}/{len(harnesses)} harness runs succeeded.")
    if skipped:
        # Named again at the end, and counted as a failure of the invocation.
        # A sweep that quietly ran six of the seven harnesses you asked for
        # reads afterwards as a sweep of six, and the missing row is indis-
        # tinguishable from one you never requested.
        print(f"Not run: {', '.join(sorted(skipped))} "
              f"-- this endpoint refuses a request shape they send.")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
